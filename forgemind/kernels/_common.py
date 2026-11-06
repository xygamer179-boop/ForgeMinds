"""Shared plumbing for the packed kernels.

Every kernel in this package is *optional*: it needs Triton and an accelerator, and every
wrapper falls back to an exactly-equivalent torch implementation when either is missing.
That is what makes the package usable on ordinary torch code -- you can import it, call it on
CPU, and get the same numbers, just slower.

Three things worth knowing
--------------------------
``TRITON_INTERPRET=1`` makes Triton run kernels on CPU through a numpy emulator. The test
suite uses that to check the *real* kernels numerically on a machine with no GPU
(``tests/test_kernels.py``), instead of only checking the fallback.

``set_backend("fallback")`` forces the torch path even on a GPU -- useful for A/B and for
proving that the fast path is numerically transparent.

``set_backend("triton")`` forces the kernel path and drops the ``is_cuda`` check. Combined
with ``TRITON_INTERPRET=1`` that is how the kernels are tested here.
"""

from __future__ import annotations

import functools
import os

import torch

try:  # Triton ships with the CUDA wheels of torch, but not with every build.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - exercised only on triton-less installs
    triton = None

    class _TLShim:  # keeps module-level annotations importable without triton
        constexpr = "tl.constexpr"

    tl = _TLShim()
    HAS_TRITON = False


INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"
_BACKEND = "auto"
# kernels are compiled for a (constexpr) row width; wider rows fall back to torch
MAX_DIM = 8192
_STATS = {"fast": 0, "fallback": 0, "reason": {}}


def set_backend(name: str) -> None:
    """``"auto"`` (default), ``"fallback"`` (force torch), ``"triton"`` (force kernels)."""
    global _BACKEND
    if name not in ("auto", "fallback", "triton"):
        raise ValueError(f"unknown kernel backend {name!r}; use auto|fallback|triton")
    _BACKEND = name


def get_backend() -> str:
    return _BACKEND


def stats() -> dict:
    """How many calls took the kernel path vs the torch path, and why."""
    return {"fast": _STATS["fast"], "fallback": _STATS["fallback"], "reasons": dict(_STATS["reason"])}


def reset_stats() -> None:
    _STATS.update(fast=0, fallback=0, reason={})


def triton_available() -> bool:
    return HAS_TRITON and not (triton is None)


def tracing() -> bool:
    """True while ``torch.compile`` is tracing this code."""
    try:
        return torch.compiler.is_compiling()
    except Exception:  # pragma: no cover - torch without the API
        return False


def _note(why: str) -> None:
    """Record *why* the torch path was taken -- but never mutate state while tracing.

    This counter used to run unconditionally, and that single dict update is enough to make
    Dynamo give up when the call happens inside a checkpointed region: activation checkpointing
    re-runs the region in the backward, and Dynamo will not trace a side effect it would have to
    apply twice. Measured cost of not having this guard: ``ac=selective + fused_ops`` went from
    1 graph / 0 breaks to 2 graphs / 1 break (PERF.md 9.2). Diagnostics are not worth a graph.
    """
    if tracing():
        return
    _STATS["fallback"] += 1
    _STATS["reason"][why] = _STATS["reason"].get(why, 0) + 1


def fast_ok(*tensors) -> bool:
    """Can the kernels run on these tensors right now?"""
    if _BACKEND == "fallback":
        _note("backend=fallback")
        return False
    if not triton_available():
        _note("no triton")
        return False
    if _BACKEND != "triton":
        for t in tensors:
            if not isinstance(t, torch.Tensor):
                continue
            if not (t.is_cuda or INTERPRET):
                _note("not cuda")
                return False
            if t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                _note(f"dtype {t.dtype}")
                return False
    if not tracing():
        _STATS["fast"] += 1
    return True


def next_pow2(n: int) -> int:
    n = max(int(n), 1)
    return 1 << (n - 1).bit_length()


@functools.lru_cache(maxsize=None)
def sm_count() -> int:
    """Streaming multiprocessors, or a sane default when there is no device."""
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            return int(getattr(props, "multi_processor_count", 108) or 108)
    except Exception:
        pass
    return 108


def nprog_for(n_rows: int, per_sm: int = 4) -> int:
    """Number of programs for a column-parallel reduction kernel.

    Partial weight gradients are `nprog x d` fp32, so this also bounds the scratch size.
    """
    return max(1, min(int(n_rows), per_sm * sm_count()))


def num_warps_for(block_elems: int, min_warps: int = 1, max_warps: int = 16) -> int:
    """Enough warps to keep a block of `block_elems` fp32 values in flight, no more."""
    elems_per_thread = 4  # one 128-bit vector each
    warps = max(1, (block_elems // 32 // elems_per_thread))
    while warps > max_warps:
        warps //= 2
    while warps < min_warps:
        warps *= 2
    return warps


def block_m_for(block_d: int, target: int = 4096) -> int:
    """Rows per program so that one tile is ~`target` elements (>=1, power of two)."""
    return max(1, next_pow2(max(1, target // max(block_d, 1))))
