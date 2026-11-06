"""Performance instrumentation used to keep ForgeMind's hot path free of accidental stalls.

The expensive mistakes in a training loop are almost never the FLOPs -- they are the little
`float(loss)`, `bool(tensor)`, `.item()` and `.tolist()` calls that force the CPU to drain the
GPU queue.  Every one of them costs a full pipeline flush (tens of microseconds, and it also
serialises host and device work so the next kernels cannot be enqueued ahead of time).

`host_syncs()` counts those calls so they can be asserted on in CI:

    from forgemind.perf import count_host_syncs

    with count_host_syncs() as syncs:
        trainer.train_step()
    assert syncs.count <= 1          # one scalar read per optimizer step is the design budget

It also hosts the small device-side helpers the trainer uses to get down to that budget.
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
from typing import List

import torch

__all__ = ["count_host_syncs", "host_syncs", "HostSyncCounter", "warn_once", "enable_fast_math", "pinned_pool"]

# --------------------------------------------------------------------------------------------- sync auditing
_PATCHED = False
_COUNTERS: List["HostSyncCounter"] = []
_TOTAL = {"n": 0}
_LOCK = threading.RLock()

_SYNC_METHODS = ("item", "tolist", "numpy", "__float__", "__int__", "__bool__", "cpu")
_ORIGINALS: dict = {}


class HostSyncCounter:
    """Counts host<->device synchronising method calls and remembers where they came from."""

    def __init__(self, capture_stack: bool = True, stack_limit: int = 4):
        self.count = 0        # every read of an accelerator-resident scalar
        self.own = 0          # ...that originated in forgemind code (what this library controls)
        self.external = 0     # ...that originated in torch itself (optimizer/checkpoint internals)
        self.capture_stack = capture_stack
        self.stack_limit = stack_limit
        self.sites: dict[str, int] = {}

    def _hit(self, name: str) -> None:
        self.count += 1
        frame = sys._getframe(2)
        own = frame.f_globals.get("__name__", "").startswith("forgemind")
        if own:
            self.own += 1
        else:
            self.external += 1
        if self.capture_stack:
            site = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno} {frame.f_code.co_name}"
            self.sites[site] = self.sites.get(site, 0) + 1

    def report(self) -> str:
        rows = sorted(self.sites.items(), key=lambda kv: -kv[1])[: self.stack_limit]
        head = f"{self.count} host sync(s) ({self.own} in forgemind, {self.external} inside torch)"
        return head if not rows else head + "\n" + "\n".join(f"  {n:>4}x {s}" for s, n in rows)


def _patch() -> None:
    global _PATCHED
    with _LOCK:
        if _PATCHED:
            return
        for name in _SYNC_METHODS:
            orig = getattr(torch.Tensor, name, None)
            if orig is None:
                continue
            _ORIGINALS[name] = orig

            def make(name=name, orig=orig):
                def wrapper(self, *args, **kwargs):
                    if self.is_cuda or _TOTAL.get("force", False):  # .item() on a CPU tensor never stalls
                        _TOTAL["n"] += 1
                        for c in list(_COUNTERS):
                            c._hit(name)
                    return orig(self, *args, **kwargs)
                wrapper._fm_sync = True
                return wrapper

            setattr(torch.Tensor, name, make())
        _TOTAL["force"] = False
        _PATCHED = True


def _unpatch() -> None:
    global _PATCHED
    with _LOCK:
        if not _PATCHED:
            return
        for name, orig in _ORIGINALS.items():
            setattr(torch.Tensor, name, orig)
        _ORIGINALS.clear()
        _PATCHED = False


@contextlib.contextmanager
def count_host_syncs(capture_stack: bool = True, count_cpu: bool = False):
    """Context manager yielding a :class:`HostSyncCounter`.

    On CPU-only builds nothing can actually stall, so purely-CPU tensor reads are ignored unless
    ``count_cpu=True`` -- the *number and location* of host reads is what we are auditing.
    """
    _patch()
    prev_force = _TOTAL.get("force", False)
    _TOTAL["force"] = bool(count_cpu)
    counter = HostSyncCounter(capture_stack=capture_stack)
    with _LOCK:
        _COUNTERS.append(counter)
    try:
        yield counter
    finally:
        with _LOCK:
            if counter in _COUNTERS:
                _COUNTERS.remove(counter)
        _TOTAL["force"] = prev_force
        if not _COUNTERS:
            _unpatch()


def host_syncs() -> int:
    """Total number of host-synchronising tensor reads since the process started (best effort)."""
    if not _PATCHED:
        return 0
    return _TOTAL["n"]


# --------------------------------------------------------------------------------------------- misc
_WARNED: set[str] = set()
_WARN_LOCK = threading.Lock()


def warn_once(msg: str, key: str | None = None) -> None:
    """Warn at most once per key -- safe to call from the batch-prefetch thread as well."""
    key = key or msg
    with _WARN_LOCK:
        if key in _WARNED:
            return
        _WARNED.add(key)
    import warnings
    warnings.warn(msg, stacklevel=3)


def enable_fast_math(enabled: bool = True) -> None:
    """bf16/tf32 friendly matmul settings; a no-op on CPU builds."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    with contextlib.suppress(Exception):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    if enabled:
        # let the fp32 matmul path use tf32 for the pieces that are not autocast (norm/loss math)
        with contextlib.suppress(Exception):
            torch.set_float32_matmul_precision("high")


class _PinnedPool:
    """Size-keyed free list of pinned host buffers (avoids one cudaHostAlloc per saved activation).

    Buffers are only handed back to the pool when the consumer is done with them, so a buffer that
    is still being read by an in-flight H2D copy can never be reused early (single-stream ordering).
    """

    def __init__(self, pin: bool = True):
        self.pin = pin
        self._free: dict[tuple, list] = {}
        self.hits = 0
        self.misses = 0

    def get(self, shape, dtype) -> torch.Tensor:
        key = (tuple(shape), dtype)
        bucket = self._free.get(key)
        if bucket:
            self.hits += 1
            return bucket.pop()
        self.misses += 1
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=self.pin)

    def put(self, buf: torch.Tensor) -> None:
        self._free.setdefault((tuple(buf.shape), buf.dtype), []).append(buf)

    def clear(self) -> None:
        self._free.clear()


def pinned_pool(pin: bool = True) -> _PinnedPool:
    return _PinnedPool(pin)
