"""Per-kernel A/B/C: eager torch, ``torch.compile`` (inductor), and the ForgeMind kernels.

``bench.py`` measures a whole training step; this measures **one op at a time**, because that is
the only honest way to answer the question the kernels exist for: *is the packed kernel actually
faster than what inductor generates for the same expression, on this GPU, at this shape?*

    python -m forgemind.cli kbench                          # every kernel, bf16, LLM-ish shapes
    python -m forgemind.cli kbench --ops rms_norm,swiglu     # just those
    python -m forgemind.cli kbench --mode fwd                # forward only (no backward)
    python -m forgemind.cli kbench --check                   # exit 3 if a kernel loses to inductor
    python -m forgemind.cli kbench --check --tolerance 0.05  # ...allow 5% slower
    python -m forgemind.cli kbench --json kernels.json

Each row is min-of-N wall time (CUDA events on a GPU, ``perf_counter`` otherwise) after warmup,
including a ``forgemind+compile`` column: the same kernel reached through ``torch.compile``,
which is how it is used inside a compiled model.

Without CUDA the kernels fall back to torch, so every column measures the same torch expression
and the numbers mean nothing -- ``--check`` therefore refuses to run on CPU unless you pass
``--allow-cpu`` (useful for exercising this tool, not for drawing conclusions).
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from ..kernels import _fallback as fb
from ..kernels.ns import _ns_torch
from .. import kernels

DEFAULT_OPS = ["rms_norm", "add_rms_norm", "swiglu", "swiglu_split", "qk_norm_rope",
               "kl_div_logits", "newton_schulz", "topk", "softmax"]


# --------------------------------------------------------------------------- workloads
class Case:
    """One op at one shape: an eager reference, the kernel, and how to differentiate it."""

    def __init__(self, name, shapes, fn, backward=True):
        self.name = name
        self.shapes = shapes
        self.fn = fn                 # fn(tensors, packed: bool) -> output(s)
        self.backward = backward


def _scalar(out):
    """The loss to differentiate: sum every output (ops may return tuples)."""
    if isinstance(out, (tuple, list)):
        return sum(o.float().sum() for o in out)
    return out.float().sum()


def _cases(op, tokens, dim, hidden, batch, seq, heads, head_dim, vocab, dtype):
    """Build the eager reference and the packed call for one op (no tensors allocated here:
    ``_case_tensors`` owns the data, so the two paths always see identical inputs)."""
    if op == "rms_norm":
        return [Case("rms_norm", f"[{tokens},{dim}] {dtype}",
                     lambda t, k: (kernels.rms_norm if k else fb.rms_norm)(t[0], t[1], 1e-6))]

    if op == "add_rms_norm":
        def f(t, k):
            if k:
                return kernels.add_rms_norm(t[0], t[1], t[2], 1e-6)
            return fb.add_rms_norm(t[0], t[1], t[2], 1e-6)
        return [Case("add_rms_norm", f"x,delta [{tokens},{dim}] {dtype}", f)]

    if op == "swiglu":
        return [Case("swiglu", f"gate,up [{tokens},{hidden}] {dtype}",
                     lambda t, k: (kernels.swiglu if k else fb.swiglu)(t[0], t[1]))]

    if op == "swiglu_split":
        def f(t, k):
            if k:
                return kernels.swiglu_split(t[0])
            h = t[0].shape[-1] // 2
            return (torch.nn.functional.silu(t[0][..., :h].float())
                    * t[0][..., h:].float()).to(t[0].dtype)
        return [Case("swiglu_split", f"[{tokens},{2 * hidden}] {dtype}", f)]

    if op == "qk_norm_rope":
        def f(t, k):
            if k:
                return kernels.qk_norm_rope(t[0], t[1], t[2], t[3], 1e-6)
            return fb.qk_norm_rope(t[0], t[1], t[2].view(1, seq, 1, -1),
                                    t[3].view(1, seq, 1, -1), 1e-6)
        return [Case("qk_norm_rope",
                     f"[{batch},{seq},{heads},{head_dim}] {dtype}, {seq} RoPE rows", f)]

    if op == "kl_div_logits":
        def f(t, k):
            if k:
                return kernels.kl_div_logits(t[0], t[1])
            return fb.kl_div_logits(t[0], t[1])[0]
        return [Case("kl_div_logits", f"[{batch * 64},{vocab}] {dtype}", f, backward=False)]

    if op == "newton_schulz":
        # no gradient: Muon calls NS inside optimizer.step under no_grad, so fwd-only rows
        # are the honest measurement; packed = fused-Triton elementwise, eager = pure torch.
        def f(t, k):
            if k:
                return kernels.newton_schulz(t[0])
            return _ns_torch(t[0], 5, None)
        return [Case("newton_schulz", f"[{dim},{dim}] {dtype}", f, backward=False)]

    if op == "topk":
        # fused selection vs torch.topk; backward flows to the values (scatter), so time both
        k_ = min(64, vocab)
        return [Case("topk", f"[{batch * 64},{vocab}] {dtype} k={k_}",
                     lambda t, k: (kernels.topk if k else torch.topk)(t[0], k_))]
    if op == "softmax":
        return [Case("softmax", f"[{tokens},{dim}] {dtype}",
                     lambda t, k: (kernels.softmax if k else torch.softmax)(t[0], -1))]

    raise ValueError(f"unknown op {op!r}, choose from {DEFAULT_OPS}")


# --------------------------------------------------------------------------- timing
def _timer(dev):
    """min-of-N timing: CUDA events on a GPU (no host sync inside the loop), wall clock else."""
    if dev.type == "cuda":  # pragma: no cover - needs CUDA
        def measure(fn, iters, warmup):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(iters):
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                best = min(best, start.elapsed_time(end))
            return best
    else:
        def measure(fn, iters, warmup):
            for _ in range(warmup):
                fn()
            best = float("inf")
            for _ in range(iters):
                t0 = time.perf_counter()
                fn()
                best = min(best, (time.perf_counter() - t0) * 1e3)
            return best
    return measure


def _step(case, tensors, packed):
    """Forward (+ backward when the case has one), with fresh grads each call."""
    ins = [t.detach().clone().requires_grad_(t.is_floating_point() and case.backward)
           for t in tensors]
    out = case.fn(ins, packed)
    if not case.backward:
        return
    _scalar(out).backward()


def _bench(case, tensors, packed, measure, iters, warmup):
    def fn():
        _step(case, tensors, packed)
    return measure(fn, iters, warmup)


def _compiled(case, packed):
    """``torch.compile`` around the op, returning a callable that produces the loss."""
    def core(t):
        return _scalar(case.fn(t, packed))
    return torch.compile(core, fullgraph=True)


def parse(argv=None):
    p = argparse.ArgumentParser(prog="forgemind-kbench",
                               description="Per-kernel benchmark: eager vs inductor vs kernels")
    p.add_argument("--ops", default=",".join(DEFAULT_OPS),
                   help=f"comma separated, from {DEFAULT_OPS}")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None, help="cpu|cuda|cuda:1 (default: cuda if available)")
    p.add_argument("--tokens", type=int, default=8192, help="rows of the norm/MLP activations")
    p.add_argument("--dim", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=11008)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--mode", default="both", choices=["fwd", "bwd", "both"],
                   help="both = forward+backward (what training pays)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--baseline", default="best", choices=["eager", "inductor", "best"],
                   help="what --check compares the kernel against")
    p.add_argument("--tolerance", type=float, default=0.0,
                   help="allowed slowdown vs the baseline before --check fails (0.05 means 5%%)")
    p.add_argument("--check", action="store_true",
                   help="exit 3 if a kernel is slower than the baseline (CI/hardware gate)")
    p.add_argument("--allow-cpu", action="store_true",
                   help="run the check on CPU, where kernels fall back and numbers are moot")
    p.add_argument("--no-compile", action="store_true",
                   help="skip the torch.compile columns (fast smoke run, no C++ toolchain)")
    p.add_argument("--json", default=None, help="write the raw measurements here")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = getattr(torch, a.dtype)
    cpu = dev.type != "cuda"
    if cpu and a.check and not a.allow_cpu:
        print("[kbench] refusing --check on CPU: without CUDA the kernels fall back to torch, so "
              "every column measures the same expression. Re-run on the target GPU (or pass "
              "--allow-cpu to exercise the tool itself).")
        return 2

    measure = _timer(dev)
    backward = a.mode in ("bwd", "both")
    rows, worst = [], None
    print(f"[kbench] device={dev} dtype={a.dtype} mode={a.mode} "
          f"iters={a.iters} kernels={'triton' if not cpu else 'torch fallback'}")
    for op in [o.strip() for o in a.ops.split(",") if o.strip()]:
        for case in _cases(op, a.tokens, a.dim, a.hidden, a.batch, a.seq, a.heads, a.head_dim,
                           a.vocab, a.dtype):
            case.backward = backward and case.backward
            tensors = _case_tensors(case, op, a, dev, dtype)
            ms_eager = _bench(case, tensors, False, measure, a.iters, a.warmup)
            ms_kernel = _bench(case, tensors, True, measure, a.iters, a.warmup)
            try:
                if a.no_compile:
                    raise RuntimeError("--no-compile")
                ms_ind = _bench_case_fn(case, tensors, _compiled(case, False), measure,
                                        a.iters, a.warmup)
                ms_kernel_c = _bench_case_fn(case, tensors, _compiled(case, True), measure,
                                             a.iters, a.warmup)
            except Exception as exc:  # pragma: no cover - inductor unavailable/CPU
                ms_ind = ms_kernel_c = float("nan")
                if not a.no_compile:
                    print(f"  ({case.name}: torch.compile unavailable: "
                          f"{type(exc).__name__}: {str(exc)[:120]})")
            rows.append(dict(op=case.name, shapes=case.shapes, eager=ms_eager, inductor=ms_ind,
                             kernel=ms_kernel, kernel_compiled=ms_kernel_c))
            base = {"eager": ms_eager, "inductor": ms_ind,
                    "best": _nanmin(ms_eager, ms_ind)}[a.baseline]
            speed = base / ms_kernel if ms_kernel else float("inf")
            flag = ""
            if a.check and base == base and ms_kernel > base * (1.0 + a.tolerance):
                flag = "  <-- SLOWER"
                if worst is None or speed < worst[1]:
                    worst = (case.name, speed)
            print(f"  {case.name:<14} {case.shapes:<44} "
                  f"eager {ms_eager:8.3f}  inductor {ms_ind:8.3f}  "
                  f"kernel {ms_kernel:8.3f}  kernel+compile {ms_kernel_c:8.3f} ms  "
                  f"({speed:5.2f}x vs {a.baseline}){flag}")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"[kbench] wrote {a.json}")

    if a.check:
        if worst is not None:
            print(f"[kbench] FAIL: {worst[0]} is {worst[1]:.2f}x vs {a.baseline} "
                  f"(tolerance {a.tolerance:.0%})")
            return 3
        print(f"[kbench] OK: every kernel is within {a.tolerance:.0%} of the {a.baseline} baseline")
    return 0


def _nanmin(a, b):
    if a != a:
        return b
    if b != b:
        return a
    return min(a, b)


def _case_tensors(case, op, a, dev, dtype):
    """The input tensors a case needs, in the order its ``f``/``packed`` take them."""
    if op in ("rms_norm", "add_rms_norm"):
        x = torch.randn(a.tokens, a.dim, device=dev, dtype=dtype)
        w = torch.randn(a.dim, device=dev, dtype=dtype)
        return [x, x.clone(), w] if op == "add_rms_norm" else [x, w]
    if op == "swiglu":
        return [torch.randn(a.tokens, a.hidden, device=dev, dtype=dtype),
                torch.randn(a.tokens, a.hidden, device=dev, dtype=dtype)]
    if op == "swiglu_split":
        return [torch.randn(a.tokens, 2 * a.hidden, device=dev, dtype=dtype)]
    if op == "qk_norm_rope":
        return [torch.randn(a.batch, a.seq, a.heads, a.head_dim, device=dev, dtype=dtype),
                torch.randn(a.head_dim, device=dev, dtype=dtype),
                torch.randn(a.seq, a.head_dim // 2, device=dev, dtype=dtype),
                torch.randn(a.seq, a.head_dim // 2, device=dev, dtype=dtype)]
    if op == "kl_div_logits":
        s = torch.randn(a.batch * 64, a.vocab, device=dev, dtype=dtype)
        return [s, torch.randn_like(s)]
    if op == "newton_schulz":
        return [torch.randn(a.dim, a.dim, device=dev, dtype=dtype)]
    if op == "topk":
        return [torch.randn(a.batch * 64, a.vocab, device=dev, dtype=dtype)]
    if op == "softmax":
        return [torch.randn(a.tokens, a.dim, device=dev, dtype=dtype)]
    raise ValueError(op)


def _bench_case_fn(case, tensors, compiled, measure, iters, warmup):
    """Time a pre-compiled step (forward + backward of the compiled graph)."""
    def fn():
        ins = [t.detach().clone().requires_grad_(t.is_floating_point() and case.backward)
               for t in tensors]
        loss = compiled(ins)
        if case.backward:
            loss.backward()
    return measure(fn, iters, warmup)


__all__ = ["main", "parse"]
