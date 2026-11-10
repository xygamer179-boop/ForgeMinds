"""Run every ForgeMind Triton kernel through TileLens (formerly Triton-Viz).

TileLens instruments plain ``@triton.jit`` kernels (its CLI patches them before this script
imports them), so the three analyses run against the *real* kernels:

```bash
TRITON_INTERPRET=1 python tools/tilelens_probe.py     # plain run through triton's interpreter
tile-profiler   tools/tilelens_probe.py               # loop unrolling, mask ratios, byte counts
tile-sanitizer  tools/tilelens_probe.py               # symbolic out-of-bounds check (aborts on OOB)
tile-race       tools/tilelens_probe.py               # cross-program race detector
```

Shapes deliberately hit the awkward paths: non-divisible rows (tail masks), non-power-of-two
widths (padded tiles), the padded SwiGLU-split epilogue, and the two-pass KL whose vocab does
not fit one tile. No GPU required -- everything runs on CPU through Triton's interpreter.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must be set *before* `import forgemind`: triton decides between @triton.jit and the CPU
# interpreter at decoration time, and the kernels decorate themselves at import.
os.environ.setdefault("TRITON_INTERPRET", "1")

import torch

import forgemind as fm
from forgemind import kernels as kk

kk.set_backend("triton")                         # force the kernel path on CPU


def go(name, fn, tensors, backward=True):
    ins = [t.clone().requires_grad_(t.is_floating_point() and backward) for t in tensors]
    out = fn(ins)
    if backward:
        loss = out.sum() if not isinstance(out, (tuple, list)) else sum(o.sum() for o in out)
        loss.float().backward()
    n = out.numel() if torch.is_tensor(out) else sum(o.numel() for o in out)
    print(f"ok {name}: {n} elements")


def main():
    torch.manual_seed(0)

    # --- rms_norm / add_rms_norm: width 40 is not a power of two -> masked tail tile --------
    x = torch.randn(6, 40)
    w = torch.randn(40)
    go("rms_norm(d=40, masked tail)", lambda t: fm.kernels.rms_norm(t[0], t[1], 1e-6), [x, w])
    delta = torch.randn_like(x)
    go("add_rms_norm(d=40)", lambda t: fm.kernels.add_rms_norm(t[0], t[1], t[2], 1e-6),
       [x, delta, w])

    # --- swiglu flat: 6*95 elements is not a multiple of the flat block --------------------
    g = torch.randn(6, 95)
    u = torch.randn(6, 95)
    go("swiglu(6x95, tail block)", lambda t: fm.kernels.swiglu(t[0], t[1]), [g, u])

    # --- swiglu_split: h=95 -> padded to 128 with masked lanes -----------------------------
    s = torch.randn(6, 190)
    go("swiglu_split(h=95, padded)", lambda t: fm.kernels.swiglu_split(t[0]), [s])

    # --- qk_norm_rope: head_dim 24 -> hd=12 padded to 16, 3 positions ----------------------
    q = torch.randn(2, 3, 3, 24)
    qw = torch.randn(24)
    cos = torch.randn(3, 12)
    sin = torch.randn(3, 12)
    go("qk_norm_rope(2x3x3x24, hd=12)",
       lambda t: fm.kernels.qk_norm_rope(t[0], t[1], t[2], t[3], 1e-6), [q, qw, cos, sin])

    # --- kl_div_logits: V=1000 fits one tile (SINGLE), V=5000 needs the two-pass loop ------
    st = torch.randn(5, 1000)
    te = torch.randn(5, 1000)
    go("kl_div_logits(V=1000, single tile)",
       lambda t: fm.kernels.kl_div_logits(t[0], t[1]), [st, te])
    st2 = torch.randn(7, 5000)
    te2 = torch.randn(7, 5000)
    go("kl_div_logits(V=5000, two-pass + tail)",
       lambda t: fm.kernels.kl_div_logits(t[0], t[1]), [st2, te2])

    # --- newton_schulz (Muon): tall matrix -> short-side Gram, flat elementwise tail blocks --
    m = torch.randn(63, 17)                      # m > n: the transposed path; Gram 17x17=289
    go("newton_schulz(63x17, transposed)",
       lambda t: fm.kernels.newton_schulz(t[0]), [m], backward=False)
    b = torch.randn(4, 31, 20)                   # Muon bucket batch; updates 1071 elems/row batch
    go("newton_schulz(4x31x20, batched)",
       lambda t: fm.kernels.newton_schulz(t[0]), [b], backward=False)

    save = os.environ.get("TILELENS_SAVE")
    if save:
        # Visualizer data path: archive the trace TileLens recorded (zip: manifest + tensors.npz)
        import tilelens
        tilelens.save(save)
        print(f"tilelens trace saved to {save}")

    print("tilelens probe done")


if __name__ == "__main__":
    main()
