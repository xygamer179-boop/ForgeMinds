"""Runs every packed Triton kernel through Triton's CPU interpreter and checks it against torch.

    TRITON_INTERPRET=1 python tests/kernel_interp_check.py

Invoked as a subprocess by ``tests/test_kernels.py`` (the flag has to be set before triton is
imported). Prints one line per check and exits non-zero on the first failure.
"""

import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from forgemind.kernels import _common, _fallback as fb  # noqa: E402
from forgemind.kernels import kl, norm, rope  # noqa: E402
from forgemind.kernels.swiglu import _SwiGLUFn, _SwiGLUSplitFn  # noqa: E402

_common.set_backend("triton")
torch.manual_seed(0)
FAILS = []
CHECKED = [0]


def check(name, got, exp, tol=2e-5):
    CHECKED[0] += 1
    if isinstance(got, bool):
        got, exp = torch.tensor(float(got)), torch.tensor(float(exp))
    err = (got.float() - exp.float()).abs().max().item()
    ok = err <= tol
    if not ok:
        FAILS.append(name)
    print(f"{'ok  ' if ok else 'FAIL'} {name:52s} err={err:.3e}")
    return ok


def grad_pair(fn_ref, fn_kernel, *tensors, up=None, n_grad=1):
    """Compare grads of `fn_kernel` against autograd on `fn_ref` for the first n_grad inputs."""
    ref_in = [t.detach().clone().requires_grad_(True) for t in tensors]
    out = fn_ref(*ref_in)
    outs = out if isinstance(out, tuple) else (out,)
    gouts = [torch.randn_like(o) for o in outs]
    torch.autograd.backward(outs, gouts)

    ker_in = [t.detach().clone().requires_grad_(True) for t in tensors]
    kout = fn_kernel(*ker_in)
    kouts = kout if isinstance(kout, tuple) else (kout,)
    for a, b in zip(kouts, outs):
        check("  fwd", a, b)
    torch.autograd.backward(kouts, gouts)
    for i in range(n_grad):
        if ref_in[i].grad is None:
            continue
        check(f"  grad[{i}]", ker_in[i].grad, ref_in[i].grad)


print("== rms_norm ==")
for n, d in ((6, 8), (5, 7), (3, 16), (1, 3)):
    x = torch.randn(n, d)
    w = torch.randn(d)
    grad_pair(lambda a, b: fb.rms_norm(a, b, 1e-6), lambda a, b: norm._RMSNormFn.apply(a, b, 1e-6),
              x, w, n_grad=2)
    y = norm._RMSNormFn.apply(x, w, 1e-6)
    check(f"dtype preserved (n={n},d={d})", y.dtype == x.dtype, True, 0)

print("== add_rms_norm ==")
for n, d in ((6, 8), (5, 7)):
    x, dl, w = torch.randn(n, d), torch.randn(n, d), torch.randn(d)
    grad_pair(lambda a, b, c: fb.add_rms_norm(a, b, c, 1e-6),
              lambda a, b, c: norm._AddRMSNormFn.apply(a, b, c, 1e-6), x, dl, w, n_grad=3)

print("== qk_norm_rope ==")
for b, t, h, hd in ((2, 5, 3, 4), (1, 7, 2, 8), (2, 3, 4, 6)):
    x = torch.randn(b, t, h, 2 * hd)
    w = torch.randn(2 * hd)
    cos, sin = torch.randn(t, hd), torch.randn(t, hd)
    ref = lambda a, ww, c, s: fb.qk_norm_rope(a, ww, c.reshape(1, t, 1, hd), s.reshape(1, t, 1, hd), 1e-6)
    grad_pair(ref, lambda a, ww, c, s: rope._QKNormRoPE.apply(a, ww, c, s, 1e-6),
              x, w, cos, sin, n_grad=2)

print("== swiglu ==")
for numel in (37, 4096):
    g, u = torch.randn(numel), torch.randn(numel)
    grad_pair(fb.swiglu, lambda a, b: _SwiGLUFn.apply(a, b), g, u, n_grad=2)

print("== swiglu_split ==")
for n, h in ((6, 8), (5, 6), (3, 32)):
    x = torch.randn(n, 2 * h)
    ref = lambda a: torch.nn.functional.silu(a[..., :h]) * a[..., h:]
    grad_pair(ref, lambda a: _SwiGLUSplitFn.apply(a), x, n_grad=1)


def ref_kl(s, t):
    ls = torch.log_softmax(s.float(), -1)
    lt = torch.log_softmax(t.float(), -1)
    return (lt.exp() * (lt - ls)).sum(-1)


print("== kl_div_logits (single tile) ==")
for v in (7, 64, 1000):
    s, t = torch.randn(4, v), torch.randn(4, v)
    check(f"per-row loss V={v}", kl.kl_div_logits(s, t, "none"), ref_kl(s, t))
    sk = s.clone().requires_grad_(True)
    kl.kl_div_logits(sk, t, "mean").backward()
    ref_g = -(torch.log_softmax(t.float(), -1).exp()
              - torch.log_softmax(s.float(), -1).exp()) / s.shape[0]
    check(f"grad V={v}", sk.grad, ref_g)
    check(f"mean V={v}", kl.kl_div_logits(s, t, "mean"), ref_kl(s, t).mean())

print("== kl_div_logits (two-pass streaming) ==")
v = 5000
s, t = torch.randn(2, v), torch.randn(2, v)
check("per-row loss V=5000", kl.kl_div_logits(s, t, "none"), ref_kl(s, t), tol=1e-4)
sk = s.clone().requires_grad_(True)
kl.kl_div_logits(sk, t, "sum").backward()
check("grad V=5000", sk.grad, torch.log_softmax(s.float(), -1).exp()
      - torch.log_softmax(t.float(), -1).exp(), tol=1e-4)

print("== kl input is not mutated / dtype respected ==")
s0 = torch.randn(3, 32)
s_copy = s0.clone()
kl.kl_div_logits(s0.clone().requires_grad_(True), torch.randn(3, 32), "mean")
check("input untouched", (s0 - s_copy).abs().max(), torch.zeros(()), 0)
sb = torch.randn(3, 32, dtype=torch.bfloat16, requires_grad=True)
kl.kl_div_logits(sb, torch.randn(3, 32, dtype=torch.bfloat16), "mean").backward()
check("bf16 grad dtype", sb.grad.dtype == torch.bfloat16, True, 0)

print("== kl_distill_loss (chunked, gradient scale) ==")
for n, d, vocab, chunk in ((6, 4, 16, 4), (6, 4, 16, 100), (5, 8, 32, 2)):
    hs, ht = torch.randn(n, d), torch.randn(n, d)
    W, WT = torch.randn(vocab, d), torch.randn(vocab, d)
    head = lambda z: z @ WT.t()
    exp, gexp = fb.kl_distill_loss_ref(hs, ht, W, head, chunk=chunk)
    got = kl.kl_distill_loss(hs, ht, W, head, chunk=chunk)
    check(f"loss n={n} V={vocab} chunk={chunk}", got, exp, tol=1e-5)
    hk = hs.clone().requires_grad_(True)
    kl.kl_distill_loss(hk, ht, W, head, chunk=chunk).backward()
    check(f"grad n={n} V={vocab} chunk={chunk}", hk.grad, gexp, tol=1e-4)

print(f"\n{CHECKED[0]} checks, {len(FAILS)} failed")
if FAILS:
    print("FAILED:", ", ".join(FAILS))
    sys.exit(1)
print("ALL KERNEL CHECKS PASSED")
