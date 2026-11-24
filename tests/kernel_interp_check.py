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
from forgemind.kernels import kl, layernorm, norm, relu2, rope  # noqa: E402
from forgemind.kernels.swiglu import _SwiGLUFn, _SwiGLUSplitFn  # noqa: E402

# `geglu` and `cross_entropy` package attributes are the public FUNCTIONS (the
# `from .geglu import geglu` rebinding shadows the submodule); grab the modules by
# importlib so the Function classes below resolve.
import importlib  # noqa: E402

cross_entropy = importlib.import_module("forgemind.kernels.cross_entropy")
geglu = importlib.import_module("forgemind.kernels.geglu")

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

print("== relu_squared ==")
for numel in (37, 4096):
    x = torch.randn(numel)
    grad_pair(fb.relu_squared, lambda a: relu2._ReLUSquaredFn.apply(a), x, n_grad=1)

print("== geglu ==")
for numel in (37, 4096):
    g, u = torch.randn(numel), torch.randn(numel)
    grad_pair(fb.geglu, lambda a, b: geglu._GEGLUFn.apply(a, b), g, u, n_grad=2)

print("== layer_norm ==")
for n, d in ((6, 8), (5, 7), (3, 16)):
    x, w, b = torch.randn(n, d), torch.randn(d), torch.randn(d)
    ref = lambda a, ww, bb: fb.layer_norm(a, ww, bb, 1e-5)
    grad_pair(ref, lambda a, ww, bb: layernorm._LayerNormFn.apply(a, ww, bb, 1e-5),
              x, w, b, n_grad=3)
    grad_pair(lambda a, ww: fb.layer_norm(a, ww, None, 1e-5),
              lambda a, ww: layernorm._LayerNormFn.apply(a, ww, None, 1e-5),
              x, w, n_grad=2)

print("== cross_entropy ==")
lg = torch.randn(6, 40)
tg = torch.randint(0, 40, (6,))
for red in ("mean", "sum", "none"):
    got = cross_entropy._CrossEntropyFn.apply(lg, tg, red, -100, 0.0)
    exp = fb.cross_entropy(lg, tg, reduction=red)
    check(f"loss {red}", got, exp)
    d = torch.randn_like(got)
    lk = lg.clone().requires_grad_(True)
    cross_entropy._CrossEntropyFn.apply(lk, tg, red, -100, 0.0).backward(d)
    lr = lg.clone().requires_grad_(True)
    fb.cross_entropy(lr, tg, reduction=red).backward(d)
    check(f"grad {red}", lk.grad, lr.grad)
ign = tg.clone()
ign[1] = -100
for ls in (0.0, 0.1):
    got = cross_entropy._CrossEntropyFn.apply(lg, ign, "mean", -100, ls)
    exp = fb.cross_entropy(lg, ign, reduction="mean", label_smoothing=ls)
    check(f"ignore+smooth ls={ls} loss", got, exp)
    lk = lg.clone().requires_grad_(True)
    cross_entropy._CrossEntropyFn.apply(lk, ign, "mean", -100, ls).backward()
    lr = lg.clone().requires_grad_(True)
    fb.cross_entropy(lr, ign, reduction="mean", label_smoothing=ls).backward()
    check(f"ignore+smooth ls={ls} grad", lk.grad, lr.grad)
# Liger #1674 regression: strided logits must not corrupt the gradient
base = torch.randn(6, 80)
strided = base[:, ::2]
check("non-contiguous loss", cross_entropy._CrossEntropyFn.apply(strided, tg, "mean", -100, 0.0),
      fb.cross_entropy(strided, tg))
lk = strided.clone().requires_grad_(True)
cross_entropy._CrossEntropyFn.apply(lk, tg, "mean", -100, 0.0).backward()
lr = strided.clone().requires_grad_(True)
fb.cross_entropy(lr, tg).backward()
check("non-contiguous grad", lk.grad, lr.grad)

print(f"\n{CHECKED[0]} checks, {len(FAILS)} failed")
if FAILS:
    print("FAILED:", ", ".join(FAILS))
    sys.exit(1)
print("ALL KERNEL CHECKS PASSED")
