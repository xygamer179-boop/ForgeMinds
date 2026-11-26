"""Tests for the packed (Triton) kernels.

Two layers:

* the **torch path** is what runs on CPU, and it is checked against the eager expressions it
  replaces (`F.rms_norm`, `F.silu`, `F.kl_div`, ...) including gradients -- this is what
  guarantees that turning the kernels on never changes your numbers;
* the **kernels themselves** are executed through Triton's CPU interpreter
  (``TRITON_INTERPRET=1``, in a subprocess because the flag must be set before triton is
  imported) and checked against the same references. That is the test that makes "bug free"
  a claim rather than a hope -- it is how the wrong-stride bug in the fused SwiGLU backward
  and the input-mutating KL kernel were caught here, on a machine with no GPU.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from forgemind import kernels
from forgemind.kernels import _fallback as fb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    kernels.set_backend("auto")


# ---------------------------------------------------------------- torch path parity
def test_rms_norm_matches_torch():
    torch.manual_seed(0)
    for d in (4, 8, 16):
        x = torch.randn(5, d)
        w = torch.randn(d)
        for eps in (1e-6, 1e-5):
            got = kernels.rms_norm(x, w, eps)
            exp = F.rms_norm(x, (d,), w, eps)
            assert torch.allclose(got, exp, atol=2e-6, rtol=1e-5)


def test_rms_norm_grad_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(6, 8, requires_grad=True)
    w = torch.randn(8, requires_grad=True)
    g = torch.randn(6, 8)
    a = kernels.rms_norm(x, w, 1e-6)
    a.backward(g, retain_graph=True)
    dx, dw = x.grad.clone(), w.grad.clone()
    x.grad = w.grad = None
    b = F.rms_norm(x, (8,), w, 1e-6)
    b.backward(g)
    assert torch.allclose(dx, x.grad, atol=2e-6, rtol=1e-5)
    assert torch.allclose(dw, w.grad, atol=2e-6, rtol=1e-5)


def test_add_rms_norm_matches_torch():
    torch.manual_seed(0)
    x, delta = torch.randn(4, 8), torch.randn(4, 8)
    w = torch.randn(8)
    xn, y = kernels.add_rms_norm(x, delta, w)
    assert torch.allclose(xn, x + delta, atol=0, rtol=0)
    assert torch.allclose(y, F.rms_norm(x + delta, (8,), w, 1e-6), atol=2e-6, rtol=1e-5)


def test_swiglu_matches_torch():
    torch.manual_seed(0)
    g, u = torch.randn(33), torch.randn(33)
    assert torch.allclose(kernels.swiglu(g, u), F.silu(g) * u, atol=2e-6, rtol=1e-5)


def test_swiglu_split_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(7, 16)
    exp = F.silu(x[..., :8]) * x[..., 8:]
    assert torch.allclose(kernels.swiglu_split(x), exp, atol=2e-6, rtol=1e-5)


def test_qk_norm_rope_matches_torch():
    torch.manual_seed(0)
    b, t, h, hd = 2, 5, 3, 4
    x = torch.randn(b, t, h, 2 * hd)
    w = torch.randn(2 * hd)
    cos, sin = torch.randn(t, hd), torch.randn(t, hd)
    exp = fb.qk_norm_rope(x, w, cos.reshape(1, t, 1, hd), sin.reshape(1, t, 1, hd), 1e-6)
    got = kernels.qk_norm_rope(x, w, cos, sin, 1e-6)
    assert torch.allclose(got, exp, atol=2e-6, rtol=1e-5)


def test_qk_norm_rope_grad_matches_torch_autograd():
    torch.manual_seed(0)
    b, t, h, hd = 2, 4, 2, 4
    x = torch.randn(b, t, h, 2 * hd)
    w = torch.randn(2 * hd)
    cos, sin = torch.randn(t, hd), torch.randn(t, hd)
    g = torch.randn(b, t, h, 2 * hd)

    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    fb.qk_norm_rope(xr, wr, cos.reshape(1, t, 1, hd), sin.reshape(1, t, 1, hd), 1e-6).backward(g)

    xk = x.clone().requires_grad_(True)
    wk = w.clone().requires_grad_(True)
    kernels.qk_norm_rope(xk, wk, cos, sin, 1e-6).backward(g)
    assert torch.allclose(xk.grad, xr.grad, atol=5e-6, rtol=1e-4)
    assert torch.allclose(wk.grad, wr.grad, atol=5e-6, rtol=1e-4)


def test_relu_squared_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(33)
    assert torch.allclose(kernels.relu_squared(x), torch.relu(x) ** 2, atol=1e-6, rtol=1e-5)


def test_relu_squared_grad_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(24, requires_grad=True)
    g = torch.randn(24)
    a = kernels.relu_squared(x)
    a.backward(g)
    xr = x.detach().clone().requires_grad_(True)
    (torch.relu(xr) ** 2).backward(g)
    assert torch.allclose(x.grad, xr.grad, atol=1e-6, rtol=1e-5)


def test_geglu_matches_torch():
    torch.manual_seed(0)
    gate, up = torch.randn(5, 16), torch.randn(5, 16)
    exp = F.gelu(gate, approximate="none") * up
    assert torch.allclose(kernels.geglu(gate, up), exp, atol=1e-6, rtol=1e-5)


def test_geglu_grad_matches_torch():
    torch.manual_seed(0)
    gate = torch.randn(12, requires_grad=True)
    up = torch.randn(12, requires_grad=True)
    g = torch.randn(12)
    kernels.geglu(gate, up).backward(g)
    gr = gate.detach().clone().requires_grad_(True)
    ur = up.detach().clone().requires_grad_(True)
    (F.gelu(gr, approximate="none") * ur).backward(g)
    assert torch.allclose(gate.grad, gr.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(up.grad, ur.grad, atol=1e-6, rtol=1e-5)


def test_layer_norm_matches_torch():
    torch.manual_seed(0)
    for n, d in ((5, 8), (4, 16)):
        x, w, b = torch.randn(n, d), torch.randn(d), torch.randn(d)
        exp = F.layer_norm(x, (d,), w, b, 1e-5)
        assert torch.allclose(kernels.layer_norm(x, w, b, 1e-5), exp, atol=1e-6, rtol=1e-5)
        exp_nb = F.layer_norm(x, (d,), w, None, 1e-5)
        assert torch.allclose(kernels.layer_norm(x, w, None, 1e-5), exp_nb,
                              atol=1e-6, rtol=1e-5)


def test_layer_norm_grads_match_torch():
    torch.manual_seed(0)
    n, d = 6, 8
    x, w, b = torch.randn(n, d), torch.randn(d), torch.randn(d)
    g = torch.randn(n, d)
    xi, wi, bi = (t.clone().requires_grad_(True) for t in (x, w, b))
    kernels.layer_norm(xi, wi, bi, 1e-5).backward(g)
    xr, wr, br = (t.clone().requires_grad_(True) for t in (x, w, b))
    F.layer_norm(xr, (d,), wr, br, 1e-5).backward(g)
    assert torch.allclose(xi.grad, xr.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(wi.grad, wr.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(bi.grad, br.grad, atol=1e-6, rtol=1e-5)


def test_cross_entropy_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(6, 40)
    targets = torch.randint(0, 40, (6,))
    for reduction in ("mean", "sum", "none"):
        got = kernels.cross_entropy(logits, targets, reduction=reduction)
        exp = F.cross_entropy(logits, targets, reduction=reduction)
        assert torch.allclose(got, exp, atol=1e-6, rtol=1e-5), reduction
    ign = targets.clone()
    ign[1] = -100
    got = kernels.cross_entropy(logits, ign, label_smoothing=0.1)
    exp = F.cross_entropy(logits, ign, label_smoothing=0.1)
    assert torch.allclose(got, exp, atol=1e-6, rtol=1e-5)


def test_cross_entropy_grad_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(6, 40)
    targets = torch.randint(0, 40, (6,))
    targets[2] = -100
    g = torch.randn(())
    li = logits.clone().requires_grad_(True)
    kernels.cross_entropy(li, targets, label_smoothing=0.1).backward(g)
    lr = logits.clone().requires_grad_(True)
    F.cross_entropy(lr, targets, label_smoothing=0.1).backward(g)
    assert torch.allclose(li.grad, lr.grad, atol=1e-6, rtol=1e-5)


def test_cross_entropy_non_contiguous_logits():
    """Regression for Liger #1674: a strided view must not corrupt the gradient."""
    torch.manual_seed(0)
    base = torch.randn(6, 80)
    logits = base[:, ::2]                      # [6, 40], stride (80, 2): non-contiguous
    assert not logits.is_contiguous()
    targets = torch.randint(0, 40, (6,))
    li = logits.clone().requires_grad_(True)
    kernels.cross_entropy(li, targets).backward()
    lr = logits.clone().requires_grad_(True)
    F.cross_entropy(lr, targets).backward()
    assert torch.allclose(li.grad, lr.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(kernels.cross_entropy(logits, targets),
                          F.cross_entropy(logits, targets), atol=1e-6, rtol=1e-5)


def test_layer_norm_multidim_input_keeps_shape():
    """Regression: the launches returned the flat [n, d] view -- a [B,T,D] activation
    came back as [B*T, D] (the fake impl promised x.shape, so a compiled graph would
    disagree with reality). 2-D tests hid it because view(-1, d) is the identity there."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    w, b = torch.randn(16), torch.randn(16)
    y = kernels.layer_norm(x, w, b, 1e-5)
    assert y.shape == x.shape
    assert torch.allclose(y, F.layer_norm(x, (16,), w, b, 1e-5), atol=1e-6, rtol=1e-5)
    xi, wi, bi = (t.clone().requires_grad_(True) for t in (x, w, b))
    g = torch.randn_like(x)
    kernels.layer_norm(xi, wi, bi, 1e-5).backward(g)
    xr, wr, br = (t.clone().requires_grad_(True) for t in (x, w, b))
    F.layer_norm(xr, (16,), wr, br, 1e-5).backward(g)
    assert xi.grad.shape == x.shape
    assert torch.allclose(xi.grad, xr.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(wi.grad, wr.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(bi.grad, br.grad, atol=1e-6, rtol=1e-5)


def test_cross_entropy_none_keeps_targets_shape():
    """Regression: reduction='none' returned a flat [N]; F returns the targets' shape
    ([B,T] for [B,T,V] logits). 1-D-target tests hid it."""
    torch.manual_seed(0)
    # 2-D logits: F itself is the oracle (it keeps the targets' shape)
    logits = torch.randn(6, 40)
    targets = torch.randint(0, 40, (6,))
    got = kernels.cross_entropy(logits, targets, reduction="none")
    exp = F.cross_entropy(logits, targets, reduction="none")
    assert got.shape == exp.shape == targets.shape
    assert torch.allclose(got, exp, atol=1e-6, rtol=1e-5)
    lk = logits.clone().requires_grad_(True)
    d = torch.randn_like(got)
    kernels.cross_entropy(lk, targets, reduction="none").backward(d)
    lr = logits.clone().requires_grad_(True)
    F.cross_entropy(lr, targets, reduction="none").backward(d)
    assert lk.grad.shape == logits.shape
    assert torch.allclose(lk.grad, lr.grad, atol=1e-6, rtol=1e-5)
    # [B,T,V] logits + [B,T] targets: our layout flattens leading dims (Liger's --
    # F would take DIM 1 as classes and reject this pairing), so the oracle is the
    # flattened 2-D call. The reduction must still return targets' shape, not [N].
    logits3 = torch.randn(2, 5, 40)
    tg3 = torch.randint(0, 40, (2, 5))
    got3 = kernels.cross_entropy(logits3, tg3, reduction="none")
    exp3 = F.cross_entropy(logits3.reshape(-1, 40), tg3.reshape(-1),
                           reduction="none").reshape(tg3.shape)
    assert got3.shape == tg3.shape
    assert torch.allclose(got3, exp3, atol=1e-6, rtol=1e-5)
    lk3 = logits3.clone().requires_grad_(True)
    d3 = torch.randn_like(got3)
    kernels.cross_entropy(lk3, tg3, reduction="none").backward(d3)
    lr3 = logits3.clone().requires_grad_(True)
    F.cross_entropy(lr3.reshape(-1, 40), tg3.reshape(-1), reduction="none").reshape(
        tg3.shape).backward(d3)
    assert lk3.grad.shape == logits3.shape
    assert torch.allclose(lk3.grad, lr3.grad, atol=1e-6, rtol=1e-5)


def test_forced_backend_still_refuses_fp64():
    """Regression: set_backend('triton') is documented to drop the *is_cuda* check
    only; it used to drop the dtype check too, so fp64 entered kernels that demote
    their loads to fp32 (silent precision loss). The dtype gate must always hold."""
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float64)
    w = torch.randn(8, dtype=torch.float64)
    saved = kernels.get_backend()
    kernels.set_backend("triton")
    try:
        y = kernels.relu_squared(x)
        assert y.dtype == torch.float64
        assert torch.equal(y, torch.relu(x) ** 2)     # exact: fallback, not a demoted kernel
        yn = kernels.layer_norm(x, w, None, 1e-5)
        assert yn.dtype == torch.float64
        assert torch.allclose(yn, F.layer_norm(x, (8,), w, None, 1e-5), atol=0, rtol=1e-12)
        assert kernels.stats()["fallback"] >= 2
    finally:
        kernels.set_backend(saved)


def test_empty_inputs_fall_back_to_torch():
    """Empty batches take the fallback (no programs would launch); F defines the answer."""
    torch.manual_seed(0)
    x, w = torch.randn(0, 16), torch.randn(16)
    y = kernels.layer_norm(x, w, None, 1e-5)
    assert y.shape == (0, 16)
    assert torch.equal(y, F.layer_norm(x, (16,), w, None, 1e-5))
    logits, tg = torch.randn(0, 40), torch.zeros(0, dtype=torch.int64)
    loss = kernels.cross_entropy(logits, tg, reduction="mean")
    exp = F.cross_entropy(logits, tg, reduction="mean")
    assert (torch.isnan(loss) and torch.isnan(exp)) or torch.equal(loss, exp)


def test_kl_matches_torch_kl_div():
    torch.manual_seed(0)
    s, t = torch.randn(6, 40), torch.randn(6, 40)
    got = kernels.kl_div_logits(s, t, "mean")
    exp = F.kl_div(F.log_softmax(s, -1), F.softmax(t, -1), reduction="sum") / s.shape[0]
    assert got.item() == pytest.approx(exp.item(), rel=1e-5, abs=1e-6)


def test_kl_grad_matches_torch():
    torch.manual_seed(0)
    s, t = torch.randn(6, 40), torch.randn(6, 40)
    s2 = s.clone().requires_grad_(True)
    kernels.kl_div_logits(s2, t, "mean").backward()
    exp = (F.softmax(s, -1) - F.softmax(t, -1)) / s.shape[0]
    assert torch.allclose(s2.grad, exp, atol=1e-6, rtol=1e-5)


def test_kl_does_not_mutate_student_logits():
    torch.manual_seed(0)
    s = torch.randn(3, 17)
    keep = s.clone()
    kernels.kl_div_logits(s.clone().requires_grad_(True), torch.randn(3, 17), "mean")
    assert torch.equal(s, keep)
    s3 = s.clone().requires_grad_(True)
    kernels.kl_div_logits(s3, torch.randn(3, 17), "mean").backward()
    assert torch.equal(s3.detach(), keep), "backward must not scribble on the logits"


@pytest.mark.parametrize("chunk", [4, 1000])
def test_kl_distill_chunk_scaling(chunk):
    torch.manual_seed(0)
    n, d, v = 8, 4, 24
    hs, ht = torch.randn(n, d), torch.randn(n, d)
    w, wt = torch.randn(v, d), torch.randn(v, d)
    head = lambda z: z @ wt.t()
    ref, gref = fb.kl_distill_loss_ref(hs, ht, w, head, chunk=chunk)
    got = kernels.kl_distill_loss(hs, ht, w, head, chunk=chunk)
    assert got.item() == pytest.approx(ref.item(), rel=1e-5, abs=1e-6)
    hk = hs.clone().requires_grad_(True)
    kernels.kl_distill_loss(hk, ht, w, head, chunk=chunk).backward()
    assert torch.allclose(hk.grad, gref, atol=1e-5, rtol=1e-4)


def test_kl_distill_does_not_train_the_teacher():
    torch.manual_seed(0)
    n, d, v = 4, 4, 8
    hs = torch.randn(n, d, requires_grad=True)
    ht = torch.randn(n, d)
    w = torch.randn(v, d)
    wt = torch.randn(v, d, requires_grad=True)
    kernels.kl_distill_loss(hs, ht, w, lambda z: z @ wt.t()).backward()
    assert wt.grad is None


# ---------------------------------------------------------------- backend selection
@pytest.mark.skipif(kernels._common.INTERPRET,
                    reason="TRITON_INTERPRET=1 runs the real kernels on CPU by design")
def test_cpu_uses_the_torch_path_and_says_so():
    kernels.reset_stats()
    x = torch.randn(4, 8)
    kernels.rms_norm(x, torch.randn(8), 1e-6)
    st = kernels.stats()
    assert st["fallback"] >= 1 and st["fast"] == 0
    assert "not cuda" in st["reasons"] or "no triton" in st["reasons"]


def test_forced_fallback_is_used_even_on_cuda():
    kernels.set_backend("fallback")
    kernels.reset_stats()
    kernels.swiglu(torch.randn(8), torch.randn(8))
    assert kernels.stats()["reasons"].get("backend=fallback") == 1


def test_stats_are_not_counted_inside_a_graph():
    """The counters are a debugging aid, so they must not become a tracing side effect: mutating
    a module-level dict while dynamo traces is what broke `ac` + `fused_ops` (activation
    checkpointing re-runs the region in the backward, and the re-run cannot be captured)."""
    kernels.set_backend("auto")
    kernels.reset_stats()
    w, x = torch.randn(8), torch.randn(4, 8)
    fn = torch.compile(lambda t: kernels.rms_norm(t, w, 1e-6),
                       backend="aot_eager", fullgraph=True)
    before = kernels.stats()
    fn(x)
    assert kernels.stats() == before, "a traced call must not touch the stats dict"


def test_stats_still_count_eager_calls():
    kernels.reset_stats()
    kernels.rms_norm(torch.randn(4, 8), torch.randn(8), 1e-6)
    st = kernels.stats()
    assert st["fast"] + st["fallback"] == 1


def test_swiglu_split_rejects_an_odd_last_dim():
    """`gate|up` needs a paired width; slicing 127 into 63+64 would be silently lopsided."""
    with pytest.raises(ValueError, match="even last dimension"):
        kernels.swiglu_split(torch.randn(4, 127))


def test_qk_norm_rope_rejects_an_odd_head_dim():
    """RoPE rotates feature pairs, so an odd head dim is not representable."""
    with pytest.raises(ValueError, match="must be even"):
        kernels.qk_norm_rope(torch.randn(1, 8, 4, 63), torch.randn(63),
                             torch.randn(8, 32), torch.randn(8, 32))


# ---------------------------------------------------------------- model wiring
def test_enable_is_numerically_transparent():
    from forgemind.models import GPT, GPTConfig

    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq=16)
    a = GPT(cfg)
    b = GPT(cfg)
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        la, _ = a(x, y)
        info = kernels.enable(b)._fm_fused
        lb, _ = b(x, y)
    assert info["norm"] > 0 and info["mlp"] > 0
    assert torch.allclose(la, lb, atol=1e-6, rtol=1e-5)


def test_trainer_flag_wires_the_kernels():
    from forgemind import ForgeConfig, GPT, GPTConfig, Trainer, SyntheticLM

    cfg = ForgeConfig()
    cfg.train.__dict__.update(max_steps=2, warmup_steps=0, log_every=99, bf16=False,
                              resume=False, fused_ops=True)
    mc = GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4, n_kv_heads=2, max_seq=8)
    tr = Trainer(lambda c: GPT(mc), cfg, SyntheticLM(64, 8, n=32), micro_batch=2)
    assert tr.model._fm_fused["norm"] > 0
    m = tr.train_step()
    assert m["loss"] == m["loss"]      # not NaN


# ---------------------------------------------------------------- the kernels themselves
def test_kernels_in_triton_interpreter():
    """Run every kernel on CPU through `TRITON_INTERPRET=1` and compare to torch."""
    import importlib.util
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONPATH=ROOT)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    out = subprocess.run([sys.executable, os.path.join(HERE, "kernel_interp_check.py")],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    tail = out.stdout[-2500:] + out.stderr[-2000:]
    assert out.returncode == 0, f"interpreter kernel checks failed:\n{tail}"
    assert "ALL KERNEL CHECKS PASSED" in out.stdout, tail
    assert "checks, 0 failed" in out.stdout


def test_compile_fullgraph_with_fused_ops():
    """`enable()` must not introduce a graph break: the torch path is fully traceable."""
    from forgemind.models import GPT, GPTConfig

    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4, n_kv_heads=2, max_seq=8)
    model = GPT(cfg)
    kernels.enable(model)
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    eager_loss, _ = model(x, y)
    eager_loss.backward()
    eager_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)

    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    loss, _ = compiled(x, y)
    loss.backward()
    assert torch.allclose(loss, eager_loss, atol=1e-6, rtol=1e-5)
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.allclose(p.grad, eager_grads[n], atol=1e-6, rtol=1e-5), n


def test_qk_norm_rope_noncontiguous_inputs_stay_correct():
    """BUG-005 (audit): _ok must reject non-contiguous x/w -- the kernels index them at
    unit stride, and under TRITON_INTERPRET a strided view used to be read as contiguous."""
    x = torch.randn(2, 4, 2, 8)
    w = torch.randn(8)
    cos = torch.randn(4, 4)
    sin = torch.randn(4, 4)
    xn = x.transpose(0, 1)                      # [4, 2, 2, 8]: non-contiguous, T is now 2
    assert not xn.is_contiguous()
    cos2, sin2 = cos[:2], sin[:2]
    # public-API comparison: the non-contiguous input must take the torch fallback while a
    # contiguous twin takes the kernel (under TRITON_INTERPRET this is kernel-vs-fallback
    # parity; before the fix the strided view was read as contiguous and diverged).
    got = kernels.qk_norm_rope(xn, w, cos2, sin2)
    ref = kernels.qk_norm_rope(xn.contiguous(), w, cos2, sin2)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
    w_strided = torch.randn(16)[::2]            # non-contiguous weight
    assert not w_strided.is_contiguous()
    got2 = kernels.qk_norm_rope(x, w_strided, cos, sin)
    ref2 = kernels.qk_norm_rope(x, w_strided.contiguous(), cos, sin)
    assert torch.allclose(got2, ref2, atol=1e-5, rtol=1e-5)


def test_rms_norm_nondefault_eps_backwards_match_torch():
    """BUG-008 (audit): the backward must work for eps != 1e-6 (the caller's real eps is
    now plumbed through) and dW must match torch's fp32 reference path."""
    x = torch.randn(6, 32)
    for eps in (1e-4, 3e-5):
        wa, wb = torch.randn(32).requires_grad_(True), torch.randn(32).requires_grad_(True)
        with torch.no_grad():                               # identical weights, isolated grads
            wb.copy_(wa)
        a, b = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        out_a = kernels.rms_norm(a, wa, eps)
        out_b = F.rms_norm(b, (32,), wb, eps)
        assert torch.allclose(out_a, out_b, atol=1e-5, rtol=1e-5), eps
        out_a.sum().backward()
        out_b.sum().backward()
        assert torch.allclose(a.grad, b.grad, atol=1e-4, rtol=1e-4), eps
        assert torch.allclose(wa.grad, wb.grad, atol=1e-4, rtol=1e-4), eps


def test_layernorm_inference_avoids_saved_statistics(monkeypatch):
    import importlib
    ln = importlib.import_module("forgemind.kernels.layernorm")
    common = importlib.import_module("forgemind.kernels._common")
    if not (common.INTERPRET or torch.cuda.is_available()):
        pytest.skip("requires the Triton interpreter or CUDA")
    device = "cpu" if common.INTERPRET else "cuda"
    x = torch.randn(7, 16, device=device)
    w = torch.randn(16, device=device)
    b = torch.randn(16, device=device)
    calls = []
    real = ln._launch_fwd

    def spy(*args, **kwargs):
        calls.append(kwargs.get("save_stats", True))
        result = real(*args, **kwargs)
        if not calls[-1]:
            assert result[1:] == (None, None)
        return result

    monkeypatch.setattr(ln, "_launch_fwd", spy)
    with torch.no_grad():
        actual = ln.layer_norm(x, w, b)
    torch.testing.assert_close(actual, torch.nn.functional.layer_norm(x, (16,), w, b))
    assert calls == [False]
    calls.clear()
    ln.layer_norm(x.requires_grad_(), w, b).sum().backward()
    assert calls == [True]
