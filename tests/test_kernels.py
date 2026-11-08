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
    try:
        import triton  # noqa: F401
    except Exception:
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
