"""The parts catalogue: one kernel, one block, one module at a time.

This is the "use ForgeMind inside *your* architecture" surface, so the tests are about drop-in
compatibility rather than about training a big model:

* ``fm.nn.RMSNorm`` must be interchangeable with ``torch.nn.RMSNorm`` -- same numbers, same
  ``state_dict`` keys, same ``eps=None`` semantics -- and differentiable (``gradcheck``);
* ``fm.nn.SwiGLU`` must compute exactly ``w2(silu(w1 x) * w3 x)``, and its fused ``split`` variant
  must be numerically identical to the three-Linear one;
* ``fm.kernels.patch`` must fuse *one* module, work on torch's own classes, report whether it
  recognised the module, and be idempotent -- because it is called on user code we cannot see;
* and a hand-written model built from these blocks must train through ``Trainer`` and ``Forge``
  without adopting ``GPT``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import forgemind as fm
from forgemind import nn as fmnn
from forgemind.kernels.apply import _is_swiglu


def _blocks():
    torch.manual_seed(0)
    x = torch.randn(4, 3, 32)
    w = torch.randn(32)
    return x, w


# ------------------------------------------------------------------------------------- RMSNorm
def test_rms_norm_module_matches_torch():
    x, _ = _blocks()
    got, ref = fmnn.RMSNorm(32), nn.RMSNorm(32)
    assert torch.allclose(got(x), ref(x), atol=1e-6)

    # eps=None means "finfo(dtype).eps" in torch; patching/using our module must not change that
    ref_bf16 = nn.RMSNorm(32).to(torch.bfloat16)
    got_bf16 = fmnn.RMSNorm(32, dtype=torch.bfloat16)
    assert torch.allclose(got_bf16(x.bfloat16()), ref_bf16(x.bfloat16()), atol=1e-2)

    explicit = fmnn.RMSNorm(32, eps=1e-5)
    assert torch.allclose(explicit(x), nn.RMSNorm(32, eps=1e-5)(x), atol=1e-6)


def test_rms_norm_state_dict_is_interchangeable():
    x, _ = _blocks()
    ours, theirs = fmnn.RMSNorm(32), nn.RMSNorm(32)
    with torch.no_grad():
        theirs.weight.uniform_(0.5, 1.5)
    ours.load_state_dict(theirs.state_dict())          # torch -> forgemind
    assert torch.allclose(ours(x), theirs(x), atol=1e-6)
    back = nn.RMSNorm(32)
    back.load_state_dict(ours.state_dict())            # forgemind -> torch
    assert torch.allclose(back(x), theirs(x), atol=1e-6)
    assert set(ours.state_dict()) == {"weight"}


def test_rms_norm_gradcheck():
    torch.manual_seed(0)
    x = torch.randn(5, 16, dtype=torch.float64, requires_grad=True)
    w = torch.randn(16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: fm.kernels.rms_norm(a, b, 1e-6), (x, w))


def test_rms_norm_module_gradcheck():
    torch.manual_seed(0)
    norm = fmnn.RMSNorm(16).double()
    x = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(norm, (x,))


def test_rms_norm_output_dtype_is_input_dtype():
    norm = fmnn.RMSNorm(8)
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(2, 8, dtype=dt)
        assert norm(x).dtype == dt


def test_rms_norm_rejects_the_wrong_last_dim():
    with pytest.raises(ValueError, match="last dim"):
        fmnn.RMSNorm(32)(torch.randn(2, 16))


def test_forward_with_residual_is_the_fused_pre_norm():
    x, res = torch.randn(4, 32), torch.randn(4, 32)
    norm = fmnn.RMSNorm(32)
    y, new_res = norm.forward_with_residual(x, res)
    assert torch.allclose(new_res, x + res)
    assert torch.allclose(y, norm(x + res), atol=1e-6)
    y0, r0 = norm.forward_with_residual(x)             # residual=None starts from zero
    assert torch.allclose(r0, x) and torch.allclose(y0, norm(x), atol=1e-6)

    # gradients must reach both the hidden state and the residual stream
    xg, rg = x.clone().requires_grad_(), res.clone().requires_grad_()
    y, _ = norm.forward_with_residual(xg, rg)
    y.sum().backward()
    assert xg.grad is not None and rg.grad is not None
    assert torch.allclose(xg.grad, rg.grad)             # both only enter through x + res


def test_add_rms_norm_returns_residual_first():
    """Documented order of the functional kernel: (new_residual, normalized)."""
    x, res, w = torch.randn(4, 16), torch.randn(4, 16), torch.ones(16)
    a, b = fm.kernels.add_rms_norm(x, res, w)
    assert torch.allclose(a, x + res)
    assert torch.allclose(b, fm.kernels.rms_norm(x + res, w), atol=1e-6)


# --------------------------------------------------------------------------------------- SwiGLU
def test_swiglu_module_matches_the_eager_expression():
    mlp = fmnn.SwiGLU(16, hidden=24)
    x = torch.randn(5, 16)
    ref = mlp.w2(F.silu(mlp.w1(x)) * mlp.w3(x))
    assert torch.allclose(mlp(x), ref, atol=1e-6)


def test_swiglu_split_variant_is_numerically_identical():
    torch.manual_seed(0)
    dense = fmnn.SwiGLU(16, hidden=24)
    fused = dense.to_split(inplace=False)
    x = torch.randn(5, 16)
    assert torch.allclose(fused(x), dense(x), atol=1e-6)
    assert hasattr(fused, "gate_up") and not hasattr(fused, "w1")
    assert hasattr(dense, "w1") and not hasattr(dense, "gate_up")   # original untouched


def test_swiglu_to_split_in_place_keeps_the_answer():
    torch.manual_seed(0)
    mlp = fmnn.SwiGLU(16, hidden=24)
    x = torch.randn(3, 16)
    before = mlp(x).detach().clone()
    mlp.to_split(inplace=True)
    assert torch.allclose(mlp(x), before, atol=1e-6)
    assert mlp.split is True


def test_swiglu_gradcheck():
    torch.manual_seed(0)
    g = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    u = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: fm.kernels.swiglu(a, b), (g, u))


def test_swiglu_module_gradcheck():
    torch.manual_seed(0)
    mlp = fmnn.SwiGLU(8, hidden=12).double()
    x = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(mlp, (x,))


# ---------------------------------------------------------------------------------- patch / enable
def test_patch_works_on_torchs_own_rmsnorm():
    for eps in (None, 1e-5):
        norm = nn.RMSNorm(32) if eps is None else nn.RMSNorm(32, eps=eps)
        x = torch.randn(4, 32)
        before = norm(x).detach().clone()
        assert fm.kernels.patch(norm) is True
        assert torch.allclose(norm(x), before, atol=1e-6)
        assert fm.kernels.patch(norm) is True          # idempotent


def test_patch_reports_modules_it_does_not_recognise():
    assert fm.kernels.patch(nn.Linear(4, 4)) is False
    assert fm.kernels.patch(nn.LayerNorm(4)) is False
    assert fm.kernels.patch(fmnn.RMSNorm(8)) is True
    assert fm.kernels.patch(fmnn.SwiGLU(8, 16)) is True


class _UserNorm(nn.Module):
    """A user's own norm class -- the case `patch` exists for."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (self.weight.numel(),), self.weight, self.eps)


class _UserMLP(nn.Module):
    def __init__(self, dim, hidden, bias=True):
        super().__init__()
        self.w1, self.w3 = nn.Linear(dim, hidden, bias=bias), nn.Linear(dim, hidden, bias=bias)
        self.w2 = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def test_patch_fuses_a_user_block_without_changing_its_output():
    torch.manual_seed(0)
    norm, mlp = _UserNorm(16), _UserMLP(16, 32)
    x, y = torch.randn(4, 16), torch.randn(4, 16)
    norm_out, mlp_out = norm(x).detach().clone(), mlp(y).detach().clone()

    assert fm.kernels.patch(norm) is False           # `_UserNorm` does not look like a norm...
    assert fm.kernels.patch(norm, kind="norm") is True   # ...but we know what it is
    assert fm.kernels.patch(mlp, kind="mlp") is True
    assert torch.allclose(norm(x), norm_out, atol=1e-6)
    assert torch.allclose(mlp(y), mlp_out, atol=1e-6)

    # the split variant rebuilds w1|w3 into one GEMM and must agree too
    torch.manual_seed(0)
    mlp2 = _UserMLP(16, 32)
    ref = mlp2(y).detach().clone()
    assert fm.kernels.patch(mlp2, split=True) is True
    assert torch.allclose(mlp2(y), ref, atol=1e-6)
    assert hasattr(mlp2, "gate_up")


def test_enable_does_not_double_count_pre_patched_modules():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = fmnn.RMSNorm(8), fmnn.RMSNorm(8)
            self.mlp = fmnn.SwiGLU(8, 16)

        def forward(self, x):
            return self.mlp(self.b(self.a(x)))

    m = Model()
    assert fm.kernels.patch(m.a) is True
    fm.kernels.enable(m)
    assert m._fm_fused == {"norm": 1, "mlp": 1}       # `a` was already fused by hand
    fm.kernels.enable(m)                              # idempotent on a second pass
    assert m._fm_fused == {"norm": 0, "mlp": 0}
    m(torch.randn(2, 8))


# ------------------------------------------------------------------- a custom architecture, trained
class _TinyLM(nn.Module):
    """Hand-written model: embedding + 2 blocks built from the parts, no GPT involved."""

    def __init__(self, vocab=64, dim=32, seq=16):
        super().__init__()
        self.cfg = type("C", (), {"max_seq": seq})()     # only used for tok/s, optional
        self.emb = nn.Embedding(vocab, dim)
        self.norms = nn.ModuleList([fmnn.RMSNorm(dim) for _ in range(2)])
        self.mlps = nn.ModuleList([fmnn.SwiGLU(dim, hidden=2 * dim) for _ in range(2)])
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, labels=None):
        h = self.emb(input_ids)
        for norm, mlp in zip(self.norms, self.mlps):
            h = h + mlp(norm(h))
        logits = self.head(h)
        if labels is None:
            return logits, None
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return ce, None


def test_custom_architecture_trains_with_trainer():
    cfg = fm.ForgeConfig()
    cfg.train.__dict__.update(max_steps=3, warmup_steps=0, lr=1e-3, log_every=99, resume=False,
                              bf16=False, eval_every=0, ckpt_every=0)
    torch.manual_seed(0)
    trainer = fm.Trainer(lambda ctx: _TinyLM(), cfg, fm.SyntheticLM(64, 16, n=128), micro_batch=4)
    metrics = [trainer.train_step() for _ in range(3)]
    assert all(m["loss"] == m["loss"] and m["loss"] > 0 for m in metrics)     # finite, positive
    assert all(torch.isfinite(torch.tensor(m["grad_norm"])) for m in metrics)
    assert [m["step"] for m in metrics] == [1, 2, 3]
    trainer.close()


def test_custom_architecture_trains_with_forge():
    torch.manual_seed(0)
    model = _TinyLM(vocab=64, dim=32, seq=16)
    forge = fm.Forge(model, torch.optim.AdamW(model.parameters(), lr=1e-3), precision="fp32",
                     accum=2, quiet=True)
    forge.loss_fn = lambda out, batch: F.cross_entropy(
        out.reshape(-1, out.shape[-1]), batch["labels"].reshape(-1)) if torch.is_tensor(out) \
        else out[0]
    ds = fm.SyntheticLM(64, 16, n=128)
    loader = [{"input_ids": ds[i]["input_ids"][None], "labels": ds[i]["labels"][None]}
              for i in range(8)]
    for batch in fm.async_loader(loader, forge.device):
        forge.step(batch)
    assert forge.step_count >= 1


def test_blocks_compile_with_torch_compile():
    block = nn.Sequential(fmnn.RMSNorm(16), fmnn.SwiGLU(16, hidden=32)).eval()
    x = torch.randn(3, 16)
    with torch.no_grad():
        ref = block(x)
        out = torch.compile(block, backend="aot_eager")(x)
    assert torch.allclose(out, ref, atol=1e-5)


def test_fusion_preserves_biases():
    """A fused gate|up GEMM must keep both biases -- the repo's own MLP has none, user MLPs do."""
    torch.manual_seed(0)
    for bias in (True, False):
        mlp = _UserMLP(16, 32, bias=bias)
        x = torch.randn(4, 16)
        ref = mlp(x).detach().clone()
        assert fm.kernels.patch(mlp, split=True) is True
        assert torch.allclose(mlp(x), ref, atol=1e-6), f"bias={bias}"

    # and the module's own to_split()
    torch.manual_seed(0)
    block = fmnn.SwiGLU(16, 32, bias=True)
    x = torch.randn(4, 16)
    ref = block(x).detach().clone()
    block.to_split(inplace=True)
    assert torch.allclose(block(x), ref, atol=1e-6)


def test_patch_split_falls_back_when_only_one_bias_exists():
    """Asymmetric biases cannot be concatenated: keep two GEMMs, fuse only the epilogue."""
    torch.manual_seed(0)
    mlp = _UserMLP(16, 32, bias=True)
    mlp.w3 = nn.Linear(16, 32, bias=False)
    x = torch.randn(4, 16)
    ref = mlp(x).detach().clone()
    assert fm.kernels.patch(mlp, split=True) is True
    assert torch.allclose(mlp(x), ref, atol=1e-6)
    assert not hasattr(mlp, "gate_up")           # fell back to the two-GEMM path


def test_patch_keeps_the_eps_of_a_module_that_has_none():
    """A module with no `eps` attribute gets 1e-6 (not torch's finfo(dtype).eps). Patching must
    leave its numbers bit-identical -- mistaking one convention for the other moves activations
    by ~2e-6 in fp32 before this was pinned down."""
    class _NoEps(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(16))

        def forward(self, x):
            return F.rms_norm(x, (16,), self.weight, 1e-6)

    torch.manual_seed(0)
    norm, x = _NoEps(), torch.randn(4, 16)
    ref = norm(x).detach().clone()
    assert fm.kernels.patch(norm, kind="norm") is True
    assert torch.equal(norm(x), ref)


def test_patch_rejects_a_gate_up_pair_with_different_input_widths():
    """`w1: 16->32`, `w3: 8->32` is not a SwiGLU this library can fuse: the one-GEMM rewrite
    assumes both projections read the same input."""
    mlp = _UserMLP(16, 32)
    mlp.w3 = nn.Linear(8, 32)
    assert _is_swiglu(mlp) is False
    assert fm.kernels.patch(mlp) is False
    with pytest.raises(TypeError):
        fm.kernels.patch(mlp, kind="mlp")


def test_split_patch_renames_parameters_so_old_checkpoints_need_mapping():
    """Documented consequence of `split=True`: `w1`/`w3` become `gate_up`. Pin it, because a
    silently incompatible state_dict is the kind of thing a user finds out at load time."""
    torch.manual_seed(0)
    mlp = _UserMLP(16, 32)
    before = set(mlp.state_dict())
    assert fm.kernels.patch(mlp, split=True) is True
    after = set(mlp.state_dict())
    assert "gate_up.weight" in after and "w1.weight" not in after
    assert before - after == {"w1.weight", "w1.bias", "w3.weight", "w3.bias"}


def test_forward_with_residual_rejects_a_broadcastable_but_wrong_shape():
    """`[1, d]` would broadcast over the batch and quietly turn the residual into a no-op."""
    norm = fmnn.RMSNorm(64)
    x = torch.randn(4, 64)
    for bad in (torch.randn(1, 64), torch.randn(4, 32)):
        with pytest.raises(ValueError, match="residual shape"):
            norm.forward_with_residual(x, bad)
    normed, res = norm.forward_with_residual(x, torch.randn(4, 64))
    assert normed.shape == res.shape == (4, 64)


def test_degenerate_sizes_are_rejected():
    """A zero-width layer is a config bug: say so at construction instead of emitting empty
    tensors that only surface as a mysterious loss of 0.0 three steps later."""
    with pytest.raises(ValueError, match="hidden > 0"):
        fmnn.SwiGLU(8, 0)
    with pytest.raises(ValueError, match="dim > 0"):
        fmnn.RMSNorm(0)
    assert fmnn.RMSNorm(8)(torch.randn(0, 8)).shape == (0, 8)   # empty *batch* is fine


def test_docstring_examples_run():
    """`>>>` in these two docstrings is executable, and CI runs it."""
    import doctest

    result = doctest.testmod(fmnn, verbose=False)
    assert result.failed == 0, f"{result.failed} failing doctest(s) in forgemind/nn.py"
    assert result.attempted >= 6
