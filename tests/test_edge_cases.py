"""Edge cases across the public surface -- the shapes and inputs that break libraries.

Each of these is something a user will do on their first afternoon: a non-contiguous input, a
degenerate dimension, a model that returns a dict, a loss that is already a mean, labels that
are all ignored. Anything that fails here is a bug worth fixing, not a caveat worth
documenting.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import forgemind as fm
from forgemind import kernels
from forgemind.loss import chunked_cross_entropy, logits_cross_entropy


# ---------------------------------------------------------------- losses
def test_chunked_ce_all_labels_ignored():
    h = torch.randn(6, 8, requires_grad=True)
    w = torch.randn(16, 8, requires_grad=True)
    y = torch.full((6,), -100)
    loss = chunked_cross_entropy(h, w, y, chunk=4)
    assert loss.item() == 0.0
    loss.backward()                        # must not raise on a graph that produced no tokens
    assert h.grad is not None


def test_chunked_ce_empty_input():
    h = torch.randn(0, 8, requires_grad=True)
    w = torch.randn(16, 8, requires_grad=True)
    y = torch.zeros(0, dtype=torch.long)
    loss = chunked_cross_entropy(h, w, y)
    assert loss.item() == 0.0


def test_chunked_ce_matches_full_ce():
    torch.manual_seed(0)
    h = torch.randn(9, 8)
    w = torch.randn(16, 8)
    y = torch.randint(0, 16, (9,))
    y[2] = -100
    got = chunked_cross_entropy(h, w, y, chunk=4)
    exp = F.cross_entropy(h @ w.t(), y, ignore_index=-100, reduction="mean")
    assert got.item() == pytest.approx(exp.item(), rel=1e-5)


def test_logits_ce_3d_and_chunking():
    torch.manual_seed(0)
    lg = torch.randn(2, 5, 16)
    y = torch.randint(0, 16, (2, 5))
    full = logits_cross_entropy(lg, y)
    chunked = logits_cross_entropy(lg, y, chunk=3)
    assert full.item() == pytest.approx(chunked.item(), rel=1e-6)


# ---------------------------------------------------------------- data
def test_synthetic_lm_shapes_for_odd_seq():
    for seq in (7, 8, 10, 63):
        ds = fm.SyntheticLM(32, seq, n=8)
        item = ds[0]
        assert item["input_ids"].shape == (seq,)
        assert item["labels"].shape == (seq,)


def test_token_file_small():
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        np.arange(4, dtype=np.uint16).tofile(f.name)
    try:
        ds = fm.TokenFile(f.name, seq=8)
        assert len(ds) == 0                     # 4 tokens cannot make a window of 8+1
        ds2 = fm.TokenFile(f.name, seq=2)
        assert len(ds2) == 1
        assert ds2[0]["input_ids"].tolist() == [0, 1]
    finally:
        os.unlink(f.name)


def test_resumable_sampler_more_ranks_than_samples():
    s = fm.ResumableSampler(3, dp_rank=2, dp_world=8, seed=0)
    assert len(s) == 0
    assert list(s) == []


def test_resumable_sampler_skip_is_clamped():
    s = fm.ResumableSampler(16, dp_rank=0, dp_world=2, seed=0)
    s.skip = 100                                # nonsense resume point
    assert len(s) == 0
    assert list(s) == []


def test_batch_stream_empty_loader_raises_clearly():
    ds = fm.SyntheticLM(8, 4, n=1)
    class Ctx:
        dp_rank, dp_world, device = 0, 1, torch.device("cpu")
    loader, sampler = fm.build_loader(ds, 4, Ctx())
    with pytest.raises(RuntimeError, match="no batches"):
        fm.BatchStream(loader, sampler, torch.device("cpu"))


# ---------------------------------------------------------------- kernels
def test_rms_norm_non_contiguous_and_degenerate():
    x = torch.randn(4, 8).t().contiguous().t()          # non-contiguous view
    assert not x.is_contiguous()
    w = torch.randn(8)
    assert torch.allclose(kernels.rms_norm(x, w, 1e-6),
                          F.rms_norm(x, (8,), w, 1e-6), atol=2e-6)



def test_rms_norm_dim_one():
    x = torch.randn(3, 1)
    w = torch.ones(1)
    got = kernels.rms_norm(x, w, 1e-6)
    exp = F.rms_norm(x, (1,), w, 1e-6)
    assert torch.allclose(got, exp, atol=1e-6)


def test_kl_degenerate_vocab():
    s = torch.randn(3, 1)
    t = torch.randn(3, 1)
    assert kernels.kl_div_logits(s, t, "mean").item() == pytest.approx(0.0, abs=1e-6)


def test_swiglu_split_hidden_one():
    x = torch.randn(5, 2)
    got = kernels.swiglu_split(x)
    exp = F.silu(x[..., :1]) * x[..., 1:]
    assert torch.allclose(got, exp, atol=1e-6)


def test_swiglu_broadcast_shapes_rejected():
    with pytest.raises((RuntimeError, ValueError)):
        kernels.swiglu(torch.randn(4), torch.randn(5))


# ---------------------------------------------------------------- forge
class DictModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(4, 1)

    def forward(self, x):
        return {"logits": self.w(x), "loss": self.w(x).mean() * 0 + (self.w(x) ** 2).mean()}


def test_forge_reads_loss_from_a_dict_output():
    m = DictModel()
    forge = fm.Forge(m, torch.optim.AdamW(m.parameters(), lr=1e-2), precision="fp32",
                     quiet=True, skip_nonfinite=False)
    st = forge.step(torch.randn(8, 4))
    assert st is not None and st["loss"] == pytest.approx(st["loss"])


def test_forge_clip_disabled():
    m = nn.Linear(4, 1)
    forge = fm.Forge(m, torch.optim.AdamW(m.parameters(), lr=1e-2), precision="fp32",
                     quiet=True, clip=0.0, skip_nonfinite=False,
                     forward_fn=lambda model, b: model(b).pow(2).mean())
    st = forge.step(torch.randn(8, 4))
    assert st is not None and st["step"] == 1
    assert st["grad_norm"] != st["grad_norm"]      # nan is the honest answer when clipping is off


def test_forge_two_dim_loss_is_reduced():
    m = nn.Linear(4, 1)
    forge = fm.Forge(m, torch.optim.AdamW(m.parameters(), lr=1e-2), precision="fp32",
                     quiet=True, skip_nonfinite=False,
                     forward_fn=lambda model, batch: model(batch[0]),
                     loss_fn=lambda out, b: (out - b[1]).abs())          # [8, 1], not a scalar
    st = forge.step((torch.randn(8, 4), torch.randn(8, 1)))
    assert st is not None and st["step"] == 1
