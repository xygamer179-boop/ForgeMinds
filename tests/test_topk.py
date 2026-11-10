"""Tests for ``fm.kernels.topk`` (FlashTopK-inspired, two-phase Triton selection).

Run with ``TRITON_INTERPRET=1`` to execute the kernel on CPU; without it the fused path
is unavailable on this machine and the tests exercise the torch fallback (the same shape
of test the other kernel suites have).
"""

import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from forgemind.kernels.topk import topk  # noqa: E402
from forgemind.kernels import _common  # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET") == "1"


def fused() -> bool:
    """True when the Triton kernels are what actually runs."""
    x = torch.randn(8, 32)
    before = dict(_common._STATS)
    topk(x, 4)
    return dict(_common._STATS).get("fast", 0) > before.get("fast", 0)


FUSED = fused()


def canon(v: torch.Tensor, idx: torch.Tensor) -> None:
    """value descending, then index ascending inside ties."""
    k = v.shape[-1]
    for row in range(v.reshape(-1, k).shape[0]):
        vv, ii = v.reshape(-1, k)[row], idx.reshape(-1, k)[row]
        for j in range(k - 1):
            assert vv[j] >= vv[j + 1] or torch.isnan(vv[j]), f"not desc at {row},{j}"
            if vv[j] == vv[j + 1] and not torch.isnan(vv[j]):
                assert ii[j] < ii[j + 1], f"tie order broken at {row},{j}"


def selects(x: torch.Tensor, v: torch.Tensor, idx: torch.Tensor) -> bool:
    k = v.shape[-1]
    flat = x.reshape(-1, x.shape[-1])
    got = torch.gather(flat, 1, idx.reshape(-1, k))
    return torch.equal(got, v.reshape(-1, k))


@pytest.mark.parametrize("shape,k", [
    ((4, 64), 8), ((2, 100), 1), ((2, 100), 99), ((1, 512), 512),
    ((3, 4096), 64), ((2, 5000), 100), ((4, 128), 17), ((5, 1), 1),
    ((2, 600), 4), ((4, 5000), 512), ((2, 300, 97), 12),
])
def test_multiset_matches_torch(shape, k):
    torch.manual_seed(0)
    x = torch.randn(*shape)
    v, idx = topk(x, k)
    tv, _ = torch.topk(x, k, dim=-1)
    assert torch.equal(v.sort(-1).values, tv.sort(-1).values), "value multiset differs"
    assert selects(x, v, idx), "indices do not reproduce the values"
    canon(v, idx)


def test_tie_band_keeps_smallest_indices():
    x = torch.tensor([[1.0, 3.0, 3.0, 2.0, 3.0]])
    v, idx = topk(x, 3)
    assert v[0].tolist() == [3.0, 3.0, 3.0]
    assert sorted(idx[0].tolist()) == [1, 2, 4]         # the right ELEMENTS either way
    if FUSED:
        assert idx[0].tolist() == [1, 2, 4]             # canonical order is ours (torch: arbitrary)


def test_nan_is_largest_and_passes_through():
    x = torch.tensor([[float("nan"), 1.0, 2.0, float("nan"), 0.0]])
    v, idx = topk(x, 3)
    assert math.isnan(v[0, 0]) and math.isnan(v[0, 1])
    assert idx[0, :2].tolist() == [0, 3]                # NaN ties: smallest indices first
    assert v[0, 2].item() == 2.0
    # torch agrees on WHICH elements (values), its tie order may differ
    tv, _ = torch.topk(x, 3)
    assert torch.isnan(tv[0, 0]) and torch.isnan(tv[0, 1])


def test_k_out_of_range_matches_torch():
    x = torch.randn(4, 8)
    with pytest.raises(RuntimeError, match="selected index k out of range"):
        topk(x, 9)
    with pytest.raises(RuntimeError, match="selected index k out of range"):
        torch.topk(x, 9)
    with pytest.raises(RuntimeError, match="selected index k out of range"):
        topk(x, 0)


def test_dim_other_than_last_is_torch():
    x = torch.randn(4, 8)
    v, idx = topk(x, 2, dim=0)
    tv, ti = torch.topk(x, 2, dim=0)
    assert torch.equal(v, tv) and torch.equal(idx, ti)
    with pytest.raises(ValueError, match="ndim"):
        topk(torch.tensor(3.0), 1)


def test_fallback_reasons_are_counted():
    before = dict(_common._STATS)
    topk(torch.randint(0, 100, (4, 64)), 8)            # int dtype -> torch
    topk(torch.randn(2, 1024), 600)                    # k > 512 -> torch
    after = dict(_common._STATS)
    assert after["fallback"] > before.get("fallback", 0)


@pytest.mark.skipif(not INTERP, reason="needs TRITON_INTERPRET=1")
def test_fused_path_is_what_runs():
    assert FUSED


def test_sorted_false_still_returns_sorted():
    torch.manual_seed(1)
    x = torch.randn(3, 256)
    v, idx = topk(x, 10, sorted=False)
    assert (v[..., :-1] >= v[..., 1:]).all()


def test_grad_matches_torch_on_distinct_values():
    torch.manual_seed(0)
    x = torch.randn(3, 128, requires_grad=True)
    v, idx = topk(x, 10)
    g1 = torch.autograd.grad(v.sum(), x)[0]
    x2 = x.detach().clone().requires_grad_()
    v2, _ = torch.topk(x2, 10)
    g2 = torch.autograd.grad(v2.sum(), x2)[0]
    assert torch.equal(g1, g2)


def test_grad_flows_only_through_values():
    x = torch.randn(2, 64, requires_grad=True)
    v, idx = topk(x, 5)
    assert idx.requires_grad is False
    (v * 2).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert int((x.grad != 0).sum()) == 2 * 5            # exactly the selected slots


def test_compile_fullgraph_and_break_count():
    import torch._dynamo

    class M(torch.nn.Module):
        def forward(self, x):
            v, _ = topk(x, 8)
            return v.sum()

    x = torch.randn(4, 64, requires_grad=True)
    torch.compile(M(), fullgraph=True, backend="eager")(x).backward()
    x2 = x.detach().clone().requires_grad_()
    M()(x2).backward()
    assert torch.equal(x.grad, x2.grad)
    ex = torch._dynamo.explain(M())(torch.randn(2, 64))
    assert ex.graph_count == 1 and ex.graph_break_count == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
