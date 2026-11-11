"""Tests for ``fm.kernels.softmax`` (fused max-sub + exp + sum + divide over the last dim)."""

import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from forgemind.kernels.softmax import softmax  # noqa: E402
from forgemind.kernels import _common  # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET") == "1"

TOLS = {torch.float32: dict(rtol=1e-5, atol=1e-6),
        torch.bfloat16: dict(rtol=2e-2, atol=1e-3),
        torch.float16: dict(rtol=2e-2, atol=1e-3)}


def fused() -> bool:
    before = dict(_common._STATS)
    softmax(torch.randn(8, 32))
    return dict(_common._STATS).get("fast", 0) > before.get("fast", 0)


FUSED = fused()


@pytest.mark.parametrize("shape", [(4, 64), (5, 1000), (3, 4096), (2, 300, 97), (7, 1)])
def test_matches_torch(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape)
    y = softmax(x)
    ref = torch.softmax(x, dim=-1)
    assert y.shape == ref.shape and y.dtype == ref.dtype
    assert torch.allclose(y, ref, **TOLS[y.dtype])
    assert torch.allclose(y.sum(-1), torch.ones_like(y.sum(-1)), atol=1e-2)


def test_backward_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(4, 256, requires_grad=True)
    y = softmax(x)
    g1 = torch.autograd.grad(y.sum(), x)[0]
    x2 = x.detach().clone().requires_grad_()
    g2 = torch.autograd.grad(torch.softmax(x2, -1).sum(), x2)[0]
    assert torch.allclose(g1, g2, **TOLS[g1.dtype])


def test_dim_other_than_last_is_torch():
    x = torch.randn(4, 8)
    assert torch.allclose(softmax(x, dim=0), torch.softmax(x, dim=0))
    with pytest.raises(ValueError, match="ndim"):
        softmax(torch.tensor(1.0))


def test_nan_propagates_like_torch():
    x = torch.tensor([[float("nan"), 1.0, 2.0]])
    y = softmax(x)
    ref = torch.softmax(x, -1)
    assert math.isnan(y[0, 0]) and math.isnan(ref[0, 0])
    assert torch.allclose(torch.nan_to_num(y), torch.nan_to_num(ref))


def test_all_neg_inf_row_is_nan_like_torch():
    x = torch.tensor([[float("-inf"), float("-inf"), float("-inf")]])
    y = softmax(x)
    assert all(math.isnan(v) for v in y[0].tolist())


def test_dtype_conversion_keeps_autograd():
    x = torch.randn(4, 64, dtype=torch.float32, requires_grad=True)
    y = softmax(x, dtype=torch.bfloat16)
    assert y.dtype == torch.bfloat16
    y.float().sum().backward()
    assert x.grad is not None and x.grad.dtype == torch.float32


def test_fallback_reasons_are_counted():
    before = dict(_common._STATS)
    # float64 is unsupported by the fused path -> counted torch fallback
    y = softmax(torch.randn(4, 64, dtype=torch.float64))
    assert y.dtype == torch.float64
    after = dict(_common._STATS)
    assert after["fallback"] > before.get("fallback", 0)


def test_compile_fullgraph_and_break_count():
    import torch._dynamo

    class M(torch.nn.Module):
        def forward(self, x):
            return softmax(x).sum()

    x = torch.randn(4, 128, requires_grad=True)
    torch.compile(M(), fullgraph=True, backend="eager")(x).backward()
    x2 = x.detach().clone().requires_grad_()
    M()(x2).backward()
    # compiled = aten::softmax, eager = the fused kernel -- same math, ulps apart
    assert torch.allclose(x.grad, x2.grad, atol=1e-6, rtol=1e-5)
    ex = torch._dynamo.explain(M())(torch.randn(2, 128))
    assert ex.graph_count == 1 and ex.graph_break_count == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
