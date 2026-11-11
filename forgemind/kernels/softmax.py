"""``fm.kernels.softmax``: fused softmax over the last dimension (Liger/inductor pattern).

One kernel does the whole thing -- max-subtract for numerical stability, ``exp``, the row
sum and the divide -- so the intermediate never touches memory. The backward pass is the
standard closed form ``g = (gy - (gy * y).sum(-1, keepdim)) * y`` expressed as tensor
operations (the same trick ``fused_linear`` uses), so it stays ``fullgraph=True``-friendly.

Dispatch (like the other kernels): eager + CUDA (or ``TRITON_INTERPRET=1``) takes the fused
path; while ``torch.compile`` traces, plain ``aten::softmax`` appears in the graph (dynamo
sees one node); unsupported dtypes, ``numel >= 2**31`` or a fallback backend go to torch
and are counted in ``fm.kernels.stats()``. ``dim`` other than the last is torch's problem
-- no copy, no transpose, just the aten call.
"""

from __future__ import annotations

import torch

from . import _common
from ._common import fast_ok, num_warps_for, tl, triton, triton_available

_BLOCK = 1024

if triton_available():

    @triton.jit
    def _softmax_kernel(O, X, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        """row-wise softmax of ``X`` into ``O``; three chunked passes (max, exp+sum, scale).

        A row can be any length, so every pass loops over ``BLOCK``-sized chunks -- a
        single-block kernel would silently leave ``O[n:]`` (or the row tail) untouched.
        Masked lanes load ``-inf`` so they never win the max and ``exp(-inf - m) = 0``
        keeps them out of the sum without a second mask.
        """
        pid = tl.program_id(0).to(tl.int32)
        row = pid * N
        lanes = tl.arange(0, BLOCK)

        # pass 1: row max (NaN propagates, exactly as torch's softmax)
        m = float("-inf")
        start = 0
        while start < N:
            off = start + lanes
            if EVEN:
                xr = tl.load(X + row + off)
            else:
                xr = tl.load(X + row + off, mask=off < N, other=float("-inf"))
            m = tl.maximum(m, tl.max(xr, axis=0))
            start += BLOCK

        # pass 2: exp(x - m) stored in place of O, summing as we go
        s = 0.0
        start = 0
        while start < N:
            off = start + lanes
            if EVEN:
                xr = tl.load(X + row + off)
                z = tl.exp(xr - m)
                tl.store(O + row + off, z)
            else:
                mm = off < N
                xr = tl.load(X + row + off, mask=mm, other=float("-inf"))
                z = tl.exp(xr - m)
                tl.store(O + row + off, z, mask=mm)
            s += tl.sum(z, axis=0)
            start += BLOCK

        # pass 3: scale every stored exp by 1 / sum
        inv = 1.0 / s
        start = 0
        while start < N:
            off = start + lanes
            if EVEN:
                z = tl.load(O + row + off)
                tl.store(O + row + off, z * inv)
            else:
                mm = off < N
                z = tl.load(O + row + off, mask=mm, other=0.0)
                tl.store(O + row + off, z * inv, mask=mm)
            start += BLOCK


def _ok(x: torch.Tensor) -> bool:
    if x.numel() >= 2 ** 31:
        _common._note("softmax numel>=2**31")
        return False
    return fast_ok(x)


def _fused(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    rows = x.reshape(-1, n)
    out = torch.empty_like(rows)
    even = n % _BLOCK == 0
    _softmax_kernel[(rows.shape[0],)](out, rows, n, _BLOCK, even,
                                      num_warps=num_warps_for(_BLOCK))
    return out.reshape(x.shape)


class _Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        with torch.no_grad():
            y = _fused(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (y,) = ctx.saved_tensors
        if grad_y is None:
            return None
        # gy - sum(gy * y) per row, times y  (the exact derivative of softmax)
        s = (grad_y * y).sum(-1, keepdim=True)
        return (grad_y - s) * y


def softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Softmax over the last dimension (``torch.softmax(x, dim=-1)`` drop-in).

    * fused max-sub + exp + sum + divide in a single kernel (no intermediate in memory);
    * backward is the closed form, so it compiles with ``fullgraph=True``;
    * NaNs propagate exactly like torch's softmax; an all-``-inf`` row yields NaN, again
      like torch;
    * ``dtype`` converts the input first (as torch does), falling through ``.to()`` so the
      cast stays in the autograd graph.

    Falls back to ``torch.softmax`` for ``dim`` other than the last, non-float dtypes,
    ``numel >= 2**31``, or while ``torch.compile`` traces (one aten node).
    """
    if x.dim() < 1:
        raise ValueError(f"softmax needs ndim >= 1, got shape {tuple(x.shape)}")
    if dim not in (-1, x.dim() - 1):
        return torch.softmax(x, dim=dim, dtype=dtype)
    if dtype is not None and dtype != x.dtype:
        x = x.to(dtype)
    if _common.tracing() or not _ok(x):
        return torch.softmax(x, dim=-1)
    return _Softmax.apply(x)


__all__ = ["softmax"]
