"""Packed SwiGLU: elementwise gate, and the fused gate|up projection epilogue.

Two entry points:

``swiglu(gate, up)``
    ``silu(gate) * up`` in one pass, with the backward fused into a single kernel.

``swiglu_split(x)``
    ``x`` is the output of a **fused** gate+up projection, ``[..., 2 * hidden]`` where the
    first half is the gate and the second half is up. This is the layout you get from
    concatenating ``w1`` and ``w3`` into one ``[2h, d]`` matrix -- one GEMM instead of two,
    which is worth more than everything else in this file on a real GPU, because the two
    projections get one launch, one shared weight stream, and better tensor-core occupancy.
    The kernel reads both halves with one address computation per row.

Bugs fixed relative to the first draft
--------------------------------------
1. **``Out`` was hard-coded to ``tl.bfloat16``** while the caller allocated
   ``torch.empty_like(gate)``. For an fp32 or fp16 model the kernel writes bf16 bit patterns
   into a wider buffer: silent corruption of half of every element. Every store now casts to
   the destination pointer's element type.
2. The same bug was in the backward, in both ``dg`` and ``du``.
3. **No tail/mask handling for the vectorised path**: the flat kernels are fine, but the
   split kernel needs a row mask, not an element mask, or it reads past the last row.
4. ``NUM_WARPS=4`` was fixed; it is derived from the block size now (a 4096-wide block with 4
   warps is 32 elements per thread, i.e. 8 vector loads each).
5. ``gate``/``up`` were assumed contiguous but not made so, so a sliced (non-contiguous) view
   would be read with the wrong stride and produce garbage. ``.contiguous()`` is explicit.

The derivative is the exact one: with ``s = sigmoid(g)``,
``d/dg [silu(g) * u] = u * s * (1 + g * (1 - s))`` and ``d/du = silu(g)``. The first draft had
it right; it is spelled out here because the fused split kernel recomputes it from ``x``.
"""

from __future__ import annotations

import torch

from . import _fallback as fb
from ._common import (fast_ok, next_pow2, num_warps_for, tl, triton, triton_available)

if triton_available():

    @triton.jit
    def _swiglu_fwd_kernel(G, U, Out, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            g = tl.load(G + off).to(tl.float32)
            u = tl.load(U + off).to(tl.float32)
            tl.store(Out + off, (g * tl.sigmoid(g) * u).to(Out.dtype.element_ty))
        else:
            m = off < N
            g = tl.load(G + off, mask=m, other=0.0).to(tl.float32)
            u = tl.load(U + off, mask=m, other=0.0).to(tl.float32)
            tl.store(Out + off, (g * tl.sigmoid(g) * u).to(Out.dtype.element_ty), mask=m)

    @triton.jit
    def _swiglu_bwd_kernel(DY, G, U, DG, DU, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            dy = tl.load(DY + off).to(tl.float32)
            g = tl.load(G + off).to(tl.float32)
            u = tl.load(U + off).to(tl.float32)
            sig = tl.sigmoid(g)
            tl.store(DU + off, (dy * g * sig).to(DU.dtype.element_ty))
            tl.store(DG + off, (dy * u * sig * (1.0 + g * (1.0 - sig))).to(DG.dtype.element_ty))
        else:
            m = off < N
            dy = tl.load(DY + off, mask=m, other=0.0).to(tl.float32)
            g = tl.load(G + off, mask=m, other=0.0).to(tl.float32)
            u = tl.load(U + off, mask=m, other=0.0).to(tl.float32)
            sig = tl.sigmoid(g)
            tl.store(DU + off, (dy * g * sig).to(DU.dtype.element_ty), mask=m)
            tl.store(DG + off, (dy * u * sig * (1.0 + g * (1.0 - sig))).to(DG.dtype.element_ty),
                     mask=m)

    @triton.jit
    def _swiglu_split_fwd(X, Out, N, h: tl.constexpr, HP: tl.constexpr,
                          ROWS: tl.constexpr, EVEN_COL: tl.constexpr, EVEN_ROW: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, HP)
        cm = cols < h
        rm = rows < N
        base = rows.to(tl.int64)[:, None] * (2 * h) + cols[None, :]
        if EVEN_COL and EVEN_ROW:
            g = tl.load(X + base).to(tl.float32)
            u = tl.load(X + base + h).to(tl.float32)
            tl.store(Out + rows.to(tl.int64)[:, None] * h + cols[None, :],
                     (g * tl.sigmoid(g) * u).to(Out.dtype.element_ty))
        else:
            m = rm[:, None] & cm[None, :]
            g = tl.load(X + base, mask=m, other=0.0).to(tl.float32)
            u = tl.load(X + base + h, mask=m, other=0.0).to(tl.float32)
            tl.store(Out + rows.to(tl.int64)[:, None] * h + cols[None, :],
                     (g * tl.sigmoid(g) * u).to(Out.dtype.element_ty), mask=m)

    @triton.jit
    def _swiglu_split_bwd(X, DY, DX, N, h: tl.constexpr, HP: tl.constexpr,
                          ROWS: tl.constexpr, EVEN_COL: tl.constexpr, EVEN_ROW: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, HP)
        cm = cols < h
        rm = rows < N
        r64 = rows.to(tl.int64)[:, None]
        src = r64 * (2 * h) + cols[None, :]      # [N, 2h] row for gate|up
        dst = r64 * (2 * h) + cols[None, :]      # gate half of the [N, 2h] gradient
        gsrc = r64 * h + cols[None, :]           # [N, h] row for dy
        if EVEN_COL and EVEN_ROW:
            dy = tl.load(DY + gsrc).to(tl.float32)
            g = tl.load(X + src).to(tl.float32)
            u = tl.load(X + src + h).to(tl.float32)
            sig = tl.sigmoid(g)
            tl.store(DX + dst, (dy * u * sig * (1.0 + g * (1.0 - sig))).to(DX.dtype.element_ty))
            tl.store(DX + dst + h, (dy * g * sig).to(DX.dtype.element_ty))
        else:
            m = rm[:, None] & cm[None, :]
            dy = tl.load(DY + gsrc, mask=m, other=0.0).to(tl.float32)
            g = tl.load(X + src, mask=m, other=0.0).to(tl.float32)
            u = tl.load(X + src + h, mask=m, other=0.0).to(tl.float32)
            sig = tl.sigmoid(g)
            tl.store(DX + dst, (dy * u * sig * (1.0 + g * (1.0 - sig))).to(DX.dtype.element_ty),
                     mask=m)
            tl.store(DX + dst + h, (dy * g * sig).to(DX.dtype.element_ty), mask=m)


def _flat_ok(gate, up):
    return (gate.shape == up.shape and gate.is_contiguous() and up.is_contiguous()
            and gate.numel() > 0 and gate.numel() < 2 ** 31 and fast_ok(gate, up))


class _SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        block = 1024
        _swiglu_fwd_kernel[(triton.cdiv(n, block),)](gate, up, out, n, block, n % block == 0,
                                                     num_warps=num_warps_for(block))
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, dy):
        gate, up = ctx.saved_tensors
        dy = dy.contiguous()
        n = gate.numel()
        block = 1024
        dg = torch.empty_like(gate)
        du = torch.empty_like(up)
        _swiglu_bwd_kernel[(triton.cdiv(n, block),)](dy, gate, up, dg, du, n, block,
                                                     n % block == 0, num_warps=num_warps_for(block))
        return dg, du


class _SwiGLUSplitFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        *lead, two_h = x.shape
        h = two_h // 2
        x = x.contiguous().view(-1, two_h)
        n = x.shape[0]
        out = torch.empty((n, h), device=x.device, dtype=x.dtype)
        hp = next_pow2(h)
        rows = max(1, 4096 // hp)
        _swiglu_split_fwd[(triton.cdiv(n, rows),)](x, out, n, h, hp, rows, h == hp, n % rows == 0,
                                                   num_warps=num_warps_for(rows * hp))
        ctx.save_for_backward(x)
        ctx.shape = tuple(lead) + (h,)
        ctx.two_h = two_h
        return out.view(ctx.shape)

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        two_h = ctx.two_h
        h = two_h // 2
        n = x.shape[0]
        dy = dy.contiguous().view(n, h)
        dx = torch.empty_like(x)
        hp = next_pow2(h)
        rows = max(1, 4096 // hp)
        _swiglu_split_bwd[(triton.cdiv(n, rows),)](x, dy, dx, n, h, hp, rows, h == hp,
                                                   n % rows == 0, num_warps=num_warps_for(rows * hp))
        return dx.view(x.shape)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up``."""
    if not _flat_ok(gate, up):
        return fb.swiglu(gate, up)
    return _SwiGLUFn.apply(gate, up)


def swiglu_split(x: torch.Tensor) -> torch.Tensor:
    """``silu(x[..., :h]) * x[..., h:]`` for a fused gate|up projection ``[..., 2h]``."""
    if x.shape[-1] % 2 or not x.is_contiguous() or x.numel() >= 2 ** 31 or not fast_ok(x):
        h = x.shape[-1] // 2
        return (torch.nn.functional.silu(x[..., :h].float()) * x[..., h:].float()).to(x.dtype)
    return _SwiGLUSplitFn.apply(x)
