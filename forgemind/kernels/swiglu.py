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

import typing

import torch

from . import _fallback as fb
from . import _ops
from ._common import (fast_ok, next_pow2, num_warps_for, tl, triton, triton_available)

if triton_available():

    @triton.jit
    def _swiglu_fwd_kernel(G, U, Out, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        # int32 offsets: `_flat_ok` refuses tensors with >= 2**31 elements, so a 32-bit
        # `pid * BLOCK + lane` cannot wrap, and it halves the index-register footprint.
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
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
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)      # int32, see _swiglu_fwd_kernel
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
        base = rows[:, None] * (2 * h) + cols[None, :]     # int32: rows * 2h < numel < 2**31
        dst = rows[:, None] * h + cols[None, :]             # computed once, used by both paths
        if EVEN_COL and EVEN_ROW:
            g = tl.load(X + base).to(tl.float32)
            u = tl.load(X + base + h).to(tl.float32)
            tl.store(Out + dst, (g * tl.sigmoid(g) * u).to(Out.dtype.element_ty))
        else:
            m = rm[:, None] & cm[None, :]
            g = tl.load(X + base, mask=m, other=0.0).to(tl.float32)
            u = tl.load(X + base + h, mask=m, other=0.0).to(tl.float32)
            tl.store(Out + dst, (g * tl.sigmoid(g) * u).to(Out.dtype.element_ty), mask=m)

    @triton.jit
    def _swiglu_split_bwd(X, DY, DX, N, h: tl.constexpr, HP: tl.constexpr,
                          ROWS: tl.constexpr, EVEN_COL: tl.constexpr, EVEN_ROW: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, HP)
        cm = cols < h
        rm = rows < N
        r32 = rows[:, None]                      # int32: rows * 2h < numel < 2**31
        src = r32 * (2 * h) + cols[None, :]      # [N, 2h] row for gate|up
        dst = src                                # gate half of the [N, 2h] gradient
        gsrc = r32 * h + cols[None, :]           # [N, h] row for dy
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


_SWIGLU_BLOCK = 1024            # elements per program for the flat elementwise kernels


def _split_rows(hp: int) -> int:
    """Rows per program for the gate|up epilogue: one row tile is ~4096 elements."""
    return max(1, 4096 // hp)


def _flat_ok(gate, up):
    return (gate.shape == up.shape and gate.is_contiguous() and up.is_contiguous()
            and gate.numel() > 0 and gate.numel() < 2 ** 31 and fast_ok(gate, up))


# --------------------------------------------------------------------------- launches
def _launch_swiglu(gate, up):
    gate = gate.contiguous()
    up = up.contiguous()
    out = torch.empty_like(gate)
    n = gate.numel()
    block = _SWIGLU_BLOCK
    _swiglu_fwd_kernel[(triton.cdiv(n, block),)](gate, up, out, n, block, n % block == 0,
                                                 num_warps=num_warps_for(block))
    return out


def _launch_swiglu_bwd(gate, up, dy):
    gate = gate.contiguous()
    up = up.contiguous()
    dy = dy.contiguous()
    n = gate.numel()
    block = _SWIGLU_BLOCK
    dg = torch.empty_like(gate)
    du = torch.empty_like(up)
    _swiglu_bwd_kernel[(triton.cdiv(n, block),)](dy, gate, up, dg, du, n, block, n % block == 0,
                                                 num_warps=num_warps_for(block))
    return dg, du


def _split_prepare(x):
    """``[..., 2h]`` -> ``([N, 2h] view, h, leading shape)``."""
    two_h = x.shape[-1]
    x2 = x.contiguous().view(-1, two_h)
    return x2, two_h // 2, tuple(x.shape[:-1])


def _launch_swiglu_split(x):
    x2, h, lead = _split_prepare(x)
    n = x2.shape[0]
    out = torch.empty((n, h), device=x2.device, dtype=x2.dtype)
    hp = next_pow2(h)
    rows = _split_rows(hp)
    _swiglu_split_fwd[(triton.cdiv(n, rows),)](x2, out, n, h, hp, rows, h == hp, n % rows == 0,
                                               num_warps=num_warps_for(rows * hp))
    return out.view(lead + (h,))


def _launch_swiglu_split_bwd(x, dy):
    x2, h, lead = _split_prepare(x)
    n = x2.shape[0]
    dy2 = dy.contiguous().view(n, h)
    dx = torch.empty_like(x2)
    hp = next_pow2(h)
    rows = _split_rows(hp)
    _swiglu_split_bwd[(triton.cdiv(n, rows),)](x2, dy2, dx, n, h, hp, rows, h == hp,
                                               n % rows == 0, num_warps=num_warps_for(rows * hp))
    return dx.view(lead + (2 * h,))


# ------------------------------------------------------------------ eager: autograd.Function
class _SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        out = _launch_swiglu(gate, up)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, dy):
        gate, up = ctx.saved_tensors
        return _launch_swiglu_bwd(gate, up, dy)


class _SwiGLUSplitFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = _launch_swiglu_split(x)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        return (_launch_swiglu_split_bwd(x, dy),)


# ------------------------------------------------------- traced: torch.library.custom_op
def _swiglu_fake(gate: torch.Tensor, up: torch.Tensor):
    return torch.empty_like(gate)


def _swiglu_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1])


def _swiglu_bwd_op(ctx, dy):
    gate, up = ctx.saved_tensors
    return _SWIGLU_BWD_OP(gate, up, dy)


def _swiglu_bwd_fake(gate: torch.Tensor, up: torch.Tensor, dy: torch.Tensor):
    return torch.empty_like(gate), torch.empty_like(up)


def _swiglu_bwd_op_fwd(gate: torch.Tensor, up: torch.Tensor, dy: torch.Tensor) -> typing.Tuple[
        torch.Tensor, torch.Tensor]:
    return _launch_swiglu_bwd(gate, up, dy)


def _split_fake(x: torch.Tensor):
    return torch.empty(x.shape[:-1] + (x.shape[-1] // 2,), device=x.device, dtype=x.dtype)


def _split_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


def _split_bwd_op(ctx, dy):
    (x,) = ctx.saved_tensors
    return (_SWIGLU_SPLIT_BWD_OP(x, dy),)


def _split_bwd_fake(x: torch.Tensor, dy: torch.Tensor):
    return torch.empty_like(x)


def _split_bwd_op_fwd(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    return _launch_swiglu_split_bwd(x, dy)


def _swiglu_op_fwd(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return _launch_swiglu(gate, up)


def _split_op_fwd(x: torch.Tensor) -> torch.Tensor:
    return _launch_swiglu_split(x)


_SWIGLU_BWD_OP = _ops.define_backward("swiglu", _swiglu_bwd_op_fwd, _swiglu_bwd_fake)
_SWIGLU_OP = _ops.define("swiglu", _swiglu_op_fwd, _swiglu_fake, _swiglu_setup, _swiglu_bwd_op)
_SWIGLU_SPLIT_BWD_OP = _ops.define_backward("swiglu_split", _split_bwd_op_fwd, _split_bwd_fake)
_SWIGLU_SPLIT_OP = _ops.define("swiglu_split", _split_op_fwd, _split_fake, _split_setup,
                               _split_bwd_op)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up``."""
    if not _flat_ok(gate, up):
        return fb.swiglu(gate, up)
    if _SWIGLU_OP is not None and _ops.wanted():
        return _SWIGLU_OP(gate, up)
    return _SwiGLUFn.apply(gate, up)


def swiglu_split(x: torch.Tensor) -> torch.Tensor:
    """``silu(x[..., :h]) * x[..., h:]`` for a fused gate|up projection ``[..., 2h]``."""
    if x.shape[-1] % 2:
        raise ValueError(f"swiglu_split needs an even last dimension (gate|up), got {x.shape[-1]}")
    if not x.is_contiguous() or x.numel() >= 2 ** 31 or not fast_ok(x):
        h = x.shape[-1] // 2
        return (torch.nn.functional.silu(x[..., :h].float()) * x[..., h:].float()).to(x.dtype)
    if _SWIGLU_SPLIT_OP is not None and _ops.wanted():
        return _SWIGLU_SPLIT_OP(x)
    return _SwiGLUSplitFn.apply(x)
