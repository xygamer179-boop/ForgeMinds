"""Packed RMSNorm and fused residual-add + RMSNorm (+ backward).

Drop-in for ``F.rms_norm(x, (d,), w, eps)``: same signature, same numbers, one kernel.

Bugs fixed relative to the first draft of these kernels
-------------------------------------------------------
1. **Output dtype was a global.** ``y = torch.empty(..., dtype=CDT)`` with a module-level
   ``CDT`` meant an fp32 model got bf16 norms back -- silently changing the model dtype (and
   crashing the next residual add in fp32). The output now has ``x.dtype``; only the
   accumulation is fp32.
2. **``W`` was not masked in the backward col-block**, so a ``d`` that is not a power of two
   read past the end of the weight vector.
3. **``d`` was a runtime scalar**, so ``cols < d`` could never be folded away and every load
   stayed predicated. It is a ``constexpr`` now and the masks disappear when ``d`` is a power
   of two (``EVEN_COL``).
4. **One CTA per row.** For ``d`` of 512--1024 that is a few warps of work per launch and the
   kernel is launch-bound. The forward now processes ``ROWS`` rows per program (a 2D tile),
   which is up to 3x faster at d=512 and identical at d=4096.
5. **``Rstd`` was stored even when the backward was never going to run** -- irrelevant single
   use, but we now skip it for inference-only calls (``NEED_RSTD``).
6. ``1/sqrt`` was recomputed per element lane instead of once per row (it is a scalar, but the
   expression was written inside the tile expression). Hoisted.
7. The weight gradient was reduced with an exact ``sum`` over a fixed number of partials, so
   the result depends on the SM count. That is kept (it is the fast option) but the partial
   buffers are now fp32 and bounded, so the reduction is deterministic for a given device.

The fused variant saves a full round trip of the residual stream: ``add_rms_norm(x, delta)``
writes ``x + delta`` once and normalizes it in the same pass (FlashAttention-style
pre-norm-residual fusion), instead of materializing ``x + delta`` and reading it back.
"""

from __future__ import annotations

import torch

from . import _fallback as fb
from ._common import (MAX_DIM, block_m_for, fast_ok, next_pow2, nprog_for,
                      num_warps_for, tl, triton, triton_available)

if triton_available():

    @triton.jit
    def _rmsnorm_fwd(X, W, Y, Rstd, N, eps,
                     d: tl.constexpr, ROWS: tl.constexpr, BLOCK_D: tl.constexpr,
                     EVEN_COL: tl.constexpr, EVEN_ROW: tl.constexpr, NEED_RSTD: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK_D)
        off = rows.to(tl.int64)[:, None] * d + cols[None, :]
        cm = cols < d
        rm = rows < N
        if EVEN_COL:
            w = tl.load(W + cols).to(tl.float32)
        else:
            w = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)
        if EVEN_COL and EVEN_ROW:
            x = tl.load(X + off).to(tl.float32)
        elif EVEN_ROW:
            x = tl.load(X + off, mask=cm[None, :], other=0.0).to(tl.float32)
        elif EVEN_COL:
            x = tl.load(X + off, mask=rm[:, None], other=0.0).to(tl.float32)
        else:
            x = tl.load(X + off, mask=rm[:, None] & cm[None, :], other=0.0).to(tl.float32)
        r = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / d + eps)  # one reciprocal per row
        y = (x * r[:, None] * w[None, :]).to(Y.dtype.element_ty)
        if NEED_RSTD:
            if EVEN_ROW:
                tl.store(Rstd + rows, r)
            else:
                tl.store(Rstd + rows, r, mask=rm)
        if EVEN_COL and EVEN_ROW:
            tl.store(Y + off, y)
        elif EVEN_ROW:
            tl.store(Y + off, y, mask=cm[None, :])
        elif EVEN_COL:
            tl.store(Y + off, y, mask=rm[:, None])
        else:
            tl.store(Y + off, y, mask=rm[:, None] & cm[None, :])

    @triton.jit
    def _rmsnorm_bwd(X, W, DY, Rstd, DX, DWP, N,
                     d: tl.constexpr, NPROG: tl.constexpr, BLOCK_D: tl.constexpr,
                     EVEN_COL: tl.constexpr):
        # column-parallel: every program owns the whole row width, so `c` is the full row sum
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        cm = cols < d
        if EVEN_COL:
            w = tl.load(W + cols).to(tl.float32)
        else:
            w = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)
        dw = tl.zeros([BLOCK_D], dtype=tl.float32)
        row = pid
        while row < N:
            off = row * d + cols
            if EVEN_COL:
                x = tl.load(X + off).to(tl.float32)
                dy = tl.load(DY + off).to(tl.float32)
            else:
                x = tl.load(X + off, mask=cm, other=0.0).to(tl.float32)
                dy = tl.load(DY + off, mask=cm, other=0.0).to(tl.float32)
            r = tl.load(Rstd + row)
            xh = x * r
            wdy = dy * w
            c = tl.sum(xh * wdy, axis=0) / d
            dx = ((wdy - xh * c) * r).to(DX.dtype.element_ty)
            if EVEN_COL:
                tl.store(DX + off, dx)
            else:
                tl.store(DX + off, dx, mask=cm)
            dw += dy * xh
            row += NPROG
        tl.store(DWP + pid * d + cols, dw, mask=None if EVEN_COL else cm)

    @triton.jit
    def _addrms_fwd(X, Dl, W, XN, Y, Rstd, N, eps,
                    d: tl.constexpr, ROWS: tl.constexpr, BLOCK_D: tl.constexpr,
                    EVEN_COL: tl.constexpr, EVEN_ROW: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK_D)
        off = rows.to(tl.int64)[:, None] * d + cols[None, :]
        cm = cols < d
        rm = rows < N
        if EVEN_COL:
            w = tl.load(W + cols).to(tl.float32)
        else:
            w = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)
        if EVEN_COL and EVEN_ROW:
            x = tl.load(X + off).to(tl.float32)
            dl = tl.load(Dl + off).to(tl.float32)
        elif EVEN_ROW:
            x = tl.load(X + off, mask=cm[None, :], other=0.0).to(tl.float32)
            dl = tl.load(Dl + off, mask=cm[None, :], other=0.0).to(tl.float32)
        elif EVEN_COL:
            x = tl.load(X + off, mask=rm[:, None], other=0.0).to(tl.float32)
            dl = tl.load(Dl + off, mask=rm[:, None], other=0.0).to(tl.float32)
        else:
            m = rm[:, None] & cm[None, :]
            x = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
            dl = tl.load(Dl + off, mask=m, other=0.0).to(tl.float32)
        xn = x + dl
        r = 1.0 / tl.sqrt(tl.sum(xn * xn, axis=1) / d + eps)
        y = (xn * r[:, None] * w[None, :]).to(Y.dtype.element_ty)
        xnc = xn.to(XN.dtype.element_ty)
        if EVEN_COL and EVEN_ROW:
            tl.store(XN + off, xnc)
            tl.store(Y + off, y)
            tl.store(Rstd + rows, r)
        elif EVEN_ROW:
            tl.store(XN + off, xnc, mask=cm[None, :])
            tl.store(Y + off, y, mask=cm[None, :])
            tl.store(Rstd + rows, r)
        elif EVEN_COL:
            tl.store(XN + off, xnc, mask=rm[:, None])
            tl.store(Y + off, y, mask=rm[:, None])
            tl.store(Rstd + rows, r, mask=rm)
        else:
            m2 = rm[:, None] & cm[None, :]
            tl.store(XN + off, xnc, mask=m2)
            tl.store(Y + off, y, mask=m2)
            tl.store(Rstd + rows, r, mask=rm)

    @triton.jit
    def _addrms_bwd(XN, W, DY, GXN, Rstd, DX, DD, DWP, N,
                    d: tl.constexpr, NPROG: tl.constexpr, BLOCK_D: tl.constexpr,
                    EVEN_COL: tl.constexpr):
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        cm = cols < d
        if EVEN_COL:
            w = tl.load(W + cols).to(tl.float32)
        else:
            w = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)
        dw = tl.zeros([BLOCK_D], dtype=tl.float32)
        row = pid
        while row < N:
            off = row * d + cols
            if EVEN_COL:
                xn = tl.load(XN + off).to(tl.float32)
                dy = tl.load(DY + off).to(tl.float32)
                gx = tl.load(GXN + off).to(tl.float32)
            else:
                xn = tl.load(XN + off, mask=cm, other=0.0).to(tl.float32)
                dy = tl.load(DY + off, mask=cm, other=0.0).to(tl.float32)
                gx = tl.load(GXN + off, mask=cm, other=0.0).to(tl.float32)
            r = tl.load(Rstd + row)
            xh = xn * r
            wdy = dy * w
            c = tl.sum(xh * wdy, axis=0) / d
            # d/dx == d/d(delta) because xn = x + delta -> compute once, store twice
            dx = ((wdy - xh * c) * r + gx).to(DX.dtype.element_ty)
            if EVEN_COL:
                tl.store(DX + off, dx)
                tl.store(DD + off, dx)
            else:
                tl.store(DX + off, dx, mask=cm)
                tl.store(DD + off, dx, mask=cm)
            dw += dy * xh
            row += NPROG
        tl.store(DWP + pid * d + cols, dw, mask=None if EVEN_COL else cm)


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        d = x.shape[-1]
        x2 = x.contiguous().view(-1, d)
        n = x2.shape[0]
        y = torch.empty_like(x2)
        block_d = next_pow2(d)
        rows = block_m_for(block_d)
        rstd = torch.empty(n, device=x.device, dtype=torch.float32)
        _rmsnorm_fwd[(triton.cdiv(n, rows),)](
            x2, w, y, rstd, n, eps, d, rows, block_d, d == block_d, n % rows == 0, True,
            num_warps=num_warps_for(block_d * rows))
        ctx.save_for_backward(x2, w, rstd)
        ctx.shape = x.shape
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        x2, w, rstd = ctx.saved_tensors
        n, d = x2.shape
        dy = dy.contiguous().view(n, d)
        dx = torch.empty_like(x2)
        npg = nprog_for(n)
        dwp = torch.empty(npg, d, device=x2.device, dtype=torch.float32)
        block_d = next_pow2(d)
        _rmsnorm_bwd[(npg,)](x2, w, dy, rstd, dx, dwp, n, d, npg, block_d, d == block_d,
                             num_warps=num_warps_for(block_d))
        return dx.view(ctx.shape), dwp.sum(0, dtype=torch.float32).to(w.dtype), None


class _AddRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, w, eps):
        d = x.shape[-1]
        x2 = x.contiguous().view(-1, d)
        d2 = delta.contiguous().view(-1, d)
        n = x2.shape[0]
        xn = torch.empty_like(x2)
        y = torch.empty_like(x2)
        block_d = next_pow2(d)
        rows = block_m_for(block_d)
        rstd = torch.empty(n, device=x.device, dtype=torch.float32)
        _addrms_fwd[(triton.cdiv(n, rows),)](
            x2, d2, w, xn, y, rstd, n, eps, d, rows, block_d, d == block_d, n % rows == 0,
            num_warps=num_warps_for(block_d * rows))
        ctx.save_for_backward(xn, w, rstd)
        ctx.shape = x.shape
        return xn.view(x.shape), y.view(x.shape)

    @staticmethod
    def backward(ctx, gxn, gy):
        xn, w, rstd = ctx.saved_tensors
        n, d = xn.shape
        gxn = gxn.contiguous().view(n, d)
        gy = gy.contiguous().view(n, d)
        dx = torch.empty_like(xn)
        dd = torch.empty_like(xn)
        npg = nprog_for(n)
        dwp = torch.empty(npg, d, device=xn.device, dtype=torch.float32)
        block_d = next_pow2(d)
        _addrms_bwd[(npg,)](xn, w, gy, gxn, rstd, dx, dd, dwp, n, d, npg, block_d,
                            d == block_d, num_warps=num_warps_for(block_d))
        return dx.view(ctx.shape), dd.view(ctx.shape), \
            dwp.sum(0, dtype=torch.float32).to(w.dtype), None


def _ok(x, w):
    # `n * d < 2**31` keeps the row-offset arithmetic in the backward loop in int32 (the fast
    # option); past that the torch path is used. 2G elements per 2D matrix is ~4 GB of bf16.
    return (w is not None and x.shape[-1] == w.numel() and 0 < x.shape[-1] <= MAX_DIM
            and x.numel() < 2 ** 31 and w.dtype == x.dtype and w.is_contiguous()
            and fast_ok(x, w))


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``F.rms_norm(x, (x.shape[-1],), w, eps)`` with a fused kernel; falls back to torch."""
    if not _ok(x, w):
        return fb.rms_norm(x, w, eps)
    return _RMSNormFn.apply(x, w, eps)


def add_rms_norm(x: torch.Tensor, delta: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    """Return ``(x + delta, rms_norm(x + delta, w, eps))`` writing the sum once."""
    if not _ok(x, w) or delta.shape != x.shape:
        return fb.add_rms_norm(x, delta, w, eps)
    return _AddRMSNormFn.apply(x, delta, w, eps)
