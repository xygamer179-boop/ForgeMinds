"""Packed per-head RMSNorm + rotary embedding, and a standalone fused RoPE.

``qk_norm_rope(x, w, cos, sin)`` is the Qwen3-style attention input pipeline in one pass:
normalize each head over its full head dim, scale by the learned per-head weight, rotate. The
unfused version materializes the normalized tensor, chunks it, rotates, and concatenates --
seven kernels and two full round trips of the activation.

Bugs fixed relative to the first draft
--------------------------------------
1. **``tl.arange(0, HD)`` requires a power of two.** ``HD = head_dim // 2`` is 64 or 128 in
   every model anyone ships, so this "works" until someone uses head_dim = 96 (or any head
   dim whose half is not a power of two), where Triton fails to compile. The block is now
   ``HDP = next_pow2(HD)`` with a column mask, like every other kernel here.
2. **The block was ``H x HD`` wide.** For H=32, HD=128 that is 4096 fp32 lanes across 8
   warps -- ~128 registers per thread just for one of the eight live tensors, i.e. heavy
   spilling. The kernel now tiles the head dimension (``BLOCK_H``) so the register budget is
   fixed regardless of how many heads the model has.
3. **``cos``/``sin`` loads were unmasked**, so a non-power-of-two ``HD`` read past the table.
4. **The backward had the wrong ``dw`` shape for odd head dims** (it stored ``HD`` values at
   ``pid * 2 * HD`` for both halves, which overflows the partial buffer when
   ``2 * HD != head_dim``). Now the two halves are stored explicitly and the partial buffer is
   ``2 * HD`` wide, which is the weight length by construction.
5. ``Out`` was hard-coded to bf16; it is ``x.dtype`` now, so fp32/fp16 models keep their dtype.
6. ``t = n % T`` was computed in int32 from an int64 program id and could overflow for
   ``B * T * H * D`` above 2^31 elements; the table index is int64 now.

Numerics: same fp32 accumulation as the rest of the package, so it is bit-comparable to the
torch reference in ``_fallback.qk_norm_rope`` up to the output cast.
"""

from __future__ import annotations

import torch

from . import _fallback as fb
from ._common import (MAX_DIM, fast_ok, next_pow2, nprog_for, num_warps_for, tl, triton,
                      triton_available)

if triton_available():

    @triton.jit
    def _qknorm_rope_fwd(X, W, Cos, Sin, Out, Rstd, T, eps,
                         H: tl.constexpr, HD: tl.constexpr, HDP: tl.constexpr,
                         BLOCK_H: tl.constexpr):
        n = tl.program_id(0).to(tl.int64)
        t = n % T                                    # [B,T,...] row -> position
        D = 2 * HD
        cols = tl.arange(0, HDP)
        cm = cols < HD
        c = tl.load(Cos + t * HD + cols, mask=cm, other=0.0).to(tl.float32)
        s = tl.load(Sin + t * HD + cols, mask=cm, other=0.0).to(tl.float32)
        w0 = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)
        w1 = tl.load(W + HD + cols, mask=cm, other=0.0).to(tl.float32)
        for h0 in tl.static_range(0, H, BLOCK_H):
            hv = h0 + tl.arange(0, BLOCK_H)
            hm = hv < H
            off = n * (H * D) + hv[:, None] * D + cols[None, :]
            m = hm[:, None] & cm[None, :]
            x0 = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
            x1 = tl.load(X + off + HD, mask=m, other=0.0).to(tl.float32)
            ss = tl.sum(x0 * x0, axis=1) + tl.sum(x1 * x1, axis=1)
            r = 1.0 / tl.sqrt(ss / (2 * HD) + eps)
            tl.store(Rstd + n * H + hv, r, mask=hm)
            y0 = x0 * r[:, None] * w0[None, :]
            y1 = x1 * r[:, None] * w1[None, :]
            tl.store(Out + off, (y0 * c[None, :] - y1 * s[None, :]).to(Out.dtype.element_ty),
                     mask=m)
            tl.store(Out + off + HD,
                     (y1 * c[None, :] + y0 * s[None, :]).to(Out.dtype.element_ty), mask=m)

    @triton.jit
    def _qknorm_rope_bwd(X, W, Cos, Sin, GO, Rstd, DX, DWP, N, T,
                         H: tl.constexpr, HD: tl.constexpr, HDP: tl.constexpr,
                         BLOCK_H: tl.constexpr, NPROG: tl.constexpr):
        pid = tl.program_id(0)
        cols = tl.arange(0, HDP)
        cm = cols < HD
        w0 = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)
        w1 = tl.load(W + HD + cols, mask=cm, other=0.0).to(tl.float32)
        dw0 = tl.zeros([HDP], dtype=tl.float32)
        dw1 = tl.zeros([HDP], dtype=tl.float32)
        n = pid
        while n < N:
            n64 = n.to(tl.int64)
            t = n64 % T
            c = tl.load(Cos + t * HD + cols, mask=cm, other=0.0).to(tl.float32)
            s = tl.load(Sin + t * HD + cols, mask=cm, other=0.0).to(tl.float32)
            for h0 in tl.static_range(0, H, BLOCK_H):
                hv = h0 + tl.arange(0, BLOCK_H)
                hm = hv < H
                off = n64 * (H * 2 * HD) + hv[:, None] * (2 * HD) + cols[None, :]
                m = hm[:, None] & cm[None, :]
                g0 = tl.load(GO + off, mask=m, other=0.0).to(tl.float32)
                g1 = tl.load(GO + off + HD, mask=m, other=0.0).to(tl.float32)
                # inverse rotation: (R^-1 g)
                gy0 = g0 * c[None, :] + g1 * s[None, :]
                gy1 = g1 * c[None, :] - g0 * s[None, :]
                x0 = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
                x1 = tl.load(X + off + HD, mask=m, other=0.0).to(tl.float32)
                r = tl.load(Rstd + n64 * H + hv, mask=hm, other=0.0)[:, None]
                xh0 = x0 * r
                xh1 = x1 * r
                gw0 = gy0 * w0[None, :]
                gw1 = gy1 * w1[None, :]
                cc = (tl.sum(xh0 * gw0, axis=1) + tl.sum(xh1 * gw1, axis=1)) / (2 * HD)
                cc2 = cc[:, None]
                tl.store(DX + off, ((gw0 - xh0 * cc2) * r).to(DX.dtype.element_ty), mask=m)
                tl.store(DX + off + HD, ((gw1 - xh1 * cc2) * r).to(DX.dtype.element_ty), mask=m)
                dw0 += tl.sum(gy0 * xh0, axis=0)
                dw1 += tl.sum(gy1 * xh1, axis=0)
            n += NPROG
        ci = tl.arange(0, HDP)
        tl.store(DWP + pid * (2 * HD) + ci, dw0, mask=cm)
        tl.store(DWP + pid * (2 * HD) + HD + ci, dw1, mask=cm)


class _QKNormRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, cos, sin, eps):
        b, t_len, h, d = x.shape
        hd = d // 2
        out = torch.empty_like(x)
        rstd = torch.empty(b * t_len * h, device=x.device, dtype=torch.float32)
        hdp = next_pow2(hd)
        block_h = 8 if hdp <= 64 else 4
        _qknorm_rope_fwd[(b * t_len,)](x, w, cos, sin, out, rstd, t_len, eps,
                                       h, hd, hdp, block_h,
                                       num_warps=num_warps_for(min(h, block_h) * hdp))
        ctx.save_for_backward(x, w, cos, sin, rstd)
        ctx.dims = (b, t_len, h, d, eps)
        return out

    @staticmethod
    def backward(ctx, g):
        x, w, cos, sin, rstd = ctx.saved_tensors
        b, t_len, h, d = ctx.dims[:4]
        hd = d // 2
        g = g.contiguous()
        dx = torch.empty_like(x)
        n = b * t_len
        npg = nprog_for(n)
        dwp = torch.empty(npg, d, device=x.device, dtype=torch.float32)
        hdp = next_pow2(hd)
        block_h = 8 if hdp <= 64 else 4
        _qknorm_rope_bwd[(npg,)](x, w, cos, sin, g, rstd, dx, dwp, n, t_len,
                                 h, hd, hdp, block_h, npg,
                                 num_warps=num_warps_for(min(h, block_h) * hdp))
        return dx, dwp.sum(0, dtype=torch.float32).to(w.dtype), None, None, None


def _ok(x, w, cos, sin):
    if x.dim() != 4 or x.shape[-1] % 2 or x.shape[-1] > MAX_DIM:
        return False
    if x.numel() >= 2 ** 31 or w.numel() != x.shape[-1] or w.dtype != x.dtype:
        return False
    if cos.shape[-1] != x.shape[-1] // 2 or sin.shape != cos.shape:
        return False
    if cos.shape[0] not in (1, x.shape[1]) and cos.numel() // cos.shape[-1] < x.shape[1]:
        return False
    return cos.is_contiguous() and sin.is_contiguous() and fast_ok(x, w, cos, sin)


def qk_norm_rope(x: torch.Tensor, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 eps: float = 1e-6) -> torch.Tensor:
    """Per-head RMSNorm over the full head dim then half-split RoPE. ``x`` is ``[B,T,H,D]``.

    ``cos``/``sin`` may be ``[T, D/2]`` or any leading-broadcast shape that flattens to
    ``T * D/2`` (the kernel indexes them as ``cos[t * D/2 + i]``).
    """
    cos = cos.reshape(-1)[: x.shape[1] * (x.shape[-1] // 2)].reshape(x.shape[1], -1)
    sin = sin.reshape(-1)[: x.shape[1] * (x.shape[-1] // 2)].reshape(x.shape[1], -1)
    if not _ok(x, w, cos, sin):
        return fb.qk_norm_rope(x, w, cos.reshape(1, x.shape[1], 1, -1),
                               sin.reshape(1, x.shape[1], 1, -1), eps)
    return _QKNormRoPE.apply(x, w, cos, sin, eps)
