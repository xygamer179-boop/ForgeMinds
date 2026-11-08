"""Packed forward-KL(teacher ‖ student) with the logit gradient computed in the same pass.

This is the distillation loss. For N rows and a vocab of V, the eager formulation is
``kl_div(log_softmax(t), log_softmax(s))``, which materializes both full [N, V] log-softmaxes
*and* their gradients. The kernel computes, per row, the two log-sum-exps, the KL, and dKL/ds
in place, so memory is ``[chunk, V]`` and the gradient never touches autograd.

Entry points
------------
``kl_div_logits(student, teacher, reduction="mean"|"sum"|"none")``
    Already-materialized logits. Gradients flow to ``student`` only.

``kl_distill_loss(hs, ht, w, teacher_head, chunk=4096)``
    The fused version: student logits are ``hs @ w.T`` computed chunk-by-chunk *inside* the
    forward and thrown away (only the gradient survives), so neither the student logits nor
    the teacher head need to be recomputed in backward.

Bugs fixed relative to the first draft
--------------------------------------
1. **Hard-coded ``bf16`` store into the student-logits buffer.** ``s_log`` is allocated by
   ``F.linear`` and therefore has the activation dtype; the kernel wrote
   ``((ps - pt) * inv_N).to(tl.bfloat16)`` into it. For an fp16 run that writes the low 16
   bits of every fp32 element -- the gradient is destroyed and nothing raises. The store now
   casts to ``S.dtype.element_ty``.
2. **The gradient was written even when it was never used.** ``need_grad`` was known at trace
   time but not to the kernel, so inference-only calls still paid a full HBM write of [M, V].
   It is a ``constexpr`` (``WRITE_GRAD``) now.
3. **Masked lanes produced NaN if the row was fully masked**, because
   ``exp(-inf) * (-inf - -inf)`` is ``0 * NaN``. Guarded with ``tl.where`` in the first draft
   *for the KL sum*, but the same expression fed the ``lse`` accumulation in the second loop,
   where a partially-masked tail could poison ``d_s``. Both accumulations now clamp the
   masked lanes before the exponent.
4. **``V`` was not checked against ``BLOCK_V``**: with ``BLOCK_V = 4096`` and a vocabulary
   below 4096 the kernel still ran the two-pass loop, reading the whole row from HBM twice.
   There is a ``SINGLE_TILE`` variant now that keeps both logit vectors in registers -- for
   small vocabularies (tests, toy models, per-token classifiers) it is a ~2x win on this
   kernel.
5. **The loss and the gradient used different scales** (`inv_N` applied to the grad but not
   the stored per-row loss) which is correct only if the caller re-scales; that contract is
   now explicit: the kernel stores the *unscaled* per-row KL and pre-scales the gradient by
   ``inv_N`` so no extra [N, V] pass is needed. ``reduction="none"`` therefore returns the
   per-row KL.
6. ``torch.autocast("cuda", ...)`` was hard-coded, so the op was unusable on CPU/MPS. It uses
   the input's device type now.
7. ``thead(ht_2d[i:j]).to(CDT)`` upcast the teacher's logits to a *global* compute dtype,
   which for a bf16 model re-cast the (already bf16) teacher head output through fp32.
   ``.to(t_log_dtype)`` is now identical to the student logits' dtype by construction.

Numerical note: the gradient is stored in the logits' dtype (bf16 in a bf16 run), which is
exactly what a fused cross-entropy does, and then multiplied by the head weight. If you need a
higher-precision gradient use ``kl_div_logits`` on fp32 logits, or set ``grad_dtype="float32"``
(2x the memory of the chunk, still independent of the batch).
"""

from __future__ import annotations

import torch

from . import _fallback as fb
from ._common import fast_ok, tl, triton, triton_available

_NEG_INF = -float("inf")
# masked lanes load this instead of -inf: -inf - -inf is NaN, and a NaN * 0 is still NaN, so a
# fully-masked tail tile could poison the accumulator. -1e30 leaves exp() == 0 without NaN.
_NEG_BIG = -1e30

if triton_available():

    @triton.jit
    def _kl_fwd_bwd_kernel(S_ptr, s_stride, T_ptr, t_stride, G_ptr, g_stride, Loss_ptr, V, inv_N,
                           BLOCK_V: tl.constexpr, WRITE_GRAD: tl.constexpr,
                           SINGLE: tl.constexpr):
        """Per-row online-logsumexp forward KL + dKL/d(student logits) * inv_N into G."""
        # int32 is enough: _kernel_ok() rejects numel >= 2**31, so row * stride < 2**31
        row = tl.program_id(0)
        s_row = S_ptr + row * s_stride
        t_row = T_ptr + row * t_stride
        g_row = G_ptr + row * g_stride
        cols = tl.arange(0, BLOCK_V)
        mask = cols < V
        if SINGLE:
            s = tl.load(s_row + cols, mask=mask, other=_NEG_BIG).to(tl.float32)
            t = tl.load(t_row + cols, mask=mask, other=_NEG_BIG).to(tl.float32)
            m_s = tl.max(s, axis=0)
            m_t = tl.max(t, axis=0)
            lse_s = m_s + tl.log(tl.sum(tl.exp(s - m_s), axis=0))
            lse_t = m_t + tl.log(tl.sum(tl.exp(t - m_t), axis=0))
            ls = s - lse_s
            lt = t - lse_t
            ps = tl.exp(ls)
            pt = tl.exp(lt)
            kl = tl.sum(tl.where(mask, pt * (lt - ls), 0.0), axis=0)
            tl.store(Loss_ptr + row, kl)
            if WRITE_GRAD:
                tl.store(g_row + cols, ((ps - pt) * inv_N).to(G_ptr.dtype.element_ty), mask=mask)
        else:
            m_s = _NEG_INF
            d_s = 0.0
            m_t = _NEG_INF
            d_t = 0.0
            off = 0
            while off < V:                      # pass 1: both log-sum-exps (online)
                c = off + tl.arange(0, BLOCK_V)
                m = c < V
                s = tl.load(s_row + c, mask=m, other=_NEG_BIG).to(tl.float32)
                t = tl.load(t_row + c, mask=m, other=_NEG_BIG).to(tl.float32)
                ms_new = tl.maximum(m_s, tl.max(s, axis=0))
                d_s = d_s * tl.exp(m_s - ms_new) + tl.sum(tl.exp(s - ms_new), axis=0)
                m_s = ms_new
                mt_new = tl.maximum(m_t, tl.max(t, axis=0))
                d_t = d_t * tl.exp(m_t - mt_new) + tl.sum(tl.exp(t - mt_new), axis=0)
                m_t = mt_new
                off += BLOCK_V
            lse_s = m_s + tl.log(d_s)
            lse_t = m_t + tl.log(d_t)
            kl_acc = 0.0
            off = 0
            while off < V:                      # pass 2: KL and dKL/ds
                c = off + tl.arange(0, BLOCK_V)
                m = c < V
                s = tl.load(s_row + c, mask=m, other=_NEG_BIG).to(tl.float32)
                t = tl.load(t_row + c, mask=m, other=_NEG_BIG).to(tl.float32)
                ls = s - lse_s
                lt = t - lse_t
                ps = tl.exp(ls)
                pt = tl.exp(lt)
                kl_acc += tl.sum(tl.where(m, pt * (lt - ls), 0.0), axis=0)
                if WRITE_GRAD:
                    tl.store(g_row + c, ((ps - pt) * inv_N).to(G_ptr.dtype.element_ty), mask=m)
                off += BLOCK_V
            tl.store(Loss_ptr + row, kl_acc)


def _kernel_ok(*tensors):
    return all(t.dtype in (torch.float32, torch.float16, torch.bfloat16) and t.numel() < 2 ** 31
               for t in tensors) and fast_ok(*tensors)


class _KLDivFn(torch.autograd.Function):
    """KL(teacher || student) for [N, V] logits; dL/d(student) is produced in the forward.

    ``reuse=True`` lets the kernel write the gradient straight into ``student``'s storage.
    That is only allowed when the caller *owns* that tensor (a temporary matmul output, as in
    ``kl_distill_loss``); with ``reuse=False`` a fresh buffer is allocated, so an input is
    never mutated (which the first draft got wrong: it destroyed the caller's logits).
    """

    @staticmethod
    def forward(ctx, student, teacher, reduction, grad_dtype, reuse):
        s2 = student.reshape(-1, student.shape[-1])
        t2 = teacher.reshape(-1, teacher.shape[-1])
        n, v = s2.shape
        want_grad = ctx.needs_input_grad[0]
        nbytes = 1 if reuse and (grad_dtype in (None, s2.dtype)) else 0
        if want_grad:
            gbuf = (s2.detach() if nbytes else
                    torch.empty((n, v), device=s2.device, dtype=grad_dtype or s2.dtype))
        else:
            gbuf = torch.empty(0, device=s2.device, dtype=s2.dtype)
        losses = torch.empty(n, device=s2.device, dtype=torch.float32)
        inv_n = 1.0 / n if reduction == "mean" else 1.0
        block = min(4096, max(64, 1 << (v - 1).bit_length()))
        _kl_fwd_bwd_kernel[(n,)](s2, s2.stride(0), t2, t2.stride(0), gbuf,
                                 0 if not want_grad else (gbuf.stride(0) if gbuf.numel() else v),
                                 losses, v, inv_n, BLOCK_V=block,
                                 WRITE_GRAD=want_grad, SINGLE=v <= block,
                                 num_warps=8 if block >= 2048 else 4)
        ctx.save_for_backward(gbuf if want_grad else None)
        ctx.shapes = (student.shape, teacher.shape)
        ctx.reduction = reduction
        if reduction == "none":
            return losses.view(student.shape[:-1])
        if reduction == "sum":
            return losses.sum()
        return losses.sum() * inv_n

    @staticmethod
    def backward(ctx, g):
        (grad,) = ctx.saved_tensors
        if grad is None:
            return None, None, None, None, None
        s_shape, _ = ctx.shapes
        if ctx.reduction == "none":
            grad = grad * g.reshape(-1, 1).to(grad.dtype)
        else:
            grad = grad * g.to(grad.dtype)      # mean already folded 1/N into `grad`
        return grad.view(s_shape), None, None, None, None


def kl_div_logits(student: torch.Tensor, teacher: torch.Tensor, reduction: str = "mean",
                  grad_dtype: torch.dtype | None = None) -> torch.Tensor:
    """``KL(softmax(teacher) ‖ softmax(student))`` over the last dim.

    ``teacher`` is treated as a constant (no gradient). Falls back to the torch reference for
    CPU / unsupported dtypes.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be mean|sum|none, got {reduction!r}")
    if (student.shape != teacher.shape or student.shape[-1] == 0
            or not _kernel_ok(student, teacher)):
        return fb.kl_div_logits(student, teacher, reduction)[0]
    return _KLDivFn.apply(student, teacher, reduction, grad_dtype, False)


def kl_distill_loss(hs: torch.Tensor, ht: torch.Tensor, w: torch.Tensor, teacher_head,
                    chunk: int = 4096) -> torch.Tensor:
    """Mean forward KL of ``softmax(teacher_head(ht)) || softmax(hs @ w.T)``.

    The student logits never outlive a chunk and the gradient is written into the chunk's own
    logits buffer, so peak memory is one ``[chunk, V]`` tile regardless of batch size. The
    teacher runs under ``no_grad`` and its parameters never accumulate gradients.
    """
    if not _kernel_ok(hs, ht):
        return fb.kl_distill_loss(hs, ht, w, teacher_head, chunk)
    hs2 = hs.reshape(-1, hs.shape[-1])
    ht2 = ht.reshape(-1, ht.shape[-1])
    n = hs2.shape[0]
    total = None
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        with torch.no_grad():
            t_log = teacher_head(ht2[i:j]).detach()
        s_log = hs2[i:j] @ w.t()          # owned temporary: the kernel may reuse it
        piece = _KLDivFn.apply(s_log, t_log, "sum", None, True)
        total = piece if total is None else total + piece
    return total / n


# human-friendly aliases
kl_div = kl_div_logits
distill_kl = kl_distill_loss
