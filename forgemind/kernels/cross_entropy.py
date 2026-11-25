"""Cross-entropy over logits: loss + exact input gradient, drop-in for ``F.cross_entropy``.

Supports ``reduction`` in ``{mean, sum, none}``, ``ignore_index`` and ``label_smoothing``
-- the features training loops actually use. Class weights, soft-capping, z-loss and the
token-accuracy extras from the source kernel are out of scope (documented adaptations).

Ported from Liger-Kernel's ``cross_entropy.py``. Four of their design choices do not
survive the port, and each one is a bug this repo either already hit or has an open
Liger issue for:

1. **the source kernel writes the gradient into the input logits** ("the forward pass
   reuses X as gradient storage"). That is the root of Liger's open #1674 (non-contiguous
   logits return the raw input as the gradient) and of the ``identify_mutated_tensors``
   warning in #174 -- under ``torch.compile`` a mutated input makes dynamo assume every
   input is mutated, which is how inductor exceptions start. This port writes ``dlogits``
   to a **separate buffer** and only ever reads the logits.
2. **reduction happens outside the kernel, in tensor math**: ``n_non_ignore`` is a 0-dim
   tensor, never a Python ``int()``/``.item()`` read (a graph break under tracing). The
   scale travels into the backward kernel as a 0-dim tensor with stride 0 (broadcast) or
   stride 1 (per-row for ``reduction='none'``).
3. **scalar arguments are never method-called**: no ``eps.to(tl.float32)`` anywhere --
   the Liger #1601 pattern that crashes ``torch.compile`` (eager gets a Triton scalar,
   ``generate_ttir`` passes a plain Python float). ``label_smoothing`` is a
   ``tl.constexpr``; the rest enter through plain arithmetic. Banned repo-wide by
   ``tests/test_kernel_hygiene.py``.
4. **the two-pass structure is kept but un-fused**: pass 1 (online softmax, Algorithm 3
   from arXiv:1805.02867) computes ``m``/``d`` for the loss; the backward recomputes
   them for the gradient. One program owns one row, so the ``-eps`` smoothing terms and
   the catastrophic-cancellation-safe fp32 rewrite of the true-class gradient stay
   exactly as derived in the source.

Also from the source and kept: int64 program ids (row offset ``pid * stride`` overflows
int32 once ``B*T*V`` is large -- triton-lang/triton#1058), the ``exp2``-based online
softmax (emits the hardware ``ex2.approx`` instruction), and the label-smoothing
derivation from Liger PR #198.
"""

# Portions of this file are adapted from Liger-Kernel's ``cross_entropy.py``:
#
#   Copyright (c) 2024-present Liger Kernel contributors. All rights reserved.
#   Licensed under the BSD 2-Clause "Simplified" License (SPDX: BSD-2-Clause);
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       https://github.com/linkedin/Liger-Kernel/blob/main/LICENSE
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import annotations

import torch

from . import _fallback as fb
from . import _ops
from ._common import fast_ok, next_pow2, tl, triton, triton_available, tune

_CE_BLOCK_CAP = 4096             # vocab tiles per pass (register budget)

_REDUCTIONS = ("mean", "sum", "none")


if triton_available():
    # constexpr-instantiated: a bare python global inside a @jit body does not compile
    # on triton >= 3.2 (same class as topk's _E_BLOCK, task-15).
    _LOG2_E = tl.constexpr(1.4426950408889634)   # e^x = 2^(x * LOG2_E): hardware ex2.approx

    @triton.jit
    def _ce_fwd(
        Loss, X, X_stride, T,
        n_cols, ignore_index,
        LS: tl.constexpr,        # label_smoothing (0.0 when off)
        BLOCK: tl.constexpr,     # vocab tile
    ):
        pid = tl.program_id(0).to(tl.int64)   # pid * stride can exceed int32 (triton#1058)
        y = tl.load(T + pid)
        if y == ignore_index:
            tl.store(Loss + pid, 0.0)
            return
        base = X + pid * X_stride

        # -- pass 1: online softmax (arXiv:1805.02867 alg. 3) -> max m, sum d, smooth term
        m = float("-inf")
        d = 0.0
        scaled_x_sum = 0.0
        eps = LS / n_cols
        i = 0
        while i < n_cols:
            off = i + tl.arange(0, BLOCK)
            msk = off < n_cols
            xb = tl.load(base + off, mask=msk, other=float("-inf")).to(tl.float32)
            if LS > 0.0:
                # scale X beforehand to avoid overflow (Liger #198 derivation)
                scaled_x_sum += tl.sum(tl.where(msk, -eps * tl.where(msk, xb, 0.0), 0.0))
            block_max = tl.max(xb)
            m_new = tl.maximum(m, block_max)
            d = d * tl.exp2((m - m_new) * _LOG2_E) + tl.sum(
                tl.exp2((xb - m_new) * _LOG2_E))
            m = m_new
            i = i + BLOCK

        # lse = max + log(sum e^(x - max)); loss = lse - x_y  (>= 1 inside the log)
        lse = m + tl.log(d)
        x_y = tl.load(base + y).to(tl.float32)
        loss = lse - x_y
        if LS > 0.0:
            # H(q', p) = (1-ls) H(q, p) + ls * (lse - mean(x)) -- see source derivation
            smooth = scaled_x_sum + LS * lse
            loss = loss * (1.0 - LS) + smooth
        tl.store(Loss + pid, loss)

    @triton.jit
    def _ce_bwd(
        DX, X, X_stride, T, Scale, scale_stride,
        n_cols, ignore_index,
        LS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Per-row gradient of the loss, multiplied by the reduction scale in-place.

        Scale is a 0-dim tensor with stride 0 (mean/sum: dloss / n_non_ignore) or an
        [N] tensor with stride 1 (reduction='none': the per-row dloss).
        """
        pid = tl.program_id(0).to(tl.int64)
        y = tl.load(T + pid)
        scale = tl.load(Scale + pid * scale_stride)
        if y == ignore_index:
            i = 0
            while i < n_cols:
                off = i + tl.arange(0, BLOCK)
                tl.store(DX + pid * X_stride + off, 0.0, mask=off < n_cols)
                i = i + BLOCK
            return
        base = X + pid * X_stride
        dbase = DX + pid * X_stride

        # -- pass 1: the same online softmax the forward used
        m = float("-inf")
        d = 0.0
        eps = LS / n_cols
        i = 0
        while i < n_cols:
            off = i + tl.arange(0, BLOCK)
            msk = off < n_cols
            xb = tl.load(base + off, mask=msk, other=float("-inf")).to(tl.float32)
            block_max = tl.max(xb)
            m_new = tl.maximum(m, block_max)
            d = d * tl.exp2((m - m_new) * _LOG2_E) + tl.sum(
                tl.exp2((xb - m_new) * _LOG2_E))
            m = m_new
            i = i + BLOCK

        # -- pass 2: dx_i = softmax_i - ls/V  (the -(1-ls) true-class term is added below)
        i = 0
        while i < n_cols:
            off = i + tl.arange(0, BLOCK)
            msk = off < n_cols
            xb = tl.load(base + off, mask=msk, other=float("-inf")).to(tl.float32)
            g = tl.exp2((xb - m) * _LOG2_E) / d - eps
            g = g * scale
            tl.store(dbase + off, g.to(DX.dtype.element_ty), mask=msk)
            i = i + BLOCK

        # true-class correction, recomputed in fp32: (p_y - 1) cancels as p_y -> 1, so a
        # read-modify-write of the loop's stored value would lose the significant bits
        # in bf16/fp16 (kept verbatim from the source kernel). Barrier first so the
        # loop's store to DX[y] cannot land after this overwrite.
        tl.debug_barrier()
        x_y = tl.load(base + y).to(tl.float32)
        p_y = tl.exp2((x_y - m) * _LOG2_E) / d
        g_y = (p_y - eps - (1.0 - LS)) * scale
        tl.store(dbase + y, g_y.to(DX.dtype.element_ty))

    _ce_fwd = tune(_ce_fwd, key=["n_cols"])
    _ce_bwd = tune(_ce_bwd, key=["n_cols"])


def _ok(logits: torch.Tensor, targets: torch.Tensor) -> bool:
    if not (logits.dim() >= 2 and logits.shape[-1] > 0 and logits.numel() > 0
            and fast_ok(logits)):
        return False
    rows = logits.numel() // logits.shape[-1]
    return targets.numel() == rows and targets.dtype == torch.int64


def _flat(logits: torch.Tensor, targets: torch.Tensor):
    v = logits.shape[-1]
    return logits.contiguous().view(-1, v), targets.contiguous().view(-1), v


def _launch_fwd(logits2, targets2, v: int, ignore_index: int,
                label_smoothing: float) -> torch.Tensor:
    rows = targets2.numel()
    block = next_pow2(min(v, _CE_BLOCK_CAP))
    loss = torch.empty(rows, dtype=torch.float32, device=logits2.device)
    _ce_fwd[(rows,)](loss, logits2, v, targets2, v, ignore_index,
                     float(label_smoothing), block)
    return loss


def _reduce(row_loss: torch.Tensor, targets: torch.Tensor, reduction: str,
            ignore_index: int, dtype: torch.dtype) -> torch.Tensor:
    """Reduction in tensor math only -- no ``.item()``, so tracing stays graph-safe."""
    if reduction == "none":
        # F returns one loss per target in the TARGETS' shape ([B,T] for [B,T,V]
        # logits), not a flat [N] -- tests with 1-D targets hide the difference.
        return row_loss.view(targets.shape).to(dtype)
    total = row_loss.sum()
    if reduction == "mean":
        total = total / (targets != ignore_index).sum()
    return total.to(dtype)


def _scale_tensor(dloss: torch.Tensor, targets: torch.Tensor, reduction: str,
                  ignore_index: int) -> tuple[torch.Tensor, int]:
    """``(Scale, stride)`` for the backward kernel: 0-dim stride-0, or [N] stride-1."""
    if reduction == "none":
        s = dloss.reshape(-1).float()
        return s, 1
    s = dloss.float()
    if reduction == "mean":
        s = s / (targets != ignore_index).sum().to(s.dtype)
    return s, 0


def _launch_bwd(logits2, targets2, v: int, dloss: torch.Tensor, reduction: str,
                ignore_index: int, label_smoothing: float) -> torch.Tensor:
    rows = targets2.numel()
    block = next_pow2(min(v, _CE_BLOCK_CAP))
    scale, stride = _scale_tensor(dloss, targets2, reduction, ignore_index)
    dx = torch.empty_like(logits2)
    _ce_bwd[(rows,)](dx, logits2, v, targets2, scale, stride, v, ignore_index,
                     float(label_smoothing), block)
    return dx


def _check_reduction(reduction: str) -> None:
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")


# ------------------------------------------------------------------ eager: autograd.Function
class _CrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, reduction, ignore_index, label_smoothing):
        logits2, targets2, v = _flat(logits, targets)
        row_loss = _launch_fwd(logits2, targets2, v, ignore_index, label_smoothing)
        loss = _reduce(row_loss, targets, reduction, ignore_index, logits.dtype)
        ctx.reduction, ctx.ignore_index, ctx.label_smoothing = (
            reduction, ignore_index, label_smoothing)
        ctx.shape = tuple(logits.shape)
        ctx.save_for_backward(logits2, targets2)
        return loss

    @staticmethod
    def backward(ctx, dloss):
        logits2, targets2 = ctx.saved_tensors
        dx = _launch_bwd(logits2, targets2, logits2.shape[-1], dloss, ctx.reduction,
                         ctx.ignore_index, ctx.label_smoothing)
        return dx.view(ctx.shape), None, None, None, None


# ------------------------------------------------------- traced: torch.library.custom_op
def _ce_fake(logits: torch.Tensor, targets: torch.Tensor, reduction: str,
             ignore_index: int, label_smoothing: float) -> torch.Tensor:
    if reduction == "none":
        return torch.zeros(targets.numel(), dtype=logits.dtype, device=logits.device)
    return torch.zeros((), dtype=logits.dtype, device=logits.device)


def _ce_setup(ctx, inputs, output):
    logits, targets, reduction, ignore_index, label_smoothing = inputs
    ctx.save_for_backward(logits, targets)
    ctx.reduction, ctx.ignore_index, ctx.label_smoothing = (
        reduction, ignore_index, label_smoothing)


def _ce_bwd_op(ctx, dloss):
    logits, targets = ctx.saved_tensors
    dx = _CE_BWD_OP(logits, targets, dloss, ctx.reduction, ctx.ignore_index,
                    ctx.label_smoothing)
    # one gradient per forward input (logits, targets, reduction, ignore_index,
    # label_smoothing); only logits is differentiable
    return dx, None, None, None, None


def _ce_bwd_fake(logits: torch.Tensor, targets: torch.Tensor, dloss: torch.Tensor,
                 reduction: str, ignore_index: int,
                 label_smoothing: float) -> torch.Tensor:
    return torch.empty_like(logits)


def _ce_bwd_op_fwd(logits: torch.Tensor, targets: torch.Tensor, dloss: torch.Tensor,
                   reduction: str, ignore_index: int,
                   label_smoothing: float) -> torch.Tensor:
    _check_reduction(reduction)
    logits2, targets2, v = _flat(logits, targets)
    dx = _launch_bwd(logits2, targets2, v, dloss, reduction, ignore_index,
                     label_smoothing)
    return dx.view(tuple(logits.shape))


def _ce_op_fwd(logits: torch.Tensor, targets: torch.Tensor, reduction: str,
               ignore_index: int, label_smoothing: float) -> torch.Tensor:
    _check_reduction(reduction)
    logits2, targets2, v = _flat(logits, targets)
    row_loss = _launch_fwd(logits2, targets2, v, ignore_index, label_smoothing)
    return _reduce(row_loss, targets, reduction, ignore_index, logits.dtype)


_CE_BWD_OP = _ops.define_backward("cross_entropy", _ce_bwd_op_fwd, _ce_bwd_fake)
_CE_OP = _ops.define("cross_entropy", _ce_op_fwd, _ce_fake, _ce_setup, _ce_bwd_op)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean",
                  ignore_index: int = -100, label_smoothing: float = 0.0) -> torch.Tensor:
    """``F.cross_entropy(logits, targets, reduction, ignore_index, label_smoothing)``.

    ``targets`` must hold class indices (``int64``), each in ``[0, C)`` or equal to
    ``ignore_index`` -- the same contract as torch, which device-asserts it.

    Layout: the LAST dim is classes and leading dims flatten (``[B,T,V]`` logits take
    ``[B,T]`` targets -- Liger's convention, what LLM training wants). F instead
    takes DIM 1 as classes for >2D inputs and *rejects* that pairing, so 3-D parity
    checks compare against the flattened 2-D call; both the kernel and the fallback
    implement this layout (the fallback must flatten too, or it raises exactly where
    the kernel succeeds).
    """
    _check_reduction(reduction)
    if not _ok(logits, targets):
        return fb.cross_entropy(logits, targets, reduction, ignore_index, label_smoothing)
    if _CE_OP is not None and _ops.wanted():
        return _CE_OP(logits, targets, reduction, ignore_index, label_smoothing)
    return _CrossEntropyFn.apply(logits, targets, reduction, ignore_index,
                                 label_smoothing)
