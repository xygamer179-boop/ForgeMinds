"""Torch reference implementations of every packed kernel.

These are not "approximations that are good enough": they define the exact semantics the
Triton kernels must reproduce (fp32 accumulation, same rounding of the output cast), so
`tests/test_kernels.py` can use them as the oracle for both the CPU path and the
interpreted-GPU path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _acc(t: torch.Tensor) -> torch.Tensor:
    """Accumulation dtype, matching the kernels: fp32 for half precision, the input dtype otherwise.

    The kernels always accumulate in fp32, which for fp16/bf16/fp32 inputs *is* the widest thing
    involved. Casting a float64 tensor to fp32 unconditionally (as this file used to) would make
    the "fallback" path silently less accurate than the model it is standing in for -- and would
    make it impossible to gradcheck, so the widening stops where the kernels stop.
    """
    return t if t.dtype in (torch.float64, torch.float32) else t.float()


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = _acc(x)
    wf = _acc(w)
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * r * wf).to(x.dtype)


def add_rms_norm(x: torch.Tensor, delta: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    xn = x + delta
    return xn, rms_norm(xn, w, eps)


def qk_norm_rope(x: torch.Tensor, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 eps: float = 1e-6) -> torch.Tensor:
    """Per-head RMSNorm over the full head dim, then half-split RoPE. x is [B,T,H,D]."""
    xf = _acc(x)
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = xf * r * _acc(w)
    y1, y2 = y.chunk(2, -1)
    return torch.cat((y1 * cos - y2 * sin, y1 * sin + y2 * cos), -1).to(x.dtype)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (F.silu(_acc(gate)) * _acc(up)).to(gate.dtype)


def kl_div_logits(student: torch.Tensor, teacher: torch.Tensor, reduction: str = "mean"):
    """KL(teacher || student) over the last dim, plus dL/d(student logits)."""
    ls = torch.log_softmax(_acc(student), -1)
    lt = torch.log_softmax(_acc(teacher), -1)
    pt = lt.exp()
    per_row = (pt * (lt - ls)).sum(-1)
    if reduction == "none":
        loss = per_row
        scale = 1.0
    elif reduction == "sum":
        loss = per_row.sum()
        scale = 1.0
    else:
        n = per_row.numel()
        loss = per_row.sum() / n
        scale = 1.0 / n
    grad = ((ls.exp() - pt) * scale).to(student.dtype)
    return loss, grad


def kl_distill_loss(hs: torch.Tensor, ht: torch.Tensor, w: torch.Tensor, teacher_head,
                    chunk: int = 4096) -> torch.Tensor:
    """Differentiable mean forward KL of teacher || student, gradients wrt `hs` only.

    Same shape of computation as the kernel path: one chunk of student logits is materialized
    at a time, and the teacher runs under `no_grad` so its parameters never accumulate grads.
    """
    hs2 = hs.reshape(-1, hs.shape[-1])
    ht2 = ht.reshape(-1, ht.shape[-1])
    n = hs2.shape[0]
    total = None
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        s_log = hs2[i:j] @ w.t()
        with torch.no_grad():
            t_log = teacher_head(ht2[i:j]).detach()
        piece = kl_div_logits(s_log, t_log, "sum")[0]
        total = piece if total is None else total + piece
    return total / n


def kl_distill_loss_ref(hs: torch.Tensor, ht: torch.Tensor, w: torch.Tensor, teacher_head,
                        chunk: int = 4096):
    """Oracle: `(loss, dL/dhs)` with the gradient built by hand, for testing the kernel."""
    hs2 = hs.reshape(-1, hs.shape[-1])
    ht2 = ht.reshape(-1, ht.shape[-1])
    grad = torch.empty_like(hs2)
    total = torch.zeros((), dtype=_acc(hs2).dtype)
    n = hs2.shape[0]
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        s_log = hs2[i:j] @ w.t()
        with torch.no_grad():
            t_log = teacher_head(ht2[i:j])
        loss, g = kl_div_logits(s_log, t_log, reduction="sum")
        total = total + loss.detach()
        grad[i:j] = _acc(g) @ _acc(w)              # chain rule through s_log = hs @ w.T
    return total / n, grad.view(hs.shape).to(hs.dtype) / n
