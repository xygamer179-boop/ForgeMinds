"""Packed (fused) kernels: norms, RoPE, SwiGLU and a distillation KL.

Every function here is a drop-in replacement for an eager torch expression, keeps the
autograd contract of that expression, and **falls back to torch** when Triton or a GPU is not
available -- so importing this package never breaks a CPU or MPS run.

    import forgemind as fm
    y = fm.kernels.rms_norm(x, w, 1e-6)          # == F.rms_norm(x, (x.shape[-1],), w, eps)
    y = fm.kernels.swiglu(gate, up)              # == F.silu(gate) * up
    y = fm.kernels.swiglu_split(gate_up)         # fused gate|up projection output
    y = fm.kernels.qk_norm_rope(q, w, cos, sin)  # per-head RMSNorm + RoPE
    loss = fm.kernels.kl_distill_loss(hs, ht, w, teacher_head)   # KL(teacher || student)

    fm.kernels.enable(model)                     # patch RMSNorm/SwiGLU modules in place
    fm.kernels.set_backend("fallback")           # force the torch path (A/B, debugging)
    fm.kernels.stats()                           # which path ran, and how often

The kernels are tested in Triton's interpreter (``TRITON_INTERPRET=1``) on CPU against the
torch reference in ``forgemind.kernels._fallback``, so "make them bug free" is a statement
with a test behind it rather than a hope.
"""

from . import _common
from ._common import HAS_TRITON, get_backend, reset_stats, set_backend, stats
from .apply import enable
from .kl import distill_kl, kl_distill_loss, kl_div, kl_div_logits
from .norm import add_rms_norm, rms_norm
from .rope import qk_norm_rope
from .swiglu import swiglu, swiglu_split

__all__ = [
    "HAS_TRITON", "set_backend", "get_backend", "stats", "reset_stats",
    "rms_norm", "add_rms_norm", "qk_norm_rope", "swiglu", "swiglu_split",
    "kl_div_logits", "kl_div", "kl_distill_loss", "distill_kl",
    "enable", "available",
]


def available() -> bool:
    """True when the fast path can actually run on this machine."""
    import torch
    return _common.HAS_TRITON and torch.cuda.is_available()
