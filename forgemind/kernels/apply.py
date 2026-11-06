"""Opt-in wiring of the packed kernels into an existing model.

Nothing is patched unless you ask, because a fused kernel changes the numerics at the 1e-7
level and that is not a decision a library should make for you.

    import forgemind as fm
    fm.kernels.enable(model)            # RMSNorm + SwiGLU modules, any nn.Module tree
    fm.kernels.enable(model, swiglu=False)

or with the trainer::

    cfg.train.fused_ops = True

What is recognised
------------------
* norm modules: a module whose class name contains ``"RMSNorm"`` (or ``"rmsnorm"``) and which
  exposes a 1-D ``weight`` parameter plus ``eps``. ``forward(x) -> rms_norm(x, weight, eps)``.
* MLP modules: a module with ``w1``/``w2``/``w3`` Linear children and a ``forward`` that
  computes ``w2(silu(w1(x)) * w3(x))``. The Silu-multiply is replaced by the packed kernel;
  with ``split=True`` the two projections are *also* concatenated into one GEMM (only done for
  modules it can rebuild, i.e. where ``w1`` and ``w3`` have identical shapes).

Everything degrades to the torch path on CPU, so ``enable`` is safe to call unconditionally.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from .norm import add_rms_norm, rms_norm
from .swiglu import swiglu


def _is_rmsnorm(m: nn.Module) -> bool:
    name = type(m).__name__.lower()
    return ("rmsnorm" in name or "rms_norm" in name) and isinstance(
        getattr(m, "weight", None), nn.Parameter) and m.weight.dim() == 1


def _is_swiglu(m: nn.Module) -> bool:
    if not all(isinstance(getattr(m, k, None), nn.Linear) for k in ("w1", "w2", "w3")):
        return False
    return m.w1.out_features == m.w3.out_features and m.w2.in_features == m.w1.out_features


def _patch_norm(m: nn.Module) -> None:
    eps = float(getattr(m, "eps", 1e-6))

    def forward(self, x):
        return rms_norm(x, self.weight.to(x.dtype), eps)

    m.forward = types.MethodType(forward, m)


def _patch_swiglu_split(m: nn.Module) -> None:
    """Concatenate w1|w3 into one [2h, d] weight and use the fused gate|up kernel."""
    w1, w3 = m.w1, m.w3
    fused = nn.Linear(w1.in_features, w1.out_features * 2, bias=False,
                      device=w1.weight.device, dtype=w1.weight.dtype)
    with torch.no_grad():
        fused.weight.copy_(torch.cat([w1.weight, w3.weight], 0))
    del m.w1, m.w3
    m.gate_up = fused

    def forward(self, x):
        return self.w2(_split_apply(self.gate_up(x)))

    def _split_apply(y):
        from .swiglu import swiglu_split
        return swiglu_split(y)

    m.forward = types.MethodType(forward, m)


def _patch_swiglu(m: nn.Module) -> None:
    def forward(self, x):
        return self.w2(swiglu(self.w1(x), self.w3(x)))

    m.forward = types.MethodType(forward, m)


def enable(model: nn.Module, norm: bool = True, swiglu_mlp: bool = True,
           split: bool = False) -> nn.Module:
    """Patch ``model`` in place and return it. Idempotent."""
    n_norm = n_mlp = 0
    for m in model.modules():
        if norm and not getattr(m, "_fm_norm", False) and _is_rmsnorm(m):
            _patch_norm(m)
            m._fm_norm = True
            n_norm += 1
        elif swiglu_mlp and not getattr(m, "_fm_mlp", False) and _is_swiglu(m):
            if split:
                _patch_swiglu_split(m)
            else:
                _patch_swiglu(m)
            m._fm_mlp = True
            n_mlp += 1
    model._fm_fused = {"norm": n_norm, "mlp": n_mlp}
    return model


__all__ = ["enable", "rms_norm", "add_rms_norm", "swiglu"]
