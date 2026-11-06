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
    # in_features must match too: with `split=True` the two weights are concatenated into one
    # [2h, in] GEMM, and a mismatched pair would silently compute w3 on the wrong input.
    return (m.w1.out_features == m.w3.out_features
            and m.w1.in_features == m.w3.in_features
            and m.w2.in_features == m.w1.out_features)


def _patch_norm(m: nn.Module) -> None:
    # Three cases, and getting them confused silently changes the model's numbers:
    #   * torch's nn.RMSNorm: eps is None -> finfo(x.dtype).eps, resolved per call;
    #   * a user module with a numeric eps -> use it;
    #   * a user module with *no* eps attribute at all -> this library's default (1e-6), not
    #     torch's dtype epsilon. Measured drift before this distinction: 1.9e-6 per activation.
    if not hasattr(m, "eps"):
        eps = 1e-6                                   # our default, not torch's dtype epsilon
    else:
        eps = getattr(m, "eps")
        eps = None if eps is None else float(eps)

    def forward(self, x):
        e = torch.finfo(x.dtype).eps if eps is None else eps
        return rms_norm(x, self.weight.to(x.dtype), e)

    m.forward = types.MethodType(forward, m)


def _patch_swiglu_split(m: nn.Module) -> None:
    """Concatenate w1|w3 into one [2h, d] weight and use the fused gate|up kernel."""
    w1, w3 = m.w1, m.w3
    if (w1.bias is None) != (w3.bias is None):
        # one bias and not the other: keep the two GEMMs and fuse only the epilogue, otherwise
        # the concatenated bias would silently drop one of them
        _patch_swiglu(m)
        return
    fused = nn.Linear(w1.in_features, w1.out_features * 2, bias=w1.bias is not None,
                      device=w1.weight.device, dtype=w1.weight.dtype)
    with torch.no_grad():
        fused.weight.copy_(torch.cat([w1.weight, w3.weight], 0))
        if fused.bias is not None:
            fused.bias.copy_(torch.cat([w1.bias, w3.bias], 0))
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


def _has_norm_weight(m: nn.Module) -> bool:
    w = getattr(m, "weight", None)
    return isinstance(w, nn.Parameter) and w.dim() == 1


def patch(module: nn.Module, split: bool = False, kind: "str | None" = None) -> bool:
    """Fuse **one** module instance in place; returns ``True`` if it was recognised.

    Use this when you have your own architecture and want to opt in block by block instead of
    running :func:`enable` over the whole tree::

        fm.kernels.patch(model.norm)             # name looks like a norm + 1-D weight + eps
        fm.kernels.patch(model.norm, kind="norm")  # my class is called `MyScale`, patch it anyway
        fm.kernels.patch(model.mlp)              # w1/w2/w3 Linears -> fused SwiGLU epilogue
        fm.kernels.patch(model.mlp, split=True)  # ...and concatenate w1|w3 into one GEMM
        fm.kernels.patch(model.mlp, kind="mlp")  # my MLP is called `FeedForward`

    ``kind`` (``"norm"`` or ``"mlp"``) skips detection and validates the structure instead, which
    is the reliable route for a class whose name says nothing. Detection by name is deliberately
    conservative so that :func:`enable` never rewrites something it merely guessed about.

    It works on **torch's** modules too -- ``fm.kernels.patch(nn.RMSNorm(768))`` is valid -- it
    preserves ``nn.RMSNorm``'s ``eps=None`` (``finfo(dtype).eps``) semantics, and it is idempotent.

    Returns ``False`` (touching nothing) when the module is not recognised, so a wrong guess fails
    at the call site instead of silently doing nothing::

        assert fm.kernels.patch(model.norm), "norm was not recognised"

    Raises ``TypeError`` when you pass an explicit ``kind`` that the module does not satisfy --
    better than patching something whose forward would then be nonsense.

    ``split=True`` **renames parameters** (``w1``/``w3`` -> ``gate_up``), so a checkpoint saved
    before patching will not load afterwards. The epilogue-only fuse (``split=False``) keeps every
    key and is the safe default.
    """
    if getattr(module, "_fm_norm", False) or getattr(module, "_fm_mlp", False):
        return True                       # idempotent
    if kind is not None:
        if kind == "norm":
            if not _has_norm_weight(module):
                raise TypeError(f"patch(kind='norm') needs a 1-D weight Parameter, "
                                f"{type(module).__name__} has {getattr(module, 'weight', None)!r}")
            _patch_norm(module)
            module._fm_norm = True
            return True
        if kind == "mlp":
            if not _is_swiglu(module):
                raise TypeError(f"patch(kind='mlp') needs Linear children w1/w2/w3, "
                                f"{type(module).__name__} does not have that layout")
            _patch_swiglu_split(module) if split else _patch_swiglu(module)
            module._fm_mlp = True
            return True
        raise ValueError(f"kind must be 'norm', 'mlp' or None, got {kind!r}")
    if _is_rmsnorm(module):
        _patch_norm(module)
        module._fm_norm = True
        return True
    if _is_swiglu(module):
        if split:
            _patch_swiglu_split(module)
        else:
            _patch_swiglu(module)
        module._fm_mlp = True
        return True
    return False


def enable(model: nn.Module, norm: bool = True, swiglu_mlp: bool = True,
           split: bool = False) -> nn.Module:
    """Patch a whole module tree in place and return it. Idempotent.

    The counts land on ``model._fm_fused`` (``{"norm": n, "mlp": n}``), which is what the
    trainer logs. Modules you have already :func:`patch`-ed by hand are not counted twice.
    """
    n_norm = n_mlp = 0
    for m in model.modules():
        if getattr(m, "_fm_norm", False) or getattr(m, "_fm_mlp", False):
            continue
        if norm and _is_rmsnorm(m):
            patch(m)
            n_norm += 1
        elif swiglu_mlp and _is_swiglu(m):
            patch(m, split=split)
            n_mlp += 1
    model._fm_fused = {"norm": n_norm, "mlp": n_mlp}
    return model


__all__ = ["patch", "enable", "rms_norm", "add_rms_norm", "swiglu"]
