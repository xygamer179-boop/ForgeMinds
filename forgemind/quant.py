"""Quantized training. FP8 via Transformer Engine on Ada/Hopper/Blackwell (sm>=8.9) when installed; torchao otherwise
(torchao float8 on sm>=8.9 without TE, INT8 mixed-precision training on older GPUs: Turing/Ampere/T4/A100/3090).

Bug fixes:
  * ``int8_wo`` used to call ``requires_grad_(False)`` on *every* parameter, which left the Trainer with an
    empty parameter list and made ``torch.optim.AdamW`` raise "optimizer got an empty parameter list". It now
    freezes only the weights that were actually quantized (norms, the router gate, embeddings and the head
    stay trainable).
  * ``importlib.util.find_spec`` probes are cached: they hit the filesystem and were repeated on every
    ``resolve_backend`` call.
  * ``_swap_te`` no longer copies a weight onto itself when a module fails the filter, and it reports how many
    layers matched instead of silently quantizing nothing.
"""
from __future__ import annotations

import contextlib
import functools
import importlib.util

import torch
import torch.nn as nn

from .config import QuantConfig
from .perf import warn_once

_TE_ERR = "--qbackend te requested but transformer_engine is not installed"
_TA_ERR = "--qbackend torchao requested but torchao is not installed"


@functools.lru_cache(maxsize=64)
def _has(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def gpu_cc():
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)


def resolve_backend(q: QuantConfig) -> str:
    if q.mode == "none":
        return "none"
    if q.backend == "te" and not _has("transformer_engine"):
        raise RuntimeError(_TE_ERR)
    if q.backend == "torchao" and not _has("torchao"):
        raise RuntimeError(_TA_ERR)
    if not torch.cuda.is_available():
        raise RuntimeError("quantized training needs a CUDA GPU")
    fp8_hw = gpu_cc() >= (8, 9)
    if q.mode == "fp8":
        if not fp8_hw:
            raise RuntimeError(f"FP8 needs sm>=8.9, this GPU is sm{gpu_cc()[0]}{gpu_cc()[1]}; use mode='int8'")
        if q.backend in ("auto", "te") and _has("transformer_engine"):
            return "te"
        if q.backend == "te":
            raise RuntimeError("transformer_engine not installed")
        if _has("torchao"):
            return "torchao"
        raise RuntimeError("install transformer-engine or torchao")
    if not _has("torchao"):
        raise RuntimeError("int8 modes need torchao (pip install torchao)")
    if q.backend == "te":
        raise RuntimeError("INT8 is only implemented via torchao; use --qbackend torchao or auto")
    return "torchao"


def _ok_linear(m, fqn, q: QuantConfig):
    return (isinstance(m, nn.Linear) and not any(x in fqn for x in q.exclude)
            and m.in_features % 16 == 0 and m.out_features % 16 == 0)


class QuantState:
    """Holds the resolved backend/recipe. The TE autocast context factory is built once, not per forward."""

    def __init__(self, backend, mode, recipe=None, group=None):
        self.backend, self.mode, self.recipe, self.group = backend, mode, recipe, group
        self._ctx = None
        if backend == "te":
            import transformer_engine.pytorch as te
            if hasattr(te, "autocast"):                       # TE >= 2.8
                self._ctx = functools.partial(te.autocast, enabled=True, recipe=recipe)
            else:
                self._ctx = functools.partial(te.fp8_autocast, enabled=True, fp8_recipe=recipe,
                                              fp8_group=group)

    def autocast(self):
        """context to wrap forward (needed by TE only); a cached no-op everywhere else"""
        return contextlib.nullcontext() if self._ctx is None else self._ctx()


def _swap_te(model, q):
    import transformer_engine.pytorch as te
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            fqn = f"{name}.{cname}" if name else cname
            if not _ok_linear(child, fqn, q):
                continue
            new = te.Linear(child.in_features, child.out_features, bias=child.bias is not None,
                            params_dtype=child.weight.dtype, device=child.weight.device)
            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None:
                    new.bias.copy_(child.bias)
            setattr(mod, cname, new)
            n += 1
    return n


def _freeze_quantized(model, q):
    """Freeze only the weights that int8-weight-only just replaced (they carry no gradient anyway)."""
    frozen = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and _ok_linear(mod, name, q):
            for p in mod.parameters(recurse=False):
                p.requires_grad_(False)
                frozen += 1
    return frozen


def quantize_model(model: nn.Module, q: QuantConfig, dp_group=None) -> QuantState:
    be = resolve_backend(q)
    if be == "none":
        return QuantState("none", "none")
    flt = lambda m, fqn: _ok_linear(m, fqn, q)
    if be == "te":
        from transformer_engine.common.recipe import DelayedScaling, Format
        n = _swap_te(model, q)
        if not n:
            warn_once("[forgemind.quant] TransformerEngine matched 0 Linear layers -- check QuantConfig.exclude",
                      key="te_noop")
        recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
        print(f"[forgemind.quant] TransformerEngine FP8: swapped {n} Linear layers")
        return QuantState("te", "fp8", recipe, dp_group)
    # torchao
    if q.mode == "fp8":
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        convert_to_float8_training(model, config=Float8LinearConfig.from_recipe_name("tensorwise"), module_filter_fn=flt)
        print("[forgemind.quant] torchao float8 training enabled")
    elif q.mode == "int8":
        from torchao.quantization import quantize_
        from torchao.prototype.quantized_training import int8_mixed_precision_training, Int8MixedPrecisionTrainingConfig
        quantize_(model, int8_mixed_precision_training(Int8MixedPrecisionTrainingConfig()), filter_fn=flt)
        print("[forgemind.quant] torchao INT8 mixed-precision training enabled (works on pre-Ada GPUs)")
    elif q.mode == "int8_wo":
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(model, Int8WeightOnlyConfig(), filter_fn=flt)
        frozen = _freeze_quantized(model, q)   # only the quantized base weights, not the whole model
        print(f"[forgemind.quant] torchao INT8 weight-only (frozen base: {frozen} tensors)")
    else:
        raise ValueError(q.mode)
    return QuantState("torchao", q.mode)


def make_optimizer_8bit(groups, lr, betas, eps):
    from torchao.optim import AdamW8bit
    return AdamW8bit(groups, lr=lr, betas=betas, eps=eps)
