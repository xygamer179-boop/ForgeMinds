"""Activation checkpointing (full / selective) with the per-call overhead removed.

Two costs dominated the original implementation:

  * `torch.utils.checkpoint` saves and restores the *RNG state* of both CPU and CUDA around every
    checkpointed block unless told otherwise. At 1 checkpoint per layer per micro-batch that is two extra
    device allocations plus two kernel launches per layer, for nothing: the reference model has no
    dropout. `preserve_rng_state=False` is now the default (opt back in with `TrainConfig.ac_preserve_rng`).
  * the `context_fn` for selective checkpointing was rebuilt (a `functools.partial` plus a policy closure)
    on every call. It is built once per block, here.

`use_reentrant=False` is required for `torch.compile` (the reentrant variant is a hard graph break and
silently disables autograd hooks), and both modes trace into a single graph.
"""
from __future__ import annotations

import functools

import torch
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy

_SAVE = {getattr(torch.ops.aten, name).default for name in (
    "mm", "addmm", "bmm", "baddbmm", "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention", "_scaled_dot_product_cudnn_attention") if hasattr(torch.ops.aten, name)}


def _policy(ctx, op, *a, **k):
    return CheckpointPolicy.MUST_SAVE if op in _SAVE else CheckpointPolicy.PREFER_RECOMPUTE


def apply_activation_checkpointing(model, mode="full", preserve_rng_state: bool = False):
    """patches block.forward in place (FQNs unchanged). 'selective' keeps matmul/attention outputs, recomputes the rest."""
    if mode == "none":
        return model
    for blk in model.layers.values():
        if getattr(blk, "_fm_ac", False):
            continue
        orig = blk.forward
        kw = dict(use_reentrant=False, preserve_rng_state=preserve_rng_state)
        if mode == "selective":
            kw["context_fn"] = functools.partial(create_selective_checkpoint_contexts, _policy)

        def fwd(*a, _o=orig, _kw=kw, **k):
            return checkpoint(_o, *a, **k, **_kw)

        blk.forward = fwd
        blk._fm_ac = True
    return model
