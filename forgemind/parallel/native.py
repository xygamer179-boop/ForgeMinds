"""Native torch parallelism: TP (DTensor), FSDP2 / HSDP, DDP, expert-parallel grad sync.

Bugs fixed here (all of them were latent crashes or silent slowdowns):
  * `apply_fsdp` indexed `ctx.mesh[dims[0]]` with an empty tuple when neither shard nor replicate degree was
    > 1 -> IndexError;
  * `sync_expert_grads` rebuilt the whole `named_parameters()` list (string work over every parameter) on
    **every optimizer step** -- the caller can now pass a cached list;
  * the per-parameter `div_()` loop over expert grads is one `_foreach_div_`;
  * `apply_ddp` can use a static graph (skips the per-iteration unused-parameter scan).
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from ..distributed import DistContext

__all__ = ["has_moe", "apply_tp", "apply_fsdp", "apply_ddp", "sync_expert_grads", "expert_params"]


def has_moe(model):
    return any(getattr(b, "is_moe", False) for b in model.layers.values())


def expert_params(model):
    """Cached ``.experts.`` parameters -- iterating named_parameters() per step is not free."""
    return [p for n, p in model.named_parameters() if ".experts." in n]


# ----------------------------------------------------------------------------------------------------------------- TP
def apply_tp(model, ctx: DistContext, plan=None):
    from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
    plan = plan or model.tp_plan()
    styles = {"col": ColwiseParallel, "row": RowwiseParallel}
    tp_mesh = ctx.mesh["tp"]
    cfg = model.cfg
    assert cfg.n_heads % ctx.pc.tp == 0 and cfg.n_kv_heads % ctx.pc.tp == 0, "heads must divide tp"
    touched = 0
    for blk in model.layers.values():
        names = {n for n, _ in blk.named_modules()}
        blk_plan = {k: styles[v]() for k, v in plan.items() if k in names}   # MoE blocks simply have no ffn.w* entries
        if blk_plan:
            parallelize_module(blk, tp_mesh, blk_plan)
            touched += len(blk_plan)
    if not touched:
        raise RuntimeError("apply_tp: no module matched the TP plan (nothing was sharded)")
    return model


# --------------------------------------------------------------------------------------------------- FSDP2 / DDP / EP
def apply_fsdp(model, ctx: DistContext, bf16=True, cpu_offload=False, reshard=True):
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
    pc = ctx.pc
    dims = tuple(n for n, s in (("dp_replicate", pc.dp_replicate), ("dp_shard", pc.dp_shard)) if s > 1)
    if not dims:
        raise ValueError("apply_fsdp called with no shard/replicate dimension")
    mesh = ctx.mesh[dims] if len(dims) > 1 else ctx.mesh[dims[0]]
    if pc.ep > 1:
        raise NotImplementedError("EP + FSDP2 needs a separate expert mesh; use DDP/dp_replicate with ep, or backend='deepspeed'")
    kw = dict(mesh=mesh)
    if bf16:
        kw["mp_policy"] = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    if cpu_offload:
        kw["offload_policy"] = CPUOffloadPolicy()
    blocks = list(model.layers.values())
    for i, b in enumerate(blocks):
        fully_shard(b, reshard_after_forward=reshard and i < len(blocks) - 1, **kw)
    fully_shard(model, reshard_after_forward=False if len(blocks) else reshard, **kw)
    return model


def apply_ddp(model, ctx: DistContext):
    from torch.nn.parallel import DistributedDataParallel as DDP
    ignore = []
    if ctx.pc.ep > 1:
        ignore = [n for n, _ in model.named_parameters() if ".experts." in n]
        DDP._set_params_and_buffers_to_ignore_for_model(model, ignore)
    dev = [ctx.local_rank] if ctx.device.type == "cuda" else None
    return DDP(model, device_ids=dev, process_group=ctx.dp_group, gradient_as_bucket_view=True,
               static_graph=ctx.pc.ddp_static_graph,
               find_unused_parameters=has_moe(model) and ctx.pc.ep == 1 and not ctx.pc.ddp_static_graph)


def sync_expert_grads(model, ctx: DistContext, params=None):
    """Expert params are owned by one ep rank; DDP ignores them. Their grads already hold the sum over the ep group's tokens
    (via all-to-all backward); finish the job: sum over edp group, divide by dp world => same scaling as dense grads."""
    pc = ctx.pc
    if pc.ep == 1 or ctx.world == 1:
        return
    if params is None:
        params = expert_params(model)
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    if ctx.edp_group is not None and dist.get_world_size(ctx.edp_group) > 1:
        for g in grads:
            dist.all_reduce(g, group=ctx.edp_group)
    torch._foreach_div_(grads, ctx.dp_world)


def broadcast_non_expert_init(model, ctx: DistContext):
    """make replicas start identical (seeded identically this is a no-op; kept as a safety net for ep>1 where expert shards differ)"""
    if ctx.world == 1:
        return
    for n, p in model.named_parameters():
        if ".experts." not in n and ctx.dp_group is not None:
            dist.broadcast(p.data, src=dist.get_global_rank(ctx.dp_group, 0), group=ctx.dp_group)
