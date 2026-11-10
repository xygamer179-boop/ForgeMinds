from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from .config import ParallelConfig


@dataclass
class DistContext:
    rank: int
    world: int
    local_rank: int
    device: torch.device
    pc: ParallelConfig
    mesh: Optional[object] = None
    dp_rank: int = 0
    dp_world: int = 1
    pp_rank: int = 0
    tp_rank: int = 0
    dp_group: Optional[object] = None
    pp_group: Optional[object] = None
    tp_group: Optional[object] = None
    ep_group: Optional[object] = None
    edp_group: Optional[object] = None   # ranks holding the same expert shard (expert data-parallel)
    ep_rank: int = 0

    @property
    def is_main(self):
        return self.rank == 0

    @property
    def device_type(self):
        return self.device.type

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    # ---- collective helpers: they all return *device* tensors so the caller decides when to pay for a
    # ---- host sync. `world == 1` short-circuits every one of them (no collective call at all).
    def mean_over_dp(self, x: torch.Tensor) -> torch.Tensor:
        """average a scalar tensor over the data-parallel group"""
        if self.dp_world > 1 and self.dp_group is not None:
            x = x.clone()
            dist.all_reduce(x, group=self.dp_group)
            x /= self.dp_world
        return x

    def sum_over_dp(self, x: torch.Tensor) -> torch.Tensor:
        if self.dp_world > 1 and self.dp_group is not None:
            x = x.clone()
            dist.all_reduce(x, group=self.dp_group)
        return x

    def sum_over_pp(self, x: torch.Tensor) -> torch.Tensor:
        if self.pc.pp > 1 and self.world > 1 and self.pp_group is not None:
            x = x.clone()
            dist.all_reduce(x, group=self.pp_group)
        return x

    def all_min_int(self, x: torch.Tensor) -> torch.Tensor:
        """0/1 flag -> 0 only if every rank says 0 (so a skip decision is unanimous)"""
        if self.world > 1:
            x = x.clone()
            dist.all_reduce(x, op=dist.ReduceOp.MIN)
        return x


def log(ctx: Optional[DistContext], *a, **k):
    if ctx is None or ctx.rank == 0:
        print(*a, flush=True, **k)


# ---------------------------------------------------------------------------
# Collectives for custom loops (``fm.launch``-ed scripts). World 1 or no process
# group: every one is a passthrough -- *no* collective call at all. These used to
# live on a Lightning-Fabric-shaped class; they are plain functions now, because
# ForgeMind is not a port of Lightning.
def all_reduce(tensor: torch.Tensor, reduce_op: str = "mean") -> torch.Tensor:
    """Reduce ``tensor`` in place across the world: ``mean`` | ``sum`` | ``max`` | ``min``."""
    op = {"sum": dist.ReduceOp.SUM, "mean": dist.ReduceOp.AVG,
          "max": dist.ReduceOp.MAX, "min": dist.ReduceOp.MIN}[reduce_op]
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast ``tensor`` from rank ``src`` (world 1: returned unchanged)."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    dist.broadcast(tensor, src=src)
    return tensor


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Gather along dim 0: ``world_size`` copies concatenated (world 1: as-is)."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    out = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(out, tensor)
    return torch.cat(out, dim=0)


def barrier() -> None:
    """No-op at world 1."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def init_distributed(pc: ParallelConfig, backend: Optional[str] = None) -> DistContext:
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1 and not dist.is_initialized():
        timeout = datetime.timedelta(seconds=pc.init_timeout_s)
        dist.init_process_group(backend or ("nccl" if cuda else "gloo"), rank=rank, world_size=world,
                                timeout=timeout, device_id=(device if cuda else None))

    fixed = pc.pp * pc.tp * pc.dp_replicate
    if pc.dp_shard == -1:
        assert world % fixed == 0, f"world {world} not divisible by pp*tp*dp_replicate={fixed}"
        pc.dp_shard = world // fixed
    assert pc.pp * pc.dp_replicate * pc.dp_shard * pc.tp == world, (
        f"pp*dp_replicate*dp_shard*tp = {pc.pp * pc.dp_replicate * pc.dp_shard * pc.tp} != world {world}")
    dp_world = pc.dp_replicate * pc.dp_shard
    assert dp_world % pc.ep == 0, f"ep={pc.ep} must divide dp world {dp_world}"

    ctx = DistContext(rank, world, local, device, pc, dp_world=dp_world)
    if world == 1:
        return ctx

    from torch.distributed.device_mesh import init_device_mesh
    names = ("pp", "dp_replicate", "dp_shard", "tp")
    mesh = init_device_mesh(device.type, (pc.pp, pc.dp_replicate, pc.dp_shard, pc.tp), mesh_dim_names=names)
    ctx.mesh = mesh
    grid = mesh.mesh.tolist()  # [pp][dr][ds][tp] of global ranks (one sync at init, not per step)
    p = r = s = t = 0
    for i, pp_slab in enumerate(grid):
        for j, dr_slab in enumerate(pp_slab):
            for k, ds_slab in enumerate(dr_slab):
                for l, grank in enumerate(ds_slab):
                    if grank == rank:
                        p, r, s, t = i, j, k, l
    ctx.pp_rank, ctx.tp_rank = p, t
    ctx.dp_rank = r * pc.dp_shard + s
    ctx.pp_group = mesh["pp"].get_group()
    ctx.tp_group = mesh["tp"].get_group()

    # groups must be created by every rank in identical order
    for pp_i in range(pc.pp):
        for tp_i in range(pc.tp):
            dp_ranks = [g for slab in grid[pp_i] for g in [row[tp_i] for row in slab]]
            g = dist.new_group(dp_ranks)
            if rank in dp_ranks:
                ctx.dp_group = g
            if pc.ep == 1:
                continue
            chunks = [dp_ranks[i:i + pc.ep] for i in range(0, len(dp_ranks), pc.ep)]
            for ch in chunks:
                g = dist.new_group(ch)
                if rank in ch:
                    ctx.ep_group, ctx.ep_rank = g, ch.index(rank)
            for j in range(pc.ep):
                ranks = [ch[j] for ch in chunks]
                g = dist.new_group(ranks)
                if rank in ranks:
                    ctx.edp_group = g
    return ctx


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
