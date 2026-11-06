"""Mixture-of-Experts: top-k router, dropless sorted dispatch, expert parallelism (all-to-all),
optional CPU expert offload (stream only the experts that received tokens, Unsloth-style).

Dispatch has three implementations, picked once at construction time (never inside `forward`, so
there is no capability `try/except` sitting in the hot path):

``grouped``  CUDA + bf16 + ``torch._grouped_mm`` (torch>=2.8): the per-expert token counts never leave
             the device, so the whole block is a single `torch.compile` graph with **zero host syncs**.
``eager``    anything else: one host read of the per-expert counts per MoE layer, then a per-expert
             loop. The loop is marked ``torch._dynamo.disable`` so ``torch.compile`` keeps the router,
             gather, weighting and scatter in a graph instead of bailing out on a data-dependent
             branch (that used to be ~12 graph breaks per MoE block).
``ep``       expert parallelism: needs host-side split sizes for `all_to_all_single`, so the dispatch
             itself stays eager by construction (one host read, not two).

`Router` no longer calls `torch.bincount`, whose output shape depends on tensor *data*: that alone was
a guaranteed graph break under `torch.compile`. A static-shape `scatter_add_` does the same job.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

__all__ = ["MoE", "Router", "Experts"]

_HAS_GROUPED_MM = hasattr(torch, "_grouped_mm")


def _counts_of(idx_flat: torch.Tensor, n: int) -> torch.Tensor:
    """Per-expert token counts with a *static* output shape.

    ``torch.bincount`` cannot be traced by torch.compile (its output shape depends on tensor data),
    which cost two guaranteed graph breaks per MoE block. A zeros+scatter_add has a static shape.
    """
    c = idx_flat.new_zeros(n, dtype=torch.int64)
    c.scatter_add_(0, idx_flat, torch.ones_like(idx_flat, dtype=torch.int64))
    return c


class _A2A(torch.autograd.Function):
    """differentiable all_to_all_single with uneven splits"""

    @staticmethod
    def forward(ctx, x, out_splits, in_splits, group):
        ctx.out_splits, ctx.in_splits, ctx.group = out_splits, in_splits, group
        out = x.new_empty((sum(out_splits),) + tuple(x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(), out_splits, in_splits, group=group)
        return out

    @staticmethod
    def backward(ctx, g):
        gi = g.new_empty((sum(ctx.in_splits),) + tuple(g.shape[1:]))
        dist.all_to_all_single(gi, g.contiguous(), ctx.in_splits, ctx.out_splits, group=ctx.group)
        return gi, None, None, None


class Router(nn.Module):
    def __init__(self, dim, n_experts, top_k):
        super().__init__()
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.n_experts, self.top_k = n_experts, top_k

    def forward(self, x):  # x [N, d]
        logits = self.gate(x).float()
        probs = logits.softmax(-1)
        w, idx = probs.topk(self.top_k, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        E = self.n_experts
        with torch.no_grad():
            frac = _counts_of(idx.reshape(-1), E) / idx.numel()
        aux = E * (frac * probs.mean(0)).sum()                # switch-style load balance
        z = logits.logsumexp(-1).pow(2).mean()                # router z-loss
        return w.to(x.dtype), idx, aux, z


class Experts(nn.Module):
    """SwiGLU experts. stacked=[E,d,h] params (fast) or per-expert params (needed for CPU offload)."""

    def __init__(self, n_local, dim, hidden, offload=False, grouped_mm=False):
        super().__init__()
        self.n_local, self.offload = n_local, offload
        # capability check happens once, here -- not per forward (no try/except in the hot path)
        self.grouped = bool(grouped_mm) and _HAS_GROUPED_MM and not offload
        if offload:
            mk = lambda a, b: nn.ParameterList([nn.Parameter(torch.empty(a, b)) for _ in range(n_local)])
        else:
            mk = lambda a, b: nn.Parameter(torch.empty(n_local, a, b))
        self.w1, self.w3, self.w2 = mk(dim, hidden), mk(dim, hidden), mk(hidden, dim)
        self.reset_parameters(dim, hidden)

    def reset_parameters(self, dim, hidden):
        def init(p, fan_in):
            nn.init.normal_(p, std=0.02)
        for w in (self.w1, self.w3, self.w2):
            for p in (w if self.offload else [w]):
                init(p.data if hasattr(p, "data") else p, 0)
        if self.offload:  # pinned host memory so H2D copies are async
            if torch.cuda.is_available():
                for w in (self.w1, self.w3, self.w2):
                    for p in w:
                        p.data = p.data.pin_memory()

    def _w(self, name, e, device):
        w = getattr(self, name)
        t = w[e]
        if self.offload and t.device != device:
            t = t.to(device, non_blocking=True)  # autograd: grad flows back to the CPU param
        return t

    # ---- fast path: no host sync, fully traceable ------------------------------------------------
    def forward(self, x, counts=None, offsets=None):
        """``offsets`` = int32 cumulative group ends on the device (grouped path, no host sync).
        ``counts``  = python list of per-expert sizes (eager path)."""
        if self.grouped and offsets is not None and x.is_cuda and x.dtype == torch.bfloat16:
            return self._forward_grouped(x, offsets)
        if counts is None:
            if offsets is None:
                counts = [x.size(0)]
            else:
                o = offsets.to(torch.int64)
                counts = torch.diff(o, prepend=o.new_zeros(1)).tolist()
        return self._forward_eager(x, counts)

    def _forward_grouped(self, x, offs):
        w1, w3, w2 = self.w1, self.w3, self.w2
        if w1.dtype != x.dtype:
            w1, w3, w2 = w1.to(x.dtype), w3.to(x.dtype), w2.to(x.dtype)
        h = torch._grouped_mm(x, w1, offs=offs)
        h = F.silu(h) * torch._grouped_mm(x, w3, offs=offs)
        return torch._grouped_mm(h, w2, offs=offs)

    @torch._dynamo.disable
    def _forward_eager(self, x, counts: List[int]):
        """Per-expert loop. Deliberately outside the compiled graph: the control flow depends on
        token *data*, which no compiler can fuse -- everything around it stays compiled."""
        sizes = [int(c) for c in counts]
        if not any(sizes):
            return x.new_zeros((0,) + tuple(x.shape[1:]))
        dt = x.dtype
        outs = []
        for e, (c, xe) in enumerate(zip(sizes, x.split(sizes))):
            if c == 0:
                continue  # offload: inactive experts never leave host memory
            a, b, o = self._w("w1", e, x.device), self._w("w3", e, x.device), self._w("w2", e, x.device)
            if a.dtype is not dt:
                a, b, o = a.to(dt), b.to(dt), o.to(dt)
            outs.append((F.silu(xe @ a) * (xe @ b)) @ o)
        if not outs:  # keep graph connected when a rank received nothing
            return x.new_zeros((0,) + tuple(x.shape[1:]))
        return torch.cat(outs, 0) if len(outs) > 1 else outs[0]


class MoE(nn.Module):
    def __init__(self, dim, hidden, n_experts, top_k, n_shared=0, ep_size=1, ep_group=None, ep_rank=0,
                 offload_experts=False, grouped_mm=False, aux_coef=1e-2, z_coef=1e-3):
        super().__init__()
        assert n_experts % ep_size == 0
        self.dim, self.n_experts, self.top_k = dim, n_experts, top_k
        self.ep_size, self.ep_group, self.ep_rank = ep_size, ep_group, ep_rank
        self.n_local = n_experts // ep_size
        self.router = Router(dim, n_experts, top_k)
        self.experts = Experts(self.n_local, dim, hidden, offload=offload_experts, grouped_mm=grouped_mm)
        self.shared = None
        if n_shared:
            from .models.gpt import SwiGLU
            self.shared = SwiGLU(dim, hidden * n_shared)
        self.aux_coef, self.z_coef = aux_coef, z_coef
        self.last_aux: Optional[torch.Tensor] = None

    @property
    def compile_friendly(self) -> bool:
        return self.ep_size == 1 and self.experts.grouped

    def probe_grouped(self, device, dtype) -> bool:
        """Verify `torch._grouped_mm` really works on this device *once*, before compilation.

        A capability probe in `forward` (`try: grouped except: loop`) would be a graph break on every
        single step; doing it up front keeps the hot path branch-free.
        """
        e = self.experts
        if not e.grouped:
            return False
        if device.type != "cuda" or dtype != torch.bfloat16:
            e.grouped = False
            return False
        try:
            n = e.w1.size(0)                                  # number of local experts (= group count)
            a = torch.zeros(2 * n, self.dim, device=device, dtype=torch.bfloat16)
            offs = (torch.arange(1, n + 1, device=device) * 2).to(torch.int32)
            torch._grouped_mm(a, e.w1.detach().to(torch.bfloat16), offs=offs)
        except Exception as ex:                                     # pragma: no cover - needs CUDA
            from ..perf import warn_once
            warn_once(f"[forgemind] torch._grouped_mm unavailable ({type(ex).__name__}: {ex}); "
                      f"falling back to the per-expert loop (one host sync per MoE layer)", key="no_grouped_mm")
            e.grouped = False
        return e.grouped

    def forward(self, x):
        shp = x.shape
        x = x.reshape(-1, self.dim)
        w, idx, aux, z = self.router(x)
        self.last_aux = self.aux_coef * aux + self.z_coef * z
        flat = idx.reshape(-1)
        order = torch.sort(flat, stable=True).indices
        tok = order // self.top_k
        xs = x[tok]                                   # [top_k * N, d], grouped by local expert
        counts = _counts_of(flat, self.n_experts)     # device tensor, static shape

        if self.ep_size == 1:
            # pure python branch on static facts (device/dtype/flag): no graph break
            if self.experts.grouped and xs.is_cuda and xs.dtype == torch.bfloat16:
                ys = self.experts(xs, offsets=counts.cumsum(0).to(torch.int32))
            else:
                ys = self.experts(xs, counts=counts.tolist())
        else:
            ys = self._ep_dispatch(xs, counts)

        ys = ys * w.reshape(-1)[order].unsqueeze(-1)
        out = torch.zeros_like(x).index_add_(0, tok, ys.to(x.dtype))
        if self.shared is not None:
            out = out + self.shared(x)
        return out.reshape(shp)

    @torch._dynamo.disable
    def _ep_dispatch(self, xs, counts):
        """Expert-parallel dispatch. Split sizes must be host-side ints for `all_to_all_single`, so
        the whole exchange runs eagerly; there is exactly one host read for both directions."""
        E_l, ep = self.n_local, self.ep_size
        cmat = counts.view(ep, E_l)
        recv_mat = torch.empty_like(cmat)
        dist.all_to_all_single(recv_mat.view(-1), cmat.contiguous().view(-1), group=self.ep_group)
        # exactly one host read for both directions and all per-expert totals
        info = torch.cat([cmat.sum(1), recv_mat.reshape(-1), recv_mat.sum(0)]).tolist()
        send = [int(v) for v in info[:ep]]
        R = [info[ep + r * E_l: ep + (r + 1) * E_l] for r in range(ep)]     # R[src][expert]
        recv = [int(sum(r)) for r in R]
        per_e_counts = [int(v) for v in info[ep + ep * E_l:]]

        xr = _A2A.apply(xs, recv, send, self.ep_group)  # layout: [src][e]
        start = 0
        seg = [[None] * E_l for _ in range(ep)]
        for r in range(ep):
            for e in range(E_l):
                c = int(R[r][e])
                seg[r][e] = (start, c)
                start += c
        per_e_in = [torch.cat([xr[a:a + c] for a, c in (seg[r][e] for r in range(ep))], 0) for e in range(E_l)]
        per_e_out = []
        for e in range(E_l):
            c = per_e_counts[e]
            per_e_out.append(self.experts(per_e_in[e], [c if i == e else 0 for i in range(E_l)]) if c
                             else per_e_in[e])
        pieces = [[None] * E_l for _ in range(ep)]
        for e in range(E_l):
            parts = per_e_out[e].split([int(R[r][e]) for r in range(ep)], 0)
            for r in range(ep):
                pieces[r][e] = parts[r]
        yr = torch.cat([pieces[r][e] for r in range(ep) for e in range(E_l)], 0)
        return _A2A.apply(yr, send, recv, self.ep_group)  # back to original sorted order
