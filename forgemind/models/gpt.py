"""Reference decoder-only transformer (RMSNorm, RoPE, GQA, SwiGLU, optional MoE). Everything ForgeMind needs from a model
is: `.layers` (ModuleDict of blocks), optional `tok_emb`/`norm`/`lm_head`, `.aux_loss()`, `.split_stage()`, `.tp_plan()`.

Performance notes (all of these were measurable costs in the original):
  * the RoPE cos/sin tables are built **once** and cached per (length, head_dim, dtype, device). They used to be
    recomputed with two fresh `arange`/`outer`/`cos`/`sin` allocations *per layer, per forward*.
  * `apply_rope` no longer upcasts q/k to fp32 and back -- the tables are cached in the compute dtype, which is
    what Megatron/NeMo do as well.
  * `forward` accumulates the MoE aux loss while walking the blocks, so there is no post-hoc scan over the
    module tree and no per-step `torch.stack` of a Python list of graph-connected tensors.
  * everything is `torch.compile` clean: no `.item()`/`.tolist()`, no shape-dependent python control flow and
    no buffer mutation inside the traced region (`GPT.prepare` warms the caches before compilation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..loss import chunked_cross_entropy

__all__ = ["GPTConfig", "GPT", "RMSNorm", "Attention", "SwiGLU", "Block", "rope_tables", "apply_rope",
           "clear_rope_cache"]


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    ffn_mult: float = 8 / 3
    max_seq: int = 2048
    rope_theta: float = 10000.0
    rope_high_precision: bool = False   # compute the rotation in fp32 (slower, marginally more accurate)
    tie_embeddings: bool = False
    loss_chunk: int = 4096
    # MoE
    n_experts: int = 0          # 0 = dense
    top_k: int = 2
    moe_every: int = 1          # every k-th layer is MoE
    n_shared_experts: int = 0
    expert_hidden_mult: float = 1.0   # expert hidden = ffn_hidden * mult
    expert_grouped_mm: bool = True    # use torch._grouped_mm for expert GEMMs when available
    aux_coef: float = 1e-2
    z_coef: float = 1e-3
    mixer: str = "attn"       # "attn" | "cell" (GatedCell recurrent mixer, O(T) time, no KV cache)


# ----------------------------------------------------------------------------------------------- RoPE cache
_ROPE: dict = {}
_ROPE_MAX_ENTRIES = 32


def rope_tables(T: int, hd: int, theta: float, device, dtype):
    """Cached cos/sin tables, shaped [1, T, 1, hd/2] so they broadcast against [B, T, H, hd/2].

    Buckets the length to a multiple of 64 so a change of batch shape does not rebuild them.
    """
    n = max(64, (int(T) + 63) // 64 * 64)
    key = (n, int(hd), float(theta), device.type, device.index, dtype)
    tabs = _ROPE.get(key)
    if tabs is None:
        if len(_ROPE) >= _ROPE_MAX_ENTRIES:
            _ROPE.clear()
        inv = theta ** (-torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd)
        ang = torch.outer(torch.arange(n, device=device, dtype=torch.float32), inv)
        tabs = (ang.cos().to(dtype)[None, :, None, :].contiguous(),
                ang.sin().to(dtype)[None, :, None, :].contiguous())
        _ROPE[key] = tabs
    return tabs[0][:, :T], tabs[1][:, :T]


def clear_rope_cache() -> None:
    _ROPE.clear()


def rope_cache(T, hd, theta, device):  # back-compat alias (fp32 tables)
    return rope_tables(T, hd, theta, device, torch.float32)


def apply_rope(x, cos, sin):  # x [B,T,H,hd]
    x1, x2 = x.chunk(2, -1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight.to(x.dtype), self.eps)


class Attention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.hd = c.dim // c.n_heads
        self.theta = c.rope_theta
        self.hp = c.rope_high_precision
        # non-persistent buffers (not in state_dict): filled by `cache_rope` before the first forward and
        # never mutated while tracing, so `torch.compile(fullgraph=True)` sees a pure function.
        self.register_buffer("_cos", torch.empty(0), persistent=False)
        self.register_buffer("_sin", torch.empty(0), persistent=False)
        self.wq = nn.Linear(c.dim, c.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(c.dim, c.n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(c.dim, c.n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(c.n_heads * self.hd, c.dim, bias=False)

    def cache_rope(self, device, dtype, seq: int) -> "Attention":
        """Materialise the rotation tables. Called by `GPT.prepare` before compilation; the buffers are
        views into a shared per-(length, head_dim) table, so every layer costs no extra memory."""
        tdt = torch.float32 if self.hp else dtype
        cos, sin = rope_tables(seq, self.hd, self.theta, device, tdt)
        self._cos, self._sin = cos, sin
        return self

    def forward(self, x):
        B, T, _ = x.shape
        if self._cos.numel() == 0 or self._cos.size(1) < T or (not self.hp and self._cos.dtype != x.dtype):
            self.cache_rope(x.device, x.dtype, T)
        q = self.wq(x).reshape(B, T, -1, self.hd)   # -1: local head count under tensor parallelism
        k = self.wk(x).reshape(B, T, -1, self.hd)
        v = self.wv(x).reshape(B, T, -1, self.hd)
        cos, sin = (self._cos, self._sin) if self._cos.size(1) == T else (self._cos[:, :T], self._sin[:, :T])
        if self.hp:
            q, k = apply_rope(q.float(), cos, sin).to(x.dtype), apply_rope(k.float(), cos, sin).to(x.dtype)
        else:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=q.size(1) != k.size(1))
        return self.wo(o.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, c: GPTConfig, layer_id: int, moe_kwargs: Optional[dict] = None):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(c.dim), RMSNorm(c.dim)
        if c.mixer == "cell":
            from .cell import CellMixer
            self.attn = CellMixer(c.dim)
        else:
            self.attn = Attention(c)
        hidden = int(c.dim * c.ffn_mult)
        hidden = (hidden + 63) // 64 * 64
        self.is_moe = c.n_experts > 0 and layer_id % c.moe_every == 0
        if self.is_moe:
            from ..moe import MoE
            mk = dict(moe_kwargs or {})
            mk.setdefault("grouped_mm", c.expert_grouped_mm)
            self.ffn = MoE(c.dim, (int(hidden * c.expert_hidden_mult) + 63) // 64 * 64, c.n_experts, c.top_k,
                           n_shared=c.n_shared_experts, aux_coef=c.aux_coef, z_coef=c.z_coef, **mk)
        else:
            self.ffn = SwiGLU(c.dim, hidden)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig, moe_kwargs: Optional[dict] = None):
        super().__init__()
        self.cfg = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.dim)
        self.layers = nn.ModuleDict({str(i): Block(c, i, moe_kwargs) for i in range(c.n_layers)})
        self.norm = RMSNorm(c.dim)
        self.lm_head = nn.Linear(c.dim, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith(("wo.weight", "w2.weight")):
                nn.init.normal_(p, std=0.02 / (2 * c.n_layers) ** 0.5)
        self._flops_cache: dict = {}

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens, labels=None):
        h = self.tok_emb(tokens) if self.tok_emb is not None else tokens
        aux = None
        for blk in self.layers.values():
            h = blk(h)
            if blk.is_moe:
                a = blk.ffn.last_aux
                if a is not None:
                    aux = a if aux is None else aux + a
        if self.lm_head is None:         # non-final pipeline stage
            return h
        h = self.norm(h)
        if labels is None:
            return self.lm_head(h)
        ce = chunked_cross_entropy(h, self.lm_head.weight, labels, self.cfg.loss_chunk)
        return ce, aux

    def aux_loss(self):
        """Sum of the router aux/z losses from the most recent forward (None if the model has no MoE blocks)."""
        a = [b.ffn.last_aux for b in self.layers.values() if b.is_moe and b.ffn.last_aux is not None]
        if not a:
            return None
        total = a[0]
        for x in a[1:]:
            total = total + x
        return total

    def num_params(self, active_only=False):
        key = ("params", active_only)
        n = self._flops_cache.get(key)
        if n is None:
            n = 0
            for name, p in self.named_parameters():
                if active_only and ".experts." in name:
                    n += p.numel() * self.cfg.top_k // self.cfg.n_experts
                else:
                    n += p.numel()
            self._flops_cache[key] = n
        return n

    def flops_per_token(self, seq):
        c = self.cfg
        key = ("flops", seq)
        v = self._flops_cache.get(key)
        if v is None:
            v = 6 * self.num_params(active_only=True) + 12 * c.n_layers * c.dim * seq
            self._flops_cache[key] = v
        return v

    # ---- hooks used by ForgeMind -------------------------------------------------------------------------------
    def prepare(self, device=None, dtype=None, seq: Optional[int] = None) -> "GPT":
        """Warm every cache that would otherwise be built lazily *inside* a compiled region:
        the RoPE tables (and the MoE's grouped-GEMM offsets buffer)."""
        p = next(self.parameters(), None)
        device = device or (p.device if p is not None else torch.device("cpu"))
        dtype = dtype or (p.dtype if p is not None else torch.float32)
        seq = int(seq or self.cfg.max_seq)
        from ..moe import MoE
        for m in self.modules():
            if isinstance(m, Attention):
                m.cache_rope(device, dtype, seq)
            elif isinstance(m, MoE):
                m.probe_grouped(device, dtype)
        return self

    def tp_plan(self):
        return {"attn.wq": "col", "attn.wk": "col", "attn.wv": "col", "attn.wo": "row",
                "ffn.w1": "col", "ffn.w3": "col", "ffn.w2": "row"}   # dense FFN only; MoE experts are not TP'd

    def split_stage(self, stage: int, n_stages: int):
        """keep only this pipeline stage's layers (FQNs are preserved because layers is a ModuleDict)"""
        n = self.cfg.n_layers
        per = [n // n_stages + (1 if i < n % n_stages else 0) for i in range(n_stages)]
        lo = sum(per[:stage]); hi = lo + per[stage]
        for k in list(self.layers.keys()):
            if not lo <= int(k) < hi:
                del self.layers[k]
        if stage != 0:
            self.tok_emb = None
        if stage != n_stages - 1:
            self.norm = None
            self.lm_head = None
        return self
