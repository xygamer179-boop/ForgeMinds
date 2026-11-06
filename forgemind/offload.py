"""Smart offloading: (1) gradients + optimizer states -> CPU, (2) saved activations -> CPU, (3) MoE experts -> CPU (see moe.py).

What changed for speed (behaviour is unchanged apart from the bug fixes listed in PERF.md):

``OffloadedAdamW``
  * gradients land **directly** in the fp32 host accumulator for the first micro-batch (it is zero, so a
    copy *is* an accumulate). The old code allocated a fresh pinned staging tensor for every parameter on
    every micro-batch -- a `cudaHostAlloc` + a copy + an `add_` per grad, forever. Steady state now does
    zero pinned allocations.
  * `flush()` synchronises the side stream **once**, and only if something is actually in flight.
  * `step()` copies the fp32 master weights back with no stream synchronisation at all (stream ordering
    already guarantees the next kernel sees them). It used to end every optimizer step with a full
    `torch.cuda.current_stream().synchronize()`.
  * `clip_grad_norm_` / zeroing use `_foreach_*` instead of per-tensor python loops.
  * the gradient hooks are installed once and never re-registered; `collect()` is idempotent.

``ActivationOffload``
  * one hook object per block instead of constructing `saved_tensors_hooks` closures on **every** forward
    (that was pure per-step python garbage on the critical path);
  * pinned staging buffers are recycled through a size-keyed pool instead of one pinned allocation per
    saved tensor per forward. A buffer only re-enters the pool when the backward that reads it has run,
    which is the same stream-ordered position as the H2D copy, so reuse can never race.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import OffloadConfig
from .perf import _PinnedPool


def _gpu_pressure() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return 1.0 - free / total


class OffloadedAdamW:
    """AdamW whose fp32 master weights and moments live in (pinned) CPU RAM.

    Gradients are streamed GPU->CPU on a side stream the moment they are produced (post-accumulate hook),
    so the GPU never holds more than a few live grads. "Smart" policy: tiny tensors stay put (one big copy
    later beats many tiny ones) and, with gpu_watermark>0, nothing is streamed until GPU memory use
    crosses the watermark. Works for single-GPU and DDP (call `.collect()` after backward when hooks are
    off). Not for FSDP (use CPUOffloadPolicy).
    """

    def __init__(self, param_groups, lr, betas=(0.9, 0.95), eps=1e-8, cfg: OffloadConfig = None, overlap=True):
        self.cfg = cfg or OffloadConfig()
        self.params = [p for g in param_groups for p in g["params"]]
        assert not any(hasattr(p, "to_local") for p in self.params), "OffloadedAdamW does not support DTensor/FSDP params"
        assert self.params, "OffloadedAdamW got no parameters (all frozen?)"
        self.cuda = any(p.is_cuda for p in self.params)
        self.stream = torch.cuda.Stream() if self.cuda else None
        self.master, self.cgrad = {}, {}
        self._staging: dict = {}      # param -> pinned host buffer, only for accum>1
        self._filled: set = set()     # params whose accumulator already holds a value
        self._pending: list = []      # (param, host buffer) copies still in flight
        self._pressure = 0.0

        groups = []
        for g in param_groups:
            ms = []
            for p in g["params"]:
                m = p.detach().to("cpu", torch.float32, copy=True)
                cg = torch.zeros_like(m)
                if self.cuda:
                    m, cg = m.pin_memory(), cg.pin_memory()
                m.grad = cg
                self.master[p], self.cgrad[p] = m, cg
                ms.append(m)
            groups.append({**{k: v for k, v in g.items() if k != "params"}, "params": ms})
        self.opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
        self.param_groups = self.opt.param_groups
        self._cgrad_list = [self.cgrad[p] for p in self.params]
        self.hooks = []
        if overlap:
            for p in self.params:
                self.hooks.append(p.register_post_accumulate_grad_hook(self._hook))

    # -- grad streaming ----------------------------------------------------------------------------------------------
    def begin_step(self) -> None:
        """Refresh the (cheap but not free) GPU memory-pressure reading once per step, not per grad."""
        if self.cfg.smart and self.cfg.gpu_watermark > 0:
            self._pressure = _gpu_pressure()

    def _eligible(self, p):
        if not self.cfg.smart:
            return True
        if p.numel() < self.cfg.min_numel:
            return False
        return self.cfg.gpu_watermark <= 0 or self._pressure >= self.cfg.gpu_watermark

    def _hook(self, p):
        if p.grad is not None and self._eligible(p):
            self._offload(p)

    def _staging_buf(self, p, dtype):
        # NOTE: takes the grad dtype explicitly -- `p.grad` is already None by the time we get here.
        buf = self._staging.get(p)
        if buf is None or buf.dtype != dtype:
            buf = torch.empty(p.shape, dtype=dtype, device="cpu", pin_memory=self.cuda)
            self._staging[p] = buf
        return buf

    def _offload(self, p):
        g = p.grad
        if g is None:
            return
        p.grad = None
        acc = self.cgrad[p]
        if not (self.cuda and g.is_cuda):
            acc.add_(g)                                  # already on the host
            self._filled.add(p)
            return
        cur = torch.cuda.current_stream()
        if p not in self._filled:
            # accumulator is zero -> a plain copy into it *is* the accumulate (no staging buffer needed)
            self.stream.wait_stream(cur)
            with torch.cuda.stream(self.stream):
                acc.copy_(g, non_blocking=True)
            g.record_stream(self.stream)
            self._filled.add(p)
            self._pending.append((p, None))
        else:
            buf = self._staging_buf(p, g.dtype)
            self.stream.wait_stream(cur)
            with torch.cuda.stream(self.stream):
                buf.copy_(g, non_blocking=True)
            g.record_stream(self.stream)
            self._pending.append((p, buf))

    def collect(self):
        """move every remaining grad off the device (call after each micro-batch backward)"""
        for p in self.params:
            if p.grad is not None:
                self._offload(p)
        self.flush()

    def flush(self):
        if not self._pending:
            return
        if self.cuda:
            self.stream.synchronize()                    # one sync, only when copies are in flight
            for p, buf in self._pending:
                if buf is not None:
                    self.cgrad[p].add_(buf)
        self._pending.clear()

    # -- optimizer API -----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def clip_grad_norm_(self, max_norm):
        self.collect()
        norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(self._cgrad_list)))
        if max_norm and max_norm > 0:
            coef = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
            torch._foreach_mul_(self._cgrad_list, coef)
        return norm

    @torch.no_grad()
    def step(self):
        self.collect()
        self.opt.step()
        if self.cuda:
            # fp32 master -> device; the copy is enqueued on the current stream, so no synchronise() needed
            for p in self.params:
                p.data.copy_(self.master[p], non_blocking=True)
        else:
            for p in self.params:
                p.data.copy_(self.master[p])
        self._reset_accumulators()

    def _reset_accumulators(self):
        torch._foreach_zero_(self._cgrad_list)
        self._filled.clear()

    def zero_grad(self, set_to_none=True):
        if self._pending:
            self.flush()
        for p in self.params:
            p.grad = None
        self._reset_accumulators()

    def state_dict(self):
        return {"opt": self.opt.state_dict(), "master": [self.master[p] for p in self.params]}

    def load_state_dict(self, sd):
        self.opt.load_state_dict(sd["opt"])
        for p, m in zip(self.params, sd["master"]):
            self.master[p].copy_(m)
            p.data.copy_(self.master[p])
        self._reset_accumulators()

    def __repr__(self):
        n = sum(p.numel() for p in self.params)
        return f"OffloadedAdamW(params={len(self.params)}, {n/1e6:.1f}M, device={'cuda' if self.cuda else 'cpu'})"


class ActivationOffload(torch.autograd.graph.saved_tensors_hooks):
    """Spill big saved-for-backward CUDA activations to pinned host memory; weights and small tensors stay on device."""

    def __init__(self, min_bytes=1 << 20, pool: _PinnedPool | None = None):
        self.pool = pool
        pin = torch.cuda.is_available()

        def pack(t):
            if (not t.is_cuda or t.numel() * t.element_size() < min_bytes
                    or isinstance(t, nn.Parameter) or (t.requires_grad and t.is_leaf)):
                return (None, t)
            buf = self.pool.get(t.shape, t.dtype) if self.pool is not None else \
                torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=pin)
            buf.copy_(t, non_blocking=True)   # same-stream ordered: safe to free t afterwards
            return (t.device, buf)

        def unpack(x):
            dev, buf = x
            if dev is None:
                return buf
            out = buf.to(dev, non_blocking=True)
            if self.pool is not None:
                self.pool.put(buf)            # readable again only after this H2D, which is stream-ordered
            return out

        super().__init__(pack, unpack)


def apply_activation_offload(model, cfg: OffloadConfig):
    blocks = list(model.layers.values())
    n = max(1, int(len(blocks) * cfg.act_layers)) if blocks else 0
    pool = _PinnedPool(pin=torch.cuda.is_available()) if cfg.act_pool else None
    for b in blocks[:n]:
        hook = ActivationOffload(cfg.act_min_bytes, pool)     # built once, reused every step
        orig = b.forward

        def fwd(*a, _o=orig, _h=hook, **k):
            with _h:
                return _o(*a, **k)
        b.forward = fwd
    return model
