"""Minimal, dependency-free pipeline-parallel runtime (GPipe + 1F1B) over point-to-point send/recv.
Static activation shape per microbatch; dtype/shape handshake happens once on the first step.

Speed work over the original:
  * activation / gradient receive buffers are allocated **once per microbatch slot** and reused, instead of
    a fresh `torch.empty` on every send and every recv (the allocator calls and the resulting memory churn
    were on the critical path of both the forward and the backward pass);
  * the handshake is re-armed automatically when the batch shape changes, so a validation pass with a
    different batch size no longer hangs or corrupts the exchange;
  * all outstanding `isend`s are waited on once, at the end of the step, keeping the tensors alive in a
    single list rather than relying on the caller;
  * the per-micro-batch losses are summed into one scalar instead of `torch.stack(...).mean()`.
"""
from __future__ import annotations

import contextlib

import torch
import torch.distributed as dist

_DT = [torch.float32, torch.bfloat16, torch.float16]


class PipelineRunner:
    def __init__(self, model, ctx, loss_fn, n_microbatches, schedule="1f1b", fwd_ctx=contextlib.nullcontext):
        self.model, self.ctx, self.loss_fn, self.M = model, ctx, loss_fn, n_microbatches
        self.schedule, self.fwd_ctx = schedule, fwd_ctx
        self.s, self.S, self.g = ctx.pp_rank, ctx.pc.pp, ctx.pp_group
        self.first, self.last = self.s == 0, self.s == self.S - 1
        self.prev = dist.get_global_rank(self.g, self.s - 1) if not self.first else None
        self.next = dist.get_global_rank(self.g, self.s + 1) if not self.last else None
        self.meta = None        # (shape, dtype) of the activation arriving from the previous stage
        self.meta_sent = False
        self._act_bufs: list = []    # per-microbatch receive buffers
        self._grad_bufs: list = []   # per-microbatch output-grad buffers
        self._batch_shape = None

    # ---- p2p -------------------------------------------------------------------------------------------------------
    def _send_header(self, t):
        h = torch.zeros(8, dtype=torch.long, device=self.ctx.device)
        h[0], h[1] = t.ndim, _DT.index(t.dtype)
        h[2:2 + t.ndim] = torch.tensor(t.shape)
        dist.send(h, self.next, group=self.g)
        self.meta_sent = True

    def reset(self):
        """Re-arm the dtype/shape handshake (call when the activation shape may change)."""
        self.meta = None
        self.meta_sent = False
        self._act_bufs.clear()
        self._grad_bufs.clear()

    def _recv_act(self, i: int):
        if self.meta is None:
            h = torch.zeros(8, dtype=torch.long, device=self.ctx.device)
            dist.recv(h, self.prev, group=self.g)
            self.meta = (tuple(h[2:2 + int(h[0])].tolist()), _DT[int(h[1])])
        while len(self._act_bufs) <= i:
            self._act_bufs.append(None)
        buf = self._act_bufs[i]
        if buf is None or tuple(buf.shape) != self.meta[0] or buf.dtype != self.meta[1]:
            buf = torch.empty(self.meta[0], dtype=self.meta[1], device=self.ctx.device)
            self._act_bufs[i] = buf
        dist.recv(buf, self.prev, group=self.g)
        return buf

    def _grad_buf(self, i: int, like):
        while len(self._grad_bufs) <= i:
            self._grad_bufs.append(None)
        buf = self._grad_bufs[i]
        if buf is None or buf.shape != like.shape or buf.dtype != like.dtype:
            buf = torch.empty_like(like).contiguous()
            self._grad_bufs[i] = buf
        return buf

    # ---- one optimizer-step worth of microbatches --------------------------------------------------------------------
    def step(self, ids, y, last_sync=True):
        M = self.M
        if ids.shape != self._batch_shape:            # static per run; re-arm the handshake if it moves
            self.reset()
            self._batch_shape = ids.shape
        xs, ys = ids.chunk(M), y.chunk(M)
        stash, keep, sends = [], [], []
        loss_sum = None
        sync_off = hasattr(self.model, "no_sync")
        fsdp = hasattr(self.model, "set_requires_gradient_sync") and not sync_off

        def fwd(i):
            nonlocal loss_sum
            if self.first:
                inp = xs[i]
            else:
                inp = self._recv_act(i).requires_grad_(True)
            with self.fwd_ctx():
                out = self.model(inp)
            if self.last:
                loss = self.loss_fn(out, ys[i])
                loss_sum = loss.detach().float() if loss_sum is None else loss_sum + loss.detach().float()
                stash.append((i, inp, loss / M))     # scale before backward so the sum matches a full-batch step
            else:
                if not self.meta_sent:
                    self._send_header(out)
                o = out.detach()
                if not o.is_contiguous():
                    o = o.contiguous()
                keep.append(o)
                sends.append(dist.isend(o, self.next, group=self.g))
                stash.append((i, inp, out))

        def bwd(final):
            i, inp, out = stash.pop(0)
            if fsdp:
                self.model.set_requires_gradient_sync(final and last_sync)
            cm = self.model.no_sync() if (sync_off and not (final and last_sync)) else contextlib.nullcontext()
            with cm:
                if self.last:
                    out.backward()
                else:
                    g = self._grad_buf(i, out)
                    dist.recv(g, self.next, group=self.g)
                    out.backward(g)
            if not self.first:
                gi = inp.grad
                if not gi.is_contiguous():
                    gi = gi.contiguous()
                keep.append(gi)
                sends.append(dist.isend(gi, self.prev, group=self.g))
                inp.grad = None      # receive buffers are reused across micro-batches/steps: start clean

        if self.schedule == "gpipe":
            for i in range(M):
                fwd(i)
            for j in range(M):
                bwd(j == M - 1)
        else:  # 1F1B
            warm = min(M, self.S - self.s - 1)
            for i in range(warm):
                fwd(i)
            done = 0
            for i in range(M - warm):
                fwd(warm + i)
                bwd(done == M - 1); done += 1
            while done < M:
                bwd(done == M - 1); done += 1
        for w in sends:
            w.wait()
        return loss_sum / M if (self.last and loss_sum is not None) else None
