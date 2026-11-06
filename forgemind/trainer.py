from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset

from .config import ForgeConfig
from .distributed import init_distributed, DistContext, log
from .data import build_loader, BatchStream
from .loss import logits_cross_entropy
from .offload import OffloadedAdamW, apply_activation_offload
from .perf import warn_once, enable_fast_math
from .quant import quantize_model, make_optimizer_8bit, QuantState
from .parallel import native
from .parallel.ac import apply_activation_checkpointing
from .parallel.pipeline import PipelineRunner


class _Both:
    def __init__(self, *cms): self.cms = cms
    def __enter__(self):
        for c in self.cms: c.__enter__()
    def __exit__(self, *a):
        for c in reversed(self.cms): c.__exit__(*a)


def moe_kwargs(ctx: DistContext, cfg: ForgeConfig) -> dict:
    """kwargs for GPT(cfg, moe_kwargs=...) wiring expert parallelism / expert offload / grouped GEMM"""
    return dict(ep_size=ctx.pc.ep, ep_group=ctx.ep_group, ep_rank=ctx.ep_rank, offload_experts=cfg.offload.experts)


class Trainer:
    """Training loop with a hard budget of one host<->device synchronisation per optimizer step.

    Everything that used to end the step with a stall is now folded into a single stacked `.tolist()`:
    grad norm, loss (already reduced over dp/pp) and the skip flag. The loss EMA and the skip decision
    are made from that one read, so nothing else has to be pulled back to the host.
    """

    def __init__(self, model_fn: Callable[[DistContext], nn.Module], cfg: ForgeConfig, train_ds: Dataset,
                 eval_ds: Optional[Dataset] = None, micro_batch: int = 8, callbacks=(), peak_flops: Optional[float] = None):
        self.cfg, tc, pc, oc = cfg, cfg.train, cfg.parallel, cfg.offload
        self.ctx = ctx = init_distributed(pc)
        self.micro_batch, self.callbacks, self.peak_flops = micro_batch, list(callbacks), peak_flops
        self.use_ds = pc.backend == "deepspeed"
        enable_fast_math(pc.backend == "native")
        torch.manual_seed(tc.seed)
        model = model_fn(ctx)
        self.raw = model
        self.seq = getattr(getattr(model, "cfg", None), "max_seq", None)

        if pc.ep > 1:  # different expert weights per ep rank
            g = torch.Generator().manual_seed(tc.seed + 1 + ctx.ep_rank)
            for n, p in model.named_parameters():
                if ".experts." in n:
                    p.data.normal_(0, 0.02, generator=g)
        mcfg = getattr(model, "cfg", None)
        if pc.pp > 1:
            model.split_stage(ctx.pp_rank, pc.pp)
            if getattr(mcfg, "aux_coef", 0) and getattr(mcfg, "n_experts", 0):
                log(ctx, "[forgemind] WARNING: MoE aux loss is not propagated across pipeline stages")
        model.to(ctx.device)

        # ---- ordering: quant -> TP -> AC / act-offload -> cache warm -> compile -> DP wrapper -> pipeline stage
        self.quant: QuantState = quantize_model(model, cfg.quant, ctx.dp_group)
        if pc.tp > 1:
            native.apply_tp(model, ctx)
        if tc.ac != "none":
            apply_activation_checkpointing(model, tc.ac, preserve_rng_state=tc.ac_preserve_rng)
        if oc.activations and ctx.device.type == "cuda":
            apply_activation_offload(model, oc)
        self.model = model
        self._warm_caches(model, tc.bf16)
        if tc.compile:
            self._compile_blocks(model, tc)

        if not self.use_ds:
            dp = ctx.dp_world > 1
            if dp and pc.dp_shard > 1:
                self.model = native.apply_fsdp(model, ctx, tc.bf16, pc.fsdp_cpu_offload, pc.fsdp_reshard_after_forward)
            elif dp and pc.dp_replicate > 1:
                self.model = native.apply_ddp(model, ctx)
        self.is_ddp = isinstance(self.model, nn.parallel.DistributedDataParallel)
        self.is_fsdp = hasattr(self.model, "set_requires_gradient_sync") and not self.is_ddp

        # ---- optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        decay = [p for p in params if p.ndim >= 2]
        nodecay = [p for p in params if p.ndim < 2]
        groups = [{"params": decay, "weight_decay": tc.weight_decay}, {"params": nodecay, "weight_decay": 0.0}]
        groups = [g for g in groups if g["params"]]
        if not groups:
            raise RuntimeError("[forgemind] no trainable parameters -- every parameter is frozen "
                               "(int8_wo freezes the base weights, so add adapters or unfreeze something)")
        self._trainable = decay + nodecay
        self.off_opt = None
        if self.use_ds:
            from .parallel.deepspeed_backend import init_deepspeed
            self.engine, self.opt = init_deepspeed(self.model, groups, cfg, ctx.world, micro_batch)
        elif oc.optimizer:
            assert pc.pp == 1, "optimizer offload + pipeline not supported"
            overlap = not (self.is_ddp or self.is_fsdp)
            self.opt = self.off_opt = OffloadedAdamW(groups, tc.lr, tc.betas, tc.eps, oc, overlap=overlap)
            self.off_overlap = overlap
        elif cfg.quant.optimizer_8bit:
            self.opt = make_optimizer_8bit(groups, tc.lr, tc.betas, tc.eps)
        else:
            self.opt = torch.optim.AdamW(groups, lr=tc.lr, betas=tc.betas, eps=tc.eps, fused=ctx.device.type == "cuda")

        # cached parameter views: walking named_parameters() every step is measurable at scale
        self._expert_params = native.expert_params(self.model) if pc.ep > 1 else []
        self._has_dtensor = any(hasattr(p, "to_local") for p in self._trainable)

        # ---- data
        self.train_dl, self.sampler = build_loader(train_ds, micro_batch, ctx, tc.seed, workers=tc.workers)
        self.feed = BatchStream(self.train_dl, self.sampler, ctx.device, prefetch=tc.prefetch)
        self.eval_dl = build_loader(eval_ds, micro_batch, ctx, tc.seed, workers=0, shuffle=False)[0] \
            if eval_ds is not None else None

        # ---- pipeline
        self.pp = None
        if pc.pp > 1:
            assert micro_batch % pc.pp_microbatches == 0, "micro_batch must be divisible by pp_microbatches"
            self.pp = PipelineRunner(self.model, ctx, logits_cross_entropy, pc.pp_microbatches, pc.pp_schedule,
                                     fwd_ctx=lambda: _Both(self._autocast(), self.quant.autocast()))

        self.step, self.loss_ema, self.history = 0, None, []
        self.skipped = 0
        n = sum(p.numel() for p in self.raw.parameters())
        log(ctx, f"[forgemind] world={ctx.world} mesh(pp,dp_rep,dp_shard,tp)=({pc.pp},{pc.dp_replicate},{pc.dp_shard},{pc.tp}) ep={pc.ep} "
                 f"backend={pc.backend} quant={cfg.quant.mode}/{self.quant.backend} params(local)={n/1e6:.1f}M device={ctx.device}")
        if tc.resume:
            self.maybe_resume()

    # ------------------------------------------------------------------------------------------------ setup helpers
    def _warm_caches(self, model, bf16):
        """Build every lazily-created table (RoPE, grouped-GEMM probe) *before* torch.compile traces the model."""
        fn = getattr(model, "prepare", None)
        if fn is None:
            return
        dtype = torch.bfloat16 if bf16 else torch.float32
        try:
            fn(device=self.ctx.device, dtype=dtype, seq=self.seq)
        except TypeError:
            fn()

    def _compile_blocks(self, model, tc):
        layers = getattr(model, "layers", None)
        blocks = list(layers.values()) if hasattr(layers, "values") else list(layers or [])
        kw = dict(mode=tc.compile_mode, fullgraph=tc.compile_fullgraph, backend=tc.compile_backend)
        if not blocks:
            model.compile(**kw)
            return
        if tc.compile and self.cfg.offload.activations:
            warn_once("[forgemind] torch.compile + activation offload: saved_tensors_hooks forces a graph break "
                      "inside each spilling block -- the hooks win, the fusion does not", key="compile_actoffload")
        if not tc.compile_fullgraph:
            moes = [b.ffn for b in blocks if getattr(b, "is_moe", False)]
            if any(not getattr(m, "compile_friendly", True) for m in moes):
                warn_once("[forgemind] torch.compile: MoE expert dispatch needs a host read on this device, so each "
                          "MoE block keeps one deliberate graph break (CUDA+bf16 enables torch._grouped_mm)",
                          key="compile_moe_dispatch")
        for b in blocks:
            if tc.compile_fullgraph and getattr(b, "is_moe", False):
                moe = b.ffn
                if hasattr(moe, "compile_friendly") and not moe.compile_friendly:
                    raise RuntimeError("compile_fullgraph=True is not possible for this MoE block: the expert "
                                       "dispatch needs a host read (use torch>=2.8 + CUDA + bf16 for grouped_mm)")
            b.compile(**kw)

    # ------------------------------------------------------------------------------------------------ helpers
    @property
    def epoch(self):
        return self.feed.epoch

    @property
    def batches_in_epoch(self):
        return self.feed.batches

    def lr_at(self, step):
        tc = self.cfg.train
        if step < tc.warmup_steps:
            return tc.lr * (step + 1) / tc.warmup_steps
        prog = min(1.0, (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps))
        return tc.lr * (tc.min_lr_ratio + (1 - tc.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))

    def _next_batch(self):
        return self.feed.next()

    def _autocast(self):
        if self.cfg.train.bf16 and self.ctx.device.type in ("cuda", "cpu"):
            return torch.autocast(self.ctx.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _clip(self):
        """Grad-norm clipping that stays on the device: returns a 0-dim fp32 tensor, never a python float."""
        tc, pc = self.cfg.train, self.cfg.parallel
        if self.off_opt is not None:
            return self.off_opt.clip_grad_norm_(tc.grad_clip).float()
        grads = [p.grad for p in self._trainable if p.grad is not None]
        if not grads:
            return torch.zeros((), device=self.ctx.device, dtype=torch.float32)
        if self._has_dtensor:
            dt = [g for g in grads if hasattr(g, "to_local")]
            pl = [g for g in grads if not hasattr(g, "to_local")]
            norm = torch.nn.utils.get_total_norm(pl).float() if pl else None
            meshes = {}
            for g in dt:                              # DTensors: one norm per mesh (FSDP mesh vs TP mesh differ)
                meshes.setdefault(id(g.device_mesh), []).append(g)
            norms = [torch.nn.utils.get_total_norm(lst) for lst in meshes.values()]
            sq = sum((n.full_tensor() if hasattr(n, "full_tensor") else n).float().pow(2) for n in norms)
            if norm is not None:
                sq = sq + norm.pow(2)
            norm = sq.sqrt()
        else:
            norm = torch.nn.utils.get_total_norm(grads).float()
        if pc.ep > 1 and self.ctx.world > 1:        # experts differ per ep rank -> their norm lives on other ranks
            e = [p.grad for p in self._expert_params if p.grad is not None]
            ee = torch.nn.utils.get_total_norm(e).float().pow(2) if e else torch.zeros((), device=norm.device)
            dense = norm.pow(2) - ee
            dist.all_reduce(ee, group=self.ctx.ep_group)
            norm = (dense + ee).clamp(min=0).sqrt()
        if pc.pp > 1:
            norm = self.ctx.sum_over_pp(norm.pow(2)).clamp(min=0).sqrt()
        if tc.grad_clip and tc.grad_clip > 0:
            coef = torch.clamp(tc.grad_clip / (norm + 1e-6), max=1.0)
            if self._has_dtensor:
                for g in grads:
                    g.mul_(coef)
            else:
                torch._foreach_mul_(grads, coef)
        return norm

    def _fwd_bwd_micro(self, last: bool):
        accum = self.cfg.train.grad_accum
        ids, y = self._next_batch()
        if self.pp is not None:
            loss = self.pp.step(ids, y, last_sync=last)
            return loss if loss is not None else torch.zeros((), device=self.ctx.device)

        if self.is_fsdp:
            self.model.set_requires_gradient_sync(last)
        cm = self.model.no_sync() if (self.is_ddp and not last) else contextlib.nullcontext()
        with cm, self._autocast(), self.quant.autocast():
            ce, aux = self.model(ids, labels=y)
            loss = ce + aux if aux is not None else ce
        if self.use_ds:
            self.engine.backward(loss)
        else:
            (loss / accum).backward()
        if self.off_opt is not None and (self.off_overlap or last):
            self.off_opt.collect()
        return ce.detach().float()

    def train_step(self):
        tc = self.cfg.train
        lr = self.lr_at(self.step)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.model.train()
        if self.off_opt is not None:
            self.off_opt.begin_step()
        t0 = time.perf_counter()
        total = None
        for m in range(tc.grad_accum):
            ce = self._fwd_bwd_micro(m == tc.grad_accum - 1)
            total = ce if total is None else total + ce
        loss = total * (1.0 / tc.grad_accum)

        if self.use_ds:
            self.engine.step()
            gn_v = float(self.engine.get_global_grad_norm() or 0.0)
            lv = float(loss)
            ok = True
        else:
            native.sync_expert_grads(self.model, self.ctx, self._expert_params)
            gn = self._clip().to(self.ctx.device)              # device scalar
            loss = self._reduce_loss(loss)                     # device scalar
            finite = (torch.isfinite(gn) & torch.isfinite(loss)).to(torch.int32) if tc.skip_nonfinite \
                else torch.ones((), dtype=torch.int32, device=loss.device)
            flag = self.ctx.all_min_int(finite)                # unanimous skip decision, zero host reads
            # v v v the *only* host synchronisation in the whole step
            flag_v, gn_v, lv = torch.stack([flag.to(torch.float32), gn.to(torch.float32), loss]).tolist()
            ok = bool(flag_v > 0.5)
            if tc.skip_spike_factor > 0 and self.loss_ema is not None and lv > tc.skip_spike_factor * self.loss_ema:
                ok = False
            if ok:
                self.opt.step()
                self.loss_ema = lv if self.loss_ema is None else 0.98 * self.loss_ema + 0.02 * lv
            else:
                self.skipped += 1
            self.opt.zero_grad(set_to_none=True)

        dt = time.perf_counter() - t0
        toks = self.micro_batch * (self.seq or 0) * tc.grad_accum * self.ctx.dp_world
        m = {"step": self.step + 1, "loss": lv, "lr": lr, "grad_norm": gn_v, "tok_s": toks / dt, "dt": dt,
             "skipped": self.skipped}
        if self.peak_flops and hasattr(self.raw, "flops_per_token") and self.seq:
            m["mfu"] = toks * self.raw.flops_per_token(self.seq) / dt / (self.peak_flops * self.ctx.world)
        if self.ctx.device.type == "cuda":
            m["mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        self.step += 1
        return m

    def _reduce_loss(self, loss):
        loss = loss.to(torch.float32)
        if self.cfg.parallel.pp > 1:      # loss only exists on the last stage
            loss = self.ctx.sum_over_pp(loss)
        return self.ctx.mean_over_dp(loss)

    @torch.no_grad()
    def evaluate(self):
        if self.eval_dl is None or self.cfg.parallel.pp > 1 or self.use_ds:
            return None
        self.model.eval()
        tot, n = torch.zeros((), device=self.ctx.device), 0
        for i, b in enumerate(self.eval_dl):
            if i >= self.cfg.train.eval_steps:
                break
            with self._autocast(), self.quant.autocast():
                ce, _ = self.model(b["input_ids"].to(self.ctx.device), labels=b["labels"].to(self.ctx.device))
            tot += ce.float(); n += 1
        self.model.train()
        return float(self.ctx.mean_over_dp(tot / max(n, 1)))

    # ------------------------------------------------------------------------------------------------ fit
    def fit(self):
        tc, ctx = self.cfg.train, self.ctx
        while self.step < tc.max_steps:
            m = self.train_step()
            self.history.append(m)
            if self.step % tc.log_every == 0 or self.step == 1:
                extra = f" mfu {m['mfu']*100:.1f}%" if "mfu" in m else ""
                extra += f" mem {m['mem_gb']:.1f}G" if "mem_gb" in m else ""
                log(ctx, f"step {m['step']:>6} | loss {m['loss']:.4f} | lr {m['lr']:.2e} | gnorm {m['grad_norm']:.2f} | "
                         f"{m['tok_s']:,.0f} tok/s | {m['dt']*1000:.0f} ms{extra}" + (f" | skipped {self.skipped}" if self.skipped else ""))
            if tc.eval_every and self.step % tc.eval_every == 0:
                ev = self.evaluate()
                if ev is not None:
                    log(ctx, f"step {self.step:>6} | eval loss {ev:.4f} | ppl {math.exp(min(ev, 20)):.2f}")
            for cb in self.callbacks:
                cb(self, m)
            if tc.ckpt_every and self.step % tc.ckpt_every == 0:
                self.save()
        if tc.ckpt_every:
            self.save()
        return self.history

    # ------------------------------------------------------------------------------------------------ checkpoint
    def _paths(self, step):
        root = self.cfg.train.ckpt_dir
        return root, os.path.join(root, f"step_{step:07d}")

    def _dcp_state(self):
        from torch.distributed.checkpoint.state_dict import get_state_dict
        opt = () if self.off_opt is not None or self.use_ds else self.opt
        return get_state_dict(self.model, opt)

    def save(self, keep_last: int = 3):
        import torch.distributed.checkpoint as dcp
        root, path = self._paths(self.step)
        os.makedirs(path, exist_ok=True)
        if self.use_ds:
            self.engine.save_checkpoint(root, tag=f"step_{self.step:07d}")
        else:
            msd, osd = self._dcp_state()
            state = {"model": msd}
            if self.off_opt is None:
                state["optim"] = osd
            dcp_dir = os.path.join(path, "dcp")
            shutil.rmtree(dcp_dir, ignore_errors=True)     # stale shards otherwise trip the "overwrite" guard
            dcp.save(state, checkpoint_id=dcp_dir)
            if self.off_opt is not None:
                torch.save(self.off_opt.state_dict(), os.path.join(path, f"offopt_rank{self.ctx.rank}.pt"))
        meta = {"step": self.step, "epoch": self.epoch, "batches_in_epoch": self.batches_in_epoch,
                "loss_ema": self.loss_ema, "micro_batch": self.micro_batch}
        with open(os.path.join(path, f"meta_rank{self.ctx.rank}.json"), "w") as f:
            json.dump(meta, f)
        self.ctx.barrier()
        if self.ctx.rank == 0:
            with open(os.path.join(root, "latest"), "w") as f:
                f.write(os.path.basename(path))
            steps = sorted(d for d in os.listdir(root) if d.startswith("step_"))
            for d in steps[:-keep_last]:
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
        self.ctx.barrier()
        log(self.ctx, f"[forgemind] saved checkpoint {path}")

    def maybe_resume(self):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import set_state_dict
        root = self.cfg.train.ckpt_dir
        latest = os.path.join(root, "latest")
        if not os.path.exists(latest):
            return
        path = os.path.join(root, open(latest).read().strip())
        if self.use_ds:
            self.engine.load_checkpoint(root, tag=os.path.basename(path))
        else:
            msd, osd = self._dcp_state()
            state = {"model": msd}
            if self.off_opt is None:
                state["optim"] = osd
            dcp.load(state, checkpoint_id=os.path.join(path, "dcp"))
            if self.off_opt is not None:
                set_state_dict(self.model, (), model_state_dict=state["model"], optim_state_dict={})
            else:
                set_state_dict(self.model, self.opt, model_state_dict=state["model"],
                               optim_state_dict=state["optim"])
            if self.off_opt is not None:
                self.off_opt.load_state_dict(torch.load(os.path.join(path, f"offopt_rank{self.ctx.rank}.pt"),
                                                        map_location="cpu", weights_only=False))
        meta = json.load(open(os.path.join(path, f"meta_rank{self.ctx.rank}.json")))
        self.step, self.loss_ema = meta["step"], meta["loss_ema"]
        self.feed.resume(meta["epoch"], meta["batches_in_epoch"])
        log(self.ctx, f"[forgemind] resumed from {path} at step {self.step}")
