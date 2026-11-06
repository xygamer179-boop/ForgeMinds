"""``Forge`` -- a two-line training loop for *any* torch model.

ForgeMind's ``Trainer`` is opinionated: it knows about GPT blocks, MoE routing, pipeline
schedules and DeepSpeed. That is great when you are training the model it was written for and
useless when you have a ``nn.Module``, a ``DataLoader`` and a loss function. ``Forge`` is the
small, boring interface for the second case -- the same one Lightning/Fabric gives you, with
the same guarantees this library is built on: **one host synchronization per optimizer step**,
device-side clipping, accumulated micro-batches, and everything optional.

Three ways to use it, in increasing order of control::

    # 1. hand it everything, let it own the loop
    forge = fm.Forge(model, optimizer, loss_fn=criterion, precision="bf16")
    forge.fit(loader, max_steps=1000)

    # 2. Fabric style: you own the loop
    out = model(x); loss = criterion(out, y)
    forge.backward(loss)        # scaled by 1/accum, autocast already applied
    forge.step()                # updates only every `accum` calls

    # 3. multi-process with one decorator (spawns, like Lightning's `trainer.fit`)
    @fm.launch(nproc=4)
    def main():
        forge = fm.Forge(build_model(), build_opt(), precision="bf16")
        forge.fit(loader)

    main()

What it does for you
--------------------
* moves tensors in the batch to the device (dict / list / tuple / nested),
* autocast (``bf16``/``fp16``, with a ``GradScaler`` for fp16) and fp32 master weights,
* gradient accumulation with the loss divided by ``accum``,
* clipping with a **device-side** coefficient -- no ``float(norm)`` stall,
* DDP wrapping when a process group is already initialized,
* LR schedule (cosine with warmup by default, or pass a ``scheduler``),
* non-finite / loss-spike skipping,
* checkpoint save/load including optimizer, scheduler and RNG state,
* async batches: the loader runs on a background thread and the copy to the device overlaps
  the step, so the GPU never waits for the CPU to build a batch,
* callbacks (``on_step``/``on_epoch``) and a history list,
* optional optimizer offload, fused kernels and ``torch.compile``.

What it will not do
-------------------
A batch is fed to the model as ``model(**batch)`` for dicts, ``model(*batch)`` for tuples/lists
and ``model(batch)`` for tensors -- pass ``forward_fn(model, batch)`` for anything else.

It will not guess your loss. If the model's output is not a scalar, pass ``loss_fn``.
"""

from __future__ import annotations

import contextlib
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn

from .config import OffloadConfig, ParallelConfig
from .data import async_loader as _async_loader
from .data import to_device as _to_device
from .distributed import cleanup, init_distributed
from .offload import OffloadedAdamW
from .perf import count_host_syncs, warn_once

__all__ = ["Forge", "launch", "ForgeState"]

_PRECISIONS = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _shutdown(loader):
    """Stop a background prefetcher, if this loader has one."""
    close = getattr(loader, "close", None)
    if callable(close):
        close()


class ForgeState(dict):
    """Metrics from the last completed optimizer step (attribute access for convenience)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:  # pragma: no cover
            raise AttributeError(k) from e


class Forge:
    """A minimal, dependency-free training façade for arbitrary torch code.

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer | None
        Built for ``model.parameters()`` *after* DDP wrapping is fine; if omitted, AdamW with
        ``lr`` is created for you.
    loss_fn : callable | None
        ``loss_fn(model_output, batch) -> scalar tensor``. Default: use the output directly if
        it is a scalar tensor, else the first element of a tuple, else ``output["loss"]``.
    precision : "fp32" | "bf16" | "fp16"
    accum : int
        Micro-batches per optimizer step.
    clip : float
        Max grad norm; 0 disables clipping.
    lr, weight_decay, warmup, schedule : optimizer/schedule defaults when ``optimizer`` is None
        or when you ask for the built-in schedule (``schedule="cosine"``).
    compile : bool
        ``torch.compile`` the module (skipped on CPU unless ``FORGEMIND_COMPILE_CPU=1``).
    offload : "none" | "optimizer"
        Offload Adam moments to pinned host memory (see ``OffloadConfig``).
    fused_ops : bool
        Patch RMSNorm/SwiGLU modules to the packed kernels.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, *,
                 loss_fn=None, device=None, precision: str = "bf16", accum: int = 1,
                 clip: float = 1.0, lr: float = 3e-4, weight_decay: float = 0.1,
                 betas=(0.9, 0.95), eps: float = 1e-8, warmup: int = 0,
                 max_steps: int = 1000, min_lr_ratio: float = 0.1,
                 schedule: str | None = None, scheduler=None, compile: bool = False,
                 compile_backend: str = "inductor", compile_mode: str = "default", forward_fn=None,
                 offload: str = "none", fused_ops: bool = False, fused_split_mlp: bool = False,
                 async_batches: bool = True, prefetch_depth: int = 2,
                 seed: int = 1234, log_every: int = 10, skip_nonfinite: bool = True,
                 skip_spike_factor: float = 0.0, callbacks=None, quiet: bool = False):
        if precision not in _PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(_PRECISIONS)}")
        if accum < 1:
            raise ValueError("accum must be >= 1")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.precision = precision
        self.amp_dtype = _PRECISIONS[precision]
        self.accum = int(accum)
        self.clip = float(clip)
        self.loss_fn = loss_fn
        # how a batch becomes a forward call: dict -> model(**batch), list/tuple -> model(*batch),
        # tensor -> model(tensor). Pass forward_fn(model, batch) for anything else.
        self.forward_fn = forward_fn
        self.async_batches = bool(async_batches)
        self.prefetch_depth = max(1, int(prefetch_depth))
        self.quiet = quiet
        self.log_every = max(1, int(log_every))
        self.seed = seed
        self.max_steps = int(max_steps)
        self.warmup = int(warmup)
        self.min_lr_ratio = float(min_lr_ratio)
        self.skip_nonfinite = skip_nonfinite
        self.skip_spike_factor = float(skip_spike_factor)
        self.callbacks = list(callbacks or [])
        self.history: list[dict] = []
        self.step_count = 0
        self.micro = 0
        self._acc_loss = None
        self.ema_loss = None
        self._skipped = 0

        torch.manual_seed(seed)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        model = model.to(self.device)
        if fused_ops:
            from .kernels.apply import enable
            enable(model, split=fused_split_mlp)
        if compile:
            backend = compile_backend
            if self.device.type == "cpu" and os.environ.get("FORGEMIND_COMPILE_CPU") != "1":
                warn_once("[forgemind] torch.compile skipped on CPU (set FORGEMIND_COMPILE_CPU=1 "
                          "to force it)", key="forge_compile_cpu")
            else:
                model = torch.compile(model, backend=backend, mode=compile_mode)
        self.raw_model = model
        self.model = self._maybe_ddp(model)

        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                          weight_decay=weight_decay, betas=betas, eps=eps)
        if offload == "optimizer":
            optimizer = OffloadedAdamW(
                optimizer.param_groups, lr=optimizer.param_groups[0]["lr"],
                betas=optimizer.param_groups[0].get("betas", betas),
                eps=optimizer.param_groups[0].get("eps", eps), cfg=OffloadConfig())
        elif offload not in ("none", None):
            raise ValueError("offload must be 'none' or 'optimizer'")
        self.optimizer = optimizer
        self.base_lrs = [g["lr"] for g in self.optimizer.param_groups]
        self.scheduler = scheduler
        if scheduler is None and schedule == "cosine":
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda s: self._cosine_lr(s))
        self.scaler = (torch.amp.GradScaler("cuda", enabled=True)
                       if precision == "fp16" and self.device.type == "cuda" else None)

    # ------------------------------------------------------------------ setup
    def _maybe_ddp(self, model):
        if self.world_size == 1:
            return model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            return model
        device_ids = [self.device] if self.device.type == "cuda" else None
        return nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

    def setup(self, model=None, optimizer=None):
        """Fabric-compatible alias: ``model, opt = forge.setup(model, opt)``."""
        return (model if model is not None else self.model,
                optimizer if optimizer is not None else self.optimizer)

    def _cosine_lr(self, step):
        if self.warmup and step < self.warmup:
            return (step + 1) / self.warmup
        span = max(1, self.max_steps - self.warmup)
        t = min(1.0, max(0.0, (step - self.warmup) / span))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t))

    # ------------------------------------------------------------------ loop
    def _autocast(self):
        if self.precision == "fp32":
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _extract_loss(self, out, batch):
        if self.loss_fn is not None:
            return self.loss_fn(out, batch)
        if torch.is_tensor(out):
            return out
        if isinstance(out, dict) and "loss" in out:
            return out["loss"]
        if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
            return out[0]
        raise TypeError("cannot infer a loss from the model output; pass loss_fn=")

    def backward(self, loss):
        """Back-propagate one micro-batch (already scaled by 1/accum)."""
        if loss.dim() != 0:
            loss = loss.mean()
        scaled = loss / self.accum
        if self.scaler is not None:
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()
        self.micro += 1
        self._last_loss = loss.detach()
        # tensor-side accumulator: the step reports the mean loss over its micro-batches
        # without costing an extra host read (it is folded into the single stacked read).
        loss_f = loss.detach().float()
        self._acc_loss = loss_f if self._acc_loss is None else self._acc_loss + loss_f

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        self.micro = 0

    @torch.no_grad()
    def _clip_coef(self):
        if self.clip <= 0:
            return None
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        return gn

    def step(self, batch=None):
        """Run one micro-batch (if ``batch`` given) and update when accumulation is done.

        Returns a :class:`ForgeState` on a completed optimizer step, else ``None``.
        """
        if batch is not None:
            batch = _to_device(batch, self.device)
            with self._autocast():
                out = self._call_with_batch(batch)
                loss = self._extract_loss(out, batch)
            self.backward(loss)
        if self.micro < self.accum:
            return None
        return self._update()

    def _call_with_batch(self, batch):
        if self.forward_fn is not None:
            return self.forward_fn(self.model, batch)
        if isinstance(batch, dict):
            return self.model(**batch)
        if isinstance(batch, (tuple, list)):
            return self.model(*batch)
        return self.model(batch)  # pragma: no cover

    def _update(self):
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        gn = self._clip_coef()
        skipped = False
        # exactly one host read per optimizer step, whatever the diagnostics: stack the scalars
        # the update needs and pull them across together.
        step_loss = (self._acc_loss / max(1, self.micro)) if self._acc_loss is not None \
            else self._last_loss.float()
        if gn is not None:
            loss_v, gn_v = torch.stack([step_loss.float(), gn.detach().float()]).tolist()
        else:
            (loss_v,), gn_v = torch.stack([step_loss.float()]).tolist(), float("nan")
        self._acc_loss = None
        if self.skip_nonfinite or self.skip_spike_factor > 0:
            finite = bool(math.isfinite(loss_v) and math.isfinite(gn_v))
            spike = (self.skip_spike_factor > 0 and self.ema_loss is not None
                     and loss_v > self.skip_spike_factor * self.ema_loss)
            skipped = (not finite) or spike
        if skipped:
            self._skipped += 1
            self.optimizer.zero_grad(set_to_none=True)
            self.micro = 0
            self._acc_loss = None
            return ForgeState(step=self.step_count, loss=loss_v, grad_norm=gn_v,
                              skipped=True, lr=self._cur_lr())
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.micro = 0
        self.step_count += 1
        if self.scheduler is not None:
            self.scheduler.step()
        state = ForgeState(step=self.step_count, loss=loss_v, grad_norm=gn_v,
                           skipped=False, lr=self._cur_lr())
        if self.skip_spike_factor > 0:
            self.ema_loss = state.loss if self.ema_loss is None else (
                0.95 * self.ema_loss + 0.05 * state.loss)
        self.history.append(dict(state))
        if not self.quiet and self.step_count % max(1, self.log_every) == 0:
            print(f"step {self.step_count:6d} | loss {state.loss:.4f} | "
                  f"gnorm {state.grad_norm:.3f} | lr {state.lr:.2e}", flush=True)
        for cb in self.callbacks:
            fn = cb.on_step if hasattr(cb, "on_step") else cb
            fn(self, state)
        return state

    def _cur_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    # ------------------------------------------------------------------ convenience loops
    def _batches(self, loader):
        """Batches on the device, produced on a background thread while the step computes."""
        if not self.async_batches:
            return loader
        return _async_loader(loader, self.device, depth=self.prefetch_depth)

    def fit(self, loader, max_steps=None, epochs: int = 1):
        """Iterate ``loader`` until ``max_steps`` optimizer steps have been taken."""
        loader = self._batches(loader)
        limit = max_steps or self.max_steps
        t0 = time.perf_counter()
        last = None
        stopped = False
        try:
            for epoch in range(epochs):
                for batch in loader:
                    last = self.step(batch)
                    if self.step_count >= limit:
                        stopped = True
                        break
                    if last is not None:
                        last = dict(last, ms=1e3 * (time.perf_counter() - t0))
                for cb in self.callbacks:
                    if hasattr(cb, "on_epoch"):
                        cb.on_epoch(self, epoch)
                if stopped:
                    break
        finally:
            _shutdown(loader)          # stop the prefetch thread, not just drop the reference
        return self.history

    @torch.no_grad()
    def evaluate(self, loader, max_batches: int | None = None):
        loader = self._batches(loader)
        self.model.eval()
        total, n = 0.0, 0
        try:
            for i, batch in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                batch = _to_device(batch, self.device)
                with self._autocast():
                    loss = self._extract_loss(self._call_with_batch(batch), batch)
                total += float(loss.detach())
                n += 1
        finally:
            _shutdown(loader)
            self.model.train()
        return total / max(1, n)

    # ------------------------------------------------------------------ state
    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        model = self.raw_model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        torch.save({
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "step": self.step_count,
            "ema_loss": self.ema_loss,
            "rng": torch.get_rng_state(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model = self.raw_model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step_count = int(ckpt.get("step", 0))
        self.ema_loss = ckpt.get("ema_loss")
        if "rng" in ckpt:
            torch.set_rng_state(ckpt["rng"])
        return self

    def count_syncs(self):
        """``with forge.count_syncs() as c: ...`` -- audit your own step for host reads."""
        return count_host_syncs()

    def __repr__(self):
        n = sum(p.numel() for p in self.raw_model.parameters())
        return (f"Forge(model_params={n/1e6:.2f}M, device={self.device.type}, "
                f"precision={self.precision}, accum={self.accum}, world={self.world_size})")


# ---------------------------------------------------------------------- launcher
def _init_ddp(backend=None):
    """Initialise the process group assuming pure data parallelism (what ``Forge`` supports)."""
    if dist.is_initialized():
        return None
    return init_distributed(ParallelConfig(dp_shard=-1), backend=backend)


def _spawn_worker(rank, world, fn, port):
    """Per-process entry point for ``launch``. Must stay module-level to be picklable."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    _init_ddp(backend="nccl" if torch.cuda.is_available() else "gloo")
    try:
        fn()
    finally:
        if dist.is_initialized():
            cleanup()


def launch(fn=None, nproc: int = 1, port: int = 29517):
    """Run ``fn`` on ``nproc`` processes of this host, like ``Lightning``'s launcher.

    ``@fm.launch(nproc=4)`` on a **module-level** function spawns the processes (the function
    must be picklable, i.e. defined at module level, and the script needs the usual
    ``if __name__ == "__main__":`` guard). Under an existing ``torchrun`` -- i.e. when
    ``WORLD_SIZE`` is already in the environment -- it simply runs ``fn`` once, so the same
    code works with ``torchrun --nproc_per_node 4 train.py``.
    """
    def decorator(f):
        def wrapper(*a, **kw):
            if nproc <= 1 or int(os.environ.get("WORLD_SIZE", "1")) > 1:
                _init_ddp(backend="nccl" if torch.cuda.is_available() else "gloo")
                try:
                    return f(*a, **kw)
                finally:
                    if dist.is_initialized():
                        cleanup()
            if torch.cuda.is_available() and torch.cuda.device_count() < nproc:
                raise RuntimeError(f"nproc={nproc} but only {torch.cuda.device_count()} "
                                   "visible devices")
            import torch.multiprocessing as mp
            return mp.spawn(_spawn_worker, args=(nproc, f, port), nprocs=nproc, join=True)
        wrapper.__name__ = getattr(f, "__name__", "launched")
        wrapper.__doc__ = f.__doc__
        return wrapper

    if fn is not None:
        return decorator(fn)()
    return decorator
