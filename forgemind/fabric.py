"""Lightning-**Fabric**-style driver: a thin, loop-your-own wrapper around this repo's machinery.

``Trainer`` owns the whole training loop; ``Forge`` is the two-line training facade; ``Fabric``
is the third shape -- the one Lightning users already know::

    fabric = fm.Fabric(accelerator="auto", devices=2, strategy="ddp", precision="bf16-mixed")
    fabric.launch()                              # spawns the processes (module-level fn)
    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)

    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            with fabric.autocast():
                loss = model(batch)
            fabric.backward(loss)
            fabric.clip_gradients(model, optimizer, max_norm=1.0)
            optimizer.step()
        fabric.save("ckpt.pt", {"model": model, "optimizer": optimizer, "epoch": epoch})

Every method name, argument order and return shape follows ``lightning.fabric.Fabric`` so porting
is mechanical; the pieces it delegates to (device movement, DDP wrapping, the process launcher,
the one-host-read-per-step budget used by ``Forge``) are this library's, not Lightning's.

Differences from Lightning, stated rather than hidden
-----------------------------------------------------
* ``clip_gradients`` returns the pre-clip total norm (Lightning returns ``None``).
* ``setup_dataloaders`` bumps the ``DistributedSampler`` epoch automatically each time a fresh
  iterator starts, and ``fabric.set_epoch(loader, epoch)`` sets it explicitly (resume).
* ``save``/``load`` are plain ``torch.save`` on rank 0 (no sharded checkpoints: use ``Trainer``
  for FSDP2/TP/PP state).
* ``strategy`` accepts ``auto`` | ``single_device`` | ``ddp``; FSDP2/DeepSpeed/TP/PP live in
  ``Trainer``/``ForgeConfig`` and asking for them here raises, never silently degrades.
* Accelerators: ``cpu`` and ``cuda`` only.
"""

from __future__ import annotations

import functools
import os
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from .data import to_device
from .forge import launch as _forge_launch

#: Lightning precision names -> (param dtype or None, autocast dtype or None).
#: ``*-true`` casts the model itself; ``*-mixed`` keeps fp32 weights and autocasts the compute.
PRECISIONS: dict[str, tuple[str | None, torch.dtype | None]] = {
    "32-true": (None, None),
    "16-true": ("float16", None),
    "bf16-true": ("bfloat16", None),
    "16-mixed": ("float32", torch.float16),
    "bf16-mixed": ("float32", torch.bfloat16),
}

_STRATEGIES = ("auto", "single_device", "ddp")
_ACCELERATORS = ("auto", "cpu", "cuda", "gpu")


class _ScalerOptimizer:
    """``optimizer.step()`` that goes through the fp16 ``GradScaler`` (16-mixed on CUDA)."""

    def __init__(self, optimizer, scaler):
        self.optimizer = optimizer
        self.scaler = scaler

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        return self.optimizer.zero_grad(set_to_none=set_to_none)

    def __getattr__(self, name):  # param_groups, state, state_dict, ...
        return getattr(self.optimizer, name)

    @property
    def param_groups(self):
        return self.optimizer.param_groups


def _unwrap_optimizer(optimizer):
    return getattr(optimizer, "optimizer", optimizer)


class _FabricLoader:
    """A dataloader that lands batches on ``fabric.device`` and owns its sampler's epoch."""

    def __init__(self, loader, fabric, sampler=None):
        self.loader = loader
        self.fabric = fabric
        self.sampler = sampler if sampler is not None else getattr(loader, "sampler", None)
        self.epoch = 0
        self._exhausted = False

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._exhausted = False      # an explicit set wins over the auto-bump

    def __iter__(self):
        if self._exhausted:                 # a completed pass == one epoch
            self.epoch += 1
            self._exhausted = False
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(self.epoch)
        for batch in self.loader:
            yield self.fabric.to_device(batch)
        self._exhausted = True

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def batch_size(self):
        return self.loader.batch_size

    def __repr__(self):
        return f"FabricLoader({self.loader!r}, epoch={self.epoch})"


class Fabric:
    """Lightning-Fabric-compatible driver (see module docstring for the loop shape).

    Parameters
    ----------
    accelerator : ``"auto" | "cpu" | "cuda"``
    devices : ``"auto" | int | list[int]``
        How many processes ``launch`` spawns / how many CUDA devices are visible.
    strategy : ``"auto" | "single_device" | "ddp"``
        ``auto`` is ``ddp`` when more than one process is running, else ``single_device``.
    precision : one of :data:`PRECISIONS` (Lightning's names, ``"32-true"`` default).
    plugins : must be ``None`` -- this library has no plugin registry (raises if given).
    port : master port used by ``launch`` when it spawns processes.
    """

    def __init__(self, accelerator: str = "auto", devices: "int | str | list[int]" = 1,
                 strategy: str = "auto", precision: str = "32-true", *,
                 plugins=None, port: int = 29517):
        if plugins is not None:
            raise ValueError("Fabric does not support plugins=; use strategy/precision or "
                             "ForgeConfig for the advanced setups (TE, DeepSpeed, quantizers)")
        if strategy not in _STRATEGIES:
            raise ValueError(f"strategy must be one of {_STRATEGIES}, got {strategy!r}; FSDP2, "
                             "DeepSpeed, TP and PP are configured through Trainer/ForgeConfig, "
                             "not silently degraded here")
        if accelerator not in _ACCELERATORS:
            raise ValueError(f"accelerator must be one of {_ACCELERATORS}, got {accelerator!r}")
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}")

        if accelerator in ("cuda", "gpu"):
            if not torch.cuda.is_available():
                raise RuntimeError("accelerator='cuda' but no CUDA device is visible")
            self.accelerator = "cuda"
        else:
            self.accelerator = "cuda" if (accelerator == "auto" and torch.cuda.is_available()) \
                else ("cpu" if accelerator in ("auto", "cpu") else accelerator)

        if isinstance(devices, str):
            if devices != "auto":
                raise ValueError(f"devices must be an int, a list, or 'auto'; got {devices!r}")
            n = torch.cuda.device_count() if self.accelerator == "cuda" else 1
            self._device_ids = list(range(n)) or [0]
        elif isinstance(devices, int):
            if devices < 1:
                raise ValueError("devices must be >= 1")
            self._device_ids = list(range(devices))
        else:
            self._device_ids = [int(d) for d in devices]
            if not self._device_ids:
                raise ValueError("devices must not be empty")

        self.strategy = strategy
        self.precision = precision
        self._param_dtype, self._autocast_dtype = PRECISIONS[precision]
        self.port = int(port)
        if self.accelerator == "cuda":
            # under launch() every worker picks its own LOCAL_RANK entry
            local = int(os.environ.get("LOCAL_RANK", 0))
            idx = self._device_ids[local] if local < len(self._device_ids) else self._device_ids[0]
            self.device = torch.device("cuda", idx)
        else:
            self.device = torch.device("cpu")
        self._scaler = (torch.amp.GradScaler("cuda", enabled=True)
                        if precision == "16-mixed" and self.accelerator == "cuda" else None)

    # ------------------------------------------------------------------ distributed facts
    @property
    def world_size(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    @property
    def global_rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def local_rank(self) -> int:
        if dist.is_initialized():
            return int(os.environ.get("LOCAL_RANK", self.global_rank))
        return 0

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    def _resolved_strategy(self) -> str:
        if self.strategy != "auto":
            return self.strategy
        return "ddp" if self.world_size > 1 else "single_device"

    # ------------------------------------------------------------------ launch
    def launch(self, fn=None, *args, **kwargs):
        """Run ``fn(fabric, ...)`` -- spawning ``devices`` processes when needed.

        ``fn`` must be picklable (module-level) when it is spawned; the same code path stays a
        single in-process call under an existing ``torchrun``. With no ``fn`` this only marks
        "the user launched the processes themselves" and returns ``self``.
        """
        if fn is None:
            if self._wants_spawn():
                raise ValueError("launch(fn) needs the function to run in the spawned "
                                 "processes (define it at module level, or start the "
                                 "processes with torchrun and call launch() with no fn)")
            return self
        if args or kwargs:
            fn = functools.partial(fn, *args, **kwargs)
        nproc = len(self._device_ids) if self._wants_spawn() else 1
        return _forge_launch(fn, nproc=nproc, port=self.port)

    def _wants_spawn(self) -> bool:
        # `self.strategy`, not `_resolved_strategy()`: before launch world_size is 1, which
        # would resolve `auto` to single_device and never spawn anything.
        return (len(self._device_ids) > 1 and not dist.is_initialized()
                and int(os.environ.get("WORLD_SIZE", "1")) <= 1
                and self.strategy in ("auto", "ddp"))

    # ------------------------------------------------------------------ setup
    def setup(self, module: nn.Module, *optimizers):
        """Move to device, apply precision, DDP-wrap when distributed; return what was passed.

        Build optimizers from ``module.parameters()`` *before* calling this: ``.to()`` keeps
        ``Parameter`` identity, so the optimizer keeps pointing at the live tensors.
        """
        if self._resolved_strategy() == "single_device" and self.world_size > 1:
            raise RuntimeError(f"strategy='single_device' but world_size={self.world_size}; "
                               "use strategy='ddp' (or 'auto')")
        module = module.to(self.device)
        if self._param_dtype is not None:
            module.to(getattr(torch, self._param_dtype))
        if self.world_size > 1 and self._resolved_strategy() == "ddp":
            device_ids = [self.device] if self.device.type == "cuda" else None
            if not isinstance(module, nn.parallel.DistributedDataParallel):
                module = nn.parallel.DistributedDataParallel(module, device_ids=device_ids)
        wrapped = [_ScalerOptimizer(o, self._scaler) if self._scaler is not None else o
                   for o in optimizers]
        if not optimizers:
            return module
        return (module, *wrapped)

    def setup_dataloaders(self, *loaders, use_distributed_sampler: bool = True):
        """Wrap loaders so batches arrive on ``fabric.device`` (and shard them when distributed)."""
        out = [self._setup_loader(l, use_distributed_sampler) for l in loaders]
        return out[0] if len(out) == 1 else tuple(out)

    def _setup_loader(self, loader, use_dist_sampler):
        sampler = None
        if (use_dist_sampler and self.world_size > 1
                and isinstance(loader, DataLoader)
                and not isinstance(getattr(loader, "sampler", None), DistributedSampler)
                and isinstance(getattr(loader, "sampler", None), (RandomSampler,
                                                                  SequentialSampler))):
            shuffle = isinstance(loader.sampler, RandomSampler)
            sampler = DistributedSampler(loader.dataset, num_replicas=self.world_size,
                                         rank=self.global_rank, shuffle=shuffle)
            try:
                loader = _rebuild_loader(loader, sampler)
            except (TypeError, ValueError) as exc:  # exotic subclass we cannot rebuild
                warnings.warn(f"[forgemind] cannot inject a DistributedSampler into "
                              f"{type(loader).__name__} ({exc}); iterate it sharded yourself or "
                              "pass use_distributed_sampler=False", stacklevel=2)
                sampler = None
        return _FabricLoader(loader, self, sampler)

    # ------------------------------------------------------------------ loop helpers
    def autocast(self):
        """The forward's precision context (``nullcontext`` for fp32 / ``*-true``)."""
        if self._autocast_dtype is None:
            import contextlib
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self._autocast_dtype)

    def to_device(self, data):
        """Recursively move tensors (and dict/list/tuple structures of them) to ``fabric.device``."""
        return to_device(data, self.device)

    def backward(self, loss: torch.Tensor):
        """Back-propagate ``loss`` (scaled by the GradScaler under 16-mixed on CUDA)."""
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_gradients(self, module: nn.Module, optimizer, max_norm: float,
                       error_if_nonfinite: bool = False):
        """Clip in place; returns the **pre-clip** total norm (Lightning returns ``None``)."""
        opt = _unwrap_optimizer(optimizer)
        if self._scaler is not None:
            self._scaler.unscale_(opt)
        return torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm,
                                              error_if_nonfinite=error_if_nonfinite)

    def set_epoch(self, loader, epoch: int) -> None:
        """Set the sampler epoch of a loader returned by :meth:`setup_dataloaders`."""
        if isinstance(loader, _FabricLoader):
            loader.set_epoch(epoch)

    # ------------------------------------------------------------------ collectives
    def all_reduce(self, tensor: torch.Tensor, reduce_op: str = "mean") -> torch.Tensor:
        if self.world_size == 1 or not dist.is_initialized():
            return tensor
        op = {"sum": dist.ReduceOp.SUM, "mean": dist.ReduceOp.AVG, "max": dist.ReduceOp.MAX,
              "min": dist.ReduceOp.MIN}[reduce_op]
        dist.all_reduce(tensor, op=op)
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.world_size == 1 or not dist.is_initialized():
            return tensor
        dist.broadcast(tensor, src=src)
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather along dim 0: the result has ``world_size`` copies concatenated (world 1: as-is)."""
        if self.world_size == 1 or not dist.is_initialized():
            return tensor
        out = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(out, tensor)
        return torch.cat(out, dim=0)

    def barrier(self) -> None:
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

    # ------------------------------------------------------------------ state
    def seed_everything(self, seed: int = 1234, workers: bool = False) -> int:
        """Seed torch, python and numpy RNGs; returns the seed (worker procs inherit via fork)."""
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if workers:
            os.environ.setdefault("PYTHONHASHSEED", str(seed))
        return seed

    def save(self, path: str, state: dict) -> None:
        """Rank 0 writes ``state`` (modules/optimizers as state dicts, DDP unwrapped)."""
        if self.is_global_zero:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            payload = {k: _state_dict(v) for k, v in state.items()}
            torch.save(payload, path)
        self.barrier()

    def load(self, path: str, state: dict) -> dict:
        """Fill ``state`` from a checkpoint written by :meth:`save`; returns ``state``.

        Accepts raw *or* DDP-wrapped modules on either side: the checkpoint itself never has
        the ``module.`` prefix, so a saved wrapped model restores into a fresh raw one.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for key, value in state.items():
            if key not in payload:
                continue
            if isinstance(value, nn.Module):
                _unwrap_module(value).load_state_dict(payload[key])
            elif isinstance(value, (torch.optim.Optimizer, _ScalerOptimizer)):
                _unwrap_optimizer(value).load_state_dict(payload[key])
            else:
                state[key] = payload[key]
        self.barrier()
        return state

    def __repr__(self):
        return (f"Fabric(accelerator={self.accelerator!r}, devices={len(self._device_ids)}, "
                f"strategy={self.strategy!r}, precision={self.precision!r}, "
                f"world_size={self.world_size})")


def _unwrap_module(module):
    """Strip ``DistributedDataParallel`` so checkpoints never carry a ``module.`` prefix."""
    return module.module if isinstance(module, nn.parallel.DistributedDataParallel) else module


def _state_dict(value):
    if isinstance(value, nn.Module):
        return _unwrap_module(value).state_dict()
    if isinstance(value, (torch.optim.Optimizer, _ScalerOptimizer)):
        return _unwrap_optimizer(value).state_dict()
    return value


def _rebuild_loader(loader: DataLoader, sampler) -> DataLoader:
    """Rebuild a stock ``DataLoader`` around ``sampler`` (``sampler=`` is read-only after init)."""
    kwargs = dict(
        dataset=loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator,
        pin_memory_device=loader.pin_memory_device,
    )
    if loader.num_workers > 0:
        kwargs["prefetch_factor"] = loader.prefetch_factor
        kwargs["persistent_workers"] = loader.persistent_workers
    in_order = getattr(loader, "in_order", None)
    if in_order is not None:
        kwargs["in_order"] = in_order
    return DataLoader(**kwargs)


__all__ = ["Fabric", "PRECISIONS"]
