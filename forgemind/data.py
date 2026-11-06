"""Datasets, a resumable data-parallel sampler and a prefetching batch stream.

The original `_next_batch()` did `batch.to(device, non_blocking=True)` on the *current* stream right before
the forward pass. Even with pinned memory that copy cannot overlap with anything, so every step started with
a host->device stall on the critical path. `BatchStream` stages the next batch on a side stream while the
current step runs, and hands back a tensor pair that is already resident.

`ResumableSampler.__len__` used to subtract the resume offset (a *sample* count) from a *batch* count; it
now reports the per-rank sample count minus the resume offset, which is what `__iter__` actually yields.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

__all__ = ["SyntheticLM", "TokenFile", "ResumableSampler", "BatchStream", "build_loader"]


class SyntheticLM(Dataset):
    """deterministic learnable toy data: repeating motifs -> loss must go down fast (used by tests/examples)"""

    def __init__(self, vocab, seq, n=4096, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.motifs = torch.randint(0, vocab, (16, 8), generator=g)
        self.pick = torch.randint(0, 16, (n, seq // 8 + 2), generator=g)
        self.seq = seq

    def __len__(self):
        return len(self.pick)

    def __getitem__(self, i):
        t = self.motifs[self.pick[i]].flatten()[: self.seq + 1]   # fancy indexing already copies
        return {"input_ids": t[:-1], "labels": t[1:]}


class TokenFile(Dataset):
    """flat uint16/uint32 token file (np.memmap), fixed-length windows"""

    def __init__(self, path, seq, dtype="uint16"):
        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.seq = seq
        self.n = (len(self.data) - 1) // seq

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = torch.from_numpy(self.data[i * self.seq:(i + 1) * self.seq + 1].astype(np.int64))
        return {"input_ids": x[:-1], "labels": x[1:]}


class ResumableSampler(Sampler):
    """shards by *data-parallel* rank (so tp/pp ranks of one replica see identical data) and can resume mid-epoch"""

    def __init__(self, n, dp_rank, dp_world, seed=0, shuffle=True):
        self.n, self.r, self.w, self.seed, self.shuffle = n, dp_rank, dp_world, seed, shuffle
        self.epoch, self.skip = 0, 0
        self.per_rank = n // dp_world

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        idx = torch.randperm(self.n, generator=g).tolist() if self.shuffle else list(range(self.n))
        idx = idx[: self.n - self.n % self.w][self.r::self.w]
        if self.skip:
            idx = idx[self.skip:]
            self.skip = 0
        return iter(idx)

    def __len__(self):
        return max(0, self.per_rank - self.skip)


class BatchStream:
    """Iterator over ``(input_ids, labels)`` pairs already staged on the device.

    With ``prefetch`` the host->device copy of the next batch is issued on a dedicated stream as soon as the
    previous batch is consumed, so it overlaps with the whole forward/backward of the current step.
    """

    def __init__(self, loader, sampler, device, prefetch: bool = True):
        self.loader, self.sampler, self.device = loader, sampler, device
        self.prefetch = bool(prefetch)
        self.stream = torch.cuda.Stream(device=device) if (self.prefetch and device.type == "cuda") else None
        self._it = iter(loader)
        self._buf = None
        self._event = None
        self.epoch, self.batches = 0, 0
        try:
            self._stage()
        except StopIteration:
            raise RuntimeError(
                "BatchStream: the loader produced no batches (is the dataset smaller than "
                f"batch_size={loader.batch_size} with drop_last=True?)") from None

    # ---- internal ----------------------------------------------------------------------------------------------
    def _move(self, b):
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                ids = b["input_ids"].to(self.device, non_blocking=True)
                labels = b["labels"].to(self.device, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.stream)
            return ids, labels, ev
        return (b["input_ids"].to(self.device, non_blocking=True),
                b["labels"].to(self.device, non_blocking=True), None)

    def _stage(self):
        try:
            b = next(self._it)
        except StopIteration:
            self.epoch += 1
            self.batches = 0
            self.sampler.set_epoch(self.epoch)
            self._it = iter(self.loader)
            b = next(self._it)
        self._buf = self._move(b)

    # ---- public ------------------------------------------------------------------------------------------------
    def next(self):
        ids, labels, ev = self._buf
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)     # ordered, not a host sync
        self.batches += 1
        self._stage()                                      # issue the *next* copy while this step computes
        return ids, labels

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def resume(self, epoch: int, batches: int):
        """rewind the sampler to ``batches`` batches into ``epoch`` (used by Trainer.maybe_resume)"""
        self.epoch = epoch
        self.batches = batches
        self.sampler.set_epoch(epoch)
        self.sampler.skip = batches * self.loader.batch_size
        self._it = iter(self.loader)
        self._stage()


def build_loader(ds, micro_batch, ctx, seed=0, workers=0, shuffle=True):
    s = ResumableSampler(len(ds), ctx.dp_rank, ctx.dp_world, seed, shuffle)
    kw = {}
    if workers and workers > 0:
        kw.update(num_workers=workers, persistent_workers=False, prefetch_factor=2)
    dl = DataLoader(ds, batch_size=micro_batch, sampler=s, drop_last=True,
                    pin_memory=ctx.device.type == "cuda", **kw)
    return dl, s
