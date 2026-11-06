"""Datasets, a resumable data-parallel sampler, and an asynchronous batch pipeline.

The data path used to be the one place where the GPU waited for the CPU. Three separate
sources of stall, in the order they hurt:

1. ``batch.to(device)`` ran on the compute stream right before the forward pass, so the
   host->device copy could not overlap with anything;
2. the *host* side -- ``next(iter(loader))``: dataset ``__getitem__``, the collate function,
   and any per-batch Python -- ran on the main thread at the top of the step, so the GPU sat
   idle while the CPU built the next batch;
3. ``DataLoader`` with ``workers=0`` has no prefetch at all, and with ``workers>0`` it still
   only prefetches *inside* the loader (the collate of the fetched batch is on the main
   thread).

``BatchStream`` fixes all three: a background thread runs the loader (1 and 2 overlap with
compute) and every batch it produces is copied to the device on a dedicated CUDA stream with an
event attached (the copy overlaps too). The consumer waits on the event *on the device* --
``current_stream().wait_event()``, not ``event.synchronize()`` -- so the host is never blocked.

What makes it production-safe
-----------------------------
* **Order is preserved.** One FIFO queue, one producer, batches come out in loader order, so a
  data-parallel rank still sees exactly the samples it saw before.
* **Errors are forwarded, not swallowed.** If the dataset raises, the exception travels through
  the queue and is re-raised from ``next()`` -- with its original traceback -- and the thread
  stops instead of spinning.
* **Epoch and resume bookkeeping is per-batch.** Each queued item carries its own
  ``(epoch, index, epoch_len)``, so ``stream.epoch`` / ``stream.batches`` are correct even
  though the producer is running ahead of the consumer, and ``resume()`` is exact.
* **Nothing outlives the stream.** ``close()`` (also ``__del__`` and the context manager) stops
  the thread, drains the queue, and joins with a timeout; the thread is a daemon so an
  abandoned stream cannot hang interpreter exit.
* **Reproducibility caveat, stated instead of hidden.** With ``workers=0`` the collate runs on
  the producer thread, which shares the *global* torch RNG with the training thread. Your data
  augmentation and your dropout then draw from one interleaved stream: still correct, not
  bit-reproducible run to run. With ``workers>0`` the RNG lives in the worker processes and this
  does not apply. ``threaded=False`` restores the old strictly-serial behaviour.
* **No allocation churn.** Batches are produced as fresh tensors (no shared staging buffer to
  get wrong) and the device tensors are registered on the compute stream with
  ``record_stream``, which is what the caching allocator needs when memory is produced on one
  stream and consumed on another.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .perf import warn_once

__all__ = ["SyntheticLM", "TokenFile", "ResumableSampler", "BatchStream", "build_loader",
           "async_loader", "to_device"]


# ----------------------------------------------------------------------------------------------- datasets
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
        self.path, self.dtype = path, np.dtype(dtype)
        self.data = np.memmap(path, dtype=self.dtype, mode="r")
        self.seq = seq
        self.n = max(0, (len(self.data) - 1) // seq)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(self.data[i * self.seq:(i + 1) * self.seq + 1],
                                        dtype=np.int64))
        return {"input_ids": x[:-1], "labels": x[1:]}

    def close(self):
        """Release the mmap (it keeps a file descriptor open on every platform)."""
        data = getattr(self, "data", None)
        if data is not None:
            del self.data
            self.data = np.empty(0, dtype=self.dtype)
        return data

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------------------------- sampler
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


# ----------------------------------------------------------------------------------------------- helpers
def to_device(obj, device, non_blocking: bool = False):
    """Recursively move the tensors in a nested batch to ``device``."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking) if obj.device != device else obj
    if isinstance(obj, dict):
        return {k: to_device(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [to_device(v, device, non_blocking) for v in obj]
        if isinstance(obj, tuple) and hasattr(obj, "_fields"):
            return type(obj)(*out)
        return tuple(out) if isinstance(obj, tuple) else out
    return obj


@dataclass
class _Item:
    """One prefetched batch plus everything needed to interpret it on the consumer side."""
    data: object
    epoch: int
    index: int
    epoch_len: "int | None"
    event: "torch.cuda.Event | None" = None
    published: float = 0.0
    tensors: list = field(default_factory=list)


@dataclass
class _Error:
    exc: BaseException


_END = object()


def _shutdown_iterator(it) -> None:
    """Release a DataLoader iterator *and* its worker pool.

    ``persistent_workers=True`` (what :func:`build_loader` sets so epochs never re-fork on the
    prefetch thread) keeps worker processes alive until the iterator is finalized. Without this,
    every ``close()`` -- and every ``Trainer.evaluate()`` -- would leave a pool behind.
    """
    shutdown = getattr(it, "_shutdown_workers", None)
    if callable(shutdown):
        with contextlib.suppress(Exception):
            shutdown()


def _bind_device(device) -> None:
    """Make the calling (background) thread use ``device`` for its allocations and streams."""
    dev = torch.device(device)
    if dev.type == "cuda":
        with contextlib.suppress(Exception):
            torch.cuda.set_device(dev)


def _loader_len(loader) -> "int | None":
    """Batches per epoch, ``0`` for a knowable-but-empty loader, ``None`` when unknown."""
    try:
        return int(len(loader))
    except (TypeError, AttributeError):
        return None


# ----------------------------------------------------------------------------------------------- BatchStream
class BatchStream:
    """Infinite, prefetched stream of ``(input_ids, labels)`` already resident on the device.

    ``prefetch``     overlap the host->device copy on a side stream (CUDA) or the copy itself.
    ``threaded``     run the loader on a background thread (this is the "release batches from
                     the CPU while the GPU computes" part). ``None`` = auto: on when nothing else
                     prefetches (``workers=0``) or when the device is CUDA, where the copy can
                     genuinely run asynchronously. On a CPU-only box with worker processes the
                     win is zero because both threads need the GIL, and this stays available as an
                     explicit opt-in instead.
    ``depth``        how many batches may be in flight (2 is enough to overlap, 3-4 smooths out
                     collate jitter at the cost of pinned/device memory).
    """

    def __init__(self, loader, sampler, device, prefetch: bool = True, depth: int = 2,
                 threaded: "bool | None" = None):
        self.loader, self.sampler, self.device = loader, sampler, device
        self.prefetch = bool(prefetch)
        self.depth = max(1, int(depth))
        self.stream = (torch.cuda.Stream(device=device)
                       if self.prefetch and torch.device(device).type == "cuda" else None)
        if threaded is None:
            workers = int(getattr(loader, "num_workers", 0) or 0)
            # Nothing else prefetches when workers=0 (the load+collate is on the critical path);
            # on CUDA the extra thread is what lets the host->device copy of batch k+1 run while
            # step k computes. Measured on CPU: 0.97-1.02x either way -- the step holds the GIL.
            cuda = torch.device(device).type == "cuda"
            threaded = self.prefetch and (cuda or workers == 0)
            if threaded:
                warn_once("[forgemind] batch prefetch thread on: dataset/collate RNG now shares the "
                          "global torch RNG with training (pass BatchStream(..., threaded=False) "
                          "for bit-exact reproducibility)", key="bs_rng")
        self.threaded = bool(threaded) and self.prefetch
        self.epoch, self.batches = 0, 0
        self._start_epoch, self._start_index = 0, 0
        self._epoch_len = _loader_len(loader)
        if self._epoch_len == 0:
            raise self._empty_error()   # knowable up front: fail at construction, not on step 1
        self.stats = {"batches": 0, "wait_ms": 0.0, "stalled": 0, "max_wait_ms": 0.0,
                      "produce_ms": 0.0, "threaded": self.threaded, "depth": self.depth}
        self._q: "queue.Queue" = queue.Queue(maxsize=self.depth)
        self._stop = threading.Event()
        self._thread: "threading.Thread | None" = None
        self._it = None
        self._last_published: "_Item | None" = None
        self._closed = False
        if self.threaded:
            self._restart()
        else:
            self._it = iter(loader)
            try:
                self._sync_stage()
            except StopIteration:
                raise self._empty_error() from None

    # ------------------------------------------------------------------------------- producer
    def _new_epoch(self, epoch: int) -> None:
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        self._it = iter(self.loader)

    def _stage(self, b, epoch: int, index: int) -> _Item:
        t0 = time.perf_counter()
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                data = (to_device(b["input_ids"], self.device, non_blocking=True),
                        to_device(b["labels"], self.device, non_blocking=True))
                ev = torch.cuda.Event()
                ev.record(self.stream)
        else:
            data = (to_device(b["input_ids"], self.device, non_blocking=True),
                    to_device(b["labels"], self.device, non_blocking=True))
            ev = None
        self.stats["produce_ms"] += (time.perf_counter() - t0) * 1e3
        return _Item(data=data, epoch=epoch, index=index, epoch_len=self._epoch_len, event=ev,
                     published=time.perf_counter())

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        # `torch.cuda.set_device` is thread-local, so a thread that inherits nothing would
        # allocate on device 0 -- the classic multi-GPU prefetcher bug.
        _bind_device(self.device)
        epoch, index = self._start_epoch, self._start_index
        try:
            while not self._stop.is_set():
                try:
                    b = next(self._it)
                except StopIteration:
                    produced = index
                    self._epoch_len = produced or None       # the real length of that epoch
                    if self._last_published is not None and self._last_published.epoch == epoch:
                        self._last_published.epoch_len = produced
                    epoch, index = epoch + 1, 0
                    self._new_epoch(epoch)
                    try:
                        b = next(self._it)
                    except StopIteration:
                        self._put(_END)                      # the loader is empty
                        return
                if self._stop.is_set():
                    return
                item = self._stage(b, epoch, index)
                if not self._put(item):
                    return
                self._last_published = item
                index += 1
                if self._epoch_len is not None and index >= self._epoch_len:
                    epoch, index = epoch + 1, 0
                    self._new_epoch(epoch)
            return
        except BaseException as exc:                          # forward the failure to the consumer
            self._put(_Error(exc))

    def _start_thread(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="forgemind-batches", daemon=True)
        self._thread.start()

    def _restart(self) -> None:
        """(Re)start the producer at ``self._start_epoch/_start_index`` without losing the queue API."""
        self._stop_worker()
        self._drain()
        self._drop_iterator()
        self._epoch_len = _loader_len(self.loader) or self._epoch_len or None
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(self._start_epoch)
        self._it = iter(self.loader)
        if self.threaded:
            self._start_thread()
        else:
            self._sync_stage()

    # ------------------------------------------------------------------------------- consumer
    def _wait(self, item: _Item):
        if item.event is not None:
            cur = torch.cuda.current_stream()
            cur.wait_event(item.event)          # device-side ordering: the host never blocks
            for t in _tensors(item.data):
                t.record_stream(cur)
        return item.data

    def next(self):
        if self.threaded:
            t0 = time.perf_counter()
            item = self._q.get()
            wait = (time.perf_counter() - t0) * 1e3
            if isinstance(item, _Error):
                self._stop_worker()
                raise item.exc
            if item is _END:
                self._stop_worker()
                if self.stats["batches"] == 0:
                    raise self._empty_error() from None
                raise StopIteration
            self.stats["wait_ms"] += wait
            self.stats["max_wait_ms"] = max(self.stats["max_wait_ms"], wait)
            if wait > 1.0:
                self.stats["stalled"] += 1
            self.epoch = item.epoch
            self.batches = item.index + 1
            if item.epoch_len is not None and item.index + 1 >= item.epoch_len:
                self.epoch, self.batches = item.epoch + 1, 0
            self.stats["batches"] += 1
            return self._wait(item)
        return self._sync_next()

    def _sync_next(self):
        ids, labels, ev = self._buf
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
        self.batches += 1
        self._sync_stage()
        return ids, labels

    def _sync_stage(self) -> None:
        try:
            b = next(self._it)
        except StopIteration:
            self.epoch += 1
            self.batches = 0
            self._new_epoch(self.epoch)
            b = next(self._it)                    # a genuinely empty loader raises here
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                ids = b["input_ids"].to(self.device, non_blocking=True)
                labels = b["labels"].to(self.device, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.stream)
            self._buf = (ids, labels, ev)
        else:
            self._buf = (b["input_ids"].to(self.device, non_blocking=True),
                         b["labels"].to(self.device, non_blocking=True), None)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    # ------------------------------------------------------------------------------- control
    def resume(self, epoch: int, batches: int) -> None:
        """Rewind the sampler to ``batches`` batches into ``epoch`` (used by ``Trainer.maybe_resume``)."""
        self.epoch, self.batches = epoch, batches
        self._start_epoch, self._start_index = epoch, batches
        self.sampler.skip = batches * int(self.loader.batch_size or 0)
        self._restart()

    def _empty_error(self) -> RuntimeError:
        return RuntimeError(
            f"BatchStream: the loader produced no batches (is the dataset smaller than "
            f"batch_size={getattr(self.loader, 'batch_size', '?')} with drop_last=True?)")

    def _drain(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def _drop_iterator(self) -> None:
        """Release the current iterator, including the worker pool of a persistent DataLoader.

        ``persistent_workers=True`` (set by :func:`build_loader` so epochs never re-fork on the
        prefetch thread) keeps its processes alive until the iterator is finalized; without this
        ``close()`` would leave worker processes behind (their queue-feeder threads exit at the
        next garbage collection, as with any ``multiprocessing.Queue``).
        """
        it, self._it = getattr(self, "_it", None), None
        if it is not None:
            _shutdown_iterator(it)

    def _stop_worker(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():  # pragma: no cover - only if a dataset call hangs
                warn_once("[forgemind] batch prefetch thread did not stop in 10s (a dataset or "
                          "collate call is blocking); it is a daemon so the process can still exit",
                          key="bs_stop")
            self._thread = None

    def close(self) -> None:
        if self._closed:
            return
        self._stop_worker()
        self._drain()
        self._drop_iterator()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    def report(self) -> dict:
        st = dict(self.stats)
        n = max(1, st["batches"])
        st["wait_ms_per_batch"] = round(st["wait_ms"] / n, 3)
        st["stall_ratio"] = round(st["stalled"] / n, 3)
        st["queued"] = self._q.qsize()
        return st


def _tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _tensors(v)


# ----------------------------------------------------------------------------------------------- generic
class _AsyncLoader:
    """Generic threaded prefetcher: yields whatever the loader yields, moved to ``device``."""

    def __init__(self, loader, device, depth: int = 2, pin: "bool | None" = None):
        self.loader, self.device = loader, device
        self.device = torch.device(device)
        self.depth = max(1, int(depth))
        self.stream = (torch.cuda.Stream(device=self.device)
                       if self.device.type == "cuda" else None)
        if pin is None:
            pin = self.device.type == "cuda"
        self.loader_pin = pin
        self.stats = {"batches": 0, "wait_ms": 0.0, "stalled": 0, "max_wait_ms": 0.0}
        self._q: "queue.Queue" = queue.Queue(maxsize=self.depth)
        self._stop = threading.Event()
        self._it = iter(loader)
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="forgemind-async-loader", daemon=True)
        self._thread.start()

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        _bind_device(self.device)
        try:
            while not self._stop.is_set():
                try:
                    batch = next(self._it)
                except StopIteration:
                    self._put(_END)
                    return
                if self.stream is not None:
                    with torch.cuda.stream(self.stream):
                        data = to_device(batch, self.device, non_blocking=True)
                        ev = torch.cuda.Event()
                        ev.record(self.stream)
                else:
                    data = to_device(batch, self.device)
                    ev = None
                if not self._put(_Item(data=data, epoch=0, index=0, epoch_len=None, event=ev)):
                    return
        except BaseException as exc:
            self._put(_Error(exc))

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        t0 = time.perf_counter()
        item = self._q.get()
        wait = (time.perf_counter() - t0) * 1e3
        if isinstance(item, _Error):
            self.close()
            raise item.exc
        if item is _END:
            self.close()
            raise StopIteration
        self.stats["batches"] += 1
        self.stats["wait_ms"] += wait
        self.stats["max_wait_ms"] = max(self.stats["max_wait_ms"], wait)
        if wait > 1.0:
            self.stats["stalled"] += 1
        if item.event is not None:
            cur = torch.cuda.current_stream()
            cur.wait_event(item.event)
            for t in _tensors(item.data):
                t.record_stream(cur)
        return item.data

    def close(self) -> None:
        if self._closed:
            return
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=10.0)
            self._thread = None
        self._closed = True
        while True:                     # anything still buffered is dead weight, never handed out
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        it, self._it = self._it, None
        if it is not None:
            _shutdown_iterator(it)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass


def async_loader(loader, device, depth: int = 2):
    """``for batch in fm.async_loader(loader, device): ...`` -- batches arrive on ``device``
    while the previous step is still computing.

    Works with any iterable of tensors/containers, so it is usable outside ``Forge``::

        for x, y in fm.async_loader(DataLoader(ds, batch_size=64, pin_memory=True), "cuda"):
            loss = criterion(model(x), y)
            ...
    """
    return _AsyncLoader(loader, device, depth=depth)


# ----------------------------------------------------------------------------------------------- loader glue
def build_loader(ds, micro_batch, ctx, seed=0, workers=0, shuffle=True,
                 persistent_workers: "bool | None" = None):
    s = ResumableSampler(len(ds), ctx.dp_rank, ctx.dp_world, seed, shuffle)
    kw = {}
    if workers and workers > 0:
        if persistent_workers is None:
            persistent_workers = True      # re-creating the worker pool every epoch is pure loss
        kw.update(num_workers=int(workers), prefetch_factor=2,
                  persistent_workers=bool(persistent_workers))
    dl = DataLoader(ds, batch_size=micro_batch, sampler=s, drop_last=True,
                    pin_memory=ctx.device.type == "cuda", **kw)
    return dl, s
