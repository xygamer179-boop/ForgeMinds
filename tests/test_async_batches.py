"""Async batch pipeline: the loader runs on a background thread while the step computes.

The tests below are the ones that would catch a *silently broken* prefetcher: wrong order, a
lost epoch boundary, a swallowed dataset error, a thread that outlives the trainer. The overlap
test measures wall time against the serial baseline, because "it has a thread" is not the same
as "it overlaps".
"""

import threading
import time

import pytest
import torch
import torch.nn as nn

import forgemind as fm
from forgemind.data import BatchStream, ResumableSampler, async_loader


class OrderedLoader:
    """Mimics ``DataLoader + ResumableSampler``: honours ``sampler.skip`` and the epoch counter."""

    def __init__(self, n_batches, sampler, batch_size=4, delay=0.0, fail_at=None):
        self.n, self.sampler, self.batch_size = n_batches, sampler, batch_size
        self.delay, self.fail_at = delay, fail_at
        self.num_workers = 0
        self.calls = 0
        self.epochs = []

    def __len__(self):
        return self.n

    def __iter__(self):
        self.epochs.append(self.sampler.epoch)
        start = self.sampler.skip // self.batch_size
        self.sampler.skip = 0
        for i in range(start, self.n):
            if self.delay:
                time.sleep(self.delay)
            if self.fail_at is not None and i == self.fail_at:
                raise ValueError(f"dataset exploded at batch {i}")
            self.calls += 1
            yield {"input_ids": torch.tensor([i] * self.batch_size),
                   "labels": torch.tensor([i + 1] * self.batch_size)}


def make(n=6, depth=2, threaded=True, delay=0.0, fail_at=None):
    sampler = ResumableSampler(n * 4, 0, 1, seed=0, shuffle=False)
    loader = OrderedLoader(n, sampler, delay=delay, fail_at=fail_at)
    stream = BatchStream(loader, sampler, torch.device("cpu"), prefetch=True, depth=depth,
                         threaded=threaded)
    return stream, loader, sampler


def drain(stream, k):
    out = []
    for _ in range(k):
        ids, _ = stream.next()
        out.append((int(ids[0]), stream.epoch, stream.batches))
    return out


def test_same_batches_same_order_as_serial():
    serial, _, _ = make(threaded=False)
    threaded, _, _ = make(threaded=True)
    try:
        a, b = drain(serial, 14), drain(threaded, 14)
    finally:
        serial.close(), threaded.close()
    assert a == b                       # same data, same epoch/batch bookkeeping


def test_epoch_boundaries_are_exact():
    stream, _, _ = make(n=4, threaded=True)
    pos = drain(stream, 9)
    stream.close()
    # the position after the last batch of an epoch is (epoch+1, 0): that is exactly what a
    # checkpoint needs, and it must match the serial path bit for bit
    assert [(e, b) for _, e, b in pos] == [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2),
                                           (1, 3), (2, 0), (2, 1)]
    # the data really did wrap
    assert [v for v, _, _ in pos] == [0, 1, 2, 3, 0, 1, 2, 3, 0]


def test_prefetch_overlaps_cpu_work_with_compute():
    """The point of the feature: total wall time < producer time + consumer time."""
    per_batch = 0.02
    n = 6

    def run(threaded):
        stream, _, _ = make(n=n, threaded=threaded, delay=per_batch)
        t0 = time.perf_counter()
        for _ in range(n):
            stream.next()
            time.sleep(per_batch)          # stand-in for the forward/backward
        stream.close()
        return time.perf_counter() - t0

    serial = run(False)
    threaded = run(True)
    assert serial > 2 * per_batch * n * 0.8, serial        # ~2x per batch, as expected
    assert threaded < serial * 0.75, (serial, threaded)     # a real overlap, not a rename
    # and the producer is never the bottleneck: the consumer's own sleep dominates
    assert threaded < per_batch * n * 1.6, threaded


def test_producer_runs_ahead_without_being_asked():
    stream, loader, _ = make(n=8, depth=3, threaded=True, delay=0.005)
    stream.next()                                   # consumer takes one batch
    deadline = time.time() + 2.0
    while loader.calls < 4 and time.time() < deadline:
        time.sleep(0.005)
    assert loader.calls >= 4, f"producer only ran {loader.calls} batches ahead"
    stream.close()


def test_dataset_error_is_forwarded_with_traceback():
    stream, _, _ = make(n=8, threaded=True, fail_at=2)
    with pytest.raises(ValueError, match="dataset exploded at batch 2"):
        for _ in range(8):
            stream.next()
    assert not any(t.name == "forgemind-batches" for t in threading.enumerate())


def test_close_is_idempotent_and_joins():
    stream, _, _ = make(n=8, threaded=True)
    stream.next()
    stream.close()
    stream.close()
    assert stream._thread is None
    assert not any(t.name == "forgemind-batches" for t in threading.enumerate())


def test_context_manager():
    with make(n=4, threaded=True)[0] as stream:
        stream.next()
    assert stream._thread is None


def test_consumer_stop_leaves_no_thread():
    before = {t.name for t in threading.enumerate()}
    for _ in range(5):
        stream, _, _ = make(n=4, threaded=True)
        stream.next()
        stream.close()
        del stream
    after = {t.name for t in threading.enumerate()}
    assert not (after - before) & {"forgemind-batches"}


def test_resume_continues_at_the_right_batch():
    stream, _, sampler = make(n=6, threaded=True)
    assert [v for v, _, _ in drain(stream, 2)] == [0, 1]
    stream.resume(0, 3)
    assert stream.epoch == 0 and stream.batches == 3
    got = [v for v, _, _ in drain(stream, 2)]
    stream.close()
    assert got == [3, 4]                                     # resumed, not restarted


def test_report_shape():
    stream, _, _ = make(n=6, threaded=True, delay=0.001)
    drain(stream, 6)
    rep = stream.report()
    for k in ("batches", "wait_ms", "wait_ms_per_batch", "stalled", "stall_ratio", "threaded",
              "depth", "queued"):
        assert k in rep
    assert rep["batches"] == 6 and rep["threaded"] is True
    stream.close()


def test_empty_loader_raises_a_clear_error_on_both_paths():
    """Serial: at construction. Threaded: on the first next(). Both explain themselves."""
    for threaded in (False, True):
        sampler = ResumableSampler(0, 0, 1)
        with pytest.raises(RuntimeError, match="no batches"):
            stream = BatchStream(OrderedLoader(0, sampler), sampler, torch.device("cpu"),
                                 threaded=threaded)
            try:
                stream.next()
            finally:
                stream.close()


class OneShot:
    """A single-use iterator (a generator, a stream): it is empty for ever after it runs out."""

    num_workers = 0
    batch_size = 1

    def __init__(self, n):
        self.batches = [{"input_ids": torch.tensor([i]), "labels": torch.tensor([i + 1])}
                        for i in range(n)]
        self.gen = iter(self.batches)

    def __iter__(self):
        return self.gen

    def __len__(self):
        raise TypeError("no length")


def test_one_shot_loader_ends_with_stopiteration():
    sampler = ResumableSampler(4, 0, 1)
    stream = BatchStream(OneShot(3), sampler, torch.device("cpu"), threaded=True)
    got = [int(stream.next()[0]) for _ in range(3)]
    assert got == [0, 1, 2]
    with pytest.raises(StopIteration):
        stream.next()
    stream.close()


# ---------------------------------------------------------------- generic prefetcher
def test_async_loader_yields_everything_and_closes():
    data = [(torch.full((2,), i), torch.full((2,), i + 1)) for i in range(5)]
    out = [int(x[0]) for x, _ in async_loader(data, torch.device("cpu"), depth=2)]
    assert out == [0, 1, 2, 3, 4]
    assert not any(t.name == "forgemind-async-loader" for t in threading.enumerate())


def test_async_loader_moves_and_forwards_errors():
    batch = async_loader([{"x": torch.zeros(2)}, (torch.ones(1),)], torch.device("cpu"))
    first = next(batch)
    assert isinstance(first, dict) and torch.is_tensor(first["x"])

    def boom():
        yield torch.zeros(1)
        raise RuntimeError("loader died")

    it = async_loader(boom(), torch.device("cpu"))
    assert torch.is_tensor(next(it))
    with pytest.raises(RuntimeError, match="loader died"):
        next(it)


def test_async_loader_accepts_plain_iterables():
    assert [int(t) for t in async_loader([torch.tensor(i) for i in range(3)],
                                         torch.device("cpu"))] == [0, 1, 2]


# ---------------------------------------------------------------- trainer / forge integration
def _train(threaded, steps=6):
    from forgemind import ForgeConfig, GPT, GPTConfig, SyntheticLM, Trainer
    cfg = ForgeConfig()
    cfg.train.__dict__.update(max_steps=steps, warmup_steps=0, lr=3e-3, log_every=99,
                              bf16=False, resume=False, ckpt_every=0, eval_every=0,
                              batch_thread=threaded, prefetch_depth=2)
    mc = GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4, n_kv_heads=2, max_seq=8)
    tr = Trainer(lambda c: GPT(mc), cfg, SyntheticLM(64, 8, n=256), micro_batch=4)
    tr.feed.close()
    tr.feed = BatchStream(*(tr.train_dl, tr.sampler, tr.ctx.device), prefetch=True, depth=2,
                          threaded=threaded)
    losses = [tr.train_step()["loss"] for _ in range(steps)]
    rep = tr.feed.report()
    tr.feed.close()
    return losses, rep


def test_trainer_threaded_matches_serial_losses():
    serial, rep_s = _train(False)
    threaded, rep_t = _train(True)
    assert rep_s["threaded"] is False and rep_t["threaded"] is True
    assert [round(x, 6) for x in serial] == [round(x, 6) for x in threaded]
    assert rep_t["batches"] >= 6


def test_forge_fit_uses_async_batches():

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

        def forward(self, x, y=None):
            return self.f(x)

    model = Net()
    forge = fm.Forge(model, torch.optim.AdamW(model.parameters(), lr=1e-2), precision="fp32",
                     quiet=True, loss_fn=lambda out, b: ((out - b[1]) ** 2).mean())
    g = torch.Generator().manual_seed(0)
    loader = [(torch.randn(8, 4, generator=g), torch.randn(8, 1, generator=g))
              for _ in range(4)]
    hist = forge.fit(loader * 2, max_steps=6)
    assert len(hist) == 6
    assert hist[-1]["loss"] < hist[0]["loss"]


def test_no_thread_leaks_from_fit_or_evaluate():
    """A prefetcher left blocked on a full queue is a leak even though it is a daemon."""

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

        def forward(self, x, y=None):
            return self.f(x)

    before = [t.name for t in threading.enumerate()].count("forgemind-async-loader")
    model = Net()
    forge = fm.Forge(model, torch.optim.AdamW(model.parameters(), lr=1e-2), precision="fp32",
                     quiet=True, loss_fn=lambda out, b: ((out - b[1]) ** 2).mean())
    g = torch.Generator().manual_seed(0)
    loader = [(torch.randn(8, 4, generator=g), torch.randn(8, 1, generator=g)) for _ in range(8)]
    forge.fit(loader, max_steps=2)                       # stops long before the loader ends
    forge.evaluate(loader, max_batches=1)                # and so does this
    after = [t.name for t in threading.enumerate()].count("forgemind-async-loader")
    assert after == before


def test_resume_after_checkpoint_is_exact_with_the_threaded_feed(tmp_path):
    """The async feed must not change what "resume" means: same data, same losses."""
    from forgemind import ForgeConfig, GPT, GPTConfig, SyntheticLM, Trainer

    def build(threaded, ckpt, resume):
        cfg = ForgeConfig()
        cfg.train.__dict__.update(max_steps=4, warmup_steps=0, lr=3e-3, log_every=99,
                                  bf16=False, resume=resume, ckpt_every=0, eval_every=0,
                                  ckpt_dir=str(ckpt), batch_thread=threaded, prefetch_depth=2)
        mc = GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4, n_kv_heads=2, max_seq=8)
        tr = Trainer(lambda c: GPT(mc), cfg, SyntheticLM(64, 8, n=256), micro_batch=4)
        return tr, cfg

    # uninterrupted 4 steps
    full, _ = build(True, tmp_path / "a", False)
    ref = [full.train_step()["loss"] for _ in range(4)]
    full.close()

    # 2 steps, checkpoint, fresh trainer, resume, 2 more
    a, _ = build(True, tmp_path / "b", False)
    first = [a.train_step()["loss"] for _ in range(2)]
    a.save()
    ckpt = sorted((tmp_path / "b").iterdir())[-1]
    a.close()

    b, _ = build(True, tmp_path / "b", True)
    second = [b.train_step()["loss"] for _ in range(2)]
    b.close()

    assert [round(x, 5) for x in first + second] == [round(x, 5) for x in ref]
    assert ckpt.exists()


def test_loader_workers_use_persistent_pool():
    """workers>0 with persistent_workers: the pool survives epoch rollovers (no fork on the
    prefetch thread, which is what makes the threaded stream safe) and close() frees it.

    The assertion is on the worker *processes*: multiprocessing's own queue-feeder threads exit
    asynchronously at collection time, so counting threads here would be a racy test, not a bug.
    """
    from forgemind import SyntheticLM, build_loader

    class Ctx:
        dp_rank, dp_world = 0, 1
        device = torch.device("cpu")

    ds = SyntheticLM(64, 8, n=64)
    dl, sampler = build_loader(ds, 4, Ctx(), workers=2)
    assert dl.persistent_workers is True
    stream = BatchStream(dl, sampler, torch.device("cpu"), threaded=True, depth=2)
    workers = list(getattr(stream._it, "_workers", []))
    assert workers, "expected a worker pool"
    seen = [int(stream.next()[0].flatten()[0]) for _ in range(10)]   # crosses an epoch boundary
    assert len(seen) == 10
    assert all(w.is_alive() for w in workers), "persistent_workers should survive the epoch"
    stream.close()
    assert all(not w.is_alive() for w in workers), "close() must terminate the worker pool"
    assert not [t for t in threading.enumerate() if t.name == "forgemind-batches"]


def test_async_loader_releases_worker_processes_on_close():
    """`persistent_workers=True` keeps its pool alive until the iterator dies; the generic
    prefetcher used by `Trainer.evaluate` must free it, every call, not at gc time."""
    from forgemind import SyntheticLM, build_loader

    class Ctx:
        dp_rank, dp_world = 0, 1
        device = torch.device("cpu")

    dl, _ = build_loader(SyntheticLM(64, 8, n=64), 4, Ctx(), workers=2)
    gen = async_loader(dl, torch.device("cpu"), depth=2)
    it = gen._it
    workers = list(getattr(it, "_workers", []))
    assert workers, "expected a worker pool"
    for _ in range(2):
        next(gen)
    gen.close()
    assert gen._it is None
    assert all(not w.is_alive() for w in workers), "worker pool still running after close()"
    with pytest.raises(StopIteration):     # a closed stream is an exhausted stream
        next(gen)
