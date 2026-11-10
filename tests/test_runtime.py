"""The runtime primitives that used to live on ``fm.Fabric``.

ForgeMind is not a port of Lightning: the driver class is gone, its *machinery* is embedded
in the main module -- collectives as plain functions (``fm.all_reduce`` & co., passthrough
at world 1), ``fm.seed_everything``, and ``Forge.save``'s rank-0 checkpoint discipline.
The DDP path itself is covered by ``tests/forge_mp_check.py`` (2 processes, which now also
exercises the collectives and the rank-0 save/load round-trip).
"""

import os
import random
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import forgemind as fm  # noqa: E402


def test_fabric_is_gone():
    """The Lightning-shaped driver was removed on purpose -- pin it so it stays removed."""
    assert not hasattr(fm, "Fabric")
    with pytest.raises(ImportError):
        __import__("forgemind.fabric")


def test_collectives_are_passthrough_at_world_one():
    t = torch.randn(4, 3)
    assert fm.all_reduce(t) is t                       # no clone, no call, same object
    assert fm.broadcast(t) is t
    assert fm.all_gather(t) is t
    assert torch.equal(fm.all_gather(t), t)
    fm.barrier()                                        # no-op, must not raise


def test_all_reduce_mean_is_the_identity_at_world_one():
    t = torch.tensor([1.0, 2.0])
    out = fm.all_reduce(t.clone(), reduce_op="mean")
    assert torch.equal(out, t)
    with pytest.raises(KeyError):
        fm.all_reduce(t.clone(), reduce_op="median")   # unknown ops still validate


def test_seed_everything_is_reproducible():
    a = fm.seed_everything(1234)
    x = (torch.randn(8), random.random(), np.random.rand())
    b = fm.seed_everything(1234)
    y = (torch.randn(8), random.random(), np.random.rand())
    assert a == b == 1234
    assert torch.equal(x[0], y[0])
    assert x[1] == y[1] and x[2] == y[2]


def test_seed_everything_workers_sets_hashseed():
    os.environ.pop("PYTHONHASHSEED", None)
    try:
        fm.seed_everything(7, workers=True)
        assert os.environ.get("PYTHONHASHSEED") == "7"
        # setdefault: an explicit PYTHONHASHSEED is never overwritten
        os.environ["PYTHONHASHSEED"] = "99"
        fm.seed_everything(8, workers=True)
        assert os.environ["PYTHONHASHSEED"] == "99"
    finally:
        os.environ.pop("PYTHONHASHSEED", None)


def test_forge_save_is_rank_zero_and_roundtrips(tmp_path):
    """world 1 writes (the rank-0 guard must not swallow the only rank)."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
    forge = fm.Forge(net, torch.optim.AdamW(net.parameters(), lr=1e-2),
                     precision="fp32", quiet=True, callbacks=[], async_batches=False)
    x = torch.randn(8, 8)
    for _ in range(3):
        forge.step(x)
    path = str(tmp_path / "ckpt.pt")
    forge.save(path)
    assert os.path.exists(path)

    net2 = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
    forge2 = fm.Forge(net2, torch.optim.AdamW(net2.parameters(), lr=1e-2),
                      precision="fp32", quiet=True, callbacks=[], async_batches=False)
    forge2.load(path)
    assert forge2.step_count == forge.step_count
    for a, b in zip(forge.raw_model.parameters(), forge2.raw_model.parameters()):
        assert torch.equal(a, b)


def test_forgemind_exports_resolve():
    for name in ("all_reduce", "all_gather", "broadcast", "barrier", "seed_everything"):
        assert callable(getattr(fm, name)), name


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
