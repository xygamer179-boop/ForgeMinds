"""``fm.Fabric``: the Lightning-Fabric-shaped driver.

These tests pin the Lightning-compatible surface (names, argument order, return shapes), the
precision mapping, checkpointing, and the sampler-epoch contract. The multi-process DDP path is
``tests/fabric_mp_check.py`` (subprocess, like ``forge_mp_check.py``).

Run:  python tests/test_fabric.py   (or pytest)
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forgemind.fabric import PRECISIONS, Fabric

HERE = os.path.dirname(os.path.abspath(__file__))


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))


def _loader(batch_size=4, n=16):
    torch.manual_seed(1)
    ds = TensorDataset(torch.randn(n, 8), torch.randn(n, 1))
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


class TestConstruction:
    def test_precision_names_match_lightning(self):
        assert set(PRECISIONS) == {"32-true", "16-true", "bf16-true", "16-mixed", "bf16-mixed"}
        with pytest.raises(ValueError, match="precision"):
            Fabric(precision="fp16")

    def test_strategy_is_never_silently_degraded(self):
        for bad in ("fsdp", "deepspeed", "xla", "ddp_spawn"):
            with pytest.raises(ValueError, match="Trainer"):
                Fabric(strategy=bad)

    def test_plugins_rejected(self):
        with pytest.raises(ValueError, match="plugins"):
            Fabric(plugins=object())

    def test_accelerator_and_devices_validation(self):
        with pytest.raises(ValueError, match="accelerator"):
            Fabric(accelerator="tpu")
        with pytest.raises(ValueError, match="devices"):
            Fabric(devices="many")
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="CUDA"):
                Fabric(accelerator="cuda")

    def test_defaults(self):
        f = Fabric()
        assert f.precision == "32-true" and f.strategy == "auto"
        assert f.world_size == 1 and f.global_rank == 0 and f.is_global_zero
        assert f.local_rank == 0 and f.device.type == ("cuda" if torch.cuda.is_available()
                                                      else "cpu")


class TestSetupAndStep:
    def test_setup_returns_what_was_passed(self):
        f = Fabric()
        m = _model()
        assert isinstance(f.setup(m), nn.Module)
        net = _model()
        m, o = f.setup(net, torch.optim.SGD(net.parameters(), lr=0.1))
        assert isinstance(o, torch.optim.SGD)

    def test_bf16_mixed_keeps_fp32_weights_and_autocasts(self):
        f = Fabric(precision="bf16-mixed")
        seen = {}

        class Probe(nn.Linear):
            def forward(self, x):
                seen["autocast"] = torch.is_autocast_enabled("cpu")
                return super().forward(x)

        probe = Probe(8, 2)
        o = torch.optim.SGD(probe.parameters(), lr=0.1)   # optimizer first, as in Lightning
        m, o = f.setup(probe, o)
        with f.autocast():
            loss = m(torch.randn(4, 8)).float().sum()
        f.backward(loss)
        f.clip_gradients(m, o, 1.0)
        o.step()
        assert seen["autocast"] is True
        assert next(m.parameters()).dtype == torch.float32

    def test_bf16_true_casts_weights(self):
        f = Fabric(precision="bf16-true")
        m = f.setup(_model())
        assert next(m.parameters()).dtype == torch.bfloat16
        with f.autocast():                       # true-half: no autocast, weights alone cast
            out = m(torch.randn(2, 8, dtype=torch.bfloat16))
        assert out.dtype == torch.bfloat16       # so *-true feeds need matching-dtype inputs

    def test_32_true_runs_a_full_step(self):
        f = Fabric(precision="32-true")
        net = _model()
        o = torch.optim.AdamW(net.parameters(), lr=1e-2)
        m, o = f.setup(net, o)
        before = [p.detach().clone() for p in m.parameters()]
        with f.autocast():
            loss = nn.functional.mse_loss(m(torch.randn(8, 8)), torch.randn(8, 1))
        f.backward(loss)
        norm = f.clip_gradients(m, o, 1.0)
        o.step()
        assert torch.isfinite(norm)
        assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))

    def test_clip_gradients_actually_clips(self):
        f = Fabric()
        m = f.setup(_model())
        o = torch.optim.SGD(m.parameters(), lr=1.0)
        with f.autocast():
            loss = m(torch.randn(4, 8)).square().mean()
        f.backward(loss)
        # crank the gradients so clipping must bite
        for p in m.parameters():
            if p.grad is not None:
                p.grad.mul_(1000.0)
        norm = f.clip_gradients(m, o, 0.01)
        assert norm > 0.01
        after = torch.norm(torch.stack([p.grad.norm() for p in m.parameters() if p.grad is not None]))
        assert after <= 0.01 + 1e-6

    def test_backward_and_step_under_fp16_mixed_cpu(self):
        f = Fabric(precision="16-mixed")         # no scaler on CPU: exercises the plain path
        assert f._scaler is None
        net = _model()
        m = f.setup(net)
        o = torch.optim.SGD(m.parameters(), lr=0.1)
        with f.autocast():
            loss = m(torch.randn(4, 8)).float().sum()
        f.backward(loss)
        o.step()


class TestDataloaders:
    def test_wrapper_moves_and_lengths(self):
        f = Fabric()
        loader = f.setup_dataloaders(_loader())
        assert len(loader) == len(_loader())
        batches = list(loader)
        assert len(batches) == 4
        x, y = batches[0]
        assert x.device == f.device and y.device == f.device

    def test_multiple_loaders_return_a_tuple(self):
        f = Fabric()
        a, b = f.setup_dataloaders(_loader(), _loader())
        assert len(a) == len(b)

    def test_epoch_bumps_after_each_full_pass(self, monkeypatch):
        monkeypatch.setattr(Fabric, "world_size", property(lambda self: 2))
        monkeypatch.setattr(Fabric, "global_rank", property(lambda self: 0))
        f = Fabric(strategy="ddp")
        loader = f.setup_dataloaders(_loader())
        from torch.utils.data import DistributedSampler
        assert isinstance(loader.sampler, DistributedSampler)
        list(loader); assert loader.sampler.epoch == 0
        list(loader); assert loader.sampler.epoch == 1        # auto-bump
        f.set_epoch(loader, 7); list(loader)
        assert loader.sampler.epoch == 7                      # explicit set wins

    def test_sampler_injection_can_be_declined(self, monkeypatch):
        monkeypatch.setattr(Fabric, "world_size", property(lambda self: 2))
        f = Fabric(strategy="ddp")
        loader = f.setup_dataloaders(_loader(), use_distributed_sampler=False)
        from torch.utils.data import RandomSampler
        assert isinstance(loader.sampler, RandomSampler)


class TestStateAndCollectives:
    def test_save_load_roundtrip(self, tmp_path):
        f = Fabric()
        net = _model()
        o = torch.optim.AdamW(net.parameters(), lr=1e-3)
        m, o = f.setup(net, o)
        with f.autocast():
            f.backward(m(torch.randn(4, 8)).square().mean())
        o.step()
        path = str(tmp_path / "ckpt.pt")
        state = {"model": m, "optimizer": o, "epoch": 3, "note": "hi"}
        f.save(path, state)

        m2 = _model()
        o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
        restored = {"model": m2, "optimizer": o2, "epoch": 0, "note": ""}
        f.load(path, restored)
        assert restored["epoch"] == 3 and restored["note"] == "hi"
        for a, b in zip(m.parameters(), m2.parameters()):
            assert torch.equal(a, b)

    def test_seed_everything_is_reproducible(self):
        f = Fabric()
        f.seed_everything(1234)
        a = torch.randn(4)
        f.seed_everything(1234)
        b = torch.randn(4)
        assert torch.equal(a, b)
        assert f.seed_everything(7) == 7

    def test_collectives_pass_through_at_world_one(self):
        f = Fabric()
        t = torch.tensor([1.0, 2.0])
        assert torch.equal(f.all_reduce(t.clone()), t)
        assert torch.equal(f.broadcast(t.clone()), t)
        assert torch.equal(f.all_gather(t.clone()), t)
        f.barrier()


class TestLaunch:
    def test_bare_launch_returns_self(self):
        assert Fabric(devices=1).launch() is not None

    def test_launch_runs_a_single_process_fn(self, tmp_path):
        marker = tmp_path / "ran.txt"
        f = Fabric(devices=1)

        def main():
            marker.write_text(f"{f.world_size}")

        f.launch(main)
        assert marker.read_text() == "1"

    def test_launch_without_fn_needs_processes_to_exist(self):
        with pytest.raises(ValueError, match="launch"):
            Fabric(devices=2).launch()

    def test_two_process_launch(self):
        r = subprocess.run([sys.executable, os.path.join(HERE, "fabric_mp_check.py")],
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
        assert "RESULT fabric_mp" in r.stdout
        assert "in_sync=yes" in r.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
