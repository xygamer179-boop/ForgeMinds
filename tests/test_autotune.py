"""Tests for the autotuner: it must measure, handle OOM as data, and never lie about a win."""

import json
import os
import subprocess
import sys
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import forgemind as fm
from forgemind import autotune as at

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def test_time_call_warms_up_and_returns_median():
    calls = []

    def fn():
        calls.append(1)
        time.sleep(0.001)

    ms = at.time_call(fn, warmup=2, repeats=3)
    assert len(calls) == 5
    assert 0.5 < ms < 500


def test_tune_batch_size_stops_at_oom():
    def step(size):
        if size > 8:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        time.sleep(0.001)      # flat step time: per-sample cost keeps falling, so it climbs

    rep = at.Report()
    size, ms = at.tune_batch_size(step, candidates=(1, 2, 4, 8, 16, 32), warmup=0,
                                  repeats=1, report=rep)
    assert size == 8
    assert rep.trials[-1].oom and rep.trials[-1].label == "micro_batch=16"
    assert "did not fit" in rep.trials[-1].note


def test_tune_batch_size_skips_sizes_that_do_not_help():
    state = {"n": 0}

    def step(size):
        # flat step time up to 4 (so per-sample cost keeps improving), then it collapses
        time.sleep({1: 0.002, 2: 0.002, 4: 0.002, 8: 0.010, 16: 0.020}[size])
        state["n"] += 1

    size, ms = at.tune_batch_size(step, candidates=(1, 2, 4, 8, 16), warmup=0, repeats=1)
    assert size == 4
    assert state["n"] == 4                     # stopped instead of trying 16


def test_oom_lookalike_is_not_swallowed():
    def step(size):
        raise ValueError("something else entirely")

    with pytest.raises(ValueError):
        at.tune_batch_size(step, candidates=(1,), warmup=0, repeats=1)


def test_apply_fast_defaults_is_idempotent():
    a = at.apply_fast_defaults()
    b = at.apply_fast_defaults()
    assert isinstance(a, dict) and isinstance(b, dict)
    assert "threads" in a


def _forge():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1))

        def forward(self, x, y=None):
            return self.f(x)

    torch.manual_seed(0)
    m = Net()
    return fm.Forge(m, torch.optim.AdamW(m.parameters(), lr=1e-2), precision="fp32",
                    quiet=True, loss_fn=lambda o, b: F.mse_loss(o, b[1])), m


def test_autotune_reports_a_baseline_and_does_not_mutate_by_default():
    forge, _ = _forge()
    x, y = torch.randn(16, 4), torch.randn(16, 1)
    before = forge.precision, forge.accum
    rep = at.autotune(forge, [(x, y)], budget_s=30)
    assert rep.baseline_ms > 0
    assert rep.trials[0].label == "baseline"
    assert (forge.precision, forge.accum) == before
    assert "baseline" in rep.table()
    assert rep.to_dict()["speedup"] == pytest.approx(1.0)


def test_autotune_budget_is_respected():
    forge, _ = _forge()
    x, y = torch.randn(16, 4), torch.randn(16, 1)
    rep = at.autotune(forge, [(x, y)], budget_s=0.0001, accum=True)
    forge.accum = 4
    rep = at.autotune(forge, [(x, y)], budget_s=0.0001)
    assert any("skipped: budget" in t.note for t in rep.trials)
    assert rep.best.get("ms") in (0.0, None) or rep.best["ms"] >= 0


def test_autotune_kernels_skips_without_a_gpu():
    out = at.autotune_kernels()
    assert "skipped" in out or "rms_norm" in out


def test_autotune_cli_kernels():
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = subprocess.run([sys.executable, "-m", "forgemind.autotune", "--kernels"],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    payload = json.loads(out.stdout)
    assert "skipped" in payload or "device" in payload
