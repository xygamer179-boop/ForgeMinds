"""Tests for the ``Forge`` façade -- the "usable on ordinary torch code" layer."""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import forgemind as fm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


class Net(nn.Module):
    """Plain torch model: takes x, optionally y (so tuple batches work)."""

    def __init__(self, d_in=8, d_hidden=16):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))

    def forward(self, x, y=None):
        return self.f(x)


def mse(out, batch):
    return F.mse_loss(out, batch[1])


def data(n=64, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randn(n, 1, generator=g)


def build(**kw):
    torch.manual_seed(0)
    m = Net()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    kw.setdefault("precision", "fp32")
    kw.setdefault("quiet", True)
    return fm.Forge(m, opt, loss_fn=kw.pop("loss_fn", mse), **kw), m, opt


def test_fit_trains_a_plain_model():
    forge, model, _ = build(max_steps=40, log_every=100)
    x, y = data()
    loader = [(x[i:i + 16], y[i:i + 16]) for i in range(0, 64, 16)]
    hist = forge.fit(loader * 10, max_steps=40)
    assert len(hist) == 40
    assert hist[-1]["loss"] < hist[0]["loss"] * 0.9


def test_step_returns_state_only_on_full_accumulation():
    forge, _, _ = build(accum=4)
    x, y = data()
    states = [forge.step((x, y)) for _ in range(8)]
    assert [s is None for s in states] == [True, True, True, False, True, True, True, False]
    assert states[3]["step"] == 1 and states[7]["step"] == 2


def test_grad_accum_matches_one_big_batch():
    x, y = data()
    a, _, _ = build(accum=1, max_steps=1)
    b, _, _ = build(accum=2, max_steps=1)
    sa = a.step((x, y))
    b.step((x[:32], y[:32]))
    sb = b.step((x[32:], y[32:]))
    assert sa["step"] == 1 and sb["step"] == 1
    # the reported loss is the mean over the step's micro-batches, so both agree
    assert sa["loss"] == pytest.approx(sb["loss"], rel=1e-5)

    # one full-batch step vs two half-batch steps: same parameters (mean loss is linear)
    pa = torch.cat([p.detach().reshape(-1) for p in a.model.parameters()])
    pb = torch.cat([p.detach().reshape(-1) for p in b.model.parameters()])
    assert torch.allclose(pa, pb, atol=1e-6, rtol=1e-5)


def test_one_host_read_per_update():
    forge, _, _ = build(accum=2, max_steps=2)
    x, y = data()
    forge.step((x, y))
    forge.step((x, y))                       # first update
    with fm.count_host_syncs(count_cpu=True) as c:
        forge.step((x, y))
        forge.step((x, y))                   # second update inside the counter
    assert c.own == 1, c.report()


def test_clipping_happens_on_device_and_is_applied():
    forge, model, _ = build(clip=0.01)
    x, y = data()
    st = forge.step((x, y))
    # the coefficient is applied under the hood: the norm must end up <= clip
    gn = torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None])
    assert gn is None or float(gn) <= 0.0101


def test_save_load_roundtrip_is_exact():
    forge, _, _ = build(max_steps=10)
    x, y = data()
    for _ in range(3):
        forge.step((x, y))
    path = os.path.join(HERE, "_forge_ckpt.pt")
    forge.save(path)
    try:
        after_three = [p.detach().clone() for p in forge.model.parameters()]
        for _ in range(3):
            forge.step((x, y))
        final = [p.detach().clone() for p in forge.model.parameters()]

        fresh, _, _ = build(max_steps=10)
        fresh.load(path)
        assert fresh.step_count == 3
        for a, b in zip(fresh.model.parameters(), after_three):
            assert torch.equal(a, b)
        for _ in range(3):
            fresh.step((x, y))
        for a, b in zip(fresh.model.parameters(), final):
            assert torch.allclose(a, b, atol=0, rtol=0)
    finally:
        os.remove(path)


def test_callbacks_fire():
    seen = []

    class Cb:
        def on_step(self, forge, state):
            seen.append(state["step"])

        def on_epoch(self, forge, epoch):
            seen.append(("epoch", epoch))

    forge, _, _ = build(callbacks=[Cb()], max_steps=3)
    x, y = data()
    forge.fit([(x, y)] * 3, max_steps=3)
    assert [s for s in seen if isinstance(s, int)] == [1, 2, 3]
    assert ("epoch", 0) in seen


def test_bf16_autocast_and_evaluate():
    forge, _, _ = build(precision="bf16")
    x, y = data()
    forge.step((x, y))
    loss = forge.evaluate([(x, y)], max_batches=1)
    assert loss == loss and loss > 0


def test_skip_nonfinite_leaves_model_untouched(monkeypatch):
    forge, model, _ = build(skip_nonfinite=True)
    before = [p.detach().clone() for p in model.parameters()]
    x, y = data()
    y = y.clone()
    y[0] = float("nan")
    st = forge.step((x, y))
    assert st["skipped"] is True
    for a, b in zip(model.parameters(), before):
        assert torch.equal(a, b)


def test_unpacking_conventions():
    torch.manual_seed(0)

    class KwModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(4, 1)

        def forward(self, x):
            return self.w(x)

    m = KwModel()
    forge = fm.Forge(m, torch.optim.AdamW(m.parameters(), lr=1e-2), precision="fp32",
                     quiet=True, skip_nonfinite=False)
    x = torch.randn(8, 4)
    st = forge.step({"x": x})           # dict -> model(**batch)
    assert st is not None

    forge2 = fm.Forge(KwModel(), torch.optim.AdamW(m.parameters(), lr=1e-2), precision="fp32",
                      quiet=True, skip_nonfinite=False,
                      forward_fn=lambda model, batch: model(batch[0] * 2))
    assert forge2.step((x,))["step"] == 1


def test_offload_optimizer_path_trains():
    forge, _, _ = build(offload="optimizer", max_steps=4)
    x, y = data()
    hist = forge.fit([(x, y)], max_steps=4)
    assert hist[-1]["loss"] == hist[-1]["loss"]
    assert hist[-1]["loss"] <= hist[0]["loss"]


def test_launch_two_processes():
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = subprocess.run([sys.executable, os.path.join(HERE, "forge_mp_check.py")],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert out.stdout.count("in_sync=yes") == 2


def test_logging_path_runs(capsys):
    """`quiet=False` is the default: the logging branch must not be the untested one."""
    torch.manual_seed(0)
    m = Net()
    forge = fm.Forge(m, torch.optim.AdamW(m.parameters(), lr=1e-2), precision="fp32",
                     loss_fn=mse, log_every=2, skip_nonfinite=False)
    x, y = data()
    for _ in range(4):
        forge.step((x, y))
    out = capsys.readouterr().out
    assert out.count("step ") == 2          # logged on steps 2 and 4
    assert "loss" in out and "gnorm" in out


# --------------------------------------------------------------------------------- v0.9 audit regressions
def test_clip_zero_still_steps():
    """BUG-001 (audit): ``clip=0`` is the documented "no clipping" setting; the NaN
    grad-norm sentinel must not fail the finite check -- it used to skip 100% of steps."""
    forge, _, _ = build(clip=0.0, max_steps=4)
    x, y = data()
    hist = forge.fit([(x, y)] * 6, max_steps=4)
    assert len(hist) == 4
    assert not any(h.get("skipped") for h in hist)


def test_skipped_step_rearms_the_scaler():
    """BUG-002 (audit): the skipped branch must call scaler.update() -- unscale_() already
    ran, and without update() the next step's unscale_() raises 'already been called'."""
    forge, _, _ = build(max_steps=2, accum=1)

    class FakeScaler:
        """Mimics GradScaler's unscale_/update arming protocol."""

        def __init__(self):
            self._armed = True
            self.updates = 0

        def scale(self, loss):
            return loss

        def unscale_(self, opt):
            if not self._armed:
                raise RuntimeError("unscale_() has already been called for this optimizer")
            self._armed = False

        def step(self, opt):
            opt.step()

        def update(self):
            self.updates += 1
            self._armed = True

    sc = FakeScaler()
    forge.scaler = sc
    x, y = data()
    forge._clip_coef = lambda: torch.tensor(float("nan"))        # force a skip
    s1 = forge.step((x, y))
    assert s1["skipped"] is True and sc.updates == 1             # update() ran on the skip
    forge._clip_coef = lambda: torch.tensor(1.0)
    s2 = forge.step((x, y))                                      # raises without the fix
    assert s2["skipped"] is False


def test_async_fit_trains_every_epoch():
    """BUG-003 (audit): _AsyncLoader is one-shot and fit() used to reuse it across epochs,
    so with the default async_batches=True epochs >= 1 silently trained on zero batches."""
    forge, _, _ = build(max_steps=100, log_every=10**9)
    x, y = data()
    loader = [(x[i:i + 16], y[i:i + 16]) for i in range(0, 64, 16)]   # 4 batches per epoch
    hist = forge.fit(loader, epochs=3, max_steps=100)
    assert len(hist) == 12              # all three epochs trained (was 4 before the fix)


def test_forge_clip_flushes_offloaded_grads():
    """BUG-007 (audit): Forge._clip_coef must use the optimizer's own clip (which collects
    pending offloaded grads first); plain clip_grad_norm_ over model.parameters() would
    miss every moved gradient and apply the update unclipped."""
    forge, _, _ = build(offload="optimizer", max_steps=2, clip=0.5)
    off = forge.optimizer
    assert hasattr(off, "clip_grad_norm_"), "offload='optimizer' should build OffloadedAdamW"
    calls = []
    real = off.clip_grad_norm_

    def spy(max_norm):
        calls.append(max_norm)
        return real(max_norm)

    off.clip_grad_norm_ = spy
    x, y = data()
    st = forge.step((x, y))
    assert st is not None
    assert calls == [0.5]                # the flushing clip ran, not just the plain one


def test_evaluate_is_a_float_mean():
    """PERF-006 (audit): evaluate() now accumulates on device (one host read). The value
    contract must not change."""
    forge, _, _ = build(max_steps=2)
    x, y = data()
    loader = [(x[i:i + 16], y[i:i + 16]) for i in range(0, 64, 16)]
    got = forge.evaluate(loader)
    ref = float(sum(forge._extract_loss(forge._call_with_batch(b), b) for b in loader) / len(loader))
    assert got == pytest.approx(ref, rel=1e-5)
