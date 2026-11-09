"""Tests for ``fm.Muon`` and ``fm.kernels.newton_schulz``.

Besides the usual correctness checks, these encode the bugs found in the reference Muon
file this feature was ported from (see ``PERF.md`` round six / ``forgemind/muon.py``):

1. ``nesterov=False`` scaled the *momentum buffer* by ``sqrt(max(1, rows/cols))`` every
   step (compounding explosion on tall matrices) -- ``test_nesterov_false_keeps_buffer_unscaled``.
2. ``nesterov=True`` mutated ``p.grad`` in place with the lerp (corrupting gradient
   accumulation) -- ``test_nesterov_does_not_mutate_grad`` and
   ``test_grad_accumulation_survives_step``.
3. unconditional ``dist`` calls and ``params[0]`` asserts (single-process / generator
   crashes) -- ``test_auto_split_groups_and_generators`` and friends.
4. strict key-set asserts rejecting extra group keys -- ``test_explicit_groups_honored``.
5. ``p.grad = zeros`` written into the model for ``None`` grads --
   ``test_none_grad_param_stays_none``.

Run of the distributed path lives in ``tests/muon_mp_check.py`` (subprocess, like the
Fabric/Forge mp checks).
"""

import copy
import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

import forgemind as fm
from forgemind import kernels as kk
from forgemind.kernels.ns import newton_schulz

HERE = os.path.dirname(os.path.abspath(__file__))
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


# --------------------------------------------------------------------------- reference
# Verbatim single-device parts of the attached reference implementation (muon_optim (1).py),
# kept in this file so the parity test pins *that* math, not a re-derivation.
def _ref_ns(G, steps=5):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _ref_muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = _ref_ns(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def _ref_adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class _RefAux(torch.optim.Optimizer):
    """``SingleDeviceMuonWithAuxAdam`` from the reference file, verbatim semantics."""

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay",
                                             "use_muon"}
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        state = self.state[p]
                        if "zero_grad" not in state:
                            state["zero_grad"] = torch.zeros_like(p)
                        p.grad = state["zero_grad"]
                    state = self.state[p]
                    if len(state) == 0 or "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = _ref_muon_update(p.grad, state["momentum_buffer"],
                                              beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = _ref_adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                              state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
        return loss


# --------------------------------------------------------------------------- ns kernel
def test_ns_shape_dtype_and_batch():
    for shape in [(17, 8), (8, 17), (8, 8), (3, 5, 9), (2, 2, 7, 11)]:
        x = torch.randn(*shape)
        out = newton_schulz(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype                     # fp32 input -> fp32 compute default
    b = torch.randn(4, 9, 6, dtype=torch.bfloat16)
    assert newton_schulz(b).dtype == torch.bfloat16     # half input -> bf16 compute
    assert newton_schulz(b, dtype=torch.float32).dtype == torch.float32  # explicit override
    assert newton_schulz(torch.randn(6, 4, dtype=torch.float16)).dtype == torch.bfloat16


@pytest.mark.parametrize("shape", [(128, 64), (64, 128), (3, 32, 16)])
def test_ns_drives_singular_values_to_one(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape)
    out = newton_schulz(x, dtype=torch.float32)
    s = torch.linalg.svdvals(out.float())
    assert torch.allclose(s, torch.ones_like(s), atol=0.35), (s.min().item(), s.max().item())


def test_ns_guards():
    with pytest.raises(ValueError, match="ndim >= 2"):
        newton_schulz(torch.randn(8))
    with pytest.raises(ValueError, match="ndim >= 2"):
        newton_schulz(torch.randn(()))
    with pytest.raises(ValueError, match="steps"):
        newton_schulz(torch.randn(4, 4), steps=0)
    with pytest.raises(ValueError, match="dtype"):
        newton_schulz(torch.randn(4, 4), dtype=torch.float64)


def test_ns_matches_reference_single_matrix():
    """torch path == attached reference, dtype-appropriate tolerance (bf16 rounding)."""
    torch.manual_seed(1)
    for shape in [(17, 8), (8, 17), (33, 33)]:
        x = torch.randn(*shape)
        ref = _ref_ns(x)                      # always bf16 inside
        ours = newton_schulz(x, dtype=torch.bfloat16, fused=False)
        assert ours.dtype == ref.dtype
        # documented re-association: b*A + c*(A@A) vs the reference's parsed
        # b*A + (c*A)@A -- mathematically equal, bf16 rounding differs a few ulps
        diff = (ours.float() - ref.float()).abs().max().item()
        assert diff < 0.05, (shape, diff)


@pytest.mark.skipif(not INTERPRET, reason="fused Triton path needs CUDA or TRITON_INTERPRET=1")
def test_ns_fused_and_torch_paths_agree():
    torch.manual_seed(2)
    for shape, dtype, tol in [((41, 23), torch.float32, 1e-5), ((23, 41), torch.float32, 1e-5),
                              ((41, 23), torch.bfloat16, 5e-2), ((2, 15, 9), torch.bfloat16, 5e-2)]:
        x = torch.randn(*shape, dtype=dtype)
        fused = newton_schulz(x)                      # auto -> fused under INTERPRET
        pure = newton_schulz(x, fused=False)          # explicit torch path
        diff = (fused.float() - pure.float()).abs().max().item()
        assert diff <= tol, (shape, dtype, diff)


def test_ns_compile_fullgraph_matches_eager():
    torch.manual_seed(3)
    x = torch.randn(12, 7)
    compiled = torch.compile(newton_schulz, backend="aot_eager", fullgraph=True)
    a, b = compiled(x), newton_schulz(x)
    # tracing routes to the torch implementation; eager on this box does too (no CUDA),
    # and under INTERPRET the fused path agrees within dtype rounding -- fp32 keeps it tight.
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=1e-5)


def test_ns_dispatch_stats_fallback():
    x = torch.randn(6, 6)
    kk.reset_stats()
    kk.set_backend("fallback")
    try:
        newton_schulz(x)
        s = kk.stats()
        assert s["fallback"] >= 1
        assert s["reasons"].get("backend=fallback", 0) >= 1
    finally:
        kk.set_backend("auto")
        kk.reset_stats()


# --------------------------------------------------------------------------- muon groups
class _Net(nn.Module):
    """Conv (4D) + tall + wide matrices + biases: every auto-split category."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.l1 = nn.Linear(8 * 4 * 4, 16)
        self.l2 = nn.Linear(16, 24)
        self.l3 = nn.Linear(24, 5)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(self.conv(x), 2)
        return self.l3(self.l2(self.l1(x.flatten(1))))


def _groups_for(net, **kw):
    mats = [p for p in net.parameters() if p.ndim >= 2]
    vecs = [p for p in net.parameters() if p.ndim < 2]
    return [dict(params=mats, use_muon=True, **kw),
            dict(params=vecs, use_muon=False, **kw)]


def test_auto_split_groups_and_generators():
    net = _Net()
    n_mats = sum(p.ndim >= 2 for p in net.parameters())
    n_vecs = sum(p.ndim < 2 for p in net.parameters())

    opt = fm.Muon(net.parameters(), lr=0.02)          # generator, no [0] assert
    assert sum(g["use_muon"] for g in opt.param_groups) == 1
    muon = next(g for g in opt.param_groups if g["use_muon"])
    adam = next(g for g in opt.param_groups if not g["use_muon"])
    assert len(muon["params"]) == n_mats
    assert len(adam["params"]) == n_vecs
    assert adam["lr"] == 0.02                          # default falls back to lr
    opt2 = fm.Muon(net.parameters(), lr=0.02, adam_lr=0.003)
    adam2 = next(g for g in opt2.param_groups if not g["use_muon"])
    assert adam2["lr"] == 0.003

    # a group dict *without* use_muon is split in place, keeping its hyperparameters
    opt3 = fm.Muon([dict(params=list(net.parameters()), lr=0.05, weight_decay=0.01)])
    assert len(opt3.param_groups) == 2
    assert all(g["lr"] == 0.05 and g["weight_decay"] == 0.01 for g in opt3.param_groups)

    with pytest.raises(ValueError, match="empty"):
        fm.Muon(iter([]))
    with pytest.raises(TypeError, match="Tensor"):
        fm.Muon([42])


def test_explicit_groups_honored():
    net = _Net()
    mats = [p for p in net.parameters() if p.ndim >= 2]
    vecs = [p for p in net.parameters() if p.ndim < 2]
    groups = [dict(params=mats, lr=0.05, momentum=0.9, weight_decay=0.02, use_muon=True,
                   ns_steps=3, ns_dtype=torch.float32),
              dict(params=vecs, lr=0.003, betas=(0.8, 0.9), eps=1e-9, weight_decay=0.0,
                   use_muon=False)]
    opt = fm.Muon(groups)
    muon = next(g for g in opt.param_groups if g["use_muon"])
    adam = next(g for g in opt.param_groups if not g["use_muon"])
    assert muon["lr"] == 0.05 and muon["momentum"] == 0.9 and muon["ns_steps"] == 3
    assert muon["ns_dtype"] == torch.float32
    assert adam["lr"] == 0.003 and adam["betas"] == (0.8, 0.9) and adam["eps"] == 1e-9

    # reference bug: set(group.keys()) == {...} asserts reject any extra key
    net2 = nn.Linear(4, 4)
    opt2 = fm.Muon([dict(params=[net2.weight], use_muon=True, scheduler_hint="cosine")])
    assert opt2.param_groups[0]["scheduler_hint"] == "cosine"


def test_vector_in_muon_group_rejected():
    net = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="use_muon=True group contains"):
        fm.Muon([dict(params=[net.bias], use_muon=True)])


def test_add_param_group_auto_splits():
    net = nn.Linear(6, 6)
    opt = fm.Muon([dict(params=[net.weight], use_muon=True)])
    opt.add_param_group(dict(params=[net.bias]))      # no flag: auto -> AdamW group
    assert len(opt.param_groups) == 2
    assert opt.param_groups[1]["use_muon"] is False


# --------------------------------------------------------------------------- muon step bugs
def test_nesterov_false_keeps_buffer_unscaled():
    """Reference bug 1: the aspect-ratio scale was written into the state buffer."""
    torch.manual_seed(0)
    lin = nn.Linear(8, 16)                             # weight (16, 8): tall, scale would be 2**0.5
    opt = fm.Muon(lin.parameters(), lr=0.02, momentum=0.9, nesterov=False, weight_decay=0.0)
    g = torch.randn_like(lin.weight)
    beta, steps = 0.9, 4
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        lin.weight.grad = g.clone()
        opt.step()
    expected = (1 - beta ** steps) * g                 # buf_t = (1 - beta^t) * g, unscaled
    buf = opt.state[lin.weight]["momentum_buffer"]
    diff = (buf - expected).abs().max().item()
    # the reference would give expected * (16/8) ** (steps/2) = 4 * expected here
    assert diff < 1e-5, diff


def test_nesterov_does_not_mutate_grad():
    """Reference bug 2: grad.lerp_(momentum, beta) corrupted the user's gradient."""
    torch.manual_seed(1)
    lin = nn.Linear(8, 8)
    opt = fm.Muon(lin.parameters(), lr=0.02)           # nesterov=True default
    lin.weight.grad = torch.randn_like(lin.weight)
    lin.bias.grad = torch.randn_like(lin.bias)
    snap_w, snap_b = lin.weight.grad.clone(), lin.bias.grad.clone()
    opt.step()
    assert torch.equal(lin.weight.grad, snap_w)
    assert torch.equal(lin.bias.grad, snap_b)


def test_grad_accumulation_survives_step():
    """set_to_none=False accumulation must not see a momentum-blended gradient.

    The reference's nesterov ``grad.lerp_(momentum, beta)`` rewrote the accumulated sum
    during ``step``; the next backward would then add to a corrupted value.
    """
    torch.manual_seed(2)
    lin = nn.Linear(8, 8)
    opt = fm.Muon(lin.parameters(), lr=0.02)
    x1, x2 = torch.randn(4, 8), torch.randn(4, 8)
    torch.nn.functional.mse_loss(lin(x1), torch.randn(4, 8)).backward()
    torch.nn.functional.mse_loss(lin(x2), torch.randn(4, 8)).backward()
    total = lin.weight.grad.clone()                    # accumulated g1 + g2
    opt.step()
    assert torch.equal(lin.weight.grad, total)         # step must not rewrite them
    opt.zero_grad(set_to_none=False)
    assert lin.weight.grad is not None                 # in-place zero, tensor kept
    assert not lin.weight.grad.any()


def test_none_grad_param_stays_none():
    """Reference wrote p.grad = zeros into the model; we keep None and leave params alone."""
    torch.manual_seed(3)
    a, b = nn.Linear(16, 16), nn.Linear(16, 16)
    opt = fm.Muon([dict(params=[a.weight, b.weight], use_muon=True, lr=0.05,
                        weight_decay=0.0)], lr=0.05)
    a.weight.grad = torch.randn_like(a.weight)         # b.weight gets no grad
    a_before, b_before = a.weight.detach().clone(), b.weight.detach().clone()
    opt.step()
    assert b.weight.grad is None
    assert torch.equal(b.weight, b_before)             # zeros -> NS(0) = 0, wd = 0
    assert a.weight.grad is not None                   # untouched by the step (no in-place lerp)
    assert not torch.equal(a.weight, a_before)         # a still trained normally


def test_frozen_param_skipped():
    torch.manual_seed(4)
    lin = nn.Linear(8, 8)
    lin.bias.requires_grad_(False)
    before = lin.bias.detach().clone()
    opt = fm.Muon(lin.parameters(), lr=0.05, weight_decay=1.0)   # wd would move it if stepped
    lin.weight.grad = torch.randn_like(lin.weight)
    opt.step()
    assert torch.equal(lin.bias, before)
    assert len(opt.state[lin.bias]) == 0               # no state created


def test_conv4d_param_trains():
    torch.manual_seed(5)
    conv = nn.Conv2d(3, 8, 3)
    opt = fm.Muon(conv.parameters(), lr=0.05)
    muon = next(g for g in opt.param_groups if g["use_muon"])
    assert len(muon["params"]) == 1                    # ndim=4 goes to Muon, not AdamW
    before = conv.weight.detach().clone()
    conv.weight.grad = torch.randn_like(conv.weight)
    conv.bias.grad = torch.randn_like(conv.bias)
    opt.step()
    assert conv.weight.shape == before.shape
    assert not torch.equal(conv.weight, before)


# --------------------------------------------------------------------------- batching & parity
def test_bucketed_matches_single_matrix_loop():
    """Batched NS over a shape bucket == one newton_schulz call per matrix."""
    torch.manual_seed(6)
    nets = [nn.Linear(16, 16, bias=False) for _ in range(3)]
    kw = dict(lr=0.05, momentum=0.9, nesterov=True, weight_decay=0.01,
              ns_steps=5, ns_dtype=torch.float32)
    opt = fm.Muon([dict(params=[n.weight for n in nets], use_muon=True, **kw)])
    grads = [torch.randn_like(n.weight) for n in nets]

    ref_nets = [copy.deepcopy(n) for n in nets]
    for n, g in zip(nets, grads):
        n.weight.grad = g.clone()
    for n, g in zip(ref_nets, grads):
        n.weight.grad = g.clone()

    ref_bufs = [torch.zeros_like(n.weight) for n in ref_nets]
    for step in range(2):
        opt.step()                                     # batched path (one bucket, N=3)
        with torch.no_grad():
            for n, g, buf in zip(ref_nets, grads, ref_bufs):
                buf.lerp_(g, 1 - 0.9)
                eff = g.lerp(buf, 0.9)
                u = newton_schulz(eff, dtype=torch.float32)
                u = u * max(1.0, 16 / 16) ** 0.5
                n.weight.mul_(1 - 0.05 * 0.01)
                n.weight.add_(u, alpha=-0.05)
    for n, r in zip(nets, ref_nets):
        diff = (n.weight - r.weight).abs().max().item()
        assert diff < 1e-4, diff


def test_parity_with_attached_reference():
    """Same model, same data, same hyperparameters: our step == the reference step.

    nesterov=True + fresh (set_to_none) grads: the reference is correct on that path,
    so it is a meaningful oracle (bf16 NS -> dtype rounding tolerance).
    """
    torch.manual_seed(7)
    net = _Net()
    ref_net = copy.deepcopy(net)
    kw = dict(lr=0.02, weight_decay=0.01)
    opt = fm.Muon(_groups_for(net, **kw), momentum=0.95,
                  betas=(0.9, 0.95), eps=1e-10)
    ref_kw = dict(lr=0.02, weight_decay=0.01)
    ref = _RefAux([dict(params=[p for p in ref_net.parameters() if p.ndim >= 2],
                        use_muon=True, momentum=0.95, **ref_kw),
                   dict(params=[p for p in ref_net.parameters() if p.ndim < 2],
                        use_muon=False, betas=(0.9, 0.95), eps=1e-10, **ref_kw)])
    g = torch.Generator().manual_seed(11)
    max_diff = 0.0
    for _ in range(3):
        x = torch.randn(6, 3, 8, 8, generator=g)
        y = torch.randn(6, 5, generator=g)
        opt.zero_grad()
        ref.zero_grad()
        torch.nn.functional.mse_loss(net(x), y).backward()
        torch.nn.functional.mse_loss(ref_net(x), y).backward()
        opt.step()
        ref.step()
        for a, b in zip(net.parameters(), ref_net.parameters()):
            max_diff = max(max_diff, (a - b).abs().max().item())
    # measured ~0.007 after 3 steps (bf16-NS re-association); real step bugs land >0.1
    assert max_diff < 0.015, max_diff


def test_muon_training_reduces_loss():
    torch.manual_seed(8)
    net = _Net()
    opt = fm.Muon(net.parameters(), lr=0.03, weight_decay=0.0)
    g = torch.Generator().manual_seed(12)
    x = torch.randn(16, 3, 8, 8, generator=g)
    y = torch.randn(16, 5, generator=g)
    first = None
    for _ in range(30):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(x), y)
        first = float(loss.detach()) if first is None else first
        loss.backward()
        opt.step()
    assert float(loss.detach()) < 0.7 * first, (first, float(loss.detach()))


def test_state_dict_roundtrip(tmp_path):
    torch.manual_seed(9)
    net = _Net()
    opt = fm.Muon(net.parameters(), lr=0.02, weight_decay=0.01)
    net(torch.randn(2, 3, 8, 8)).sum().backward()
    opt.step()
    path = tmp_path / "muon.pt"
    torch.save({"opt": opt.state_dict()}, path)

    net2 = copy.deepcopy(net)
    opt2 = fm.Muon(net2.parameters(), lr=0.02, weight_decay=0.01)
    opt2.load_state_dict(torch.load(path)["opt"])
    for p, q in zip(net.parameters(), net2.parameters()):
        assert torch.equal(p, q)
    # momentum buffers restored: identical next step on identical grads
    for p in net.parameters():
        if p.grad is not None:
            p.grad = torch.randn_like(p)
    for p, q in zip(net.parameters(), net2.parameters()):
        if p.grad is not None:
            q.grad = p.grad.clone()
    opt.step()
    opt2.step()
    for p, q in zip(net.parameters(), net2.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


# --------------------------------------------------------------------------- distributed
def test_two_process_muon_check():
    r = subprocess.run([sys.executable, os.path.join(HERE, "muon_mp_check.py")],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    assert "RESULT muon_mp" in r.stdout
    assert "in_sync=yes" in r.stdout
    assert "ref_ok=yes" in r.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
