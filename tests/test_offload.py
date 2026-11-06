"""Correctness of the offload utilities, including the paths the single-process suite never covered
(gradient accumulation through the CPU accumulator, zeroing with grads still in flight, activation spill).

Run:  python tests/test_offload.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn

from forgemind import ForgeConfig, GPT, GPTConfig, Trainer, SyntheticLM
from forgemind.config import OffloadConfig
from forgemind.offload import OffloadedAdamW, apply_activation_offload
from forgemind.perf import _PinnedPool

MC = dict(vocab_size=64, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq=32)


def _train(offload, accum, steps=8, seed=0):
    torch.manual_seed(seed)
    cfg = ForgeConfig()
    cfg.train.__dict__.update(max_steps=steps, warmup_steps=0, lr=3e-3, bf16=False, resume=False,
                              log_every=10 ** 9, grad_clip=0.0, grad_accum=accum)
    cfg.offload.optimizer = offload
    ds = SyntheticLM(64, 32, n=256)
    tr = Trainer(lambda c: GPT(GPTConfig(**MC)), cfg, ds, micro_batch=8)
    tr.fit()
    return tr


def test_accumulated_offload_matches_adamw():
    for accum in (1, 2, 3):
        a = _train(False, accum)
        b = _train(True, accum)
        d = max((p1 - p2).abs().max().item() for p1, p2 in zip(a.raw.parameters(), b.raw.parameters()))
        assert d < 1e-4, (accum, d)
        print(f"grad_accum={accum}: offloaded optimizer matches AdamW (max diff {d:.2e})")


def test_zero_grad_with_copies_in_flight_keeps_data():
    m = nn.Linear(8, 4)
    m.weight.grad = torch.ones_like(m.weight)
    opt = OffloadedAdamW([{"params": [m.weight]}], 1e-3, cfg=OffloadConfig(), overlap=False)
    opt._offload(m.weight)              # queued, not yet flushed
    assert m.weight.grad is None
    opt.zero_grad()                     # must flush first, then clear
    assert torch.count_nonzero(opt.cgrad[m.weight]) == 0
    print("zero_grad() flushes pending copies before clearing")


def test_offload_accumulates_across_collect_calls():
    torch.manual_seed(0)
    lin = nn.Linear(16, 16)
    ref = torch.zeros_like(lin.weight)
    acc_opt = OffloadedAdamW([{"params": [lin.weight]}], 1e-3, cfg=OffloadConfig(), overlap=False)
    for _ in range(3):                  # three micro-batches of gradient
        lin.weight.grad = None
        lin(torch.randn(4, 16)).sum().backward()
        ref.add_(lin.weight.grad)
        acc_opt.collect()
    acc = acc_opt.cgrad[lin.weight]
    assert torch.allclose(acc, ref, atol=1e-6), (acc - ref).abs().max()
    print("three micro-batches accumulate exactly")


def test_clip_matches_torch_clip_grad_norm():
    torch.manual_seed(0)
    lin = nn.Linear(32, 32)
    lin(torch.randn(8, 32)).sum().backward()
    grads = lin.weight.grad.clone()
    n_ref = grads.norm()
    expected = grads * torch.clamp(1.0 / (n_ref + 1e-6), max=1.0)
    opt = OffloadedAdamW([{"params": [lin.weight]}], 1e-3, cfg=OffloadConfig(), overlap=False)
    n_off = opt.clip_grad_norm_(1.0)          # also consumes p.grad into the CPU accumulator
    assert torch.allclose(n_off.float(), n_ref.float(), rtol=1e-5), (n_off, n_ref)
    assert torch.allclose(opt.cgrad[lin.weight], expected, rtol=1e-5)
    print("clip_grad_norm_ matches torch (norm and clipped grads)")


def test_activation_offload_is_transparent():
    """On CPU the hook only observes; outputs and gradients must be bit-identical to the plain model."""
    torch.manual_seed(0)
    m = GPT(GPTConfig(**MC))
    ref = GPT(GPTConfig(**MC))
    ref.load_state_dict(m.state_dict())
    apply_activation_offload(m, OffloadConfig(act_min_bytes=1))
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    a, _ = m(x, labels=y); b, _ = ref(x, labels=y)
    assert torch.allclose(a, b, atol=1e-6)
    a.backward(); b.backward()
    for p, q in zip(m.parameters(), ref.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-6)
    print("activation offload is numerically transparent")


def test_pinned_pool_recycles():
    pool = _PinnedPool(pin=False)
    a = pool.get((4, 8), torch.float32)
    pool.put(a)
    b = pool.get((4, 8), torch.float32)
    assert b is a and pool.hits == 1 and pool.misses == 1
    assert pool.get((4, 9), torch.float32) is not a
    print("pinned pool recycles buffers per (shape, dtype)")


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            print("==", k)
            f()
    print("ALL OFFLOAD TESTS PASSED")
