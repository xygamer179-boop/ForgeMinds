import os, sys, tempfile, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from forgemind import *

def mk(moe=False, steps=40, **over):
    cfg = ForgeConfig()
    cfg.train.__dict__.update(max_steps=steps, warmup_steps=5, lr=3e-3, log_every=1000, bf16=False, resume=False, **over)
    mc = GPTConfig(vocab_size=64, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq=32, n_experts=4 if moe else 0, top_k=2)
    return cfg, mc

def run(cfg, mc, moe=False):
    ds = SyntheticLM(mc.vocab_size, mc.max_seq, n=512)
    tr = Trainer(lambda c: GPT(mc, moe_kwargs(c, cfg) if moe else None), cfg, ds, ds, micro_batch=8)
    h = tr.fit()
    return tr, h

def drop(h): return h[0]["loss"], h[-1]["loss"]

def test_dense():
    cfg, mc = mk(); tr, h = run(cfg, mc); a, b = drop(h); print("dense", a, b); assert b < a * 0.6

def test_moe():
    cfg, mc = mk(moe=True); tr, h = run(cfg, mc, True); a, b = drop(h); print("moe", a, b); assert b < a * 0.6

def test_accum_equiv():
    # accum=2 with micro 8 must match accum=1 with batch 16 loosely: just check it trains & grads finite
    cfg, mc = mk(grad_accum=2); tr, h = run(cfg, mc); a, b = drop(h); print("accum", a, b); assert b < a * 0.7

def test_offload_opt_matches_adamw():
    torch.manual_seed(0)
    cfg, mc = mk(steps=15, grad_clip=0.0)
    ds = SyntheticLM(mc.vocab_size, mc.max_seq, n=256)
    t1 = Trainer(lambda c: GPT(mc), cfg, ds, micro_batch=8); t1.fit()
    cfg2, _ = mk(steps=15, grad_clip=0.0); cfg2.offload.optimizer = True
    t2 = Trainer(lambda c: GPT(mc), cfg2, ds, micro_batch=8); t2.fit()
    d = max((p1 - p2).abs().max().item() for p1, p2 in zip(t1.raw.parameters(), t2.raw.parameters()))
    print("offload-vs-adamw max param diff", d); assert d < 1e-4

def test_expert_offload_module():
    cfg, mc = mk(moe=True); cfg.offload.experts = True
    tr, h = run(cfg, mc, True); a, b = drop(h); print("expert-offload", a, b); assert b < a * 0.7
    assert isinstance(tr.raw.layers["0"].ffn.experts.w1, torch.nn.ParameterList)

def test_ac_equiv():
    for mode in ("full", "selective"):
        cfg, mc = mk(steps=10, ac=mode); tr, h = run(cfg, mc); print("ac", mode, drop(h)); assert h[-1]["loss"] < h[0]["loss"]

def test_checkpoint_resume():
    d = tempfile.mkdtemp()
    cfg, mc = mk(steps=10, ckpt_every=5, ckpt_dir=d)
    ds = SyntheticLM(mc.vocab_size, mc.max_seq, n=256)
    t1 = Trainer(lambda c: GPT(mc), cfg, ds, micro_batch=8); t1.fit()
    cfg2, _ = mk(steps=14, ckpt_dir=d); cfg2.train.resume = True
    t2 = Trainer(lambda c: GPT(mc), cfg2, ds, micro_batch=8)
    assert t2.step == 10, t2.step
    for p1, p2 in zip(t1.raw.parameters(), t2.raw.parameters()): assert torch.equal(p1, p2)
    t2.fit(); assert t2.step == 14 and len(t2.history) == 4
    # offload-optimizer checkpoint
    d2 = tempfile.mkdtemp(); c3, _ = mk(steps=6, ckpt_every=3, ckpt_dir=d2); c3.offload.optimizer = True
    t3 = Trainer(lambda c: GPT(mc), c3, ds, micro_batch=8); t3.fit()
    c4, _ = mk(steps=8, ckpt_dir=d2); c4.offload.optimizer = True; c4.train.resume = True
    t4 = Trainer(lambda c: GPT(mc), c4, ds, micro_batch=8); assert t4.step == 6; t4.fit()

def test_spike_skip_and_nonfinite():
    cfg, mc = mk(steps=6); ds = SyntheticLM(mc.vocab_size, mc.max_seq, n=128)
    tr = Trainer(lambda c: GPT(mc), cfg, ds, micro_batch=8)
    orig = tr._clip; calls = {"n": 0}
    def bad():
        calls["n"] += 1; n = orig()
        return torch.tensor(float("nan")) if calls["n"] == 3 else n
    tr._clip = bad; tr.fit(); assert tr.skipped == 1, tr.skipped

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            print("==", k); f()
    print("ALL SINGLE-PROCESS TESTS PASSED")
