"""Host<->device synchronisation budget for one training step.

Every `.item()` / `.tolist()` / `float(tensor)` / `bool(tensor)` / `.cpu()` on a device tensor drains the
accelerator queue. The design target is **one** such read per optimizer step (grad norm + loss + skip flag,
stacked into a single tensor and read once). This test asserts that budget so it cannot silently regress.

Run:  python tests/test_sync_audit.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from forgemind import GPT, GPTConfig, Trainer, SyntheticLM, ForgeConfig as FC
from forgemind.perf import count_host_syncs

MC = dict(vocab_size=64, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq=32)


def make(moe=False, steps=4, **train):
    cfg = FC()
    cfg.train.__dict__.update(max_steps=steps, warmup_steps=2, lr=3e-3, log_every=10 ** 9, bf16=False,
                              resume=False, ckpt_every=0, **train)
    mc = GPTConfig(**MC, n_experts=4 if moe else 0, top_k=2)
    ds = SyntheticLM(mc.vocab_size, mc.max_seq, n=256)
    tr = Trainer(lambda c: GPT(mc, None if not moe else dict(ep_size=1)), cfg, ds, micro_batch=8)
    tr.train_step()                          # warm up lazy paths before measuring
    return tr


def measure(tr, n=3):
    with count_host_syncs(capture_stack=True, count_cpu=True) as c:
        for _ in range(n):
            tr.train_step()
    return c


def test_dense_step_has_a_single_host_sync():
    tr = make()
    c = measure(tr)
    assert c.own <= 3, f"{c.own} host syncs over 3 steps:\n{c.report()}"
    print(f"dense: {c.own / 3:.2f} host sync(s)/step")
    if c.sites:
        print("  " + c.report().replace("\n", "\n  "))


def test_grad_accum_does_not_add_syncs():
    tr = make(grad_accum=2)
    c = measure(tr)
    assert c.own <= 3, f"accumulation added syncs: {c.report()}"
    print(f"grad_accum=2: {c.own / 3:.2f} host sync(s)/step")


def test_skip_logic_reuses_the_same_read():
    # skip_spike_factor forces the EMA comparison to run, which must not cost an extra read
    tr = make(skip_spike_factor=2.0)
    c = measure(tr)
    assert c.own <= 3, f"spike guard added syncs:\n{c.report()}"
    print(f"spike guard: {c.own / 3:.2f} host sync(s)/step")


def test_moe_has_one_read_per_layer():
    # Without CUDA+bf16 the expert dispatch cannot use torch._grouped_mm, so it needs one host read per
    # MoE layer to size the per-expert GEMMs. That is the documented fallback; nothing else may sync.
    tr = make(moe=True)
    n_moe = sum(1 for b in tr.raw.layers.values() if b.is_moe)
    c = measure(tr)
    assert c.own <= (1 + n_moe) * 3, f"MoE exceeded its sync budget:\n{c.report()}"
    print(f"moe: {c.own / 3:.2f} host sync(s)/step for {n_moe} MoE layer(s) [eager dispatch fallback]")


def test_activation_checkpointing_is_free():
    a = make(); b = make(ac="full")
    ca, cb = measure(a, 2), measure(b, 2)
    assert cb.own <= ca.own, f"AC added host syncs:\n{cb.report()}"
    print(f"ac=full: {cb.own / 2:.2f} vs {ca.own / 2:.2f} baseline sync(s)/step")


def test_offloaded_optimizer_is_free():
    mc = GPTConfig(**MC)
    ds = SyntheticLM(mc.vocab_size, mc.max_seq, n=256)
    cfg = FC()
    cfg.train.__dict__.update(max_steps=4, warmup_steps=2, lr=3e-3, log_every=10 ** 9, bf16=False, resume=False)
    cfg.offload.optimizer = True
    tr = Trainer(lambda c: GPT(mc), cfg, ds, micro_batch=8)
    tr.train_step()
    c = measure(tr, 2)
    assert c.own <= 2, f"OffloadedAdamW added host syncs:\n{c.report()}"
    print(f"offload optimizer: {c.own / 2:.2f} host sync(s)/step")


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            print("==", k)
            f()
    print("ALL SYNC AUDIT TESTS PASSED")
