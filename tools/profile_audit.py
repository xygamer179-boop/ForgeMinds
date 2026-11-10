"""torch.profiler + dynamo graph-break audit of a ForgeMind training step (CPU-capable).

    python tools/profile_audit.py                     # op table, host-sync audit, graph breaks
    python tools/profile_audit.py --trace /tmp/step.json   # also write a chrome trace
    python tools/profile_audit.py --forge             # audit a Forge step instead of Trainer

What it looks for:

* **host reads / data-dependent branches** -- ``aten::_local_scalar_dense`` (the ``.item()``,
  ``.tolist()``, ``bool(t)`` family) is the primitive that forces a device sync and turns a
  GPU-side ``if`` into a pipeline stall. The repo budgets *one* stacked read per optimizer
  step; anything more shows up here with its call stack (``with_stack=True``).
* **launch/branch churn** -- ops by count: thousands of tiny ``aten`` calls per step are the
  ``if``/foreach/python-loop tax paid on the host between kernels.
* **torch.compile graph breaks** -- ``torch._dynamo.explain`` lists every break with its
  reason, so a "fullgraph" claim can be checked instead of asserted (dense path should be
  0; MoE dispatch breaks are deliberate, see documentation.md section 10).

No GPU required: the profiler runs in CPU mode; numbers are host-side structure, not speed.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from forgemind import GPT, GPTConfig, ForgeConfig, SyntheticLM, Trainer, moe_kwargs


def _trainer(moe: bool, steps: int = 4):
    cfg = ForgeConfig()
    cfg.train.__dict__.update(max_steps=steps, warmup_steps=0, lr=3e-3, bf16=False,
                              resume=False, log_every=1000)
    mc = GPTConfig(vocab_size=256, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq=32,
                   n_experts=4 if moe else 0, top_k=2)
    ds = SyntheticLM(mc.vocab_size, mc.max_seq, n=128)
    return Trainer(lambda c: GPT(mc, moe_kwargs(c, cfg) if moe else None), cfg, ds,
                   micro_batch=4)


def _forge_step():
    from forgemind import Forge
    torch.manual_seed(0)
    mc = GPTConfig(vocab_size=256, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq=32)
    model = GPT(mc)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    forge = Forge(model, opt, precision="fp32", quiet=True, callbacks=[],
                  async_batches=False)
    g = torch.Generator().manual_seed(0)
    batch = (torch.randint(0, 256, (4, 32), generator=g),
             torch.randint(0, 256, (4, 32), generator=g))
    return forge, batch


SYNC_OPS = {"aten::_local_scalar_dense", "aten::item", "aten::tolist"}


def host_reads(prof) -> list:
    """Every aten op that materialises a scalar on the host (device sync when on GPU),
    attributed to the innermost forgemind/torch frame that asked for it."""
    from collections import Counter
    where = Counter()
    for evt in getattr(prof.profiler, "function_events", []):
        if evt.key not in SYNC_OPS:
            continue
        frame = next((f for f in (evt.stack or []) if "/forgemind/" in f
                      or f.startswith("forgemind")), None)
        if frame is None:
            frame = next((f for f in (evt.stack or []) if "torch" in f), "<unknown>")
        where[(evt.key, frame.split("/")[-1])] += 1
    return where


def op_table(prof, n: int = 20) -> list:
    rows = [(e.key, int(e.count), e.self_cpu_time_total, e.self_cpu_time_total / max(1, e.count))
            for e in prof.key_averages() if e.count]
    rows.sort(key=lambda r: -r[2])
    return rows[:n]


def churn_table(prof, n: int = 15) -> list:
    """High-count ops: the per-step if/foreach/python-loop tax."""
    rows = [(e.key, int(e.count), e.self_cpu_time_total)
            for e in prof.key_averages() if e.count and not e.key.startswith("aten::empty")]
    rows.sort(key=lambda r: -r[1])
    return rows[:n]


def profile_step(run_step, label: str, trace_path: str | None):
    # warmup (async loader, caches, lazy inits) then profile the steady state
    for _ in range(3):
        run_step()
    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(activities=activities, with_stack=True,
                                record_shapes=False) as prof:
        for _ in range(3):
            run_step()
    if trace_path:
        prof.export_chrome_trace(trace_path)
        print(f"chrome trace: {trace_path}")

    print(f"\n=== {label}: host-read / data-dependent ops (per profiled window of 3 steps) ===")
    reads = host_reads(prof)
    if not reads:
        print("  (none)")
    for (key, frame), count in reads.most_common():
        print(f"  {count:5d}  {key:34s} <- {frame}")

    print(f"\n=== {label}: top ops by self CPU time ===")
    print(f"  {'count':>7s}  {'self ms':>9s}  {'us/call':>8s}  op")
    for key, count, t, per in op_table(prof):
        print(f"  {count:7d}  {t / 1e3:9.2f}  {per:8.1f}  {key}")

    print(f"\n=== {label}: highest-count ops (launch/python churn) ===")
    print(f"  {'count':>7s}  {'self ms':>9s}  op")
    for key, count, t in churn_table(prof):
        print(f"  {count:7d}  {t / 1e3:9.2f}  {key}")
    return prof


def graph_breaks(model, batch, label: str):
    """torch._dynamo.explain: every break and its reason (dense path should be zero)."""
    import torch._dynamo as dynamo

    model.eval()
    with torch.no_grad():
        try:
            rep = dynamo.explain(model)(*batch) if isinstance(batch, (tuple, list)) \
                else dynamo.explain(model)(batch)
        except Exception as e:                       # explain itself failing is a finding
            print(f"\n=== {label}: dynamo.explain FAILED: {type(e).__name__}: {e}")
            return
    breaks = getattr(rep, "graph_break_count", None)
    reasons = getattr(rep, "break_reasons", None) or []
    graphs = getattr(rep, "graph_count", "?")
    print(f"\n=== {label}: torch._dynamo.explain -> {graphs} graph(s), {breaks} break(s) ===")
    if not reasons:
        print("  (no breaks)")
    seen = set()
    for r in reasons:
        msg = str(getattr(r, "reason", r))
        if msg in seen:
            continue
        seen.add(msg)
        print(f"  - {msg.splitlines()[0][:160]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=None, help="write a chrome trace JSON here")
    ap.add_argument("--forge", action="store_true", help="audit a Forge step instead of Trainer")
    ap.add_argument("--moe", action="store_true", help="also run the MoE trainer (has the "
                    "deliberate dispatch breaks)")
    args = ap.parse_args()

    if args.forge:
        forge, batch = _forge_step()

        def step():
            st = None
            while st is None:
                st = forge.step(batch)
            return st
        label = "Forge step (dense GPT)"
        profile_step(step, label, args.trace)
        graph_breaks(forge.raw_model, batch, label)
        return

    for moe in ([False, True] if args.moe else [False]):
        trainer = _trainer(moe)
        label = f"Trainer step ({'MoE' if moe else 'dense'} GPT)"
        profile_step(trainer.train_step, label, args.trace if not moe else None)
        # graph breaks on the raw module with a representative batch
        raw = trainer.model
        while hasattr(raw, "module") or hasattr(raw, "_orig_mod"):
            raw = getattr(raw, "module", None) or getattr(raw, "_orig_mod")
        g = torch.Generator().manual_seed(0)
        batch = (torch.randint(0, 256, (4, 32), generator=g),
                 torch.randint(0, 256, (4, 32), generator=g))
        graph_breaks(raw, batch, label)


if __name__ == "__main__":
    main()
