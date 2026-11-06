"""ForgeMind benchmark CLI.

  python -m forgemind.cli.bench --quant fp8 --qbackend te          # force Transformer Engine
  python -m forgemind.cli.bench --quant fp8 --qbackend torchao     # force torchao
  python -m forgemind.cli.bench --quant int8 --qbackend auto --json out.json
  torchrun --nproc_per_node=4 -m forgemind.cli.bench --dp-shard 4 --moe

Measures tokens/s, step latency (p50/p90), MFU, and peak memory over timed steps after warmup.
Rank 0 prints the table and writes the JSON if requested.
"""
from __future__ import annotations
import argparse
import json
import statistics

import torch

from .. import ForgeConfig, GPT, GPTConfig, Trainer, SyntheticLM, moe_kwargs, cleanup
from ..quant import resolve_backend


def parse(argv=None):
    p = argparse.ArgumentParser(prog="forgemind-bench", description="Benchmark ForgeMind training")
    m = p.add_argument_group("model")
    m.add_argument("--preset", default="small", choices=["tiny", "small", "medium"])
    m.add_argument("--seq", type=int, default=512)
    m.add_argument("--micro-batch", type=int, default=4)
    m.add_argument("--moe", action="store_true")
    m.add_argument("--experts", type=int, default=8)
    m.add_argument("--top-k", type=int, default=2)

    q = p.add_argument_group("quantization")
    q.add_argument("--quant", default="none", choices=["none", "fp8", "int8", "int8_wo"])
    q.add_argument("--qbackend", default="auto", choices=["auto", "te", "torchao"],
                   help="te = Transformer Engine only; torchao = torchao only; auto = TE if installed else torchao")
    q.add_argument("--optimizer-8bit", action="store_true")

    par = p.add_argument_group("parallelism")
    par.add_argument("--dp-replicate", type=int, default=1)
    par.add_argument("--dp-shard", type=int, default=1)
    par.add_argument("--tp", type=int, default=1)
    par.add_argument("--pp", type=int, default=1)
    par.add_argument("--ep", type=int, default=1)
    par.add_argument("--backend", default="native", choices=["native", "deepspeed"])
    par.add_argument("--zero", type=int, default=2)

    o = p.add_argument_group("offload / memory")
    o.add_argument("--offload-opt", action="store_true")
    o.add_argument("--offload-act", action="store_true")
    o.add_argument("--offload-experts", action="store_true")
    o.add_argument("--ac", default="none", choices=["none", "full", "selective"])

    r = p.add_argument_group("run")
    r.add_argument("--warmup", type=int, default=5)
    r.add_argument("--steps", type=int, default=30)
    r.add_argument("--peak-tflops", type=float, default=0.0, help="device peak bf16 TFLOPS for MFU (0 = skip)")
    r.add_argument("--json", default=None, help="write results to this file (rank 0)")
    r.add_argument("--profile", default=None, metavar="DIR", help="record torch.profiler trace to DIR (TensorBoard/Perfetto)")
    r.add_argument("--profile-top", type=int, default=15, help="print the N most expensive ops (with --profile)")
    r.add_argument("--compile", action="store_true", help="torch.compile the transformer blocks")
    r.add_argument("--no-prefetch", action="store_true", help="disable the overlapped host->device batch copy")
    r.add_argument("--syncs", action="store_true",
                   help="count host<->device synchronisations per step (the stall budget)")
    return p.parse_args(argv)


PRESETS = {
    "tiny": dict(dim=256, n_layers=4, n_heads=8, n_kv_heads=4),
    "small": dict(dim=768, n_layers=12, n_heads=12, n_kv_heads=4),
    "medium": dict(dim=1024, n_layers=24, n_heads=16, n_kv_heads=4),
}


def build_config(a) -> ForgeConfig:
    cfg = ForgeConfig()
    cfg.parallel.__dict__.update(dp_replicate=a.dp_replicate, dp_shard=a.dp_shard, tp=a.tp, pp=a.pp,
                                 ep=a.ep, backend=a.backend, zero_stage=a.zero)
    cfg.quant.__dict__.update(mode=a.quant, backend=a.qbackend, optimizer_8bit=a.optimizer_8bit)
    cfg.offload.optimizer, cfg.offload.activations, cfg.offload.experts = a.offload_opt, a.offload_act, a.offload_experts
    cfg.train.__dict__.update(max_steps=a.warmup + a.steps, warmup_steps=0, log_every=10**9,
                              ac=a.ac, ckpt_every=0, eval_every=0, resume=False, grad_clip=1.0,
                              compile=a.compile, prefetch=not a.no_prefetch)
    return cfg


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main(argv=None):
    a = parse(argv)
    cfg = build_config(a)
    mc = GPTConfig(vocab_size=32000, max_seq=a.seq, n_experts=a.experts if a.moe else 0, top_k=a.top_k,
                   **PRESETS[a.preset])
    ds = SyntheticLM(mc.vocab_size, a.seq, n=max(256, a.micro_batch * (a.warmup + a.steps) * 2))
    tr = Trainer(lambda ctx: GPT(mc, moe_kwargs(ctx, cfg) if a.moe else None), cfg, ds,
                 micro_batch=a.micro_batch, peak_flops=a.peak_tflops * 1e12 or None)
    ctx = tr.ctx
    if ctx.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Resolve backend up front so the report states what actually ran (fails loudly if forced backend is missing).
    resolved = resolve_backend(cfg.quant) if a.quant != "none" else "none"

    times, toks, losses = [], [], []
    prof = None
    if a.profile:
        from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if ctx.device.type == "cuda" else [])
        prof = profile(activities=acts, record_shapes=True, profile_memory=True, with_stack=False,
                       schedule=schedule(wait=0, warmup=1, active=3, repeat=1),
                       on_trace_ready=tensorboard_trace_handler(a.profile) if ctx.rank == 0 else None)
        prof.__enter__()
    sync_cm = sync_counter = None
    for i in range(a.warmup + a.steps):
        if i == a.warmup and a.syncs:
            from ..perf import count_host_syncs
            sync_cm = count_host_syncs(capture_stack=False, count_cpu=ctx.device.type != "cuda")
            sync_counter = sync_cm.__enter__()
        m = tr.train_step()
        if i >= a.warmup:
            times.append(m["dt"]); toks.append(m.get("tok_s", 0.0)); losses.append(m["loss"])
        if prof is not None:
            if ctx.device.type == "cuda":
                torch.cuda.synchronize()
            prof.step()
    if sync_cm is not None:
        sync_cm.__exit__(None, None, None)
    syncs_total = sync_counter.own if sync_counter is not None else 0
    syncs_ext = sync_counter.external if sync_counter is not None else 0
    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    if prof is not None:
        prof.__exit__(None, None, None)
        if ctx.rank == 0:
            sort = "cuda_time_total" if ctx.device.type == "cuda" else "cpu_time_total"
            print(f"\n=== top {a.profile_top} ops by {sort} (profiled steps only) ===")
            print(prof.key_averages(group_by_input_shape=False).table(sort_by=sort, row_limit=a.profile_top))
            print(f"trace(s) written to {a.profile}/ (open .json in https://ui.perfetto.dev or TensorBoard)")

    med = statistics.median(times)
    res = {
        "preset": a.preset, "seq": a.seq, "micro_batch": a.micro_batch, "moe": a.moe,
        "world": ctx.world, "device": str(ctx.device), "gpu": (torch.cuda.get_device_name(ctx.device)
                                                              if ctx.device.type == "cuda" else "cpu"),
        "quant_mode": a.quant, "quant_requested": a.qbackend, "quant_resolved": resolved,
        "parallel": {"dp_replicate": a.dp_replicate, "dp_shard": a.dp_shard, "tp": a.tp, "pp": a.pp, "ep": a.ep},
        "offload": {"opt": a.offload_opt, "act": a.offload_act, "experts": a.offload_experts},
        "ac": a.ac, "backend": a.backend,
        "params_M": round(sum(p.numel() for p in tr.raw.parameters()) / 1e6, 2),
        "step_ms_p50": round(med * 1000, 2),
        "step_ms_p90": round(pct(times, 0.9) * 1000, 2),
        "tok_s": round(statistics.mean(toks), 1),
        "loss_last": round(losses[-1], 4),
    }
    if ctx.device.type == "cuda":
        res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    if a.syncs:
        # "own" = synchronisations issued by forgemind; "external" = torch internals (e.g. AdamW's
        # bias-correction reads a CPU-hosted step counter, which is free, and never fires with fused=True).
        res["host_syncs_per_step"] = round(syncs_total / max(1, a.steps), 2)
        res["host_syncs_torch_per_step"] = round(syncs_ext / max(1, a.steps), 2)
    if a.peak_tflops:
        res["mfu_pct"] = round(statistics.mean(
            [tr.raw.flops_per_token(a.seq) * t / (a.peak_tflops * 1e12 * ctx.world) for t in toks]) * 100, 2)

    if ctx.rank == 0:
        print("\n=== ForgeMind benchmark ===")
        for k, v in res.items():
            print(f"{k:>18}: {v}")
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=2)
            print(f"wrote {a.json}")
    cleanup()
    return res


if __name__ == "__main__":
    main()
