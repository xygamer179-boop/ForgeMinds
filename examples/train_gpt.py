"""torchrun --nproc_per_node=N examples/train_gpt.py --preset tiny [--moe] [--dp-shard N] [--tp N] [--pp N] [--ep N] [--fp8] [--offload-opt]"""
import argparse
import forgemind as fm
from forgemind import ForgeConfig, GPT, GPTConfig, Trainer, SyntheticLM, moe_kwargs, cleanup

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--moe", action="store_true")
ap.add_argument("--dp-replicate", type=int, default=1)
ap.add_argument("--dp-shard", type=int, default=1)
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--pp", type=int, default=1)
ap.add_argument("--ep", type=int, default=1)
ap.add_argument("--quant", default="none", choices=["none", "fp8", "int8"])
ap.add_argument("--offload-opt", action="store_true")
ap.add_argument("--offload-act", action="store_true")
ap.add_argument("--offload-experts", action="store_true")
ap.add_argument("--ac", default="none")
ap.add_argument("--backend", default="native", choices=["native", "deepspeed"])
ap.add_argument("--zero", type=int, default=2)
ap.add_argument("--data", default=None, help="flat uint16 token file; default = synthetic")
a = ap.parse_args()

cfg = ForgeConfig()
cfg.parallel.__dict__.update(dp_replicate=a.dp_replicate, dp_shard=a.dp_shard, tp=a.tp, pp=a.pp, ep=a.ep,
                             backend=a.backend, zero_stage=a.zero)
cfg.quant.mode = a.quant
cfg.offload.optimizer, cfg.offload.activations, cfg.offload.experts = a.offload_opt, a.offload_act, a.offload_experts
cfg.train.__dict__.update(max_steps=a.steps, warmup_steps=10, lr=1e-3, ac=a.ac, ckpt_every=0, log_every=10)

mc = GPTConfig(vocab_size=512, dim=256, n_layers=4, n_heads=8, n_kv_heads=4, max_seq=128,
               n_experts=8 if a.moe else 0, top_k=2)
ds = SyntheticLM(mc.vocab_size, mc.max_seq)
tr = Trainer(lambda ctx: GPT(mc, moe_kwargs(ctx, cfg) if a.moe else None), cfg, ds, ds, micro_batch=8)
tr.fit()
cleanup()
