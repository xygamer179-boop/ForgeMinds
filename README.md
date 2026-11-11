# ForgeMind

PyTorch training library: FSDP2/HSDP/DDP, tensor/pipeline/expert parallelism (3D), DeepSpeed ZeRO backend,
FP8 (Transformer Engine / torchao) and INT8 training, smart gradient/activation/expert offload, MoE, packed
Triton kernels (torch.compile-native), device autotuning, an asynchronous batch pipeline, and a full trainer.
It is an **extension, not a framework**: every kernel and block below is usable on its own,
inside an architecture you already have.

**Docs:** [documentation.md](documentation.md) — guide, API and config reference ·
[PERF.md](PERF.md) — performance/sync audit · [OFFLOAD.md](OFFLOAD.md) — offload design and rejected ideas.

## Install

```bash
pip install -e .                 # torch>=2.6, numpy, Python >= 3.10
pip install -e ".[kernels]"      # + triton (packed kernels)
pip install -e ".[torchao]"      # + torchao (INT8 training, AdamW8bit)
pip install -e ".[te]"           # + transformer-engine (FP8)
pip install -e ".[deepspeed]"    # + deepspeed (ZeRO backend)
pip install -e ".[test]"         # + pytest
```

Everything optional is imported lazily: `fm.kernels.available()` is `False` on a CPU box and the
kernels fall back to torch with identical numerics.

## Quick checks (from the repo root)

```bash
torchrun --nproc_per_node=4 examples/train_gpt.py --moe --dp-replicate 4 --ep 2
torchrun --nproc_per_node=4 examples/train_gpt.py --tp 2 --dp-shard 2
pytest tests/               # 218 tests: trainer, parallel, kernels, topk/softmax, forge, runtime, autotune, offload, muon
torchrun --standalone --nproc_per_node=2 tests/mp_check.py ep_equiv   # also ddp, fsdp, pp, tp, prefetch, ...
```

## Trainer: the full stack from one config

```python
import forgemind as fm

cfg = fm.ForgeConfig()
cfg.train.__dict__.update(max_steps=5000, warmup_steps=200, lr=3e-4, grad_accum=4, bf16=True,
                          log_every=10, eval_every=500, ckpt_every=500, compile=True, ac="selective")
cfg.parallel.__dict__.update(dp_shard=8)                       # FSDP2 across 8 GPUs

mc = fm.GPTConfig(vocab_size=32000, dim=1024, n_layers=24, n_heads=16, n_kv_heads=4, max_seq=2048)
trainer = fm.Trainer(lambda ctx: fm.GPT(mc), cfg,
                     train_ds=fm.TokenFile("train.bin", seq=2048),
                     eval_ds=fm.TokenFile("val.bin", seq=2048),
                     micro_batch=4, peak_flops=989e12)
history = trainer.fit()          # logs, evals, checkpoints, resumable; returns per-step metrics
trainer.close()
```

The model factory receives the `DistContext`, so it can size itself per rank. A dataset is any
`torch.utils.data.Dataset` returning `{"input_ids": ..., "labels": ...}`. `train_step()` returns
`{"step", "loss", "lr", "grad_norm", "dt", "skipped"}`, plus `tok_s` (needs `model.cfg.max_seq`),
`mfu` and `mem_gb` when they apply — see
[documentation.md §5.1](documentation.md#51-train-a-gpt-with-trainer).

## Your own model, two lines

`Forge` is the two-line façade: any `nn.Module`, any `DataLoader`, and it keeps the
guarantees this library is built on (one host sync per step, device-side clipping, accumulation, DDP).

```python
import forgemind as fm

forge = fm.Forge(model, optimizer, loss_fn=criterion, precision="bf16", accum=4)
forge.fit(loader, max_steps=1000)

# or keep your own loop:
loss = criterion(model(**batch), batch["labels"])
forge.backward(loss)
forge.step()          # ForgeState on an optimizer step, None while accumulating
```

Multi-process is one decorator away (it spawns, or defers to an existing `torchrun`):

```python
@fm.launch(nproc=4)
def main():
    fm.Forge(build_model(), build_opt(), precision="bf16").fit(loader)
main()                # module-level function; works unchanged under torchrun
```

Mesh: world = pp * dp_replicate * dp_shard * tp; `ep` divides the DP world.

## Your loop, with the runtime primitives

When you keep the loop yourself, the machinery is plain functions on `fm` -- collectives,
seeding, checkpoints. There is no driver class to learn, because ForgeMind is not a port of
Lightning; `Forge` already owns device movement, precision, DDP and the one-sync-per-step
discipline for the two-line case:

```python
import forgemind as fm

fm.seed_everything(1234)                    # torch + python + numpy (+ PYTHONHASHSEED with workers=True)

@fm.launch(nproc=2)                        # spawns; a no-op under an existing torchrun
def main():
    for epoch in range(epochs):
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)   # DistributedSampler: reshuffle per epoch
        for batch in loader:
            ...                            # your forward / backward / optimizer.step
        grads = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
        fm.all_reduce(grads, reduce_op="mean")   # world 1: returns `grads` untouched
        fm.barrier()
    forge.save("ckpt.pt")                  # rank 0 writes; the other ranks wait at the barrier
```

The collectives are `fm.all_reduce` (`mean|sum|max|min`), `fm.broadcast`, `fm.all_gather`
(concatenated on dim 0) and `fm.barrier` -- every one a **passthrough at world 1** (no
collective call at all), returning device tensors so you decide when to pay for a host
sync. `forge.save` guards itself: under DDP every rank holds identical weights, so letting
all of them write the same file is a race for zero benefit. Checkpoints are plain
`torch.save` with DDP unwrapped (no `module.` prefix); sharded samplers are the standard
`torch.utils.data.DistributedSampler`, not an injected wrapper.

Three shapes, one library: **`Trainer`** owns the whole loop, **`Forge`** is the two-line
façade, and the primitives above are for the loop you write yourself. Verified
single-process and as a real 2-process DDP run (`tests/test_runtime.py`,
`tests/forge_mp_check.py`: bit-identical replicas, `fm.all_gather` agreement, rank-0
checkpoint restored on both ranks).

## Bring your own architecture

Nothing requires `GPT`, `Trainer` or `Forge`. Take one op, one block, or fuse one module of your
own model.

```python
import torch, torch.nn as nn
import forgemind as fm
from forgemind import nn as fmnn

y = fm.kernels.rms_norm(x, w, eps=1e-6)        # == F.rms_norm(x, (d,), w, eps)
y = fm.kernels.swiglu(gate, up)                # == F.silu(gate) * up
xn, y = fm.kernels.add_rms_norm(x, delta, w)   # residual + norm in one pass

class MyBlock(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = fmnn.RMSNorm(dim)          # drop-in for nn.RMSNorm (same `weight`, same state_dict)
        self.mlp = fmnn.SwiGLU(dim, hidden)    # w2(silu(w1 x) * w3 x), fused epilogue

    def forward(self, h):
        return h + self.mlp(self.norm(h))

fm.kernels.patch(model.norm)                   # ...or fuse a module you already wrote
fm.kernels.patch(model.mlp, split=True)        #     (kind="norm"/"mlp" if the name says nothing)
```

| you want | use | notes |
|---|---|---|
| one fused op | `fm.kernels.rms_norm` / `swiglu` / `swiglu_split` / `add_rms_norm` / `qk_norm_rope` | differentiable, dtype-preserving, compile/export-native, torch expression when Triton is absent |
| one block | `fmnn.RMSNorm(dim)` / `fmnn.SwiGLU(dim, hidden)` | plain `nn.Module`s; `RMSNorm` is `nn.RMSNorm`-compatible, `SwiGLU(split=True)` fuses gate\|up into one GEMM |
| your own module | `fm.kernels.patch(m, split=False, kind=None)` | returns `True` if recognised, idempotent, preserves biases, works on torch's classes too |
| the whole tree | `fm.kernels.enable(model)` | what `Trainer` does when `fused_ops=True` |

`Trainer` accepts any factory as long as `model(tokens, labels=None) -> (ce, aux)`; DDP/FSDP2,
accumulation, AC, `torch.compile`, offload, async batches and checkpointing all apply unchanged
(TP/PP/EP still need this repo's model). Guide:
[documentation.md §5.3](documentation.md#53-bring-your-own-architecture).

## Asynchronous batches

The loader runs on a background thread, so while step *k* computes, batch *k+1* is being assembled
**and** copied to the device on a side stream. Training and eval both use it; nothing to configure.

```python
trainer = Trainer(model_fn, cfg, dataset, micro_batch=8)   # on by default (auto)
for step in range(1000):
    trainer.train_step()
print(trainer.feed.report())     # {'wait_ms_per_batch': 0.04, 'stalled': 0, 'produce_ms': ..., 'depth': 2}

# your own loop, same machinery (any loader, any container):
for x, y in fm.async_loader(DataLoader(ds, 64, pin_memory=True), "cuda", depth=2):
    loss = criterion(model(x), y)
```

Pipeline depth is `TrainConfig.prefetch_depth` (2 = double buffer). `batch_thread=False` restores
strictly serial order and RNG for bit-exact comparisons, and `Trainer.close()` / `Forge.fit()`
always stop the thread (no leaked workers, no leaked CUDA streams). On a CPU-only box this is
GIL-bound and therefore a no-op — the win is measured for the GPU-shaped case, see
[PERF.md §8](PERF.md).

Cost model: a step costs `max(load, step)` instead of `load + step` — measured 40.5 → 23.0 ms/batch
(0.57x) when the step doesn't need the interpreter. `fm.async_loader`, `fm.to_device` and the
dataset/sampler contract are documented in
[documentation.md §5.5](documentation.md#55-datasets-and-loading).

## Parallelism

| goal | `ParallelConfig` |
|---|---|
| replicas (DDP) | `dp_replicate=N` |
| sharded (FSDP2) | `dp_shard=N` |
| hybrid sharding (HSDP) | `dp_replicate=R, dp_shard=S` |
| tensor / pipeline / expert parallel | `tp=T`, `pp=P` (+`pp_schedule`), `ep=E` |
| 3D | `pp=P, tp=T, dp_shard=S` |
| DeepSpeed ZeRO | `backend="deepspeed", zero_stage=2/3` |

Every layout is checked against a single-process reference:

```bash
torchrun --standalone --nproc_per_node=2 tests/mp_check.py ddp     # ddp, ddp_accum_offload, fsdp, hsdp,
torchrun --standalone --nproc_per_node=2 tests/mp_check.py tp      # fsdp_accum, ep_ddp, pp, tp, tp_fsdp,
torchrun --standalone --nproc_per_node=4 tests/mp_check.py 3d      # pp_fsdp, 3d, 3d_full, ep_equiv,
torchrun --standalone --nproc_per_node=2 tests/mp_check.py prefetch # tp_equiv, pp_equiv, prefetch
python tests/forge_mp_check.py                                     # Forge + DDP + rank-0 ckpt, 2 processes
python tests/muon_mp_check.py                                      # Muon + DDP across 2 processes
```

## Packed kernels

`fm.kernels` has fused Triton kernels for the ops that are usually *several* eager kernels:
`rms_norm`, `add_rms_norm` (residual+norm in one pass), `qk_norm_rope` (per-head norm + RoPE),
`swiglu` / `swiglu_split` (fused gate|up projection epilogue), a distillation KL that
computes the logit gradient in the forward pass, `newton_schulz` (the quintic
orthogonalization that gives `fm.Muon` its spectral-norm steps), `topk` (FlashTopK-inspired
exact selection: strided-sample threshold, canonical value-desc/index-asc order, smallest
indices win ties) and `softmax` (one kernel: max-sub, exp, sum, divide). Every one is a drop-in for the eager expression,
degrades to torch on CPU/MPS, and is verified against the torch reference by running the *actual
kernels* through `TRITON_INTERPRET=1` on CPU (`tests/test_kernels.py`, 66 numeric checks).

```python
import forgemind as fm
y = fm.kernels.rms_norm(x, w, 1e-6)         # == F.rms_norm(x, (d,), w, eps)
fm.kernels.enable(model)                    # or patch RMSNorm/SwiGLU modules in place
loss = fm.kernels.kl_distill_loss(hs, ht, w, teacher_head)   # teacher_head = teacher's lm_head (callable)
```

They are `torch.compile`-native: eager calls go through `autograd.Function`, but while a graph is
traced each kernel is a `torch.library.custom_op` with a fake implementation and a registered
backward, so it stays **inside** the graph (`fullgraph=True`) and `torch.export` captures it —
which is what an AOTInductor serving stack needs. Same launch either way, bit-identical numbers.

`newton_schulz`, `topk` and `softmax` are *not* `custom_op`s: the first runs under `no_grad`
inside `Optimizer.step`, the other two substitute `torch.topk` / `aten::softmax` while a graph
is traced (one aten node, `fullgraph=True` holds), while eager still takes the Triton path.
`stats()` counts every fallback, including `topk`'s `k > 512` limit.

```bash
python -m forgemind.cli kbench --check   # per-op A/B/C vs inductor on your GPU; exit 3 if we lose
```

## Muon optimizer

```python
opt = fm.Muon(model.parameters(), lr=0.02)     # matrices -> Muon, vectors -> AdamW
```

`fm.Muon` is one optimizer for the whole model: `ndim >= 2` parameters (Linear, conv
filters, ...) get momentum orthogonalized by batched quintic Newton-Schulz — same-shape
matrices share one batched call per step — and everything else gets decoupled AdamW.
Explicit param-group dicts with `use_muon=True/False` are honored (per-group `lr`,
`weight_decay`, `momentum`, `nesterov`, `ns_steps`, `ns_dtype`, `betas`, `eps`). Multi-GPU
shards each shape bucket across ranks and combines the disjoint shares with a single
`all_reduce`; single-process runs make **no** `torch.distributed` calls at all. Unlike the
reference implementation it neither mutates `p.grad` in the nesterov blend nor scales the
momentum buffer every step — six bugs were found in the reference file and fixed, each with
a regression test (see [PERF.md section 11](PERF.md)).

Verified by `tests/test_muon.py` (reference parity, singular values driven to 1,
batched==per-matrix, the bug regressions, compile/tracing dispatch) and a 2-process DDP
check (`tests/muon_mp_check.py`: bit-identical replicas == a single-process run). The NS
kernel joins the speed gate via `python -m forgemind.cli kbench --ops newton_schulz`.

## Autotuning

```python
from forgemind import autotune as at
at.apply_fast_defaults()                    # tf32, cudnn.benchmark, expandable segments
size, ms = at.tune_batch_size(lambda s: step(s))     # doubles until it stops paying off / OOMs
report = at.autotune(forge, loader, budget_s=60)     # precision / fused / accum, printed as a table
print(report.table())
```

Batch sizes are scored by **per-sample** cost, not step time (a bigger batch is usually the same
wall time, so step time alone picks the smallest batch that fits). OOM is caught and reported as
data. `python -m forgemind.autotune --kernels` sweeps the Triton launch configs on your GPU.

## Performance

One host↔device synchronisation per optimizer step (down from five), zero `torch.compile` graph breaks on
the dense path, and RoPE tables built once instead of per layer per forward. See **[PERF.md](PERF.md)** for
the full audit (every stall site, every graph break, every bug, with measurements).

```bash
python -m forgemind.cli.bench --syncs --moe --experts 8        # tokens/s, p50/p90, MFU, host syncs/step
python -m forgemind.cli.bench --compile --quant fp8 --qbackend auto
python -m forgemind.cli.bench --profile /tmp/trace             # top ops + Perfetto/TensorBoard trace
python -m forgemind.cli.matrix --quant none fp8 --backends auto te   # comparison table
```

```python
from forgemind.perf import count_host_syncs, host_syncs

with count_host_syncs() as c:      # audit your own step: counts + call sites, own vs torch-internal
    trainer.train_step()
print(c.report())                  # "1 host sync(s) (1 in forgemind, 0 inside torch)" + call sites
print(trainer.feed.report())       # batch pipeline: wait_ms_per_batch, stalled, stall_ratio, produce_ms
```

| knob | default | effect |
|---|---|---|
| `TrainConfig.fused_ops` / `fused_split_mlp` | off / off | swap RMSNorm/SwiGLU for the packed kernels; `split` also fuses the gate and up projections into one GEMM |
| `TrainConfig.compile` / `compile_mode` / `compile_backend` / `compile_fullgraph` | off / `default` / `inductor` / off | compiles each transformer block (caches are warmed first); `fullgraph=True` refuses to fall back |
| `TrainConfig.ac` / `ac_preserve_rng` | `none` / `False` | full or selective checkpointing; leave RNG preservation off unless you use dropout |
| `TrainConfig.prefetch` / `workers` | `True` / `0` | overlap the next batch's H2D copy on a side stream; `workers>0` adds a DataLoader pool |
| `TrainConfig.batch_thread` / `prefetch_depth` | auto (workers==0 or CUDA) / 2 | run the **loader** on a background thread so batch *k+1* is assembled (and copied) while step *k* computes; `False` restores bit-exact serial order |
| `OffloadConfig.optimizer` / `smart` / `gpu_watermark` | off / on / 0 | fp32 master + Adam states in pinned host RAM, grads streamed on a side stream |
| `OffloadConfig.pipeline_chunks` | 4 | overlap the grad DMA of chunk *k+1* with the host Adam update of chunk *k*; `opt.report()` shows the split (see [OFFLOAD.md](OFFLOAD.md)) |
| `OffloadConfig.activations` / `act_pool` | off / on | spill large saved activations to pinned host memory, recycling buffers |
| `OffloadConfig.experts` | off | keep MoE experts in pinned host RAM, stream only the active ones |
| `GPTConfig.expert_grouped_mm` | `True` | `torch._grouped_mm` expert GEMMs (CUDA+bf16): removes the last host read from the MoE dispatch |
| `GPTConfig.rope_high_precision` | `False` | compute the rotation in fp32 (slower, marginally more accurate) |
| `ParallelConfig.ddp_static_graph` | `False` | skip DDP's per-iteration unused-parameter scan |
| `TrainConfig.bf16` | `True` | autocast for forward/loss; `fp16` + GradScaler is available through `Forge(precision="fp16")` |
| `QuantConfig.mode` / `backend` / `optimizer_8bit` | `none` / `auto` / off | FP8 (TE/torchao) or INT8 (torchao) training, optional 8-bit AdamW |

## Where to read what

| you want | read |
|---|---|
| a working example for each API | [documentation.md §3–5](documentation.md#3-the-two-minute-tour) |
| using one kernel or block in your own model | [documentation.md §5.3](documentation.md#53-bring-your-own-architecture) |
| every function and its signature | [documentation.md §6](documentation.md#6-api-reference) |
| every config field and its default | [documentation.md §7](documentation.md#7-configuration-reference) |
| the kernel API | [documentation.md §8](documentation.md#8-kernels-reference) |
| writing your own loop (collectives, seed, checkpoints) | [documentation.md §5.4](documentation.md#54-your-own-loop) |
| a Muon optimizer | [documentation.md §5.14](documentation.md#514-muon-one-optimizer-for-the-whole-model) |
| CLI flags (`bench`, `matrix`, `autotune`, examples, checks) | [documentation.md §9](documentation.md#9-command-line-reference) |
| what is guaranteed, and what is *not* verified | [documentation.md §10–11](documentation.md#10-guarantees-and-invariants) |
| an error message you just got | [documentation.md §12](documentation.md#12-troubleshooting) |
| what each source file does | [documentation.md §13](documentation.md#13-repository-map) |

## Troubleshooting (the five that actually happen)

| symptom | fix |
|---|---|
| `BatchStream: the loader produced no batches` | dataset smaller than `micro_batch` with `drop_last=True` — shrink the batch or grow the dataset |
| `feed.report()['stall_ratio']` is high | the loader is the bottleneck: raise `workers`, then `prefetch_depth` |
| `pp*dp_replicate*dp_shard*tp = N != world M` | the mesh must multiply to the world size; `dp_shard=-1` fills the remainder |
| `micro_batch must be divisible by pp_microbatches` | make `micro_batch % pp_microbatches == 0` |
| `compile_fullgraph=True is not possible for this MoE block` | that dispatch needs a host read; use CUDA+bf16 `torch._grouped_mm`, or leave `fullgraph=False` |
| `no trainable parameters` | everything is frozen — typical with `quant.mode="int8_wo"`; add adapters or unfreeze |

The rest (quantization backends, offload+PP, empty `evaluate()`, OOM tuning) is in
[documentation.md §12](documentation.md#12-troubleshooting).

## Status

Verified on CPU/gloo (see `tests/`, 218 tests): dense/MoE training, grad accumulation, checkpoint+resume,
offload optimizer equivalence (and the chunked/pipelined path), AC modes, DDP/FSDP2/HSDP/TP/PP/3D, EP
equivalence, `torch.compile` (`aot_eager`) with `fullgraph=True` and eager-gradient equivalence, the
packed Triton kernels run through Triton's CPU interpreter (forward *and* backward vs torch) **and**
through TileLens (formerly Triton-Viz): symbolic memory sanitizer, race detector and profiler across
all eight packed kernels (see [PERF.md §10](PERF.md)) plus `newton_schulz`, `topk` and `softmax`
(TileLens scenarios + interpreter parity), the `Forge` façade and the runtime primitives (2-process
DDP launch covering `fm.all_gather` and rank-0 save/load), `fm.Muon` (reference parity, the six fixed reference bugs as regression tests,
a 2-process DDP check), autotuning, the asynchronous batch pipeline (order,
overlap, rank alignment, resume, thread/worker cleanup), and a one-host-sync-per-step budget.

Untested here (no accelerator): CUDA offload streams and the pipelined overlap they enable, async H2D
overlap and pinned-buffer reuse in the batch pipeline, kernel launches on real hardware (correctness is
verified through the interpreter, *speed* is not), fused kernels under `torch.compile` on CUDA,
`torch._grouped_mm` dispatch, TE/torchao FP8/INT8 training, DeepSpeed engine, NCCL.

Limits: EP+FSDP2 unsupported; MoE aux loss not propagated across PP stages; experts not TP'd or quantized;
OffloadedAdamW not for FSDP/PP; no eval under PP.
