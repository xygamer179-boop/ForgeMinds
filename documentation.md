# ForgeMind documentation

Reference and guide for **forgemind 0.7.0** — a PyTorch training library with FSDP2/HSDP/DDP,
tensor/pipeline/expert parallelism, DeepSpeed ZeRO, FP8/INT8 quantized training, gradient /
activation / expert offload, MoE, packed Triton kernels, device autotuning, an asynchronous batch
pipeline, and two levels of trainer API.

Every number and every claim in this file was either measured in this repository or is marked as
**unverified**; see [§11](#11-verified-vs-not-verified). The measurements themselves, the stall
sites and the bug list live in [PERF.md](PERF.md); the offload design (including what was rejected
and why) is in [OFFLOAD.md](OFFLOAD.md).

---

## Contents

1. [What ForgeMind is](#1-what-forgemind-is)
2. [Install and requirements](#2-install-and-requirements)
3. [The two-minute tour](#3-the-two-minute-tour)
4. [Concepts](#4-concepts)
   - [4.1 The mesh](#41-the-mesh)
   - [4.2 One host sync per optimizer step](#42-one-host-sync-per-optimizer-step)
   - [4.3 The data pipeline](#43-the-data-pipeline)
   - [4.4 What the logs tell you](#44-what-the-logs-tell-you)
5. [Guides](#5-guides)
   - [5.1 Train a GPT with `Trainer`](#51-train-a-gpt-with-trainer)
   - [5.2 Bring your own model with `Forge`](#52-bring-your-own-model-with-forge)
   - [5.3 Bring your own architecture](#53-bring-your-own-architecture)
   - [5.4 Your own loop](#54-your-own-loop)
   - [5.5 Datasets and loading](#55-datasets-and-loading)
   - [5.6 Multi-GPU: layouts and how to prove them](#56-multi-gpu-layouts-and-how-to-prove-them)
   - [5.7 Memory: activation checkpointing and offload](#57-memory-activation-checkpointing-and-offload)
   - [5.8 Precision and quantized training](#58-precision-and-quantized-training)
   - [5.9 Mixture of experts](#59-mixture-of-experts)
   - [5.10 Speed: compile, fused kernels, autotune, prefetch](#510-speed-compile-fused-kernels-autotune-prefetch)
   - [5.11 Checkpoint and resume](#511-checkpoint-and-resume)
   - [5.12 Evaluation](#512-evaluation)
   - [5.13 Measuring](#513-measuring)
   - [5.14 Muon: one optimizer for the whole model](#514-muon-one-optimizer-for-the-whole-model)
6. [API reference](#6-api-reference)
7. [Configuration reference](#7-configuration-reference)
8. [Kernels reference](#8-kernels-reference)
9. [Command line reference](#9-command-line-reference)
10. [Guarantees and invariants](#10-guarantees-and-invariants)
11. [Verified vs not verified](#11-verified-vs-not-verified)
12. [Troubleshooting](#12-troubleshooting)
13. [Repository map](#13-repository-map)

---

## 1. What ForgeMind is

Three ways in, same guarantees underneath:

| Entry point | For | You get |
|---|---|---|
| `fm.Trainer` | the `GPT`/`MoE` model in this repo, or any `nn.Module` that speaks the `(input_ids, labels) -> (ce, aux)` contract | the whole stack configured from one `ForgeConfig`: parallelism, quantization, offload, AC, compile, checkpoints, eval, logging |
| `fm.Forge` | any `nn.Module`, any `DataLoader`, any loss | `fit()` / `step()` / `backward()`, with the one-sync-per-step and clipping discipline |

What the library deliberately optimises for: **the accelerator should never wait on the host.**
That shows up as three concrete budgets you can check yourself ([§5.13](#513-measuring)):

* exactly **one host↔device synchronisation per optimizer step** (grad-norm + loss + skip flag
  stacked into a single `.tolist()`);
* **no `torch.compile` graph breaks** on the dense path (six deliberate ones inside MoE dispatch on
  devices without `torch._grouped_mm`);
* **no batch-shaped stall**: the loader runs on a background thread and the H2D copy runs on a side
  stream, so a step costs `max(load, step)` rather than `load + step`.

---

## 2. Install and requirements

```bash
pip install -e .                 # core: torch>=2.6, numpy (Python >= 3.10)
pip install -e ".[kernels]"      # + triton, for the packed kernels
pip install -e ".[torchao]"      # + torchao (INT8 training, AdamW8bit)
pip install -e ".[te]"           # + transformer-engine (FP8)
pip install -e ".[deepspeed]"    # + deepspeed (ZeRO backend)
pip install -e ".[test]"         # + pytest
```

Nothing optional is imported at `import forgemind` time: `fm.kernels` imports triton lazily, and
`fm.kernels.available()` is `False` on a CPU-only box (the kernels then fall back to torch and stay
numerically equivalent). Quantization raises a clear error if its backend is missing
([§12](#12-troubleshooting)).

---

## 3. The two-minute tour

```python
import torch, forgemind as fm

cfg = fm.ForgeConfig()
cfg.train.__dict__.update(max_steps=200, lr=3e-4, bf16=True, ckpt_every=100)
cfg.parallel.__dict__.update(dp_replicate=torch.cuda.device_count())   # plain DDP

mc = fm.GPTConfig(vocab_size=32000, dim=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq=1024)
trainer = fm.Trainer(lambda ctx: fm.GPT(mc), cfg, train_ds, eval_ds=val_ds, micro_batch=8)
history = trainer.fit()

print(history[-1]["loss"], history[-1].get("tok_s"))    # tok_s needs model.cfg.max_seq
trainer.close()
```

That is: one config object, one model factory (it receives the `DistContext`, so a model can size
itself per rank), one dataset, and `fit()` does the loop, the logging, the schedule, the eval, the
checkpoints and the background batch pipeline.

Prefer your own model and your own loop? See [§5.2](#52-bring-your-own-model-with-forge).

---

## 4. Concepts

### 4.1 The mesh

`ParallelConfig` describes one mesh, and the world size must equal the product:

```
world_size = pp * dp_replicate * dp_shard * tp
```

* `dp_replicate` — DDP degree (replicas, gradients all-reduced).
* `dp_shard` — FSDP2 degree (parameters sharded, `-1` means "fill what is left").
* `tp` — tensor parallelism (`apply_tp` rewrites the module graph).
* `pp` — pipeline stages (the model must implement `split_stage(rank, n_stages)`).
* `ep` — experts carved out of the DP world; must divide `dp_replicate * dp_shard`.

Rank bookkeeping lives in `DistContext`: `rank/world/local_rank/device`, plus `dp_rank/dp_world`,
`pp_rank`, `tp_rank`, `ep_rank` and the matching process groups. `init_distributed(pc)` builds it
from the environment (`RANK`/`WORLD_SIZE`/`LOCAL_RANK`), so the same script works under `torchrun`
or plain `python`.

Two rules that follow from the mesh:

* **Data is sharded by `dp_rank` only** (`ResumableSampler`), so the `tp`/`pp` ranks that form one
  replica see identical batches — which is what makes TP and PP numerically checkable.
* **The DP group is what averages.** `dp_shard`/`dp_replicate` both count as "data parallel" for
  loss reduction and gradient norms.

### 4.2 One host sync per optimizer step

A "host sync" is any `.item()`, `.tolist()`, `float(tensor)`, `.cpu()`, `torch.cuda.synchronize()`
or equivalent that forces the accelerator to drain. Each one turns an asynchronous pipeline into a
synchronous one and is usually worth 5-20% of step time.

The step is built so that exactly one read happens, at the end, with everything the host still
needs stacked into a single tensor:

```python
flag_v, gn_v, lv = torch.stack([flag.to(torch.float32), gn.to(torch.float32), loss]).tolist()
```

`grad_norm` is computed **on the device** (FSDP/TP/EP aware), clipping is a device-side coefficient
(`clamp(max=1)` then `_foreach_mul_`), the skip decision is an `all_min_int` collective rather than
a host read, and the loss is reduced over DP/PP with collectives. Audit any step with:

```python
from forgemind.perf import count_host_syncs

with count_host_syncs(capture_stack=True) as c:
    trainer.train_step()
print(c.count, c.own, c.external)    # total, inside forgemind, inside torch
print(c.report())                    # per-call-site breakdown
```

`tests/test_sync_audit.py` asserts this budget for the dense, accumulation, spike-guard, MoE, AC and
offload paths, so it cannot silently regress.

### 4.3 The data pipeline

Three stages, each able to overlap with the one before it:

```
dataset __getitem__  ──▶  collate  ──▶  to(device, non_blocking)  ──▶  forward/backward
        (producer thread)              (side CUDA stream, event)      (compute stream)
```

* `ResumableSampler` shards by DP rank and can rewind mid-epoch (`skip`), which is what makes
  `save()`/`maybe_resume()` exact.
* `build_loader(ds, micro_batch, ctx, seed, workers)` returns `(DataLoader, sampler)` with
  `drop_last=True`, `pin_memory` on CUDA, and `persistent_workers=True` + `prefetch_factor=2` when
  `workers > 0` (so epochs never re-fork on the prefetch thread).
* `BatchStream` is the producer/consumer: one FIFO queue of depth `prefetch_depth`, per-item
  `(epoch, index, epoch_len)` bookkeeping so epoch counters stay right while the producer runs
  ahead, error forwarding, and `close()` that stops the thread *and* the worker pool.
* `fm.async_loader(loader, device, depth)` is the same machinery for an arbitrary loader, yielding
  tuples or nested containers — usable outside this library entirely.

Cost model: a step costs `max(load, step)` instead of `load + step` (measured 40.5 → 23.0 ms/batch
for the GIL-free case, [PERF.md §8](PERF.md)). On a CPU-only box the thread is neutral because both
sides need the interpreter lock, which is why the default is *auto* and `batch_thread=False`
restores strictly serial order.

### 4.4 What the logs tell you

One line at startup, then one per `log_every` steps:

```
[forgemind] world=8 mesh(pp,dp_rep,dp_shard,tp)=(1,4,1,1) ep=2 backend=native quant=none/none params(local)=354.2M device=cuda:3
step      1 | loss 4.1672 | lr 1.00e-03 | gnorm 1.24 | 16,494 tok/s | 31 ms mfu 12.4% mem 7.1G
```

`mesh(...)` is the actual layout (a mistyped flag shows up here immediately), `params(local)` counts
this rank only, and `tok_s` is per-optimizer-step throughput across the DP world (the column is
omitted for a model with no `cfg.max_seq`). `mfu` appears when
`peak_flops` is passed to `Trainer`; `mem_gb` is `cuda.max_memory_allocated()`.

---

## 5. Guides

### 5.1 Train a GPT with `Trainer`

```python
import forgemind as fm

cfg = fm.ForgeConfig()
cfg.train.__dict__.update(
    max_steps=5000, warmup_steps=200, lr=3e-4, weight_decay=0.1,
    grad_accum=4, bf16=True, grad_clip=1.0,
    log_every=10, eval_every=500, eval_steps=20, ckpt_every=500, ckpt_dir="ckpts",
    compile=True, ac="selective", fused_ops=True,
)
cfg.parallel.__dict__.update(dp_shard=8)                  # FSDP2 across 8 GPUs

mc = fm.GPTConfig(vocab_size=32000, dim=1024, n_layers=24, n_heads=16, n_kv_heads=4, max_seq=2048)
trainer = fm.Trainer(lambda ctx: fm.GPT(mc), cfg,
                     train_ds=fm.TokenFile("train.bin", seq=2048),
                     eval_ds=fm.TokenFile("val.bin", seq=2048),
                     micro_batch=4, peak_flops=989e12)
history = trainer.fit()          # blocks until max_steps; returns the per-step metric list
trainer.close()
```

Run it under `torchrun --nproc_per_node=8 your_script.py`. The model factory receives the
`DistContext`, so a model can be sized per rank (`ctx.pp_rank`, `ctx.tp_rank`, ...).

**`train_step()` returns a dict** — you can drive the loop yourself:

| key | meaning |
|---|---|
| `step` | 1-based step number |
| `loss` | mean cross-entropy over the accumulation window (DP/PP reduced) |
| `lr` | learning rate used |
| `grad_norm` | norm *before* clipping (EP/PP aware) |
| `tok_s` | tokens/second: `micro_batch * seq * grad_accum * dp_world / dt`; only present when the model exposes `cfg.max_seq` |
| `dt` | seconds for this step |
| `skipped` | cumulative count of skipped updates (non-finite or spike) |
| `mfu`, `mem_gb` | only when `peak_flops` is given / on CUDA |

`fit()` adds: logging every `log_every`, `evaluate()` every `eval_every`, callbacks
(`cb(trainer, metrics)`), and `save()` every `ckpt_every`.

### 5.2 Bring your own model with `Forge`

```python
import torch, torch.nn as nn, forgemind as fm

model = MyModel()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

forge = fm.Forge(
    model, opt,
    loss_fn=lambda out, batch: torch.nn.functional.cross_entropy(out, batch["labels"]),
    precision="bf16",        # "fp32" | "bf16" | "fp16" (fp16 gets a GradScaler)
    accum=4,                 # gradient accumulation
    clip=1.0,                # device-side global-norm clipping
    compile=True,            # torch.compile the module
    async_batches=True,      # background loader thread + side-stream copies
)

forge.fit(loader, max_steps=2000)
forge.save("ckpt.pt")
```

How a batch becomes a forward call (or pass `forward_fn(model, batch)` for anything else):

| your batch is | the call is |
|---|---|
| `dict` | `model(**batch)` |
| `tuple` / `list` | `model(*batch)` |
| `Tensor` | `model(tensor)` |

Loss: `loss_fn(out, batch)` wins; otherwise a tensor output *is* the loss, a `dict` with `"loss"` is
used, and a tuple/list uses its first tensor.

Nothing in `Forge` assumes a transformer, a tokenizer, or a schedule: `max_steps`, `warmup` and
`min_lr_ratio` only shape the built-in cosine LR (pass `scheduler=` or `schedule=None` to take over).

### 5.3 Bring your own architecture

ForgeMind is an extension, not a framework. `GPT`, `Trainer` and `Forge` are conveniences layered
on top of parts you can take individually — and nothing in this section needs any of them.

**One kernel, as a function.** The lowest level takes tensors and returns tensors:

```python
import forgemind as fm

y = fm.kernels.rms_norm(x, w, eps=1e-6)          # == F.rms_norm(x, (d,), w, eps)
y = fm.kernels.swiglu(gate, up)                  # == F.silu(gate) * up
y = fm.kernels.swiglu_split(x)                   # == silu(x[..., :h]) * x[..., h:]
xn, y = fm.kernels.add_rms_norm(x, delta, w)     # residual + norm in one pass: (x+delta, normed)
y = fm.kernels.qk_norm_rope(q, w, cos, sin)      # per-head RMSNorm + RoPE, x is [B,T,H,D]
loss = fm.kernels.kl_distill_loss(hs, ht, w, teacher_head)
```

Every one is differentiable (exact gradients, `gradcheck`ed for the blocks), keeps the input
dtype, and **is the torch expression when Triton is not available** — so the same line of code
runs on CPU, MPS and CUDA.

Two dispatch paths, both your friend: **eager** goes through `torch.autograd.Function` (the
cheapest way to call a kernel from Python), **while `torch.compile` traces** the same launch is
a `torch.library.custom_op` with a fake implementation and a registered backward. That is why a
compiled model keeps the kernel inside one graph, `fullgraph=True` passes, and `torch.export`
can capture the op instead of hitting an opaque `autograd.Function` (measurements in
[PERF.md §9](PERF.md)). Both paths call the same launch, so the numbers are bit-identical —
`tests/test_kernel_ops.py` asserts exactly that.

**One block, as a module.** `fm.nn` holds the two building blocks most architectures need, as
ordinary `nn.Module`s you can put inside your own model:

```python
from forgemind import nn as fmnn
import torch, torch.nn as nn

class MyBlock(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = fmnn.RMSNorm(dim)             # fused RMSNorm
        self.mlp = fmnn.SwiGLU(dim, hidden)       # w2(silu(w1 x) * w3 x), fused epilogue

    def forward(self, h):
        h = h + self.mlp(self.norm(h))
        return h
```

* `fmnn.RMSNorm` is a drop-in for `torch.nn.RMSNorm`: same `weight` parameter, so an existing
  checkpoint loads straight into it and back out. It also keeps torch's `eps=None` semantics
  (`finfo(dtype).eps`) instead of quietly substituting `1e-6`.
* `fmnn.RMSNorm.forward_with_residual(x, residual)` returns `(normed, x + residual)`, writing the
  residual stream once — the pre-norm fusion that saves a full round trip of the hidden state.
* `fmnn.SwiGLU(..., split=True)` (or `.to_split()`) concatenates the gate and up projections into
  one `gate_up` GEMM and uses the fused `swiglu_split` epilogue: one launch and one weight stream
  instead of two. It is numerically identical; only the parameter names change, which is why it is
  opt-in.

**Your own module, fused in place.** When you already have the class you want, fuse it rather than
rewrite it:

```python
fm.kernels.patch(model.norm)                 # name looks like a norm, 1-D weight + eps
fm.kernels.patch(model.scale, kind="norm")   # my class is called `Scale`: say what it is
fm.kernels.patch(model.mlp)                  # w1/w2/w3 Linears -> fused SwiGLU epilogue
fm.kernels.patch(model.mlp, split=True)      # ...and one gate|up GEMM
```

| call | what it fuses | refuses when |
|---|---|---|
| `patch(m)` / `patch(m, kind="norm")` | `forward` becomes `rms_norm(x, m.weight, m.eps)` | no 1-D `weight` Parameter (`kind="norm"` raises `TypeError`) |
| `patch(m)` / `patch(m, kind="mlp")` | `w2(silu(w1 x) * w3 x)` with the packed epilogue | no `w1`/`w2`/`w3` Linear layout |
| `patch(m, split=True)` | also rebuilds `w1\|w3` into one `gate_up` Linear | the two biases differ, or the shapes do not match (falls back to the epilogue-only fuse) |

`patch` returns `True` when it recognised the module and `False` when it did not, so a wrong guess
fails at the call site: `assert fm.kernels.patch(model.norm)`. It is idempotent, it works on
**torch's own** classes (`patch(nn.RMSNorm(768))`), and it preserves biases — a fused gate|up GEMM
carries both.

**Training it.** `Trainer` takes any model factory; it never looks inside the model. The contract:

| you provide | required? | used for |
|---|---|---|
| `model(tokens, labels=None) -> (ce, aux)` | yes | the training step; `aux` is an extra loss or `None` |
| `model.cfg.max_seq` | no | tokens/s |
| `model.prepare(device, dtype, seq)` | no | warming lazily built tables before `torch.compile` |
| `model.split_stage(rank, n_stages)` | only for `pp > 1` | pipeline partitioning |
| `model.flops_per_token(seq)` | no | MFU when `peak_flops` is given |

```python
class MyLM(nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([MyBlock(dim, 4 * dim) for _ in range(4)])
        self.head = nn.Linear(dim, vocab)

    def forward(self, tokens, labels=None):
        h = self.emb(tokens)
        for b in self.blocks:
            h = b(h)
        logits = self.head(h)
        if labels is None:
            return logits, None
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)), None

trainer = fm.Trainer(lambda ctx: MyLM(32000, 768), cfg, train_ds, micro_batch=8)
trainer.fit()
```

DDP/FSDP2, gradient accumulation, activation checkpointing, `torch.compile`, offload, the async
batch pipeline, checkpoints and the one-sync-per-step budget all apply unchanged. What still needs
this repo's model is **TP** (`apply_tp` rewrites the attention/MLP naming it knows), **PP** (needs
`split_stage`) and **EP** (needs `MoE`).

**Same numbers, both paths.** Every block delegates to `fm.kernels`, which falls back to torch when
Triton or CUDA is missing:

```python
fm.kernels.available()             # False on CPU (the torch expression is used, not an error)
fm.kernels.set_backend("fallback") # force the torch path to A/B your own change
fm.kernels.stats()                 # which path ran, and how often
fm.kernels.use_custom_ops(True)    # force the torch.library path (export, AOTI, debugging)
```

### 5.4 Your own loop

```python
forge = fm.Forge(model, opt, precision="bf16", accum=8, quiet=True)

for batch in fm.async_loader(loader, forge.device, depth=2):
    out = model(**batch)                      # your forward, your shapes
    loss = criterion(out, batch["labels"])
    forge.backward(loss)                      # scaled for accumulation
    state = forge.step()                      # None while accumulating
    if state is not None:
        print(state.step, state.loss, state.grad_norm)
```

`forge.step()` returns `ForgeState` (a `dict` with attribute access) on a completed optimizer step
and `None` mid-accumulation — that is the contract `tests/forge_mp_check.py` relies on. `step(batch)`
is the same thing with the forward/backward folded in. `forge.zero_grad()`, `forge.setup(model, opt)`
and `forge.count_syncs()` round out the loop surface.

Multi-process:

```python
@fm.launch(nproc=4)
def main():
    ...                                    # runs on 4 processes, WORLD_SIZE already set up

if __name__ == "__main__":
    main()
```

`launch` spawns `nproc` processes when it owns the world, and simply runs the function when
`WORLD_SIZE > 1` is already in the environment, so the same script works under `torchrun`.
The decorated function must be **module-level** (it has to be picklable).

**Runtime primitives — plain functions, no driver class.** The machinery a hand-written loop
needs lives on `fm` itself (it used to sit on a Lightning-Fabric-shaped class; ForgeMind is
not a port of Lightning, so the class is gone and its *useful parts* are embedded):

```python
fm.seed_everything(1234, workers=False)     # torch + python + numpy; returns the seed
fm.all_reduce(t, reduce_op="mean")          # in place: mean | sum | max | min
fm.broadcast(t, src=0)                      # from rank src
fm.all_gather(t)                            # dim-0 concatenation of every rank's copy
fm.barrier()
```

Every collective is a **passthrough at world 1** — no collective call at all — and returns
the device tensor it was given, so the caller decides when to pay for a host sync (the
one-sync-per-step budget is yours to keep). Under `fm.launch`, DDP wraps the model in the
world the launcher built; a sharded `DataLoader` is the standard
`torch.utils.data.DistributedSampler` (call `set_epoch` yourself each epoch), not an
injected wrapper.

`forge.save(path)` is rank-0: rank 0 writes, the other ranks wait at the barrier (under DDP
every rank holds identical weights, so an unguarded multi-rank write is a race for zero
benefit); `forge.load` reads it back on every rank with DDP unwrapped — no `module.`
prefix. `tests/test_runtime.py` pins the world-1 behaviour of all of it, and
`tests/forge_mp_check.py` runs the real thing: two processes, bit-identical replicas,
`fm.all_gather` agreement and the rank-0 checkpoint restored on both ranks.

### 5.5 Datasets and loading

The `Trainer` contract is a `torch.utils.data.Dataset` whose `__getitem__` returns a dict with
`input_ids` and `labels`:

```python
class MyDataset(torch.utils.data.Dataset):
    def __len__(self): return len(self.tokens) - self.seq
    def __getitem__(self, i):
        x = self.tokens[i : i + self.seq + 1]
        return {"input_ids": x[:-1], "labels": x[1:]}
```

Two ready-made ones: `fm.SyntheticLM(vocab, seq, n)` (deterministic, learnable, used by the tests and
the examples — loss drops fast, so it is the right dataset for a smoke test) and
`fm.TokenFile(path, seq, dtype="uint16")` (a `np.memmap` over a flat token file; call `.close()` —
or let the destructor do it — to release the mapping).

Rolling your own loader for `Forge` is free-form: anything iterable of tensors/dicts works, and
`fm.async_loader` will move it to the device for you.

**Resume is exact** because the sampler can be rewound: `Trainer.save()` records
`(step, epoch, batches_in_epoch)`, and `maybe_resume()` rewinds both the sampler (`skip`) and the
stream, so the batch sequence after a resume is the batch sequence of an uninterrupted run — asserted
to 5 decimals in `tests/test_async_batches.py`.

### 5.6 Multi-GPU: layouts and how to prove them

| goal | `ParallelConfig` |
|---|---|
| replicas (DDP) | `dp_replicate=N` |
| sharded (FSDP2 / ZeRO-3-like) | `dp_shard=N` |
| hybrid sharding (HSDP) | `dp_replicate=R, dp_shard=S` |
| tensor parallel | `tp=T` |
| pipeline parallel | `pp=P` (+ `pp_schedule="1f1b"` or `"gpipe"`, `pp_microbatches`) |
| expert parallel | `ep=E` (MoE models; divides the DP world) |
| 3D | `pp=P, tp=T, dp_shard=S` |
| DeepSpeed ZeRO | `backend="deepspeed", zero_stage=2/3`, `zero_offload_optimizer/param` |

Every combination is verified the same way — a two-or-more-process script that compares against a
single-process reference:

```bash
torchrun --standalone --nproc_per_node=2 tests/mp_check.py ddp
torchrun --standalone --nproc_per_node=2 tests/mp_check.py fsdp
torchrun --standalone --nproc_per_node=2 tests/mp_check.py pp
torchrun --standalone --nproc_per_node=4 tests/mp_check.py 3d
torchrun --standalone --nproc_per_node=2 tests/mp_check.py ep_equiv
torchrun --standalone --nproc_per_node=2 tests/mp_check.py prefetch     # rank alignment of the batch stream
python tests/forge_mp_check.py                                         # Forge + launch, 2 processes
```

Full scenario list: `ddp`, `ddp_accum_offload`, `fsdp`, `hsdp`, `fsdp_accum`, `ep_ddp`, `ep_ddp4`,
`pp`, `tp`, `tp_fsdp`, `pp_fsdp`, `3d`, `3d_full`, `ep_equiv`, `tp_equiv`, `pp_equiv`, `prefetch`.
Each prints a `RESULT ...` line and exits non-zero on mismatch.

### 5.7 Memory: activation checkpointing and offload

```python
cfg.train.ac = "full"              # "none" | "full" | "selective"
cfg.train.ac_preserve_rng = False  # True only if the model has dropout inside checkpointed blocks

cfg.offload.optimizer = True       # fp32 master weights + Adam moments in pinned host RAM
cfg.offload.pipeline_chunks = 4    # overlap grad D2H of chunk k+1 with the host Adam of chunk k
cfg.offload.activations = True     # spill large saved activations to pinned host RAM
cfg.offload.act_layers = 0.5       # only the first half of the blocks
cfg.offload.experts = True         # stream only the *active* experts for MoE
```

`OffloadedAdamW` is numerically identical to `AdamW` (bit-exact against the serial path, chunk counts
1/2/3/5) and reports where its time goes:

```python
trainer.opt.report()
```

Rules that will bite if ignored: optimizer offload is not supported together with pipeline
parallelism (and not with FSDP/PP — see [OFFLOAD.md](OFFLOAD.md)), and `act_layers` counts blocks
from the front, so `0.5` means "the first half".

### 5.8 Precision and quantized training

```python
cfg.train.bf16 = True              # autocast context around forward/loss (CPU included, for tests)
cfg.quant.mode = "fp8"             # "none" | "fp8" | "int8" | "int8_wo"
cfg.quant.backend = "auto"         # "auto" | "te" | "torchao"
cfg.quant.optimizer_8bit = True    # torchao AdamW8bit
```

* `fp8` needs CUDA + sm≥8.9 and a backend (Transformer Engine or torchao); otherwise you get an
  explicit error telling you which one is missing and which mode to use instead.
* `int8` is quantized *training* through torchao; `int8_wo` freezes the base weights
  (weight-only) and is for adapters — with everything frozen you get
  `no trainable parameters`, by design.
* `QuantConfig.exclude` keeps `lm_head`, `gate`, `router`, `tok_emb` out of quantization.

### 5.9 Mixture of experts

```python
mc = fm.GPTConfig(vocab_size=32000, dim=1024, n_layers=12, n_heads=16, n_kv_heads=4, max_seq=2048,
                  n_experts=8, top_k=2, moe_every=2, n_shared_experts=1,
                  expert_grouped_mm=True, aux_coef=0.01, z_coef=0.001)

cfg.parallel.ep = 2                             # expert parallel
cfg.offload.experts = True                      # keep expert weights in host RAM
trainer = fm.Trainer(lambda ctx: fm.GPT(mc, fm.moe_kwargs(ctx, cfg)), cfg, ds, micro_batch=8)
```

`fm.moe_kwargs(ctx, cfg)` is the glue that passes the EP group, EP rank and the expert-offload flag —
use it or wire `fm.MoE(...)` yourself. `expert_grouped_mm=True` uses `torch._grouped_mm` (CUDA+bf16)
and removes the last host read from the dispatch; on other devices each MoE block keeps **one
deliberate graph break**, which `compile_fullgraph=True` will refuse (with an explanation).

Known limits: experts are not TP'd or quantized, the aux loss is not propagated across pipeline
stages (a warning is logged at construction), and EP+FSDP2 is unsupported.

### 5.10 Speed: compile, fused kernels, autotune, prefetch

```python
cfg.train.compile = True
cfg.train.compile_mode = "default"        # "default" | "reduce-overhead" | "max-autotune"
cfg.train.compile_backend = "inductor"    # "aot_eager" for boxes without a C++ toolchain
cfg.train.compile_fullgraph = False       # True = refuse to fall back on a graph break

cfg.train.fused_ops = True                # packed Triton RMSNorm/SwiGLU, when available
cfg.train.fused_split_mlp = True          # also fuse w1|w3 into one GEMM (changes the module)

cfg.train.prefetch = True                 # copy the next batch on a side stream
cfg.train.prefetch_depth = 2              # batches in flight
cfg.train.batch_thread = None             # None = auto, True/False = force
cfg.train.workers = 4                     # DataLoader worker processes
```

Autotuning tries configurations on your hardware instead of guessing:

```python
from forgemind import autotune as at

at.apply_fast_defaults()                                  # tf32, cudnn.benchmark, expandable segments
size, ms = at.tune_batch_size(lambda s: step(s))          # per-sample scoring, OOM caught
report = at.autotune(forge, loader, budget_s=60)          # precision / accum / fused / compile
print(report.table())
```

Batches are scored by **per-sample** cost: a bigger micro-batch usually costs the same wall time, so
step-time scoring picks the smallest batch that fits. `python -m forgemind.autotune --kernels` sweeps
Triton launch configurations instead.

Two selection/normalization kernels sit outside the module patching, as plain callables:

```python
values, indices = fm.kernels.topk(x, 64)     # last dim; torch.topk drop-in, k <= 512 fused
y = fm.kernels.softmax(logits)               # last dim; torch.softmax drop-in
```

`topk` follows FlashTopK's structure in portable Triton — strided-sample threshold (never
above the true k-th), exact pivot inside `{x > t}`, canonical **value-descending /
index-ascending** output so ties at the k-th score deterministically keep the smallest
indices (which `torch.topk` does not promise), NaN treated as largest, scatter backward.
`softmax` fuses max-sub + exp + sum + divide with the closed-form backward. Both fall back
to torch while `torch.compile` traces (one aten node, `fullgraph=True` holds), on CPU/MPS,
for unsupported dtypes, and — for `topk` — for `k > 512`; every fallback is counted in
`fm.kernels.stats()`.

### 5.11 Checkpoint and resume

`Trainer` (sharded, distributed-checkpoint format):

```python
trainer.save()            # ckpt_dir/step_0001234/{dcp/, meta_rank*.json}; keeps the last 3
trainer.maybe_resume()    # reads ckpt_dir/latest and restores model+optim+step+position
```

* Restores model, optimizer, `step`, `loss_ema` and the exact data position; `resume=True`
  (default) calls it at construction.
* `keep_last=3` step directories are retained; older ones are deleted on rank 0 after a barrier.
* With `OffloadedAdamW` the optimizer state is written per rank next to the DCP directory
  (`offopt_rank{N}.pt`), because its state lives in host memory.
* DeepSpeed uses `engine.save_checkpoint/load_checkpoint` with the same directory layout.

`Forge` (single-file):

```python
forge.save("ckpt.pt")     # model, optimizer, scheduler, step, ema_loss, RNG state
forge.load("ckpt.pt")
```

### 5.12 Evaluation

```python
cfg.train.eval_every = 500      # call evaluate() every N steps from fit()
cfg.train.eval_steps = 20       # batches per eval
trainer = fm.Trainer(..., eval_ds=val_ds)
print(trainer.evaluate())       # mean CE, or None
```

`evaluate()` runs on the same async pipeline as training, under `torch.no_grad()`, and reduces the
loss over the DP group. It returns `None` (with no exception) when there is no `eval_ds`, when
`pp > 1`, or on the DeepSpeed backend — those combinations are not implemented. `Forge.evaluate()`
takes `max_batches=` and computes the mean of your `loss_fn`.

### 5.13 Measuring

```bash
python -m forgemind.cli.bench --syncs --moe --experts 8        # tok/s, p50/p90, MFU, syncs/step
python -m forgemind.cli.bench --compile --quant fp8 --qbackend auto
python -m forgemind.cli.bench --profile /tmp/trace             # top ops + Perfetto/TensorBoard trace
python -m forgemind.cli.matrix --quant none fp8 --backends auto te   # comparison table
```

In-process:

```python
from forgemind.perf import count_host_syncs, host_syncs

with count_host_syncs() as c:
    trainer.train_step()
print(c.report())                 # count + per-call-site breakdown

print(trainer.feed.report())      # the batch pipeline: wait_ms_per_batch, stalled, stall_ratio,
                                  # produce_ms, depth, queued, threaded
```

`host_syncs()` returns a running total (no stack capture, lowest overhead — usable inside a step).
`feed.report()['wait_ms_per_batch']` near zero means the producer is keeping up; a `stall_ratio` well
above zero means the loader is the bottleneck and `workers` / `prefetch_depth` are the knobs.

### 5.14 Muon: one optimizer for the whole model

```python
opt = fm.Muon(model.parameters(), lr=0.02, weight_decay=0.01)
for batch in loader:
    opt.zero_grad()
    loss = model(batch).sum()
    loss.backward()
    opt.step()
```

One optimizer, two update rules, decided by shape: parameters with `ndim >= 2` (Linear
weights, conv filters, ...) get **Muon** -- momentum orthogonalized by quintic Newton-Schulz
(`fm.kernels.newton_schulz`), i.e. steepest descent under the spectral norm; everything else
(`ndim < 2`: biases, norms, ...) gets a decoupled **AdamW** with `betas` / `eps` / `adam_lr`.
Same-shape matrices are stacked and orthogonalized in one batched NS call, so a layer of
identically shaped matrices costs one call per shape bucket, not one per matrix.

* explicit control: pass param-group dicts with `use_muon=True/False` (a group without the
  flag is auto-split by `ndim`, keeping its hyperparameters); per-group `lr`,
  `weight_decay`, `momentum`, `nesterov`, `ns_steps`, `ns_dtype` are honored;
* `ns_dtype=torch.bfloat16` by default (the canonical fast choice); pass
  `torch.float32` for maximum accuracy -- the kernels accept either;
* multi-GPU: each shape bucket is computed on a disjoint `rank::world` share and combined
  with **one** `all_reduce`; with `world == 1` (or no `torch.distributed` at all) there is
  not a single collective call;
* the update never touches `p.grad` or the momentum buffer in place (the reference Muon
  file this was ported from did both -- the six found bugs are listed in
  [PERF.md section 11](PERF.md) and each has a regression test in `tests/test_muon.py`).

Verified: `tests/test_muon.py` (24 tests: orthogonality, reference parity, the bug
regressions, batched-vs-single equality, compile/tracing dispatch) plus a 2-process DDP
check (`tests/muon_mp_check.py`) that requires bit-identical replicas *and* equality with a
single-process run. `python -m forgemind.cli kbench --ops newton_schulz` includes the NS
kernel in the speed gate.

---

## 6. API reference

Everything below is importable from the package root; `fm.kernels`, `fm.autotune`, `fm.perf`,
`fm.data`, `fm.parallel`, `fm.quant`, `fm.loss` and `fm.models` are also reachable as submodules.

### Models and experts

```python
fm.GPT(c: GPTConfig, moe_kwargs: dict | None = None)
fm.GPTConfig(vocab_size=32000, dim=512, n_layers=8, n_heads=8, n_kv_heads=4, ffn_mult=8/3,
             max_seq=2048, rope_theta=10000.0, rope_high_precision=False, tie_embeddings=False,
             loss_chunk=4096, n_experts=0, top_k=2, moe_every=1, n_shared_experts=0,
             expert_hidden_mult=1.0, expert_grouped_mm=True, aux_coef=0.01, z_coef=0.001,
             mixer="attn")
fm.MoE(dim, hidden, n_experts, top_k, n_shared=0, ep_size=1, ep_group=None, ep_rank=0,
       offload_experts=False, grouped_mm=False, aux_coef=0.01, z_coef=0.001)
fm.moe_kwargs(ctx, cfg) -> dict
```

`GPT(tokens, labels=None) -> (ce, aux)`; `aux` is the MoE aux/z loss or `None`.
`GPT.split_stage(pp_rank, n_stages)` is what pipeline parallelism calls; `GPT.prepare(device, dtype,
seq)` warms the lazily built RoPE/grouped-GEMM tables before `torch.compile` traces the model.

### Composable blocks (`fm.nn`)

The "bring your own architecture" surface: ordinary modules, kernel-backed where possible.

```python
fm.nn.RMSNorm(dim, eps=1e-6, elementwise_affine=True, device=None, dtype=None)
fm.nn.SwiGLU(dim, hidden=None, bias=False, split=False, device=None, dtype=None)
```

| member | description |
|---|---|
| `RMSNorm.forward(x)` | `x / sqrt(mean(x**2) + eps) * weight` over the last dim; `state_dict` compatible with `nn.RMSNorm` (parameter `weight`); output dtype is the input dtype |
| `RMSNorm.forward_with_residual(x, residual=None)` | returns `(normed, x + residual)`, writing the residual stream once |
| `SwiGLU.forward(x)` | `w2(silu(w1(x)) * w3(x))`; with `split=True`, `w2(swiglu_split(gate_up(x)))` |
| `SwiGLU.to_split(inplace=True)` | rebuild `w1\|w3` into one `gate_up` Linear; same output, new parameter names |

The last-dim check is unconditional: a wrong width raises a message instead of a broadcast error
three frames deep in torch. `bias=True` is honoured by both variants; `to_split` raises if the two
projections disagree about having a bias.

### Trainer

```python
fm.Trainer(model_fn: Callable[[DistContext], nn.Module], cfg: ForgeConfig, train_ds: Dataset,
           eval_ds: Dataset | None = None, micro_batch: int = 8, callbacks=(),
           peak_flops: float | None = None)
```

| member | description |
|---|---|
| `.fit() -> list[dict]` | run to `cfg.train.max_steps`; logging, eval, callbacks, checkpoints |
| `.train_step() -> dict` | one optimizer step (accumulation inside); see [§5.1](#51-train-a-gpt-with-trainer) |
| `.evaluate() -> float \| None` | mean CE over `eval_steps` batches, on the async pipeline |
| `.save(keep_last=3)` / `.maybe_resume()` | DCP checkpoint / resume from `ckpt_dir/latest` |
| `.lr_at(step) -> float` | warmup + cosine schedule value |
| `.close()` | stop the background batch prefetcher (idempotent; `__del__` also does it) |
| `.feed` | the `BatchStream`: `.report()`, `.epoch`, `.batches`, `.resume()`, `.close()` |
| `.epoch` / `.batches_in_epoch` | properties proxying the stream (exact mid-epoch position) |
| `.raw` / `.model` / `.opt` / `.ctx` / `.history` / `.skipped` | model, wrapped model, optimizer, context, metrics, skip count |

### Forge

```python
fm.Forge(model, optimizer=None, *, loss_fn=None, device=None, precision="bf16", accum=1, clip=1.0,
         lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8, warmup=0, max_steps=1000,
         min_lr_ratio=0.1, schedule=None, scheduler=None, compile=False, compile_backend="inductor",
         compile_mode="default", forward_fn=None, offload="none", fused_ops=False,
         fused_split_mlp=False, async_batches=True, prefetch_depth=2, seed=1234, log_every=10,
         skip_nonfinite=True, skip_spike_factor=0.0, callbacks=None, quiet=False)
```

| member | description |
|---|---|
| `.fit(loader, max_steps=None, epochs=1)` | the loop, with the async prefetcher and a `finally` that stops it |
| `.step(batch=None) -> ForgeState \| None` | one micro-batch; `None` while accumulating |
| `.backward(loss)` | accumulation-scaled backward |
| `.evaluate(loader, max_batches=None) -> float` | mean loss, no grad, prefetched |
| `.save(path)` / `.load(path)` | single-file checkpoint (model, opt, sched, step, RNG); rank 0 writes, the others wait at the barrier |
| `.setup(model, opt)` | returns `(model, opt)` — the stored ones for anything omitted |
| `.zero_grad(set_to_none=True)`, `.count_syncs()` | passthroughs |
| `.history` / `.step_count` / `.device` / `.raw_model` | state |

```python
fm.launch(fn=None, nproc=1, port=29517)
```
Decorator (or direct call). Spawns `nproc` processes unless `WORLD_SIZE > 1` is already set, in
which case it runs `fn` once inside the existing world. The function must be module-level.

### Runtime primitives (module-level)

```python
fm.seed_everything(seed=1234, workers=False) -> int
fm.all_reduce(tensor, reduce_op="mean") -> tensor      # mean | sum | max | min, in place
fm.broadcast(tensor, src=0) -> tensor
fm.all_gather(tensor) -> tensor                        # dim-0 concatenation of world copies
fm.barrier() -> None
```

Plain functions on the package (no driver class): at world 1 — or with no process group at
all — every collective is a passthrough that returns the *same tensor* without calling into
`torch.distributed`, and `all_reduce` validates `reduce_op` before the passthrough so typos
fail fast on every rank. `seed_everything` seeds torch/python/numpy (plus
`PYTHONHASHSEED` when `workers=True`) and returns the seed. Covered by
`tests/test_runtime.py` (world-1 contract) and `tests/forge_mp_check.py` (real 2-process
`all_gather` + rank-0 `forge.save`/`forge.load`).

### Muon (optimizer)

```python
fm.Muon(params, lr=0.02, weight_decay=0.0, momentum=0.95, nesterov=True, ns_steps=5,
        ns_dtype=torch.bfloat16, betas=(0.9, 0.95), eps=1e-10, adam_lr=None)
```

| detail | behavior |
|---|---|
| `params` | iterable of tensors (auto-split: `ndim >= 2` -> Muon, else AdamW) or param-group dicts; generators are fine |
| group dicts | `use_muon=True/False` selects the update; without the flag the group is split by `ndim` keeping its hyperparameters; unknown keys pass through |
| state | Muon: `momentum_buffer`; AdamW: `exp_avg`, `exp_avg_sq`, `step` (per parameter) |
| `step(closure=None)` | `@torch.no_grad()`; bucketed batched NS per shape; one `all_reduce` per bucket only when `dist` is initialized with `world > 1` |
| decay | decoupled on both paths: `p *= 1 - lr * weight_decay`, then the step |
| `add_param_group` | same auto-split/validation as the constructor |
| rejects | a `use_muon=True` group containing `ndim < 2` parameters (ValueError saying what to do instead) |

### Data

```python
fm.SyntheticLM(vocab, seq, n=4096, seed=0)          # deterministic, learnable toy dataset
fm.TokenFile(path, seq, dtype="uint16")             # memmap over a flat token file; .close()
fm.ResumableSampler(n, dp_rank, dp_world, seed=0, shuffle=True)
fm.build_loader(ds, micro_batch, ctx, seed=0, workers=0, shuffle=True,
                persistent_workers=None) -> (DataLoader, ResumableSampler)
fm.BatchStream(loader, sampler, device, prefetch=True, depth=2, threaded=None)
fm.async_loader(loader, device, depth=2) -> iterator     # generic prefetcher, any loader
fm.to_device(obj, device, non_blocking=False)            # recursive tensor mover
```

`BatchStream`: `.next() -> (input_ids, labels)`, `.resume(epoch, batches)`, `.close()`,
`.report() -> dict` with `batches, wait_ms, wait_ms_per_batch, max_wait_ms, stalled, stall_ratio,
produce_ms, threaded, depth, queued`. It is an iterator, a context manager, and raises a clear
`RuntimeError` if the loader can never produce a batch (typically `drop_last=True` with a dataset
smaller than `micro_batch`).

### Distributed

```python
fm.init_distributed(pc: ParallelConfig, backend=None) -> DistContext
fm.DistContext      # rank, world, local_rank, device, pc, mesh, dp_rank/dp_world, pp_rank, tp_rank,
                    # ep_rank, and the process groups (dp/dp_shard/pp/tp/ep/edp)
fm.cleanup()
```

### Performance and offload

```python
fm.count_host_syncs(capture_stack=True, count_cpu=False)   # context manager; .count/.own/.external/.report()
fm.host_syncs() -> int                                     # running total
fm.OffloadedAdamW(param_groups, lr, betas=(0.9, 0.95), eps=1e-8, cfg: OffloadConfig = None,
                  overlap=True)                            # .report(), .state_dict()
fm.autotune.apply_fast_defaults(device=None, allow_tf32=True, allocator=True) -> dict
fm.autotune.tune_batch_size(step_fn, candidates=(1,2,4,...), *, warmup=1, repeats=3,
                            min_gain=0.02, footprint=None, report=None) -> (int, float)
fm.autotune.autotune(target, loader=None, *, build=None, budget_s=60.0, apply=False,
                     precision=True, accum=True, compiled=False, fused=True, report=None) -> Report
```

Lower-level (not re-exported at the root, import from `forgemind.parallel.native`):
`apply_tp(model, ctx, plan=None)`, `apply_ddp(model, ctx)`, `apply_fsdp(model, ctx, bf16=True,
cpu_offload=False, reshard=True)`, `expert_params(model)`, `sync_expert_grads(model, ctx, params=None)`.

---

## 7. Configuration reference

`fm.ForgeConfig` bundles four dataclasses. Every field, every default:

### 7.1 `ParallelConfig`

| field | default | meaning |
|---|---|---|
| `backend` | `"native"` | `"native"` (torch FSDP2/DDP/TP/PP) or `"deepspeed"` (ZeRO + AutoTP/PP) |
| `dp_replicate` | `1` | DDP degree (HSDP replicate dim when `dp_shard > 1`) |
| `dp_shard` | `1` | FSDP2 shard degree; `-1` fills the remainder |
| `tp` | `1` | tensor parallel degree |
| `pp` | `1` | pipeline stages |
| `ep` | `1` | expert parallel degree; must divide `dp_replicate * dp_shard` |
| `pp_microbatches` | `4` | micro-batches per pipeline step (`micro_batch % pp_microbatches == 0`) |
| `pp_schedule` | `"1f1b"` | `"1f1b"` or `"gpipe"` |
| `fsdp_reshard_after_forward` | `True` | FSDP2 `reshard_after_forward` |
| `fsdp_cpu_offload` | `False` | FSDP2 CPU offload of parameters |
| `ddp_static_graph` | `False` | skip DDP's per-iteration unused-parameter scan (only if the graph never changes) |
| `init_timeout_s` | `1800` | process-group init timeout |
| `zero_stage` | `2` | DeepSpeed ZeRO stage |
| `zero_offload_optimizer` / `zero_offload_param` | `False` / `False` | DeepSpeed CPU offload |
| `ds_extra` | `{}` | extra kwargs for the DeepSpeed config |

### 7.2 `QuantConfig`

| field | default | meaning |
|---|---|---|
| `mode` | `"none"` | `"none"`, `"fp8"`, `"int8"` (quantized training), `"int8_wo"` (frozen weight-only) |
| `backend` | `"auto"` | `"auto"`, `"te"`, `"torchao"` |
| `exclude` | `("lm_head", "gate", "router", "tok_emb")` | module names kept in full precision |
| `optimizer_8bit` | `False` | torchao `AdamW8bit` |

### 7.3 `OffloadConfig`

| field | default | meaning |
|---|---|---|
| `optimizer` | `False` | fp32 master + Adam states in pinned host RAM, grads streamed |
| `smart` | `True` | only stream large grads / only under GPU memory pressure |
| `min_numel` | `262144` | "large grad" threshold for `smart` |
| `gpu_watermark` | `0.0` | `0` = always offload eligible grads; `0.8` = only above 80% GPU memory |
| `activations` | `False` | spill large saved activations to pinned host memory |
| `act_min_bytes` | `1048576` | size threshold for activation spilling |
| `act_layers` | `1.0` | fraction of blocks (from the front) whose activations are offloaded |
| `act_pool` | `True` | recycle pinned buffers instead of allocating per saved tensor |
| `pipeline_chunks` | `4` | overlap grad D2H of chunk *k+1* with the host Adam update of chunk *k* |
| `experts` | `False` | keep MoE expert weights in host RAM, stream only the active ones |

### 7.4 `TrainConfig`

| field | default | meaning |
|---|---|---|
| `max_steps` | `1000` | optimizer steps to run |
| `grad_accum` | `1` | micro-batches per optimizer step |
| `lr` / `min_lr_ratio` / `warmup_steps` | `3e-4` / `0.1` / `100` | warmup + cosine schedule |
| `weight_decay` / `betas` / `eps` | `0.1` / `(0.9, 0.95)` / `1e-8` | AdamW; decay only on `ndim >= 2` params |
| `grad_clip` | `1.0` | global-norm clip, computed and applied on the device (`0` disables) |
| `bf16` | `True` | autocast dtype for forward/loss |
| `compile` / `compile_mode` / `compile_backend` / `compile_fullgraph` | `False` / `"default"` / `"inductor"` / `False` | per-block `torch.compile` |
| `ac` / `ac_preserve_rng` | `"none"` / `False` | activation checkpointing mode and RNG preservation |
| `seed` | `1234` | seeds torch, the sampler and the EP weight init |
| `log_every` | `10` | steps between log lines |
| `eval_every` / `eval_steps` | `0` / `20` | eval cadence and batches per eval |
| `ckpt_every` / `ckpt_dir` / `resume` | `0` / `"checkpoints"` / `True` | checkpoint cadence, directory, auto-resume |
| `skip_spike_factor` | `0.0` | skip an update when `loss > factor * EMA(loss)` |
| `skip_nonfinite` | `True` | skip updates on non-finite loss/grad-norm |
| `prefetch` | `True` | move the next batch to the device on a side stream |
| `prefetch_depth` | `2` | batches in flight in the prefetch queue |
| `batch_thread` | `None` | run the loader on a background thread; `None` = auto (`workers == 0` or CUDA) |
| `workers` | `0` | DataLoader worker processes |
| `fused_ops` / `fused_split_mlp` | `False` / `False` | packed Triton norm/MLP; `split` fuses the gate\|up GEMM |

### 7.5 `GPTConfig`

| field | default | meaning |
|---|---|---|
| `vocab_size` / `dim` / `n_layers` | `32000` / `512` / `8` | model size |
| `n_heads` / `n_kv_heads` | `8` / `4` | attention heads, KV heads (GQA when smaller) |
| `ffn_mult` | `8/3` | MLP hidden = `dim * ffn_mult` |
| `max_seq` | `2048` | sequence length (RoPE tables are sized for it) |
| `rope_theta` / `rope_high_precision` | `10000.0` / `False` | RoPE base; fp32 rotation is slower but more accurate |
| `tie_embeddings` | `False` | share input/output embeddings |
| `loss_chunk` | `4096` | rows per chunk in the chunked cross-entropy (memory knob, not a numerics knob) |
| `n_experts` / `top_k` / `moe_every` / `n_shared_experts` | `0` / `2` / `1` / `0` | MoE layout |
| `expert_hidden_mult` | `1.0` | expert hidden = `hidden * expert_hidden_mult` |
| `expert_grouped_mm` | `True` | use `torch._grouped_mm` when available (CUDA+bf16) |
| `aux_coef` / `z_coef` | `0.01` / `0.001` | load-balance and router z-loss coefficients |
| `mixer` | `"attn"` | attention variant |

---

## 8. Kernels reference

`fm.kernels` — packed Triton kernels, each a drop-in for the eager expression, each falling back to
torch transparently on CPU/MPS or when triton is missing.

```python
fm.kernels.rms_norm(x, w, eps=1e-6)                       # F.rms_norm(x, (d,), w, eps)
fm.kernels.add_rms_norm(x, delta, w, eps=1e-6)            # residual add + norm in one pass
fm.kernels.qk_norm_rope(x, w, cos, sin, eps=1e-6)         # per-head QK norm + RoPE
fm.kernels.swiglu(gate, up)                               # silu(gate) * up
fm.kernels.swiglu_split(x)                                # same, from one [.., 2h] buffer
fm.kernels.kl_distill_loss(hs, ht, w, teacher_head, chunk=4096)   # KL on logits, grad in the forward
#   hs: student hidden states, w: student lm_head weight, ht: teacher hidden states,
#   teacher_head: a callable (the teacher's lm_head) -- run under no_grad, never accumulates grads
fm.kernels.kl_div_logits(student, teacher, reduction="mean", grad_dtype=None)
fm.kernels.kl_div(student, teacher, reduction="mean", grad_dtype=None)
fm.kernels.newton_schulz(g, steps=5, dtype=None, fused=True)  # batched (..., m, n) -> same shape, Muon's polar step

fm.kernels.enable(model, norm=True, swiglu_mlp=True, split=False)  # patch a whole tree in place
fm.kernels.patch(module, split=False, kind=None) -> bool     # one module; False = not recognised
fm.kernels.available() -> bool
fm.kernels.set_backend("auto" | "fallback" | "triton")
fm.kernels.get_backend() / fm.kernels.stats()
fm.kernels.use_custom_ops(None | True | False) -> bool   # None = automatic (on while tracing)
```

Correctness, not just speed: the kernels had ten bugs fixed before they were wired in (wrong dtype
on the store, input mutation in the KL kernel, a non-power-of-two `tl.arange`, wrong strides in the
`swiglu_split` backward, `-inf` sentinel NaN on non-divisible vocab, and more — the table is in
[PERF.md §7.1](PERF.md)). They are verified by running the **actual kernels** through Triton's
interpreter on CPU:

```bash
TRITON_INTERPRET=1 python tests/kernel_interp_check.py     # 66 checks, forward and backward
```

`tests/test_kernels.py` drives the same comparisons through pytest (19 tests, including
non-contiguous inputs, `dim=1` RMSNorm, odd `head_dim`, bf16 gradients and `V=5000` streaming).

**`torch.compile` / `torch.export`.** Each kernel is registered as `torch.ops.forgemind.<name>`
(fake implementation + registered backward) and is used whenever `torch.compile` traces, so:

* `fullgraph=True` passes with the kernel in the graph (no break, no eager fallback);
* the backward is a second op (`forgemind::<name>_backward`), so AOTAutograd compiles it too — a raw
  launch inside a registered backward would run against FakeTensors during the joint trace and blow up;
* `torch.export` emits `forgemind.rms_norm.default` as a node, which is what AOTInductor serving
  stacks need. Second-order gradients are not supported (the Triton backward has no `grad_fn`), the
  same as before this change.

In eager the kernels still go through `torch.autograd.Function`: a custom op costs ~59 µs/call of
Python against ~24 µs for `Function.apply`, which would be a real regression for small tensors, so the
op path is entered only where that overhead is amortised — inside a compiled graph.

**`newton_schulz` is deliberately not a custom op.** It is an optimizer utility called under
`no_grad`, so there is nothing to differentiate; while `torch.compile` traces it, it simply runs
the pure-torch implementation — dynamo sees ordinary tensor ops (`fullgraph=True` holds) and
inductor fuses the elementwise chain from its side. Eager on CUDA (or under
`TRITON_INTERPRET=1`) keeps the three GEMMs per iteration in torch (cuBLAS, the benchmark) and
fuses the five elementwise ops into two Triton kernels (`B = b*A + c*A@A`, then
`X = a*X + B@X`): 10 elementwise launches per five-step NS instead of 25, same FLOPs — a
structural no-regression claim whose speed half is measured by
`kbench --ops newton_schulz` + `--check` on hardware. `fused=False` forces the torch path
(the A/B baseline); `set_backend("fallback")` and the `stats()` counters behave as for every
other kernel. Numerics: the quintic is evaluated as `b*A + c*(A@A)`, while the reference
source line `b*A + c*A @ A` parses left-to-right as `b*A + (c*A)@A` — mathematically
identical, a few bf16 ulps apart (measured ≤ 0.04 absolute on the output), pinned by tests
on both sides; the two dispatch paths agree to ≤ 1e-5 in fp32.

Speed is a separate question, and it is answered per op against the thing it has to beat,
`torch.compile`'s inductor output, on the hardware you actually train on:

```bash
python -m forgemind.cli kbench                       # eager vs inductor vs kernel, fwd+bwd
python -m forgemind.cli kbench --ops rms_norm --mode fwd
python -m forgemind.cli kbench --check --tolerance 0 # exit 3 if any kernel loses to inductor
```

`--check` is a gate, not a report: it fails the run when a kernel is slower than the baseline, so
"no GPU speed decrease" is a command rather than a claim. (It refuses to run on CPU, where the
kernels fall back and every column measures the same expression.)

---

## 9. Command line reference

### `python -m forgemind.cli.bench`

| flag | meaning |
|---|---|
| `--preset {tiny,small,medium}`, `--seq`, `--micro-batch` | model shape |
| `--moe`, `--experts N`, `--top-k K` | MoE |
| `--quant {none,fp8,int8,int8_wo}`, `--qbackend {auto,te,torchao}`, `--optimizer-8bit` | quantization |
| `--dp-replicate`, `--dp-shard`, `--tp`, `--pp`, `--ep`, `--backend {native,deepspeed}`, `--zero` | layout |
| `--offload-opt`, `--offload-act`, `--offload-experts`, `--ac {none,full,selective}` | memory |
| `--warmup`, `--steps`, `--peak-tflops` | run shape (MFU needs peak TFLOPS) |
| `--compile`, `--no-prefetch`, `--syncs` | speed / auditing |
| `--json FILE`, `--profile DIR`, `--profile-top N` | machine-readable output, profiler trace |

### `python -m forgemind.cli.matrix`

Runs the bench over several quantization configurations and prints a comparison table; a failing
combination is reported, not fatal.

```bash
python -m forgemind.cli.matrix --quant none fp8 --backends auto te --preset small --seq 1024
```

### `python -m forgemind.autotune`

`--kernels` sweeps Triton launch configurations on the current GPU and prints the winners.

### Examples and checks

```bash
torchrun --nproc_per_node=4 examples/train_gpt.py --moe --dp-replicate 4 --ep 2
python examples/train_gpt.py --help          # steps, moe, dp-*, tp, pp, ep, quant, offload-*, ac, backend, zero, data
python tests/forge_mp_check.py               # Forge + launch across 2 processes
pytest tests/                                # the whole suite
```

---

## 10. Guarantees and invariants

These are enforced by tests, not just intentions.

1. **One host sync per optimizer step.** `count_host_syncs` reports `own <= 1` per step for dense,
   accumulation, spike-guard, MoE, AC and offload paths (`tests/test_sync_audit.py`).
2. **Clipping happens on the device.** `_clip()` returns a 0-dim tensor; no python float enters the
   hot path.
3. **Ranks stay aligned.** `ResumableSampler` shards by `dp_rank`, so `tp`/`pp` ranks of a replica
   see identical batches; `mp_check.py prefetch` asserts a threaded stream is batch-for-batch
   `torch.equal` to the serial stream, on both ranks.
4. **Resume is exact.** Model, optimizer, step, loss EMA and data position all restore; the
   remaining batch sequence equals an uninterrupted run (`tests/test_async_batches.py`,
   `tests/test_edge_cases.py`).
5. **No leaks.** The prefetch thread, its queue and the DataLoader worker pool are released by
   `close()`, which is idempotent and also called from `__del__`; `fit`/`evaluate` close in a
   `finally` so an early stop cannot leave a producer blocked on a full queue.
6. **Numerical equivalence.** Offloaded Adam == AdamW (bit-exact); chunked pipeline == serial; AC on
   == AC off (allowing for RNG); `torch.compile` == eager gradients; TP/PP/EP match the
   single-process reference.
7. **Fallbacks are correct, not approximate.** Kernels, grouped GEMM, quantization and offload all
   degrade to a torch path with the same semantics, which is what makes the CPU test suite
   meaningful.

---

## 11. Verified vs not verified

Verified on CPU/gloo in this repository (`pytest tests/`, 218 tests — and 236 with
`TRITON_INTERPRET=1`, where the real kernels run inside the test suite too):

* dense and MoE training, gradient accumulation, checkpoint + resume;
* `OffloadedAdamW` equivalence including the chunked/pipelined path, activation offload;
* AC modes (`full`, `selective`, RNG preservation);
* DDP / FSDP2 / HSDP / TP / PP / 3D and EP equivalence (`mp_check.py`, up to 8 ranks);
* `torch.compile` with `aot_eager` (graph breaks counted, `fullgraph=True`, eager-gradient equality);
* the packed Triton kernels through Triton's CPU interpreter, forward and backward (66 checks);
* the kernels as `torch.library.custom_op`s: identical values and gradients to the eager path,
  `fullgraph=True` forward+backward, `torch.export` capture, and fake-impl shape/dtype agreement;
* the `Forge` façade including a 2-process `fm.launch` run and bit-identical replicas;
* `fm.Muon` end to end: reference parity (bf16 tolerance), singular values driven toward 1
  after five NS steps, batched bucket == per-matrix loop, one regression test per bug found
  in the reference file, and a real 2-process DDP run whose replicas are bit-identical
  *and* equal to a single-process `fm.Muon` run (no collective at world 1);
* the `newton_schulz` kernels through **TileLens** as well (0 findings; a planted
  out-of-bounds in the update kernel aborts the sanitizer, proving the NS kernels are
  instrumented) and through `kbench`'s op table;
* the eight packed kernels plus `newton_schulz`, `topk` and `softmax` through **TileLens**
  (formerly Triton-Viz) on CPU: symbolic out-of-bounds sanitizer (0 findings, tool validated
  against a planted bug), race detector, and the profiler's mask/loop tables — see
  [PERF.md §10](PERF.md) and [PERF.md §12](PERF.md#12-round-seven-profiler-audit--topk--softmax);
* the embedded runtime primitives: world-1 passthrough collectives, `seed_everything`
  reproducibility, rank-0 `forge.save` guarded by a barrier, and a real 2-process DDP run
  (bit-identical replicas, `fm.all_gather` agreement, checkpoint restored on both ranks);
* autotuning (batch-size search, OOM handling, report shape);
* the asynchronous batch pipeline: order, epoch bookkeeping, overlap timing, error forwarding,
  resume, thread/worker cleanup, rank alignment;
* the one-host-sync-per-step budget.

**Not verified here — no accelerator in this environment:**

* CUDA offload streams and the pipelined overlap they enable;
* async H2D overlap, pinned-buffer reuse and `record_stream` under memory pressure in the batch
  pipeline (the CPU code paths are exercised; the GPU win is unmeasured);
* kernel launch *speed* on real hardware (correctness is verified through the interpreter; the
  `kbench --check` gate exists to be run there and is **not** a measured claim here);
* fused kernels under `torch.compile` on CUDA, `torch._grouped_mm` dispatch;
* Transformer Engine FP8, torchao INT8 training, the DeepSpeed engine, NCCL.

Feature limits, stated rather than hidden: EP+FSDP2 unsupported; MoE aux loss not propagated across
PP stages; experts not TP'd or quantized; optimizer offload not for FSDP/PP; no eval under PP.

---

## 12. Troubleshooting

| symptom | cause and fix |
|---|---|
| `RuntimeError: BatchStream: the loader produced no batches` | dataset smaller than `micro_batch` with `drop_last=True`. Shrink `micro_batch`, enlarge the dataset, or set `drop_last=False` in your own loader |
| everything is slow, `feed.report()['stall_ratio']` high | the loader is the bottleneck: raise `workers`, then `prefetch_depth` |
| `batch_thread` thread warnings about RNG | with `workers=0` the collate shares the global torch RNG with training. Set `batch_thread=False` for bit-exact runs, or `workers>0` to move RNG into worker processes (warning is emitted once) |
| `RuntimeError: compile_fullgraph=True is not possible for this MoE block` | the expert dispatch needs a host read on this device. Use `torch>=2.8` + CUDA + bf16 (`torch._grouped_mm`), or leave `compile_fullgraph=False` |
| `RuntimeError: quantized training needs a CUDA GPU` / `int8 modes need torchao` | install the backend (`pip install torchao` / `transformer-engine`) and check `cfg.quant.mode` |
| `FP8 needs sm>=8.9, this GPU is smX` | use `mode="int8"` on older GPUs |
| `AssertionError: pp*dp_replicate*dp_shard*tp = N != world M` | the mesh must multiply to the world size; use `dp_shard=-1` to let it fill the remainder |
| `AssertionError: ep=E must divide dp world W` | `ep` carves expert groups out of the DP world, so it must divide `dp_replicate * dp_shard` |
| `AssertionError: micro_batch must be divisible by pp_microbatches` | make `micro_batch % pp_microbatches == 0` |
| `AssertionError: optimizer offload + pipeline not supported` | use FSDP2/HSDP sharding instead of optimizer offload, or drop `pp` |
| `RuntimeError: no trainable parameters` | every parameter is frozen — typical with `quant.mode="int8_wo"`; add adapters or unfreeze something |
| `apply_tp: no module matched the TP plan` | the model is not a `GPT` (or has no recognized attention/MLP names); `apply_tp` works on this repo's model layout |
| `ValueError: unknown kernel backend` | `fm.kernels.set_backend` takes `auto`, `fallback` or `triton` |
| step time is flat no matter what you change | check `count_host_syncs`: something in your loop (or your loss) is reading the device per step |
| `evaluate()` returns `None` | no `eval_ds`, or `pp > 1`, or DeepSpeed — those paths are not implemented |
| OOM when raising `micro_batch` | `tune_batch_size` catches OOM as data; also try `ac="selective"`, `offload.activations`, `offload.optimizer` |

---

## 13. Repository map

| path | contents |
|---|---|
| `forgemind/trainer.py` | `Trainer`: config → mesh → quant → TP → AC → compile → DP → pipeline, optimizer, data, step, eval, checkpoint |
| `forgemind/forge.py` | `Forge` façade, `launch`, `seed_everything`, rank-0 `save` |
| `forgemind/muon.py` | `Muon`: Muon (spectral-norm momentum + batched Newton-Schulz) for matrices, AdamW for vectors |
| `forgemind/data.py` | datasets, `ResumableSampler`, `BatchStream`, `async_loader`, `to_device`, `build_loader` |
| `forgemind/config.py` | `ForgeConfig` and the four sub-configs |
| `forgemind/models/` | `GPT`, blocks, RoPE, chunked cross-entropy |
| `forgemind/moe.py` | `MoE` with EP groups, expert offload, grouped GEMM |
| `forgemind/parallel/` | `native.py` (TP/DDP/FSDP2/PP), `pipeline.py`, `ac.py`, `deepspeed_backend.py` |
| `forgemind/quant.py` | quantized training (TE fp8, torchao int8) and the 8-bit optimizer |
| `forgemind/offload.py` | `OffloadedAdamW`, activation offload, pinned buffer pooling |
| `forgemind/perf.py` | host-sync counting, `warn_once`, fast-math defaults, pinned pool |
| `forgemind/kernels/` | Triton kernels (`ns.py` Newton-Schulz, `topk` FlashTopK-style selection, `softmax`), torch fallbacks, `apply.patch()` / `apply.enable()` |
| `forgemind/nn.py` | composable blocks: `RMSNorm` (with fused residual), `SwiGLU` (dense or fused gate\|up) |
| `forgemind/autotune.py` | `apply_fast_defaults`, `tune_batch_size`, `autotune`, kernel sweep |
| `forgemind/distributed.py`, `forgemind/loss.py` | mesh/groups init, module-level collectives (`all_reduce`/`broadcast`/`all_gather`/`barrier`), chunked CE/KL losses |
| `forgemind/cli/` | `bench.py`, `matrix.py`, `__main__` |
| `examples/train_gpt.py` | runnable GPT/MoE training script with the full flag set |
| `tests/` | unit suite plus `mp_check.py`, `forge_mp_check.py`, `muon_mp_check.py`, `kernel_interp_check.py` |
| `documentation.md` | this file: guide, API/config reference, troubleshooting, verified/unverified matrix |
| `PERF.md` | the performance audit: stall sites, graph breaks, both measurement rounds, bug tables |
| `OFFLOAD.md` | offload design and rejected alternatives |

Test files by concern: `test_single.py` (trainer basics), `test_edge_cases.py`, `test_sync_audit.py`
(sync budget), `test_compile.py` (graph breaks), `test_forge.py`, `test_async_batches.py` (data
pipeline), `test_offload.py`, `test_kernels.py`, `test_kernel_ops.py` (the `torch.library` op path:
eager-vs-op equality, `fullgraph`, export), `test_kernel_bench.py` (the `kbench` harness),
`test_runtime.py` (the embedded collectives, `seed_everything`, rank-0 save),
`test_topk.py` / `test_softmax.py` (the selection and softmax kernels: multiset parity,
canonical ties, NaN, autograd, `fullgraph`),
`test_muon.py` (the Muon optimizer and the NS kernel: reference parity, the found-bug
regressions, bucketed batching, compile dispatch, subprocess mp check), `test_autotune.py`, `test_ds_config.py`
(DeepSpeed config generation), `test_nn_blocks.py` (the blocks and `patch` are drop-in
compatible with torch, differentiable, and trainable in a hand-written model), `test_docs.py` (these documents: every documented `fm.*` name must
exist, every anchor must resolve, every config field must be documented).
