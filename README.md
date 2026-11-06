# ForgeMind

PyTorch training library: FSDP2/HSDP/DDP, tensor/pipeline/expert parallelism (3D), DeepSpeed ZeRO backend,
FP8 (Transformer Engine / torchao) and INT8 training, smart gradient/activation/expert offload, MoE, packed
Triton kernels, device autotuning, and a full trainer.

```bash
pip install -e .            # + torchao / transformer-engine / deepspeed as needed
torchrun --nproc_per_node=4 examples/train_gpt.py --moe --dp-replicate 4 --ep 2
torchrun --nproc_per_node=4 examples/train_gpt.py --tp 2 --dp-shard 2
pytest tests/               # 90 tests: trainer, parallel, kernels, forge, autotune, offload
torchrun --standalone --nproc_per_node=2 tests/mp_check.py ep_equiv   # also tp_equiv, pp_equiv, ddp, fsdp, ...
```

## Your own model, two lines

`Forge` is the Lightning/Fabric-style façade: any `nn.Module`, any `DataLoader`, and it keeps the
guarantees this library is built on (one host sync per step, device-side clipping, accumulation, DDP).

```python
import forgemind as fm

forge = fm.Forge(model, optimizer, loss_fn=criterion, precision="bf16", accum=4)
forge.fit(loader, max_steps=1000)

# or keep your own loop, Fabric-style:
loss = criterion(model(**batch), batch["labels"])
forge.backward(loss)
forge.step()
```

Multi-process is one decorator away (it spawns, or defers to an existing `torchrun`):

```python
@fm.launch(nproc=4)
def main():
    fm.Forge(build_model(), build_opt(), precision="bf16").fit(loader)
main()
```

Mesh: world = pp * dp_replicate * dp_shard * tp; `ep` divides the DP world.

## Packed kernels

`fm.kernels` has fused Triton kernels for the ops that are usually *several* eager kernels:
`rms_norm`, `add_rms_norm` (residual+norm in one pass), `qk_norm_rope` (per-head norm + RoPE),
`swiglu` / `swiglu_split` (fused gate|up projection epilogue), and a distillation KL that
computes the logit gradient in the forward pass. Every one is a drop-in for the eager expression,
degrades to torch on CPU/MPS, and is verified against the torch reference by running the *actual
kernels* through `TRITON_INTERPRET=1` on CPU (`tests/test_kernels.py`, 66 numeric checks).

```python
import forgemind as fm
y = fm.kernels.rms_norm(x, w, 1e-6)         # == F.rms_norm(x, (d,), w, eps)
fm.kernels.enable(model)                    # or patch RMSNorm/SwiGLU modules in place
loss = fm.kernels.kl_distill_loss(hs, ht, w, teacher_head)
```

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
```

```python
from forgemind.perf import count_host_syncs

with count_host_syncs() as c:      # audit your own step: counts + call sites, own vs torch-internal
    trainer.train_step()
print(c.report())
```

| knob | default | effect |
|---|---|---|
| `TrainConfig.fused_ops` / `fused_split_mlp` | off / off | swap RMSNorm/SwiGLU for the packed kernels; `split` also fuses the gate|up GEMM |
| `TrainConfig.compile` / `compile_mode` / `compile_backend` / `compile_fullgraph` | off / `default` / `inductor` / off | compiles each transformer block (caches are warmed first); `fullgraph=True` refuses to fall back |
| `TrainConfig.ac` / `ac_preserve_rng` | `none` / `False` | full or selective checkpointing; leave RNG preservation off unless you use dropout |
| `TrainConfig.prefetch` / `workers` | `True` / `0` | overlap the next batch's H2D copy on a side stream; `workers>0` adds a DataLoader pool |
| `OffloadConfig.optimizer` / `smart` / `gpu_watermark` | off / on / 0 | fp32 master + Adam states in pinned host RAM, grads streamed on a side stream |
| `OffloadConfig.pipeline_chunks` | 4 | overlap the grad DMA of chunk *k+1* with the host Adam update of chunk *k*; `opt.report()` shows the split (see [OFFLOAD.md](OFFLOAD.md)) |
| `OffloadConfig.activations` / `act_pool` | off / on | spill large saved activations to pinned host memory, recycling buffers |
| `OffloadConfig.experts` | off | keep MoE experts in pinned host RAM, stream only the active ones |
| `GPTConfig.expert_grouped_mm` | `True` | `torch._grouped_mm` expert GEMMs (CUDA+bf16): removes the last host read from the MoE dispatch |
| `GPTConfig.rope_high_precision` | `False` | compute the rotation in fp32 (slower, marginally more accurate) |
| `ParallelConfig.ddp_static_graph` | `False` | skip DDP's per-iteration unused-parameter scan |

## Status

Verified on CPU/gloo (see `tests/`, 90 tests): dense/MoE training, grad accumulation, checkpoint+resume,
offload optimizer equivalence (and the chunked/pipelined path), AC modes, DDP/FSDP2/HSDP/TP/PP/3D, EP
equivalence, `torch.compile` (`aot_eager`) with `fullgraph=True` and eager-gradient equivalence, the
packed Triton kernels run through Triton's CPU interpreter (forward *and* backward vs torch), the
`Forge` façade (including a 2-process DDP launch), autotuning, and a one-host-sync-per-step budget.

Untested here (no accelerator): CUDA offload streams and the pipelined overlap they enable, kernel
launches on real hardware (correctness is verified through the interpreter, *speed* is not), fused
kernels under `torch.compile` on CUDA, `torch._grouped_mm` dispatch, TE/torchao FP8/INT8 training,
DeepSpeed engine, NCCL.

Limits: EP+FSDP2 unsupported; MoE aux loss not propagated across PP stages; experts not TP'd or quantized;
OffloadedAdamW not for FSDP/PP; no eval under PP.
