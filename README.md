# ForgeMind

PyTorch training library: FSDP2/HSDP/DDP, tensor/pipeline/expert parallelism (3D), DeepSpeed ZeRO backend,
FP8 (Transformer Engine / torchao) and INT8 training, smart gradient/activation/expert offload, MoE, and a full trainer.

```bash
pip install -e .            # + torchao / transformer-engine / deepspeed as needed
torchrun --nproc_per_node=4 examples/train_gpt.py --moe --dp-replicate 4 --ep 2
torchrun --nproc_per_node=4 examples/train_gpt.py --tp 2 --dp-shard 2
python tests/test_single.py; python tests/test_ds_config.py
python tests/test_sync_audit.py; python tests/test_compile.py; python tests/test_offload.py   # or: pytest tests/
torchrun --standalone --nproc_per_node=2 tests/mp_check.py ep_equiv   # also tp_equiv, pp_equiv, ddp, fsdp, ...
```

Mesh: world = pp * dp_replicate * dp_shard * tp; `ep` divides the DP world.

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
| `TrainConfig.compile` / `compile_mode` / `compile_backend` / `compile_fullgraph` | off / `default` / `inductor` / off | compiles each transformer block (caches are warmed first); `fullgraph=True` refuses to fall back |
| `TrainConfig.ac` / `ac_preserve_rng` | `none` / `False` | full or selective checkpointing; leave RNG preservation off unless you use dropout |
| `TrainConfig.prefetch` / `workers` | `True` / `0` | overlap the next batch's H2D copy on a side stream; `workers>0` adds a DataLoader pool |
| `OffloadConfig.optimizer` / `smart` / `gpu_watermark` | off / on / 0 | fp32 master + Adam states in pinned host RAM, grads streamed on a side stream |
| `OffloadConfig.activations` / `act_pool` | off / on | spill large saved activations to pinned host memory, recycling buffers |
| `OffloadConfig.experts` | off | keep MoE experts in pinned host RAM, stream only the active ones |
| `GPTConfig.expert_grouped_mm` | `True` | `torch._grouped_mm` expert GEMMs (CUDA+bf16): removes the last host read from the MoE dispatch |
| `GPTConfig.rope_high_precision` | `False` | compute the rotation in fp32 (slower, marginally more accurate) |
| `ParallelConfig.ddp_static_graph` | `False` | skip DDP's per-iteration unused-parameter scan |

## Status

Verified on CPU/gloo (see `tests/`): dense/MoE training, grad accumulation, checkpoint+resume, offload
optimizer equivalence, AC modes, DDP/FSDP2/HSDP/TP/PP/3D, EP equivalence, `torch.compile`
(`aot_eager`) with `fullgraph=True` and eager-gradient equivalence, and a one-host-sync-per-step budget.

Untested here (no accelerator): CUDA offload streams, `torch._grouped_mm` dispatch, TE/torchao FP8/INT8
training, DeepSpeed engine, NCCL.

Limits: EP+FSDP2 unsupported; MoE aux loss not propagated across PP stages; experts not TP'd or quantized;
OffloadedAdamW not for FSDP/PP; no eval under PP.
