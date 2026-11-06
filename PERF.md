# ForgeMind performance & `torch.compile` audit

Three things were asked for: **stop the utilities from costing speed**, **remove the `torch.compile`
breaks**, and **find the GPU↔CPU synchronisations**. This document is the complete list of what was
found, what it cost, and what replaced it.

Everything below is either measured in this repo (numbers marked *measured*) or a structural argument
that can be checked by reading the diff. CUDA-only paths are labelled as such — this audit was executed
on a CPU-only box, so anything that requires an accelerator to *time* is stated as a design consequence
rather than a measurement.

---

## 1. Headline numbers (measured)

CPU, `torch 2.9.1`, `dim=128 / 4 layers / 8 heads / 4 kv-heads`, `micro_batch=8`, `seq=64`, fp32.
Latency is the **best of 3 runs of the median of 20 optimizer steps** (a 2-vCPU box is noisy; a single
run moves ±5%), old code = the shipped `forgemind.zip`, new code = this tree:

| configuration | host syncs / step | median step latency | aten ops / step |
|---|---|---|---|
| dense | **5 → 1** | 69.9 ms → **64.6 ms** (−7.7%) | 1545 → 1409 |
| dense + activation checkpointing (full) | **5 → 1** | 91.0 ms → **87.8 ms** (−3.5%) | — |
| MoE, 8 experts / 4 MoE layers | **9 → 5** | 190.5 ms → **185.5 ms** (−2.6%) | — |

Read the sync column as the result and the latency column as a lower bound on its value: the CPU has no
asynchronous queue, so a removed host read only saves the read plus whatever the op-count cut removes
(the dense row's 136 fewer aten ops per step are the RoPE rewrite). On a GPU the same sync removal is
worth far more, because every one of those reads drained the queue rather than the CPU's own backlog.

The MoE row is the honest one: on CPU the remaining 4 reads are the per-layer expert dispatch, and CPU
step time cannot improve from removing syncs because there is no asynchronous queue to drain. The MoE
win is elsewhere — 12 graph breaks per MoE block went to 5 deliberate ones, and on `CUDA + bf16` the
dispatch is fully device-side (`torch._grouped_mm`), which removes those 4 reads *and* their stalls.

Op-count detail for the dense step (`TorchDispatchMode` count), old → new:

| op | before | after | where it came from |
|---|---|---|---|
| `aten.arange / outer / cos / sin` | 16 | **0** | RoPE tables rebuilt per layer per forward |
| `aten.unsqueeze` | 32 | **0** | `cos[None, :, None, :]` built per layer per forward |
| `aten.clone` | 16 | **0** | `f.cos()`/`f.sin()` temporaries in `rope_cache` |
| `aten._local_scalar_dense` | 44 | 39 | `.item()`-class host reads |

Reproduce with `python -m forgemind.cli.bench --syncs` (new `--syncs` flag, prints
`host_syncs_per_step` for ForgeMind and `host_syncs_torch_per_step` for a plain torch loop as a
baseline) or `pytest tests/test_sync_audit.py`.

---

## 2. GPU↔CPU synchronisations (the stalls)

A host read of a device scalar (`float(t)`, `t.item()`, `t.tolist()`, `bool(t)`, `t.cpu()`) blocks
until every previously enqueued kernel has completed. It is not just the ~10-30 µs of the read: it
stops the CPU from running *ahead* of the GPU, so the next kernels are enqueued later than they could
have been. The loop had **five** per step plus one per MoE layer.

### 2.1 Trainer, per optimizer step — 5 → 1

| # | site (before) | why it was there | what it is now |
|---|---|---|---|
| 1 | `trainer.py` `_clip()` → `float(min(1.0, clip/(float(norm)+1e-6)))` | to decide *whether* to clip | the coefficient is a 0-dim device tensor: `torch.clamp(clip/(norm+1e-6), max=1.0)` applied with `_foreach_mul_`. Clipping always runs, which is a no-op when `coef == 1`. |
| 2 | `trainer.py` `lv = float(self.ctx.mean_over_dp(loss))` inside the non-DeepSpeed branch | to feed the loss EMA | the EMA is computed from the single stacked read (§2.1.5). |
| 3 | `trainer.py` — the *same* quantity read **again** two statements later (`lv = float(...)` for the metrics dict) | metrics dict | removed: one reduction, one value. |
| 4 | `trainer.py` `ok = bool(flag.item())` after `all_reduce(MIN)` | all ranks must agree to skip | the flag is folded into the same stack: `torch.stack([flag, grad_norm, loss]).tolist()`. |
| 5 | *(new)* | the single read | `flag, grad_norm, loss = torch.stack([...]).tolist()` — **exactly one** host round-trip per step, asserted by `tests/test_sync_audit.py`. |

The skip decision (`skip_nonfinite`, `skip_spike_factor`) is evaluated from that one read, so it is
free: `isfinite(grad_norm) & isfinite(loss)` is computed and all-reduced **on the device**, and the
spike test is a pure-Python comparison against the already-hosted loss.

Also removed: the doubled loss reduction over the pipeline group (it ran `all_reduce` twice per step),
and `mean_over_dp`'s unconditional `clone()` is skipped entirely when `dp_world == 1`.

### 2.2 MoE dispatch — one read per layer (or zero)

`moe.py` `MoE.forward` did `counts.tolist()` on every forward to size the per-expert GEMMs, and the EP
path did **two** reads (`cmat.sum(1).tolist()` and `recv_mat.tolist()`) plus one `int()` per local expert.

* `ep_size == 1` + `CUDA` + `bf16` + `torch._grouped_mm` → **zero** reads: the cumulative offsets stay a
  device tensor (`counts.cumsum(0).to(torch.int32)`).
* otherwise (the eager fallback) → one read per MoE layer, which is the theoretical minimum for
  data-dependent per-expert GEMM sizes.
* the EP path → **one** read for both directions and the per-expert totals
  (`torch.cat([send, recv_flat, recv_totals]).tolist()`), down from 2 + n_local.

One consequence worth knowing: a fallback MoE layer pays that read **twice per step** under activation
checkpointing, because the recompute re-runs the dispatch. Measured: `--moe --experts 4 --ac full` gives
9 own reads/step (1 trainer + 4 layers x 2 passes). With `expert_grouped_mm` it is 1.

The `torch._grouped_mm` capability probe now happens **once**, in `MoE.probe_grouped()`, called from
`GPT.prepare()` before compilation — instead of a `try/except` around the op on every forward.

### 2.3 Pipeline

`parallel/pipeline.py` had one read (the shape/dtype handshake), which is fine — but it was *never
re-armed*, so a different activation shape (evaluation with another batch size, or a second run in the
same process) silently reused the stale shape. Now `PipelineRunner` re-arms automatically when
`x.shape` changes. The blocking `dist.send` for the header stayed (once per run); activations and
gradients were already `isend` + `wait()` at the end of the step.

### 2.4 Utilities

| component | per-step cost before | after |
|---|---|---|
| `OffloadedAdamW.flush()` | `stream.synchronize()` per micro-batch **and** `torch.cuda.current_stream().synchronize()` at the end of every `step()` | one stream sync, only when copies are actually in flight; the `step()` sync is gone (stream ordering already guarantees the next kernel sees the new weights) |
| `OffloadedAdamW._offload()` | a fresh **pinned** staging tensor (`cudaHostAlloc`) per parameter per micro-batch, plus a `.add_()` per parameter | first micro-batch copies straight into the (zeroed) fp32 accumulator — no staging at all in the common `grad_accum=1` case; later micro-batches reuse one per-parameter buffer |
| `OffloadedAdamW._eligible()` | `torch.cuda.mem_get_info()` per gradient hook | read once per step (`begin_step()`) |
| `OffloadedAdamW` grad norm / zeroing | python loop per parameter | `_foreach_norm` / `_foreach_zero_` / `_foreach_div_` |
| `ActivationOffload` | a **new** `saved_tensors_hooks` object + closures on every block forward; one pinned allocation per saved tensor per forward | one hook object per block, built once; pinned buffers recycled through a size-keyed pool (`OffloadConfig.act_pool`) |
| activation checkpointing | `preserve_rng_state=True` → CPU **and** CUDA RNG state saved + restored per checkpointed block per micro-batch | `TrainConfig.ac_preserve_rng=False` by default (the reference model has no dropout); the selective-checkpoint `context_fn` is built once per block instead of per call |
| data loading | `batch.to(device, non_blocking=True)` on the **current** stream, immediately before the forward | `BatchStream` stages batch *N+1* on a side stream (event-ordered, no host sync) while step *N* computes (`TrainConfig.prefetch`) |
| loss | `(hc @ w.t().to(hc.dtype)).float()` per chunk — the weight was re-cast, and a full fp32 `[chunk, vocab]` temp was allocated, inside the recompute region | the weight cast happens once per call, outside the checkpointed region; logits stay in the compute dtype (`log_softmax` already accumulates in fp32 for bf16); no checkpoint is installed when nothing requires grad |
| pipeline loss | `F.cross_entropy` on the whole `[N, vocab]` logits (materialises a second full-size fp32 log-softmax) | reduced in 4096-token chunks |
| `Trainer._clip()` | `self.model.named_parameters()` scanned (string matching over every parameter) **every step** for the EP expert-norm split; same in `sync_expert_grads` | cached once at construction (`self._expert_params`) |
| RoPE | 2 (`arange`) + 2 (`outer`) + 2 trig + 1 cast + 2 `unsqueeze` per layer per forward | tables cached per (length, head_dim, theta, dtype, device) as **views shared by all layers**, materialised once in `GPT.prepare()` |
| `apply_rope` | upcast `q`/`k` to fp32, `cat`, `type_as` | computed in the activation dtype (what Megatron does); `GPTConfig.rope_high_precision=True` restores the fp32 path |
| quant | `importlib.util.find_spec` on every `resolve_backend`; TE autocast context rebuilt per forward | `lru_cache`d; the context factory is built once in `QuantState` |
| init | `new_group` created ep and edp groups even for `ep == 1` (`pp*tp*2` wasted process-group creations) | skipped when `ep == 1` |

---

## 3. `torch.compile`

Measured with `torch._dynamo.explain` (`tests/test_compile.py`):

| model | before | after |
|---|---|---|
| dense GPT, forward+loss | 1 graph, 0 breaks | 1 graph, 0 breaks |
| dense GPT + AC full / selective | 1 graph, 0 breaks | 1 graph, 0 breaks |
| MoE GPT | **13 graphs, 12 breaks** | 6 graphs, 5 breaks — all `torch.compiler.disable()`d expert dispatches, i.e. the deliberate boundary, nothing else |
| MoE on CUDA+bf16 (`_grouped_mm`) | — | 1 graph, 0 breaks |

Fixed breaks, in the order they fire:

1. **`torch.bincount` for the routing histogram** (`Router.forward`, `MoE.forward`) — *"Dynamic shape
   operator: `aten.bincount`'s output shape depends on input Tensor data"*, two guaranteed breaks per
   MoE block. Replaced by `zeros(n) + scatter_add_`, which has a static shape and is the same kernel
   count.
2. **`counts.tolist()` + `if c == 0: continue` in the expert loop** — *"Could not guard on data-dependent
   expression `Eq(u0, 0)`"*. An unrollable data-dependent branch is not compilable, so the loop now
   lives in `Experts._forward_eager`, marked `@torch._dynamo.disable`. Everything around it — router,
   top-k, sort, gather, weighting, `index_add_` scatter — stays in one graph.
3. **`try: torch._grouped_mm(...) except Exception: pass` on every forward** — an exception region on the
   hot path, and it silently hid every failure (a broken grouped GEMM looked like "slow but working").
   The capability check happens once at `prepare()` time; a failure now prints a one-time warning and
   picks the fallback permanently.
4. **`torch.tensor(counts, device=dev)`** built from a Python list inside `forward` — now
   `counts.cumsum(0).to(torch.int32)` on the device.
5. **RoPE tables built *and the module-level cache mutated* inside the traced region** —
   `Mutating a variable not in the current scope (SideEffects)`, which is a hard error under
   `fullgraph=True`. The tables are now non-persistent **buffers** filled by `GPT.prepare()` before
   compilation, so the forward is a pure function (`tests/test_compile.py::test_rope_cache_is_not_rebuilt_inside_the_graph`).
6. **`rope_cache()` allocating `arange/outer/cos/sin` per layer per forward** — not a break by itself,
   but it depended on the module-level dict write above.
7. **`GPT.aux_loss()` walking `self.layers` after the forward** to find graph-connected tensors — the aux
   losses are now summed while walking the blocks inside `forward`, so nothing outside the graph reads
   module state.
8. **`CellMixer.scan`** would have been unrolled `T` times (2048 cells for a 2k context). It is a
   sequential dependency chain, so it is now explicitly eager instead of a compile trap.
9. `torch.utils.checkpoint` stays non-reentrant everywhere (`use_reentrant=False`), which dynamo can
   trace; the reentrant variant would have been both a graph break and an autograd-hook hazard.

Verification is not "it did not crash": `tests/test_compile.py` runs the whole forward **and backward**
with `fullgraph=True` and asserts that the loss and *every* gradient match the eager run bit-for-bit
within 1e-5, for dense, AC-full and AC-selective. `Trainer` config gains
`compile_backend` / `compile_mode` / `compile_fullgraph`.

---

## 4. Bugs fixed

1. **`QuantConfig.mode="int8_wo"` could not train at all.** `quantize_model` called
   `requires_grad_(False)` on *every* parameter, so the trainer built an empty parameter group and
   `torch.optim.AdamW` raised *"optimizer got an empty parameter list"*. Only the weights that were
   actually quantized are frozen now; norms, the router gate, embeddings and the head stay trainable,
   and `Trainer` raises a clear error if nothing is trainable.
2. **`OffloadedAdamW.zero_grad()` discarded in-flight gradients** (`self.pending.clear()` without a
   flush). Latent in the default flow, immediate data loss for anyone calling `zero_grad()` directly.
   It flushes first now (`tests/test_offload.py::test_zero_grad_with_copies_in_flight_keeps_data`).
3. **`apply_fsdp` raised `IndexError`** when neither `dp_shard` nor `dp_replicate` was > 1
   (`ctx.mesh[dims[0]]` on an empty tuple). It now fails with an explanation.
4. **`ResumableSampler.__len__` mixed units** — it subtracted a sample offset (`skip`) from a batch count,
   so `len(dataloader)` was wrong after a mid-epoch resume, and negative when `skip > per_rank`.
5. **`PipelineRunner` never re-armed its shape handshake** — a different activation shape reused the
   stale `(shape, dtype)` and produced a shape error or a hang.
6. **Pipeline receive buffers** are reused now, so `inp.grad` had to be cleared after the send — without
   it, activation gradients accumulate across steps. Covered by `mp_check pp_equiv`.
7. **`Trainer.__init__` assumed the model exposes `.cfg`** (`getattr(model.cfg, ...)` for the PP+MoE
   warning) — `AttributeError` for any custom model without one.
8. **`dcp.save` overwrote an existing checkpoint directory** and warned that this becomes a hard error in
   a future PyTorch; the directory is removed first.
9. **`_swap_te` silently quantized zero layers** when the filter excluded everything (wrong `exclude`,
   non-multiple-of-16 dims) and still printed "FP8 enabled". It reports the count and warns at zero.
10. **`torch.load` of the offloaded-optimizer checkpoint** had no `map_location`, so a checkpoint written
    on GPU could not be resumed on CPU; it also pins `weights_only` explicitly.
11. **Loss/EMA double reduction**: the same DP-mean was computed twice per step (2 `all_reduce`s,
    2 host reads). Fixed as part of §2.1.
12. **`Trainer._clip()` recomputed the expert parameter list every step** — an EP-scaling hazard *and* a
    slowdown on models with thousands of parameters. Cached.
13. `data.SyntheticLM.__getitem__` cloned every window twice; `index_add_`-style empty-expert returns in
    MoE no longer build a dead `x.sum() * 0` graph node.
14. **The pipeline backward used the unscaled loss.** Each micro-batch's loss was back-propagated at full
    strength, so with `grad_accum = micro-batches per step > 1` every pipeline gradient came out M times too
    large — a silently wrong LR, invisible in a loss curve, and `mp_check pp`/`pp_equiv` only caught it once
    the reference comparison was added. The loss is divided by `M` before `backward()` and the reported value
    is still the sum over micro-batches divided by `M`.
15. **EP gradient norm was squared twice.** `clip_grad_norm_`'s EP path reduced the *squared* norms a second
    time, so the reported norm was `sqrt(sum of squares of squares)` — and clipping therefore either never
    fired or cut far too hard. Now the per-rank squared norms are summed once.
16. **Expert gradients were not averaged over the expert-data-parallel group, and HSDP ignored
    `fsdp_reshard_after_forward`.** Expert params are owned by one rank per EP group, so DDP skips them: their
    grads held the sum over the ep group's tokens but were never summed over `edp` nor divided by the dp world
    factor that dense grads get, i.e. the two parameter classes trained at different effective learning rates
    under `ep > 1`. `sum_expert_grads()` now completes the reduction, and `reshard_after_forward` is threaded
    through to every `fully_shard` call (the last block and the root never reshard).

---

## 5. How to check this yourself

```bash
pytest tests/ -q                    # everything: 90 tests, ~35s on CPU
python tests/kernel_interp_check.py # every Triton kernel vs torch, via TRITON_INTERPRET=1 (66 checks)
python tests/forge_mp_check.py      # Forge + DDP, 2 processes
python tests/test_sync_audit.py     # the sync budget: 1 read/step (dense), 1/MoE-layer (fallback)
python tests/test_compile.py        # graph-break counts + fullgraph == eager equivalence
python tests/test_offload.py        # offload numerics, accumulation, in-flight zeroing
python tests/test_single.py         # end-to-end training, checkpoints, AC modes
torchrun --standalone --nproc_per_node=2 tests/mp_check.py pp_equiv   # also ddp, fsdp, tp, ep_equiv, ...

python -m forgemind.cli.bench --syncs --moe --experts 8 --steps 30   # on a GPU box
python -m forgemind.cli.bench --compile --syncs --quant fp8 --qbackend auto
```

`forgemind.perf.count_host_syncs()` is the reusable piece: it patches the synchronising tensor methods
and reports `own` (your code) vs `external` (torch internals) counts with call sites, so a regression in
a new kernel is one test away from being caught.

```python
from forgemind.perf import count_host_syncs

with count_host_syncs() as c:
    trainer.train_step()
print(c.report())      # 1 host sync(s) (1 in forgemind, 0 inside torch)
                       #    1x trainer.py:281 train_step
```

Note on `external`: PyTorch's non-fused AdamW calls `_get_value(step)` once per parameter
(`torch/optim/adam.py`, bias correction). Those `step` tensors are deliberately hosted on the CPU when
`capturable=False`, so the reads are free there; the trainer's default `fused=True` on CUDA path does
not do this at all.

---


---

## 6. Known limits / not verified here

* No accelerator in this environment: the CUDA-only code paths (`torch._grouped_mm` dispatch, pinned
  staging, side-stream copies, `cudaMemcpyAsync` overlap) are written and guarded but were **not timed**.
  They are all covered by the CPU fallbacks that the tests do exercise.
* `torch.compile` was validated with the `aot_eager` backend (`FORGEMIND_TEST_BACKEND=inductor` switches
  it): inductor codegen needs a C++ toolchain and `Python.h`, which this sandbox does not have. The graph
  boundaries — which is what this audit is about — are identical for both backends.
* TE / torchao FP8+INT8, the DeepSpeed engine and NCCL remain unverified, exactly as before this change.
* `experts` are still not TP'd, EP+FSDP2 is still unsupported, and the MoE aux loss is still not
  propagated across pipeline stages (that one now warns at construction *and* is documented).

---

## 7. Round two: packed kernels, an accessible façade, autotuning, faster offload

Same rules as above -- measured, or explicitly marked unmeasured.

### 7.1 Packed Triton kernels (`forgemind/kernels/`)

Six kernels, each replacing 2--7 eager ops: `rms_norm`, `add_rms_norm` (residual + norm in one
pass), `qk_norm_rope` (per-head norm + RoPE), `swiglu`, `swiglu_split` (epilogue for a fused
gate|up GEMM) and a distillation KL that computes `dKL/d(logits)` inside the forward pass.

Real bugs found in the incoming draft, all fixed:

| # | bug | consequence |
|---|---|---|
| 1 | gradient stored into the student-logits buffer as **hard-coded bf16** | fp16/fp32 runs get half of every gradient element overwritten: silent, and nothing raises |
| 2 | `swiglu` forward *and* backward stored `tl.bfloat16` into an `empty_like` buffer | same corruption, both directions |
| 3 | the KL kernel **mutated its input** (`student` logits) | the caller's tensor is destroyed; with the fused distill path this was by design, now it is an explicit flag |
| 4 | `tl.arange(0, HD)` in the QK-norm kernel | `head_dim/2` must be a power of two or Triton fails to compile -- i.e. head_dim 96 breaks |
| 5 | the QK-norm block was `H x HD` wide entirely | 32 heads x 128 = 4096 fp32 lanes per CTA: ~128 registers/thread per live tensor, heavy spills; now tiled over heads |
| 6 | `swiglu_split` backward wrote `dgate`/`dup` with the **wrong stride** (used the `[N,h]` offset for the `[N,2h]` buffer) | gradients land in the neighbouring row -- caught by the interpreter test, not by shape checks |
| 7 | output dtype was a module-global `CDT` | an fp32 model got bf16 norms back, changing the model dtype mid-graph |
| 8 | masked tail lanes loaded `-inf`, so `ln - ls = -inf - -inf = NaN` and `0 * NaN = NaN` | poisoning the accumulator on every non-multiple-of-block vocab; now a finite sentinel |
| 9 | the KL gradient was written even when it was not needed | a full `[chunk, vocab]` HBM write for inference-only calls |
| 10 | `V <= BLOCK_V` still ran the two-pass loop | the whole row read from HBM twice; there is a single-tile path now |

How these were found with no GPU: Triton's interpreter (`TRITON_INTERPRET=1`) runs the real
kernels on CPU through a numpy emulator. `tests/kernel_interp_check.py` drives every kernel
(forward and backward, including non-power-of-two `D`, non-divisible `N`, `V=5000` streaming,
bf16 grads, input mutation) and compares against the torch reference in `_fallback.py`: **66
checks, 0 failures**. It is wired into `pytest` as a subprocess test.

### 7.2 `Forge` -- the accessible API

Any `nn.Module`, any loader, two lines, with this library's guarantees kept (one host read per
optimizer step, device-side clip coefficient, accumulation, DDP, checkpoint/resume). Verified:
grad-accumulation equals a single big batch, `count_host_syncs` reports exactly **1** read per
update, save/load resumes bit-identically, and a 2-process `fm.launch(nproc=2)` run leaves both
ranks with identical parameters.

### 7.3 Autotuning

`tune_batch_size` scores candidates by **per-sample** cost rather than step time -- a bigger
micro-batch usually costs the same wall time, so step-time scoring happily returns the smallest
batch that fits. OOM is caught and reported as a trial, not raised. `autotune(forge, loader)`
prints the table and only mutates the object when `apply=True`.

### 7.4 Faster offloading

`OffloadedAdamW.step()` is now chunked: the host Adam update of chunk *k* overlaps the D2H of
chunk *k+1* (`OffloadConfig.pipeline_chunks`, default 4), so the critical path is
`max(copy, update) + one chunk` instead of `copy + update`. Numerically identical to the serial
path (`chunks` 1/2/3/5, bit-exact). `opt.report()` reports the split between stream wait and
host update so you can tell whether the link or the host is your bottleneck. The full idea list
-- including what was rejected and why (NVMe, 8-bit moments, reduce-scatter into host memory,
a second Adam thread) -- is in **[OFFLOAD.md](OFFLOAD.md)**.
