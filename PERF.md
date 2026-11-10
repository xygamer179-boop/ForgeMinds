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

---

## 8. Round three: asynchronous batches ("release the batch while the compute runs")

Before this round the "prefetch" was a *staging* trick: `BatchStream.next()` copied the next batch
to the device on a side stream, so the H2D copy overlapped the step, but **the loader itself**
(iterate → decode/augment → collate) still ran on the main thread, between the optimizer step and
the next forward. If a batch took 8 ms of host time, every step paid those 8 ms on the critical
path — the accelerator sat idle while the CPU assembled data.

### 8.1 What changed

`forgemind/data.py` now has a producer/consumer split for both training and eval:

* `BatchStream(..., threaded=True)` starts one daemon thread that owns the loader. It pulls batch
  *k+1*, stages it (`to(device, non_blocking=True)` on the side stream, event recorded) and pushes
  it into a `queue.Queue(maxsize=depth)`. The consumer only pops and waits on the event, so the
  copy and the collate of batch *k+1* both happen **while step *k* is computing**.
* `depth` (`TrainConfig.prefetch_depth`, default 2) is the number of batches allowed in flight.
* `Trainer.evaluate()` uses the same machinery (`async_loader`, depth 2) instead of a bare
  `for batch in dl:` loop — eval is often 10-30% of a real run.
* `fm.async_loader(loader, device, depth=2)` exposes the same thing generically, so it is usable
  outside this library:
  `for x, y in fm.async_loader(DataLoader(ds, 64, pin_memory=True), "cuda"): ...`
* `Forge(..., async_batches=True, prefetch_depth=2)` and `TrainConfig.batch_thread` are the knobs.
  `batch_thread=None` (auto) turns the thread on when nothing else prefetches (`workers=0`) or when
  the device is CUDA; `batch_thread=False` restores bit-exact serial order and RNG.

CUDA safety, which is where a prefetcher usually goes wrong: every staged tensor has
`record_stream(current_stream)` called on it before the producer's buffer can be reused, the
consumer does `current_stream().wait_event(ev)`, and the producer thread calls `torch.cuda.set_device`
on entry — `torch.cuda.set_device` is **thread-local**, so without that a rank pinned to `cuda:1`
would allocate on `cuda:0` from the worker thread. (That was one of the bugs found, see 8.4.)

### 8.2 Measured overlap

Two measurements, because they answer different questions.

**(a) Is the overlap real?** A loader with 20 ms of simulated host work per batch against a step
that takes 20 ms of wall time and does not need the GIL (this is what a GPU step looks like from
Python's point of view — the CPU is free while the kernels run):

| | total | per batch |
|---|---|---|
| serial | 324.2 ms / 40 batches | 40.5 ms |
| threaded, depth 2 | **184.3 ms** | **23.0 ms** (0.57x) |

Producer wait 2.74 ms/batch, one stall over 40 batches. i.e. the pipeline runs at
`max(load, step)` instead of `load + step`, which is exactly the intent.

**(b) What does it cost on a CPU-only box?** No overlap is possible when *both* sides need the GIL:
a 1 ms/sample dataset (8 ms/batch) with a 18 ms CPU step, on the real `Trainer` path, interleaved
A/B/A/B, min of 4 rounds:

| config | ms/step | vs serial |
|---|---|---|
| serial, `workers=0` | 18.17 | 1.00x |
| threaded d2, `workers=0` | 18.65 | 0.97x |
| threaded d4, `workers=0` | 18.43 | 0.99x |
| serial, `workers=2` | 10.91 | 1.67x |
| threaded d2, `workers=2` | 11.17 | 1.63x |
| **auto**, `workers=0` | 17.82 | 1.02x |
| **auto**, `workers=2` | 11.04 | 1.65x |

Read this as: **the thread costs nothing and buys nothing on CPU** (all differences are inside the
±5% noise floor; `workers=2` is 1.6x faster than `workers=0` because that is the DataLoader's own
process pool, not this change). It buys real time only when the step lets go of the interpreter:

* on CUDA, where the copy (`non_blocking=True` on a side stream) and the collate both run while the
  kernels execute — **not verified here, no GPU in this environment**;
* with the CPU step itself blocked on I/O (checkpoint writes, tokenizer calls that release the GIL).

That is why the default is *auto* rather than always-on and why `batch_thread=False` exists: on a
CPU box it is a no-op, and "no regression" beats "sounds faster".

### 8.3 Invariants kept

* **One host read per optimizer step.** The prefetch path adds no `.item()`, no
  `cudaSynchronize`, no `torch.cuda.synchronize`: `tests/test_sync_audit.py` still counts 1, and
  the only device→host traffic is the existing grad-norm read.
* **Same batches, same order.** `mp_check.py prefetch` runs two ranks with `dp_replicate=2` and
  asserts that each rank's threaded stream is *batch-for-batch identical* to its own serial stream
  (12/12, `torch.equal`), and that both ranks consumed the same number of batches.
* **Resume is unaffected.** `test_async_batches.py` trains 4 steps, checkpoints at 2, resumes in a
  fresh `Trainer` with the threaded feed, and matches the uninterrupted run to 5 decimals; the
  sampler rewind (`ResumableSampler.skip`) is unchanged.
* **No leaks.** `close()` stops the thread, drains the queue and shuts down the DataLoader worker
  pool (`persistent_workers=True` would otherwise keep it alive until gc), and is idempotent;
  `Forge.fit`/`evaluate` and `Trainer.evaluate` close the stream in a `finally`, because stopping
  early (`max_steps`, `eval_steps`) leaves the producer blocked on a full queue.

### 8.4 Bugs found and fixed while doing this

| # | bug | consequence |
|---|---|---|
| 1 | the stream re-built the batch as a raw dict instead of `(input_ids, labels)` | the step silently received the wrong tuple — the kind of bug that trains on garbage and never raises |
| 2 | `Forge.fit()` / `evaluate()` never closed the stream when `max_steps` cut it short | a daemon thread stayed blocked on a full queue for the rest of the process |
| 3 | `evaluate(max_batches=…)` left the producer blocked forever | same, once per eval |
| 4 | the "empty loader" error was only raised on the serial path | with the thread on, an empty loader looked like a normal end of data |
| 5 | a one-shot (generator) loader raised "no batches" instead of `StopIteration` | a confusing error at the end of every epoch |
| 6 | `threaded=None` did not call `torch.cuda.set_device` on the producer thread | on a multi-GPU box every rank would have staged its batches on device 0 — OOM or silent cross-device copies |
| 7 | `close()` did not shut down `persistent_workers` pools | one leaked worker pool per `Trainer.evaluate()` call with `workers>0` |
| 8 | `async_loader.close()` neither drained nor marked itself closed | `next()` after `close()` returned a stale batch instead of stopping |
| 9 | `warn_once` mutated a module-level set from the producer thread | a race could print the same warning twice (benign, but it is now lock-guarded) |
| 10 | an empty loader with a known length was only detected on the first `next()` | a `drop_last` misconfiguration surfaced mid-step instead of at construction |
| 11 | `TokenFile` never released its memmap | the file stayed mapped (and un-deletable on Windows) after the trainer was gone |
| 12 | the auto policy enabled the thread even when worker processes already prefetch | second hand-off for nothing; auto now keys off "does anything else prefetch" |

### 8.5 What is *not* verified

The CUDA half: async copy overlap, `record_stream` correctness under memory pressure, and the
multi-GPU device binding are written and reasoned about, and the CPU paths that exercise the same
code are tested, but there is no accelerator in this environment. Treat the GPU speedup as
**unmeasured** — measure it with `trainer.feed.report()`, which returns
`wait_ms_per_batch`, `stalled`, `stall_ratio` and `produce_ms` for exactly this purpose.

---

## 9. Round four: kernels as `torch.library` ops (compile-native, no eager regression)

The kernels were already correct and already graph-break-free in the dense path. What they were not
was *first class* to `torch.compile`: an `autograd.Function` is opaque to `torch.export`, and
AOTInductor (the ahead-of-time serving path) cannot capture it at all. Round four fixes that without
making the eager path any slower.

### 9.1 The dispatch rule, and why it is not one path

Same Triton launch, two entry points:

| path | used when | cost per call (measured, CPU, 20k calls of a 64x64 op) |
|---|---|---|
| `torch.autograd.Function.apply` | eager | **24.2 µs** |
| `torch.library.custom_op` | while `torch.compile` traces (or forced) | **59.0 µs** |
| plain torch expression, for scale | — | 11.7 µs |

A custom op is the slowest way to call Python-level torch code: every call re-enters the dispatcher
through a boxed Python kernel. Using it unconditionally would have been a genuine **speed
regression** in eager, which is exactly what these kernels are not allowed to cause. But inside a
compiled graph the overhead is gone — the op is captured once and the graph replays from C++ — and
that is where the op buys things a `Function` cannot:

1. a **fake implementation**, so dynamo/inductor/export propagate shapes and dtypes without running
   the kernel;
2. a **registered backward** that AOTAutograd places in the backward graph like any other node,
   including under activation checkpointing;
3. **`torch.export` compatibility** — the exported graph contains `forgemind.rms_norm.default`.

So: `Function` in eager, `custom_op` while tracing. `fm.kernels.use_custom_ops(True)` forces the op
path (tests, exporters), `FORGEMIND_KERNEL_OPS=ops|eager` does it from the environment, and both
paths call the same launcher — `tests/test_kernel_ops.py` asserts the values *and* the gradients are
bit-identical, not merely close.

### 9.2 The backward has to be an op too

The first version registered only the forward and let the backward launch Triton directly. That
fails the moment AOTAutograd traces the *joint* graph:

```
RuntimeError: Attempted to access the data pointer on an invalid python storage.
```

The registered backward is ordinary Python that dynamo traces — with FakeTensors. A raw launch
inside it runs against fake storage during compilation. The fix is the pattern production kernel
libraries use: every kernel registers **two** ops, `forgemind::<name>` (fake + autograd) and
`forgemind::<name>_backward` (fake, no autograd), and the Python backward is a single op call. Eleven
ops in total. The same exercise surfaced the gradient-count convention: a registered backward returns
one gradient per *input* (torch appends its metadata slot itself), which is not obvious from the
error message (`expected 6, got 3`).

### 9.3 Kernel-level work that can only remove work

No GPU here, so nothing in this list is a measured claim — each item strictly *removes* instructions
or registers, which is why it was safe to make without a stopwatch:

* **32-bit address arithmetic** in `rms_norm`, `add_rms_norm`, `swiglu`, `swiglu_split`,
  `qk_norm_rope` and `kl`. Every one of those kernels already refuses tensors with `>= 2**31`
  elements in its Python guard, so no offset can wrap; a 64-bit index was doubling the register
  footprint of every live address tile for nothing (and in `kl`, an `int64` row for a per-program
  scalar that is bounded by `N`).
* **one address computation** for the `swiglu_split` store instead of two, and the same for the
  backward's `src`/`dst` (they are the same offset by construction).
* the mask-free `EVEN_*` paths, the `ROWS` tiling and the fp32-only-where-needed stores were already
  in place from round two and were left alone — re-tuning them blind is how a "no regression"
  promise gets broken.

### 9.4 Verifying it on real hardware

`python -m forgemind.cli kbench` times each kernel against the eager expression *and* against
inductor's output for the same expression (plus a `kernel+compile` column), min-of-N with CUDA
events, forward or forward+backward. `--check` turns it into a gate:

```bash
python -m forgemind.cli kbench --check --tolerance 0   # exit 3 if any kernel loses to inductor
```

That is the honest form of "no GPU speed decrease": the harness refuses to run on CPU (where the
kernels fall back and every column measures the same expression), and no number in this file is
claimed from a machine without an accelerator.

Covered by tests on CPU: `tests/test_kernel_ops.py` (20 tests: path equality, `fullgraph=True`
forward+backward, `torch.export` capture, fake-impl agreement, dispatch flag, inference under
`no_grad`), `tests/test_kernel_bench.py` (the harness and its refusal), and the existing
interpreter suite — **66 checks, 0 failures** — plus 150 earlier tests, unchanged.

---

## 10. Round five: TileLens critique (sanitizer / profiler / race on the real kernels)

[TileLens](https://github.com/Deep-Learning-Profiling-Tools/tilelens) (formerly *Triton-Viz*,
PyPI `tilelens` 3.1.1) instruments plain `@triton.jit` kernels — its CLIs patch `triton.jit`
before the script imports ForgeMind, so the analysis runs over **our actual kernels**, on CPU
through Triton's interpreter. The probe is committed as `tools/tilelens_probe.py`.

```bash
tile-sanitizer tools/tilelens_probe.py     # symbolic out-of-bounds check (aborts on OOB)
tile-profiler  tools/tilelens_probe.py     # mask ratios, loop unroll stats, byte counts
tile-race      tools/tilelens_probe.py     # Z3 happens-before race detector
TILELENS_SAVE=/tmp/probe.tvz tile-profiler tools/tilelens_probe.py   # archive for tile-visualizer
```

The probe runs all six kernels forward **and** backward on deliberately awkward shapes:
`d=40` (non-power-of-two width), `6x95` flat SwiGLU (non-divisible element count), `h=95`
SwiGLU-split (padded tile), `head_dim=24` (padded RoPE tile), and KL with `V=1000` (single
tile) plus `V=5000` (two-pass loop with a tail).

### 10.1 Environment (this is the recipe, not an afterthought)

* `tilelens` requires `triton>=3.6`. Installed alongside `torch==2.9.1` (which pins
  `triton==3.5.1`) you must pick one; **3.6.0 works, 3.9.0 does not**: TileLens's profiler
  `load_overrider(ptr, mask, other, cache_modifier, eviction_policy, is_volatile)` no longer
  matches triton 3.9's `create_masked_load(ptr, mask, other, cache_policy, volatile)` and every
  kernel dies with `missing 1 required positional argument: 'is_volatile'`.
* Everything below ran with `torch 2.9.1 + triton 3.6.0 + tilelens 3.1.1`, CPU interpreter.
* `tilelens.save()`/`load()` round-trip verified (`manifest.json` + `tensors.npz` inside the
  `.tvz`), which is what `tile-visualizer` consumes.

### 10.2 Results

| analysis | result | scope notes |
|---|---|---|
| **Sanitizer** (symbolic OOB) | **0 findings** on all six kernels, fwd+bwd, all probe shapes; exit 0 | Meaningful, not vacuous: a planted bug (store masked as fully valid past a 10-element tensor) makes it exit 1 with a full `IMA Diagnostic Report` (tensor metadata, call stack, expression tree) |
| **Race detector** (Z3 happens-before, on by default) | **No race reported** on any kernel | Our writes are disjoint by construction — every store targets a unique `(program, element)` — so the structural argument and the tool agree. Caveat: a planted *write-write* race was **not** flagged either, so on this version the detector's write-write sensitivity is unverified; treat the clean run as "no read/write hazard found", not as a proof |
| **Profiler** — mask ratios | High *false-element* ratios **only** on the padded/tail tiles of the probe's awkward shapes: `norm.py:68` 94.1% false at `d=40` in a 64x64 tile (6x40 real), `swiglu_split` 86% at `h=95`, KL two-pass 36.8% (the `V=5000` tail tile) | At production shapes the ratios collapse to the boundary tile only: `block_m_for(block_d)` tiles `4096/block_d` rows per program, so `d=4096` is 1 row x 4096 cols per program with **no** column mask at all. No code change: the masks are the tails, not waste inside valid data |
| **Profiler** — loops | `qk_norm_rope`: 1 loop, `tl_static_range`, 1 step. Norm/SwiGLU: no loops | The KL `while off < V` loops are *not* reported — the profiler hooks Python `range` iteration only — so "no for-loops" for KL means "not visible to this check", not "unrolled" |
| **Profiler** — buffer-load warning | Fires on **every** kernel | False positive on this machine: the check only passes when native `amdgcn` ASM was compiled and contains `buffer_load`; in interpreter mode there is no ASM, so the warning says "no AMD GPU was compiled", nothing about the code. Actionable only on an AMD GPU |

### 10.3 What changed because of it

Nothing structural — and that is the finding. The round-four work (int32 addressing, the
two-op backward pattern) survives an independent symbolic memory-safety check, the race
detector sees no read/write hazard, and every mask the profiler flags is a boundary tile at a
shape we chose to be awkward. Two candidate "polish" changes were considered and **rejected
rather than merged blind**:

1. rewriting KL's `while off < V` as `tl.range(..., num_stages=...)` to expose loop pipelining —
   plausible, but a generated-IR change with no GPU to A/B, i.e. exactly the kind of unverified
   speed claim this file refuses to make;
2. adding `eviction_policy="evict_first"` to streamed loads — L2 policy, invisible to the
   interpreter, measurable only with `python -m forgemind.cli kbench --check` on the target GPU.

The probe itself is the durable artifact: it is committed, documented in its docstring, and
re-runnable in one line whenever a kernel changes. Speed acceptance stays where round four left
it: `kbench --check` on real hardware.

---

## 11. Round six: Muon + `newton_schulz` (bug hunt on the reference optimizer)

Task: *"optimize it further, find bugs, and add Muon to the kernels."* The attachment was a
Keller-Jordan-style Muon reference (478 lines: distributed / SingleDevice / WithAuxAdam
variants plus a Fast path with a Triton Gram kernel). It was audited line by line; every
finding below is encoded as a regression test in `tests/test_muon.py` rather than left as
prose.

### 11.1 Bugs found in the reference file

| # | where | what happens | our fix (test) |
|---|---|---|---|
| 1 | `muon_update`, `nesterov=False` | `update = momentum` aliases the state buffer, then `update *= sqrt(max(1, rows/cols))` **scales the buffer every step** -- on a tall matrix the factor compounds as `(m/n)^(t/2)` (4x after 4 steps at 16x8) | the scale is applied to the NS output only; the buffer is never written (`test_nesterov_false_keeps_buffer_unscaled`) |
| 2 | `muon_update`, `nesterov=True` | `grad.lerp_(momentum, beta)` **mutates `p.grad` in place** -- under `zero_grad(set_to_none=False)` or any post-step grad read, accumulation sees a momentum-blended value | out-of-place `g.lerp(buf, beta)` (`test_nesterov_does_not_mutate_grad`, `test_grad_accumulation_survives_step`) |
| 3 | `Muon.__init__` / `step` | `dist.get_world_size()` unconditional (single-process runs crash); pad count `world - len % world` adds a whole extra pad block when `len % world == 0`; `isinstance(params[0], Parameter)` rejects generators and param-group dicts; the NS `ndim >= 2` assert dies on the first bias of an unsplit model | `dist` consulted only when initialized with world > 1; **no padding** (disjoint `rank::world` shares summed by one `all_reduce`); any iterable or group spec accepted; auto-split by `ndim` (`test_auto_split_groups_and_generators`) |
| 4 | `MuonWithAuxAdam.__init__` | `set(group.keys()) == {...}` asserts reject any extra key (a scheduler leftover, `ns_steps`, ...) | unknown keys pass through (`test_explicit_groups_honored`) |
| 5 | `None`-grad handling | `p.grad = state["zero_grad"]` writes a cached zero tensor **into the model**, and the nesterov lerp then corrupts that cache for later `None`-grad steps | local zeros; `p.grad` stays `None` (`test_none_grad_param_stays_none`) |
| 6 | `SingleDeviceMuon` step | same class of problems as 1-2 on the single-device path (buffer scale, grad mutation, `p.grad` writes) | one implementation covers single and multi process (`test_parity_with_attached_reference` pins the reference semantics where it was correct) |

The file's own `FastMuonWithAuxAdam` already fixed the grad mutation and the dist-at-world-1
problem, and it contains the two patterns worth porting: shape-bucketed batched NS and the
disjoint-slice `all_reduce`. Its Triton Gram kernel (`_mmt_kernel`, upper-tiles-only +
transpose store) was **not** ported: cuBLAS `bmm` is the benchmark to beat, there is no GPU
here to A/B it, and shipping an unverifiable "fast" path is how regressions sneak in.

### 11.2 The `newton_schulz` kernel

Per NS iteration the math is 3 GEMMs plus 5 elementwise ops in eager torch. The GEMMs are
cuBLAS either way (the same backend inductor would pick), so the elementwise chain becomes
**two** Triton kernels -- `B = b*A + c*A@A`, then `X = a*X + (B@X)` -- for **10 vs 25
elementwise launches** over five iterations (5 vs 8 launches per iteration counting GEMMs),
same FLOPs, one fewer HBM round trip per intermediate. That is a *structural*
no-regression argument (launches + HBM on identical math); the speed half stays with
`python -m forgemind.cli kbench --ops newton_schulz` + `kbench --check` on hardware.

Dispatch deliberately does **not** use a `torch.library` op: NS is a no-grad optimizer
utility, so while dynamo traces, `newton_schulz` returns the pure-torch implementation --
dynamo sees ordinary tensor ops (`fullgraph=True` holds; measured in
`test_ns_compile_fullgraph_matches_eager`) and inductor fuses the elementwise chain from
its side. Eager + CUDA (or `TRITON_INTERPRET=1`) takes the fused path; `fused=False`
forces torch for A/B; `set_backend("fallback")` and the `stats()` counters behave as for
every other kernel (no state is mutated while tracing -- same rule as round four).

Numerics: both paths agree in fp32 to <= 1e-5, and with the reference within a few bf16
ulps -- measured <= 0.04 absolute on the NS output after five iterations. The quintic is
evaluated as `b*A + c*(A@A)`; the reference source line `b*A + c*A @ A` parses
left-to-right as `b*A + (c*A)@A`. The two are mathematically identical; re-associating is
exactly what buys two launches, because the reference order needs a third elementwise pass
to materialize `c*A` before the second GEMM. Both associations are pinned by tests
(`test_ns_matches_reference_single_matrix` <= 0.05 bf16, path agreement <= 1e-5 fp32).

### 11.3 Measured (CPU only; this sandbox has no GPU)

| check | result |
|---|---|
| `pytest tests/test_muon.py` | 23 passed + 1 skipped (CPU: fused-Triton case skips); 24 passed under `TRITON_INTERPRET=1` (fused path live) |
| NS singular values after 5 steps (fp32) | all within `[0.68, 1.13]` across tall/wide/batched shapes (test bound 1 +/- 0.35) |
| reference parity, 3 steps, conv + 3xLinear | max param diff 0.0071 (threshold 0.015; dominated by the documented bf16 re-association) |
| batched bucket vs per-matrix loop, 2 steps (fp32) | max diff < 1e-4 |
| 2-process DDP check (`tests/muon_mp_check.py`, gloo) | replicas bit-identical **and** bit-equal to a single-process `fm.Muon` run; also passes under `TRITON_INTERPRET=1` |
| `tile-sanitizer tools/tilelens_probe.py` | 0 findings on the NS kernels; a planted out-of-bounds in `_ns_update_kernel` aborts the run (exit 1), proving the NS kernels are instrumented -- then reverted |
| `tile-profiler` | mask tables reported for `ns.py` (tail-block masks, e.g. 57% masked on the 63x17 scenario); the buffer-load warning is the same known CPU/AMD false positive as section 10 |
| `kbench --ops newton_schulz --allow-cpu` | the row prints (eager ~= kernel on CPU, because every column is torch there); `inductor = nan` on this box for **all** ops -- the sandbox has no `Python.h` for inductor's C++ toolchain (environment limitation, pre-existing) |
| `kbench --check` | still refuses CPU (the documented exit-2 guard): speed acceptance is GPU-only |

GPU-unverified by policy: fused-vs-torch launch speed on real shapes, bf16 tensor-core
occupancy of the GEMM chain, and `num_warps` tuning for the flat kernels. The claim that
survives without a GPU is the structural one: fewer launches and HBM round trips over the
same arithmetic, measured by a command (`kbench --check`) that exists to be run there.

---

## 12. Round seven: profiler audit + topk + softmax

### 12.1 Method

`tools/profile_audit.py` (committed) drives three measurements from one entry point:

* **op table** — `torch.profiler` with `table_sort_by_self_cpu_time_ms`, top 25 ops over a
  3-step steady-state window (after warmup steps that trigger compile/cudagraph effects);
* **host-read audit** — the same profiler's `SYNC_OPS` (`aten::item`,
  `aten::_local_scalar_dense`, `aten::tolist`) plus `torch._dynamo.explain` for graph breaks;
* **churn table** — highest-count ops (launch/python/view tax rather than flops).

Flags: default = dense `Trainer` step, `--forge` = `Forge.step`, `--moe` = the MoE config,
`--trace` = chrome-trace JSON dump.

**Attribution caveat (tooling finding).** In torch 2.9.1 the profiler's stack capture is
empty even with `with_stack=True`: `function_events[].stack` is `[]` and the chrome trace's
`args.frames` are absent, for every op we checked. Source attribution was therefore done by
wrapping `Tensor.item` / `Tensor.tolist` with a Python-stack recorder (the
`count_host_syncs` idea, extended to `.item`) — that is how §12.3's numbers got their
caller. Recorded here so nobody re-tries the stack path.

### 12.2 Graph breaks (the compile/cudagraph question)

| model | graphs | breaks | verdict |
|---|---|---|---|
| dense `Trainer` step | 1 | **0** | `fullgraph=True` claim holds on the real loop |
| dense `Forge` step | 1 | **0** | same loop, façade entry |
| MoE `Trainer` step | 6 | 5 | every break = `Skip inlining torch.compiler.disable()d function` — the **deliberate** MoE dispatch disables, reported not "fixed" |

### 12.3 Host reads: the one-sync budget, and what else calls `.item()`

Steady state, dense Trainer, 3-step window:

| read | per step | source |
|---|---|---|
| `aten::tolist` | **1** | `trainer.py:301` — the stacked `flag/gn/loss` read. Budget held. |
| `aten::item` | **21** | `torch.optim.AdamW.step` → `_single_tensor_adam`: one bias-correction `item` per parameter tensor (21 params in the audit model) |

The 21 are **not** ForgeMind code and **not** the documented budget item. On CUDA,
non-capturable AdamW keeps `state["step"]` on the CPU, so the math is host-side rather than
a device sync — *unverified on GPU, flagged as such*. Options if it ever shows up on a
profile: `fm.Muon`/our Adam (Python-int step, zero `item`s) or the fused/foreach path on
CUDA. No code change: torch's optimizer is not ours to patch.

### 12.4 Churn (per step, dense, CPU)

`aten::to` ~197, `as_strided` ~195, `copy_` ~142, `_to_copy` ~136, `transpose` ~122,
`resolve_conj` ~108 — the view/copy tax of eager aten. With **0 graph breaks** these are
exactly what inductor fuses, so the actionable item is `compile=True` on GPU (already the
recommended config), not a source edit. On CPU each is sub-µs; python-side `cProfile` shows
`train_step` at ~9 ms/step cumulative with no pathological branch in `forgemind/` — the hot
frames are AdamW's per-parameter loop (§12.3 again).

**Audit verdict: no unsafe if/else "speed killers" in this repo's code.** The structures the
profiler was asked to check are intact: one stacked host read per step, zero dense graph
breaks, MoE's breaks deliberate.

### 12.5 `topk` (FlashTopK-inspired) and `softmax`

Two new kernels under the standing rules (interpreter-tested, torch fallback, fullgraph,
TileLens scenarios, kbench op):

* **`fm.kernels.topk`** — strided-sample threshold (k-th of `m = min(n, max(2k, 256))`
  samples; subset rule proves it never exceeds the true k-th) → exact pivot inside
  `{x > t}` by k masked compound maxes → canonical (value desc, index asc) output, smallest
  indices win ties, NaN treated as largest. 2 launches/row-program, `k ≤ 512` fused
  (counted fallback beyond), scatter backward. Parity: 13 shape cases × multiset-exact vs
  `torch.topk` + gather-self-check + tie order, all green under `TRITON_INTERPRET=1`;
  grad parity exact on distinct values; `explain` → 1 graph / 0 breaks.
* **`fm.kernels.softmax`** — three chunked passes (row max, exp+sum, scale); backward is
  the closed form as tensor ops (compile-friendly, like `fused_linear`).

**Bugs found while building them** (both caught by the interpreter tests before commit):

| # | bug | symptom | fix |
|---|---|---|---|
| 1 | topk case-B kill tracked one position (`bpos`) that each iteration *overwrote*, resurrecting earlier kills | pivot off by iterations (k-th came back as 2nd) | carry the kill as "strictly worse than the last (value, index)" — one scalar pair encodes the whole history |
| 2 | softmax had no chunk loop: one 1024-block per row | `n > 1024` left the row tail **uninitialized** (garbage 1e-41/1e30 outputs) | three-pass chunked loop over the row |

### 12.6 TileLens (CPU, 3.1.1 + triton 3.6.0) — extended

`tile-sanitizer tools/tilelens_probe.py` → **exit 0**, all prior scenarios still clean, plus
`softmax(5x1000)` (masked tail), `softmax(4x4096)` (even path) and the topk **threshold**
kernel (K1, standalone launch). **Known tool gap:** K2 (the select kernel) branches on a
reduction (`if c >= k`), and TileLens's symbolic engine raises
`NotImplementedError: Concretize for op sum` when a reduction feeds a Python bool — the
probe prints an explicit `skip` line instead of crashing, and K2's correctness rests on the
13-case interpreter parity suite instead. (First attempt at silencing this — moving
reductions to `axis=0`/fp32 — was still insufficient: the failure is reduction→branch, not
the dtype. Reductions *do* now consistently pass `axis=0`, matching `kl.py`/`norm.py`.)

### 12.7 kbench (CPU only — no GPU in this sandbox)

`kbench --ops topk,softmax --dim 128 --iters 3 --warmup 1 --allow-cpu`:

| op | shape | eager | kernel | note |
|---|---|---|---|---|
| topk | [512, 32000] bf16 k=64 | 85.095 ms | 76.286 ms | CPU; fallback path≈ eager — **no speed claim** |
| softmax | [8192, 128] bf16 | 2.568 ms | 2.446 ms | CPU; within noise of eager |

`inductor=nan` env-wide (missing `Python.h`, §6). Speed gates stay GPU-only:
`kbench --ops topk,softmax --check` on hardware decides wins; nothing here is claimed.
