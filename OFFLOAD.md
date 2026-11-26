# Faster offloading — ideas, what shipped, what didn't

Offloading trades GPU memory for bandwidth. Everything below is judged on one question: **does
it shorten the critical path of a step, or does it just move work around?** A step has three
phases that can overlap — device→host gradient copies (D2H), host-side optimizer math, and
host→device weight copies (H2D) — plus, for activations, D2H before backward and H2D during
backward.

Legend: **✅ shipped** (in this repo, with a test) · **🔜 viable, not implemented** (with the
reason) · **❌ rejected** (with the reason, so nobody re-litigates it).

---

## 1. The current critical path

`OffloadedAdamW.step()` drains any pending D2H copies, then calls host AdamW
**once for all parameters**, then queues per-parameter H2D copies. Chunk events
permit individual waits, but there is no per-chunk optimizer update and therefore
no D2H/Adam pipeline. In the Trainer path `collect()` flushes after each
micro-batch and clipping collects again before `step()`, so pending copies are
normally already drained. D2H may overlap backward via hooks, but the final
update remains serialized after the waits. CUDA measurements are needed to
quantify the costs; `pipeline_chunks` does not promise a speedup.

Instrumentation: `opt.report()` returns `d2h_MB`, `h2d_MB`, `syncs`, `stream_ms`,
`host_update_ms`, `chunks`. If `host_update_ms` dominates, offload is cheap for you; if
`stream_ms` dominates, the link is the bottleneck and you should send **less** (see §3).

Tests: `tests/test_offload.py::test_chunked_step_matches_serial_step` (chunks = 2/3/5 give
bit-identical parameters to the serial path) and `test_chunked_step_drains_all_param_groups`.

---

## 2. Move less: fewer, larger transfers

| idea | verdict | why |
|---|---|---|
| Only offload tensors above `min_numel` | **✅ shipped before this round** (`OffloadConfig.smart`, `min_numel=2^18`) | a 64 KB copy costs ~2 µs of DMA and ~5 µs of Python/event overhead; a 4 KB copy is pure loss |
| Gate on GPU memory pressure (`gpu_watermark`) | **✅ already there** | if you fit, don't pay |
| One arena for all staging buffers | **🔜** | would remove per-tensor `cudaMalloc` from the pinned pool, but pinned allocation happens once at startup, not per step — no step-time win |
| `_foreach_copy_` for the whole group | **🔜** | fewer Python/launch calls (measurable at ~10 µs × #params) but the same number of DMAs; worth it only for models with >2k tensors |
| Merge tiny grads into one buffer before D2H | **❌** | the merge itself is a device kernel + a copy, so you pay one kernel and one copy to save several copies — net neutral, and it breaks per-parameter accumulators |

## 3. Move less data: state compression

| idea | verdict | why |
|---|---|---|
| bf16 optimizer moments (M, V) | **🔜** | halves host RAM **and** D2H/H2D bytes. Costs accuracy: moments are small-magnitude and slowly varying, so bf16 (8-bit mantissa) is usually fine, but it is a real numerical change, so it needs an opt-in flag + tolerance test rather than being a default. Implementing it means owning the Adam math (torch's `AdamW` won't store bf16 states), i.e. ~60 lines of duplicated math — worth doing, not worth doing silently |
| 8-bit moments with error feedback | **🔜** | 4× the bandwidth win of bf16 with the same "own the math" cost, plus the feedback buffer. This is what DeepSpeed/8-bit Adam (bitsandbytes) does; correctness is subtle |
| Skip the H2D of unchanged weights | **❌** | every weight changes every step; sparse updates exist but Adam with weight decay touches all of them |
| fp32 master **only for params that need it** | **🔜** | for a bf16 model the fp32 master *is* the extra memory; alternative is bf16 master + Kahan-style compensation, which is exactly the "own the math" trade above |

## 4. Overlap harder

| idea | verdict | why |
|---|---|---|
| Side stream for D2H, event per chunk, host Adam on chunk k while k+1 copies | **🔜 not implemented** | events and chunk waits exist, but the single Adam step runs only after all waits (§1) |
| Prefetch the H2D of chunk k+1 while Adam runs on chunk k | **🔜 not implemented** | H2D copies are enqueued after the full host update on the current stream; following compute waits for them |
| Second host thread for Adam (double-buffered) | **❌** | host Adam releases the GIL only inside ATen, and the two halves would contend for the same memory bandwidth; measured wins for this pattern are ~0 outside of CPU-bound Adam on huge models |
| Copy grads during the backward (post-accumulate hooks) | **✅ already there** | this is what turns "copy after backward" into "copy overlapping backward" |
| Reduce-scatter straight into host memory (NCCL `ReduceScatter` with a pinned output) | **🔜** | removes a D2H pass entirely: the gradient ends up host-side as a side effect of the collective. Needs a custom comm hook + `nccl` support for host buffers; unverifiable in this sandbox (no NCCL) |
| `cudaMemcpyAsync` on a **second** stream per chunk | **❌** | PCIe is the shared resource; two streams do not double bandwidth, they just interleave |

## 5. Activations

| idea | verdict | why |
|---|---|---|
| Spill only tensors above `act_min_bytes` | **✅ already there** | same argument as §2 |
| Recycle pinned buffers (`act_pool`) | **✅ already there** | removes the allocator from the backward critical path |
| Pack several saved tensors into one pinned arena, one copy each | **🔜** | the win is allocation/fragmentation, not bandwidth; requires a bump allocator whose lifetime is a backward pass. Viable, ~40 lines, no numerical risk |
| Recomputation instead of offload for cheap ops | **✅ already there** (`ac="full"/"selective"`) | for anything cheap to recompute, recompute: it is DRAM traffic only, no PCIe |
| Offload *and* recompute for the biggest tensors | **🔜** | store the small, recompute the large — needs per-tensor cost estimates; `act_min_bytes` is the crude version of this |
| bf16 activation spill | **🔜** | halves the traffic, adds a rounding error into the backward pass. Fine for most activations, dangerous for anything that feeds a norm |

## 6. Storage tiering

| idea | verdict | why |
|---|---|---|
| NVMe (DeepSpeed-ZeRO-Infinity style) | **❌ for this library** | ~3 GB/s against ~25 GB/s PCIe and ~1 TB/s HBM; it only wins at scales where the optimizer state genuinely does not fit in host RAM (100B+ params, many nodes). The plumbing (io_uring, prefetch threads, `posix_fadvise`) is a project of its own |
| `mmap`'d host state | **❌** | same bandwidth as a plain pinned buffer once pages are resident, plus fault handling; the only benefit is overcommit, which `expandable_segments` already gives you |
| Keep "hot" params resident, evict cold ones | **🔜** | Adam touches every parameter every step, so there is no hot set. It becomes interesting with sparse/adam-update-before-step optimizers (Adafactor's factored second moment is the real version of this idea: 2 states → O(√n)) |
| Adafactor-style factored states | **🔜** | this is the only *compression* scheme with a real algorithmic win (row+column statistics instead of a full moment). Changes the optimizer, so it belongs behind a flag |

---

## 7. What "faster offloading" actually means here

On this box (CPU only) the chunked path cannot overlap anything. On a GPU the current
implementation still waits for all chunks before the optimizer update; tests establish
numerical identity, not copy/update overlap. `opt.report()` and `tests/test_offload.py`
cover the cost and correctness side:

* offloaded AdamW ≡ `torch.optim.AdamW` to **0.00e+00** for `grad_accum` 1/2/3 (and now for
  `pipeline_chunks` 1/2/3/5),
* `zero_grad()` never discards in-flight gradients,
* pending copies are synchronized by a flush, not individually per parameter; repeated
  micro-batch collects can nevertheless synchronize more than once per step.

The remaining roadmap items that need an accelerator to evaluate honestly are: NCCL
reduce-scatter into pinned host memory (§4), bf16 moments (§3), and activation packing (§5).
They are left out rather than shipped untested — the same rule this library applies to
`torch._grouped_mm`, DeepSpeed and FP8.

---

## 8. Automatic selection (design — not implemented)

Sections 2–7 each pick *one* lever. The open question is who moves the levers: today it is
the human reading `OffloadConfig`. This section sketches a controller that picks for you —
cost model first, telemetry second, and a hard rule that it may never add a host sync to a
step (the one-sync budget in §1 is load-bearing).

**The signal.** `torch.cuda.mem_get_info()` returns `(free, total)` from the driver — a
cheap query, no device sync, safe to call at a step boundary every *N* steps (say, 50).
One number is not enough to act on: `free` conflates weights, activations, fragmentation
and the caching allocator's reserved-but-idle blocks, so the controller treats it as a
*trend* signal (slope over a window), never as an instant trigger.

**Hysteresis, always.** Two thresholds and a dwell time, or the system thrashes:

```
free < low  (e.g. 15% of total) for ≥ D steps  →  move to the next-smaller resident set
free > high (e.g. 30% of total) for ≥ D steps  →  move back
min dwell between transitions ≥ D steps        →  no policy flip-flops within a window
```

The asymmetry (low ≠ high) is the point: every transition costs a full weight transfer in
each direction, so a policy that oscillates is strictly worse than either static policy.
`D` absorbs allocator noise (grad freeing, activation peaks at layer boundaries).

**Cost model under a memory budget.** For each candidate resident set *S* (which modules
keep weights on the device):

```
peak(S)  = resident_weights(S) + activations + workspace + safety_margin
time(S)  = compute + Σ bytes_moved(S) / bw          (simplified; overlap must be measured, not assumed)
choose S* = argmin time(S)  subject to  peak(S) ≤ budget
```

Two terms are calibrated online rather than assumed: `bw` is measured with a probe
H2D/D2H of a known buffer every K steps (PCIe varies with contention), and `compute` is
the EMA of the last N step times. The search itself is greedy, not a solver: rank modules
by *bytes freed per µs of added transfer* and take from the top until the budget holds —
module sizes are known statically, transfer cost is linear in bytes, and the greedy order
is exactly what a knapsack relaxation would pick anyway.

**Prefetch scheduling.** While module *k* computes, H2D module *k+1* on a side stream —
the same double-buffering shape as `pipeline_chunks` (§1), but for weights. Rules:

* issue copies only for the *next* module in execution order (look-ahead of 1–2), never
  speculative beyond the resident set;
* record a CUDA event per copy and `wait` it at the first use — events are stream-side,
  **no host sync**, so the one-sync budget is untouched;
* only prefetch when telemetry says the copy queue is empty and the destination is already
  allocated (prefetching into memory the allocator must grow for is how a controller makes
  things worse).

**Recompute vs offload — a decision table, not a single rule.** The break-even for a tensor
class is `flops_per_byte > bw / flops_rate`: recompute pays compute, offload pays bytes.

| tensor class | verdict | why |
|---|---|---|
| optimizer state | **offload (shipped, §1)** | zero flops, streamed anyway, the host has the memory |
| gradients | **stream (shipped)** | already D2H once per step; chunked, §1 |
| activations | **recompute** | attention blocks are byte-heavy, flops-light per byte → recompute wins almost always; §5's D2H staging stays rejected until pinned+async is evaluated on hardware |
| weights | **offload only if `peak > budget`** | on-device is always fastest; this is the knob the controller owns |

**Emergency tier.** On `torch.cuda.OutOfMemoryError`: drop to the smallest resident set,
`empty_cache`, retry the step once. A retry is a policy miss — count it, surface it in
`opt.report()`-style stats, and let it feed the controller (a persistent miss rate means
the budget model is mis-calibrated). This is the only place the controller is allowed to
be reactive; everything else is predictive.

**Honest status: 🔜 design only.** None of it can be validated on this box (no GPU, and
`mem_get_info` is meaningless on CPU), so nothing here ships untested — the same rule as
§4–§5. What the repo *does* provide as prerequisites: the one-sync step budget the
controller must respect, `pipeline_chunks` events as copy-wait machinery, and
`OffloadConfig` as the static policy the dynamic one would wrap rather than replace.

## 9. Measuring VRAM alongside offload

For CUDA whole-training-step peak **allocated** memory, run the same workload in
separate processes with `python -m forgemind.cli bench --preset tiny --json torch.json`
and then with `--fused-ops --json fused.json`. Both runs should use identical
`--offload-opt`, `--offload-act`, `--ac`, `--compile` and other settings when
comparing kernel patches; to compare an offload option, change only that option.
Subtract the second `peak_mem_bytes` from the first, then divide by the first for
the percentage. The field is rank-0 peak allocated memory during measured steps
after warmup—not reserved VRAM, host RAM, or a sum across ranks. It includes
live model/optimizer allocations. Kernel savings and offload savings cannot be
added independently because their tensor lifetimes can overlap.

`kbench --memory --no-compile` separately measures isolated eager operators, not
whole-model or offload peaks. CUDA is required for both actual VRAM comparisons;
no result has been measured on a GPU in this environment. See [PERF.md](PERF.md#vram-measuring-kernel-and-whole-model-impact).
