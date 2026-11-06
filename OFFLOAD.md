# Faster offloading — ideas, what shipped, what didn't

Offloading trades GPU memory for bandwidth. Everything below is judged on one question: **does
it shorten the critical path of a step, or does it just move work around?** A step has three
phases that can overlap — device→host gradient copies (D2H), host-side optimizer math, and
host→device weight copies (H2D) — plus, for activations, D2H before backward and H2D during
backward.

Legend: **✅ shipped** (in this repo, with a test) · **🔜 viable, not implemented** (with the
reason) · **❌ rejected** (with the reason, so nobody re-litigates it).

---

## 1. The critical path, before and after

```
serial (before)                        pipelined (now)
─────────────────────────────────      ─────────────────────────────────────────
wait(all D2H)                          [chunk 0 D2H] ──► update 0
host Adam (everything)                       [chunk 1 D2H] ──► update 1
enqueue H2D (everything)                          [chunk 2 D2H] ──► update 2
─────────────────────────────────      ─────────────────────────────────────────
T = copy + update                      T ≈ max(copy, update) + copy/K
```

**✅ shipped: chunked/pipelined `OffloadedAdamW.step()`**
(`OffloadConfig.pipeline_chunks`, default 4, `offload.py`). Gradients are grouped once at
construction; `_offload()` records a CUDA event per chunk on the copy stream, and `step()`
waits only for chunk *k*'s event before running Adam on chunk *k*. With the old code the host
was idle for the entire DMA phase and the PCIe link was idle for the entire update phase; now
the two overlap, and the tail is one chunk instead of all of them.

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
| Side stream for D2H, event per chunk, host Adam on chunk k while k+1 copies | **✅ shipped** | §1 |
| Prefetch the H2D of chunk k+1 while Adam runs on chunk k | **✅ effectively free now** | the H2D is enqueued on the *current* stream after the last chunk, and the next step's forward kernels queue behind it, so it overlaps the next step's compute; doing it mid-`step()` would need a second host thread to be worth anything |
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

On this box (CPU only) the chunked path cannot overlap anything, so the honest summary is:
the step is now **structured** so that on a GPU the copy and the update overlap, and the
pipeline is tested for numerical identity with the serial path. What *is* measured here is the
cost side: `opt.report()` and `tests/test_offload.py` pin down that

* offloaded AdamW ≡ `torch.optim.AdamW` to **0.00e+00** for `grad_accum` 1/2/3 (and now for
  `pipeline_chunks` 1/2/3/5),
* `zero_grad()` never discards in-flight gradients,
* the number of sync points is one per step, not one per parameter.

The remaining roadmap items that need an accelerator to evaluate honestly are: NCCL
reduce-scatter into pinned host memory (§4), bf16 moments (§3), and activation packing (§5).
They are left out rather than shipped untested — the same rule this library applies to
`torch._grouped_mm`, DeepSpeed and FP8.
