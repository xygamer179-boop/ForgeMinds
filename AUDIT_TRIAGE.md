# Forensic Code Audit — Triage Report

**Source:** "Forensic Code Audit: xygamer179-boop/ForgeMinds (forgemind/ v0.9.0)" —
user-generated site, self-labelled *"user-generated and unverified"*:
<https://01a1eeeb-a158-7fca-9fa0-2085fc693f44.arena.site/>
**Audited tree:** `main` @ `1091c9a` (post PR #11), branch `arena/8525974e-forgeminds`.
**Method:** every one of the 20 findings was verified line-by-line against the actual code;
each verdict below cites the evidence. Verdicts: **CONFIRMED**, **PARTIAL** (sub-claims
differ), **REBUTTED** (claim does not hold in code).

**Scoreboard: 14 CONFIRMED · 4 PARTIAL · 2 REBUTTED-in-part · 0×20 fully false.**
No finding was dismissed; every confirmed one gets a fix or an honest status note.

---

## Verdict table

| # | Finding (headline) | Location | Verdict |
|---|---|---|---|
| BS-001 | Autotune "fused sweep" is placebo; cache file dead | `autotune.py:275-334` | **CONFIRMED** |
| BS-002 | Offload D2H/Adam overlap never engages | `offload.py:163-228`, `trainer.py` | **CONFIRMED** (structural) |
| BS-003 | kbench shared `try` clobbers inductor `ms` with NaN | `kbench.py:151-159,230-241,310` | **CONFIRMED** |
| BS-004 | "lazy" imports are fake; `import forgemind` pulls triton | `__init__.py:4-29`, `trainer.py:19` | **CONFIRMED** |
| BS-005 | `enable()` wires only norm+SwiGLU; showcase kernels unused | `apply.py`, `moe.py:69-70`, `gpt.py` | **PARTIAL** — no false README claim; gaps real |
| BS-006 | Sync counter misses `cuda/stream/event.synchronize`; autotune `enable` not reverted | `perf.py:29-88`, `autotune.py:247-250` | **CONFIRMED** (both) |
| BUG-001 | `clip=0` + `skip_nonfinite` → every step skipped forever | `forge.py:257-316` | **CONFIRMED** (default is 1.0, not 0 — audit wrong on default) |
| BUG-002 | fp16 skip path misses `scaler.update()` → next step `RuntimeError` | `forge.py:288-316` | **CONFIRMED** |
| BUG-003 | `_AsyncLoader` one-shot → `epochs>1` silently trains nothing | `forge.py:347-370`, `data.py:508-558` | **CONFIRMED** (default `async_batches=True`) |
| BUG-004 | Resume poisons `_epoch_len` (skip-aware len) → epoch truncated/rolled early | `data.py:131,292,316-318,333,410-415` | **CONFIRMED** |
| BUG-005 | Kernel contiguity: `reshape` "raises"; rope `_ok` misses x/w contig; kl strides | `topk.py:228`, `softmax.py:91`, `rope.py:217`, `kl.py:136` | **PARTIAL** — rope **REAL**; reshape-raises **REBUTTED**; kl **REBUTTED** (reshape copies) |
| BUG-006 | Muon orthogonalizes embeddings; WORLD `all_reduce` breaks TP/PP/EP | `muon.py:133-138,185-239` | **CONFIRMED** as facts; docs never promised exclusion — design+doc fix |
| BUG-007 | Forge clip ignores offloaded grads; DDP local skip divergence | `forge.py:182-186,257-261,301-306` | **CONFIRMED** (first); divergence only via loss/spike (gn is synced) |
| BUG-008 | Port: bwd `dW` keeps `.to(W.dtype)` cast; bwd hardcodes `eps=1e-6` | `norm.py:174,220,441,496,1114` | **CONFIRMED** (both — my port's leftovers) |
| PERF-001 | topk serial kill-loops re-read row from HBM (K·N/BLOCK passes) | `topk.py:52-211` | **CONFIRMED** (structural; by-design cost) |
| PERF-002 | softmax 3 HBM passes; exp truncated through `O.dtype` before scaling | `softmax.py:26-81` | **CONFIRMED** (structural; docstring already explains passes) |
| PERF-003 | chunked KL `save_for_backward` accumulates → N×V peak, not one tile | `kl.py:149-176,205-226` | **CONFIRMED** (docstring overclaims) |
| PERF-004 | `add_rms` bwd stores `dx` twice (identical values) | `norm.py:1077-1085,1141-1147` | **CONFIRMED** (by design; aliasing return unsafe) |
| PERF-005 | EP dispatch: `experts()` per expert in loop, one-hot counts; per-param `all_reduce` | `moe.py:256-267`, `parallel/native.py:92-95` | **CONFIRMED** |
| PERF-006 | Forge omits DDP `no_sync`; eval host-stalls per batch; "double moves" | `forge.py:236-245,269,381-384`, `muon.py:226-267` | **PARTIAL** — no_sync+eval **CONFIRMED**; double-move **REBUTTED** (same-device `.to` is a no-op) |

---

## Evidence & remediation per finding

### BS-001 — autotune fused sweep is placebo · CONFIRMED
`autotune.py` fused block loops `block_m / warps / block` trial names but every trial runs
the **same** `norm.rms_norm(x,w,1e-6)` / `swiglu.swiglu(g,u)` call — the loop variables are
never passed to any launch. The only place they could apply is `tune(...)` launch metadata
chosen inside the kernel wrappers, which this code never influences. `triton.json` is
written (`autotune.py:327`) but read back only by the same merge code — nothing in the
trainer, `enable()`, or kernels consumes it.
**Fix:** make the sweep actually vary the launch metadata the kernels expose (or remove the
fake trials and record honestly which knob is kernel-internal); wire the cache file to a
consumer or drop it.

### BS-002 — offload overlap never engages · CONFIRMED (structural)
`OffloadedAdamW.step()` runs *after* the whole chunk loop, and by then `_pending` is
always empty: `Trainer._fwd_bwd_micro` calls `off_opt.collect()` every micro-batch when
`off_overlap` is set, and `Trainer._clip` → `clip_grad_norm_` → `collect()` flushes the
rest before `step()` is ever reached. First-touch path writes straight into `cgrad`
(`(p, None)` pending entries); other cases stage through a per-param pinned buffer — which
is also why deferred flushing cannot be enabled naively (buffers are reused). Net effect:
correct, but **serialized D2H → clip → Adam**, i.e. the chunked pipeline in
`offload.py:163-228` is dead code in the standard flow, and the overlap docstring's
timings math is never achieved.
**Fix:** documentation honesty (state serial semantics + measured expectations); keep the
chunk machinery behind an explicitly-supported path only if a safe per-chunk buffer design
is added later.

### BS-003 — kbench NaN clobber · CONFIRMED
`kbench.py` wraps **both** the inductor timing and our-op compiled timing in one
`try/except`: if `_compiled(case, True)` raises after `_compiled(case, False)` succeeded,
the `except` sets `ms_ind = ms_kernel_c = nan`, erasing a valid inductor measurement. The
qk_norm_rope row's `inductor = nan (compile failed)` in `PERF.md` §13 is therefore not
evidence that **inductor** failed — the failure cause is whichever of the two compiles
raised inside the shared block, and the report attributes it to the wrong column.
**Fix:** separate `try` per compile, per-column error notes in the table; annotate the
`PERF.md` §13 row.

### BS-004 — fake lazy imports · CONFIRMED
`forgemind/__init__.py:4` eagerly does `from .trainer import …`; `trainer.py:19` eagerly
does `from .kernels import available` → `kernels/__init__` imports `_common` (triton),
all kernel modules and registers the custom ops. The `__getattr__` lazy hook
(`"kernels", "forge", …`) is therefore dead for `kernels` (and `forge`, imported at
`__init__.py:12`). `fm.nn` (→ `kernels`) and the docs' "triton only when you touch it"
promise do not hold.
**Fix:** move `kernels_available` to a function-local import in `trainer.py` (its single
use is a warning message), drop the dead `"forge"` entry, correct the `__getattr__`
docstring; verify with `-X importtime` that `import forgemind` no longer loads `triton`.

### BS-005 — enable() scope vs showcase · PARTIAL
The "deception" sub-claim is **not supported**: README:238 says exactly *"patch
RMSNorm/SwiGLU modules in place"*, docs:1038 spells `enable(model, norm=True,
swiglu_mlp=True)` — scope is documented truthfully. The observed gaps are real, though:
* `Router.forward` (`moe.py:69-70`) uses `logits.softmax` / `probs.topk` — never
  `fm.kernels.softmax/topk`.
* `GPT`/`Block.forward` (`gpt.py:174-175`) never uses `add_rms_norm` or `qk_norm_rope`
  (GPT has its own rope tables; the pre-norm pattern only matches `add_rms` on the FFN
  sublayer).
* `kl.py` registers no `custom_op` (uses plain `autograd.Function`), unlike norm/rope/
  swiglu — so export/AOTI graphs for KL break out.
**Fix:** document the wiring table (which models use which kernels); optional router
rewiring is a GPU-benchmarkable change, not safe blind.

### BS-006 — sync counter blind spots + non-reverted enable · CONFIRMED (both)
* `perf.py:_SYNC_METHODS = (item, tolist, numpy, __float__, __int__, __bool__, cpu)` misses
  `torch.cuda.synchronize`, `Stream.synchronize`, `Event.synchronize` — and **offload
  itself** calls `Event.synchronize()` (`offload.py:160`) and `Stream.synchronize()`
  (`offload.py:175`), so the library's own host stalls are invisible to
  `count_host_syncs()`.
* `autotune.py:247-250`: `measure("fused_kernels=on", lambda: enable(target.raw_model))`
  passes `setup=` with **no `teardown=`** (`measure(label, setup, teardown, …)`,
  `autotune.py:201`) — a sweep run under `apply=False` leaves the model permanently
  patched.
**Fix:** patch the three CUDA sync methods in `_SYNC_METHODS` machinery (counted with the
same own/external split); add a `revert()` in `apply.py` (store original forwards on
patch) and pass it as `teardown=`.

### BUG-001 — clip=0 skips every step · CONFIRMED (with correction)
Default `Forge(clip=1.0)` — the audit's "defaults to clip=0" is wrong. But the docstring
*"0 disables clipping"* makes `clip=0` a supported configuration, and with the default
`skip_nonfinite=True`: `_clip_coef` returns `None` → `gn_v = nan` (sentinel) →
`finite = isfinite(loss) and isfinite(nan) = False` → `skipped=True` → `zero_grad`,
return — **100 % of steps skipped forever**, silently (`_skipped` counts up).
**Fix:** when clipping is disabled, exclude `gn_v` from the finiteness test.

### BUG-002 — fp16 skip path skips `scaler.update()` · CONFIRMED
`_update` calls `scaler.unscale_` first; the `skipped` branch returns **without**
`scaler.update()`. Next `_update` → `scaler.unscale_` raises
`RuntimeError: unscale_() has already been called …` (fp16 + scaler only; bf16 typically
runs without scaler).
**Fix:** call `scaler.update()` on the skipped branch when a scaler exists.

### BUG-003 — one-shot async loader kills epochs>1 · CONFIRMED
`_AsyncLoader` captures `self._it = iter(loader)` once; the producer thread ends on
`StopIteration` → `_END` → `__next__` closes the loader and `_closed` short-circuits all
later reads. `Forge.fit` wraps the loader **once** outside the epoch loop, so with the
default `async_batches=True`, epoch ≥ 1 iterates an immediately-exhausted loader —
**silently zero batches** (the `for` swallows `StopIteration`).
**Fix:** build the wrapper per epoch inside `fit` and `_shutdown` it at each epoch end
(under the existing `finally`), or make `_AsyncLoader` restartable; add a regression test
that epochs=2 actually runs twice the batches.

### BUG-004 — resume poisons `_epoch_len` · CONFIRMED
`resume()` sets `sampler.skip = batches·batch_size` and `_start_index = batches`.
`_restart` then sets `self._epoch_len = _loader_len(loader)` — but `len(loader)` goes
through `ResumableSampler.__len__ = max(0, per_rank - skip)`, i.e. the **remaining**
length (e.g. 40) while `_run` starts at `index = batches` (e.g. 60): `60 ≥ 40` (indeed
`61 ≥ 40` after the first batch) → premature `_new_epoch` roll — the resumed epoch
truncates after ~1 batch and later epochs are mislabeled/short.
**Fix:** `_epoch_len` must be the *full* epoch length (`sampler.per_rank` when present,
minus nothing); the skip only offsets the start. Regression test: resume mid-epoch,
assert remaining batches are produced before the epoch rolls.

### BUG-005 — contiguity guards · PARTIAL
* **rope: REAL.** `_ok` checks only `cos/sin.is_contiguous()`; `_launch_qk_norm_rope`
  passes raw `x, w` to the Triton kernel (the only `.contiguous()` in rope.py is on `g`
  in backward). Non-contiguous `x`/`w` → silently wrong results (no copy, no fallback).
  **Fix:** require `x.is_contiguous() and w.is_contiguous()` in `_ok`, plus defensive
  `.contiguous()` inside `_rope_op_fwd` (the custom-op path can bypass `_ok`).
* **topk/softmax `reshape(-1, n)` "raises": REBUTTED.** `Tensor.reshape` *views when
  possible, otherwise copies* — it never raises for non-contiguous input (only `view`
  does); the kernel then receives correct contiguous rows. Cost: a hidden copy on
  non-contiguous inputs — worth a `stride(-1)==1` fast-gate note, not a correctness bug.
* **kl "no `stride(-1)==1`": REBUTTED for correctness.** `s2/t2 = …reshape(-1, v)` gives
  the kernel contiguous rows (copy when needed), `gbuf` is fresh contiguous or the copy
  itself; row stride is passed explicitly (`s2.stride(0)`). Only effect: `reuse=True`
  silently loses its alloc saving on non-contiguous students (documented, harmless).

### BUG-006 — Muon scope · CONFIRMED as facts, docs nuance
* `mats = [p for p in items if p.ndim >= 2]` — token embedding (`[V, d]`) and `lm_head`
  **do** enter the Muon group; documentation never promised exclusion (`ndim>=2` rule is
  documented at README:262/docs:871), but README's "(Linear, conv filters, …)" leaves it
  ambiguous. Embedding-NS is a legitimate design (used by several production Muons) — the
  bug is ambiguity + no per-architecture default.
* Distributed: `dist.all_reduce(out)` on the **WORLD** group assumes identical bucket
  lists on every rank — true for DDP replication, **false for TP/PP/EP** (different
  param subsets/shapes per rank → shape mismatch or garbage mix). No guard, no warning.
**Fix:** docs state the ndim rule for embeddings explicitly + add a `group=` parameter
(default WORLD) so TP/PP users pass their replication group; note the limitation in
`documentation.md`.

### BUG-007 — Forge skips offloaded grads · CONFIRMED (first part)
`Forge._clip_coef` → `clip_grad_norm_(self.model.parameters(), clip)` — with
`OffloadedAdamW` the moved grads are `None`, so clipping sees only resident grads:
offloaded rows are applied **unclipped**, and `gn` under-reports. (`Trainer._clip`
handles this via `off_opt.clip_grad_norm_` — Forge doesn't.)
Second part (DDP divergence): `gn` itself is computed from allreduced grads → identical
on all ranks; only loss/spike conditions are rank-local, so divergence is possible but
narrower than claimed (spike/loss-driven skips).
**Fix:** in `Forge._clip_coef`, flush first (duck-typed `collect()` on the optimizer)
before clipping; under `dist`, align the skip decision with an allreduce-min of the
device-side finite flag (still one host read).

### BUG-008 — my port's leftovers · CONFIRMED (both)
* `norm.py:220` and `:496`: backward `dW` still does
  `normed.to(W.dtype.element_ty).to(tl.float32)` — but the forward now multiplies in fp32
  with a single rounding (comment at `:174`), so `dW = Σ dY·round_W(normed)` is
  inconsistent with the forward's `normed` and adds bf16 rounding error to the weight
  gradient. The stale comment *"The forward multiplies W by normed rounded to W's
  dtype"* contradicts the forward it describes.
* `_launch_rms_bwd` (`norm.py:1114`) hardcodes `eps=1e-6` — it only seeds the
  `_multirow_checked` self-check key, but a real `eps≠1e-6` run then performs the GPU
  self-check **twice** under two keys (cache pollution, extra launch cost).
**Fix:** drop the two stale casts (keep fp32 `normed` in `dW`), correct the comments, and
plumb the true `eps` through `ctx` into `_launch_rms_bwd`.

### PERF-001/002/004 — real structure, no correctness bug
* **PERF-001 (topk):** threshold kernel's K serial `argmax` iterations are
  register-resident (cheap); the selection kernel re-scans the row's `N` in 1024-lane
  chunks **once per kill** → ~(K+1)·N/BLOCK global-memory passes per row, serial inside
  one program. Real latency cost for large `n`, correct results. Documented cost; a fix
  needs a redesigned selection pass and GPU validation (out of scope without hardware).
* **PERF-002 (softmax):** three chunked passes by design (any row length — already
  documented in the docstring); `z` is stored to `O` in `O.dtype` and reloaded for
  scaling → one extra bf16 rounding of `exp` before `1/sum`. Small precision cost;
  two-pass rewrite needs GPU validation.
* **PERF-004 (add_rms bwd):** `dx == ddelta` mathematically (comment says so); computed
  once, stored twice = +1 write of traffic. Returning the *same tensor* for both inputs
  is an autograd aliasing hazard (later in-place grad accumulation would corrupt the
  other branch); inductor "wins" by dedup at graph level. Keep; document.

### PERF-003 — KL peak memory docstring · CONFIRMED
`kl_distill_loss` chunks with `_KLDivFn.apply` per chunk; each ctx saves its `gbuf`
(`[chunk, V]`) until backward — graph accumulation ⇒ peak **N×V** when grads are needed,
exactly what the *"peak memory is one [chunk, V] tile"* docstring denies. (Same peak as
torch's own path, but the docstring must be honest.)
**Fix:** docstring correction (+ note the grad-free case really is one tile).

### PERF-005 — EP dispatch & grad reduction · CONFIRMED
* `moe.py:256-267`: `self.experts(per_e_in[e], [c if i == e else 0 …])` — one module
  invocation **per local expert** with a fabricated one-hot count vector; a single call
  with the full count vector would dispatch once. Refactor changes token-order logic
  (needs GPU tests) → document now, refactor only with hardware.
* `parallel/native.py:92-95`: `for g in grads: dist.all_reduce(g, …)` — one collective
  **per parameter** (latency-bound). **Fix now:** `all_reduce_coalesced(grads, group)`
  with a per-tensor fallback (semantics-preserving, no timing claim without GPU).

### PERF-006 — Forge loop overheads · PARTIAL
* **`no_sync()` missing in Forge: CONFIRMED** — `Forge.fit/step` never wraps accumulating
  micro-steps in `model.no_sync()` (Trainer does, `trainer.py:261`), so DDP-wrapped Forge
  allreduces every micro-batch. **Fix:** mirror Trainer's `no_sync` pattern.
* **Eval host stall: CONFIRMED** — `evaluate` does `total += float(loss.detach())` per
  batch (D2H sync each iteration). **Fix:** accumulate on device, one `.item()` at the
  end.
* **"Double batch moves": REBUTTED** — `Forge.step`'s `_to_device(batch, self.device)`
  after an async loader already moved the batch is a *same-device `.to()` = no-op*
  (no copy). Not a cost.

---

## Fix plan (this branch)

**Correctness (must fix):** BUG-001, BUG-002, BUG-003, BUG-004, BUG-005(rope),
BUG-007(collect-before-clip), BUG-008(both), BS-003 (kbench + PERF §13 annotation).
**Scope/safety fixes:** BS-004 (lazy imports), BS-006 (sync methods + revert teardown),
BUG-006 (`group=` + docs), PERF-005 (coalesced all_reduce), PERF-006 (`no_sync`, eval
device accumulation).
**Honest documentation (no blind perf rewrites without GPU):** BS-001, BS-002,
BS-005 (wiring table), PERF-001/002/003/004, plus the false sub-claims rebutted with
tests (BUG-005 reshape, PERF-006 double-move).

## Remediation status (final, this branch)

**Fixed in code** (each with a regression test unless noted):
BS-001 (`autotune_kernels` rewritten: real measurements, placebo loops and the dead
`triton.json` write removed) · BS-003 (kbench: one `try` per compile; `PERF.md` §13
footnote re-attributes the `nan`) · BS-004 (`trainer` + `moe.py` dynamo decorators made
lazy; `import forgemind` no longer loads `forgemind.kernels`, `triton` or `torch._dynamo`
— measured 3.9 s → 1.4 s; subprocess test pins it) · BS-006 (sync counter covers
`torch.cuda`/`Stream`/`Event.synchronize`; fused autotune trial gained `teardown=` via the
new `fm.kernels.revert()`) · BUG-001 (`gn is None` sentinel excluded from the finite
check) · BUG-002 (`scaler.update()` on the skipped branch, arming-protocol test) ·
BUG-003 (`fit` builds one loader wrapper per epoch) · BUG-004 (`_restart` measures the
full epoch length with the skip held at 0; threaded resume regression test) · BUG-005
rope part (`_ok` requires contiguous `x`/`w`; defensive `.contiguous()` in the
custom-op path) · BUG-006 (`Muon(..., group=)` + docs: `ndim ≥ 2` embedding rule and the
TP/PP/EP caveat) · BUG-007 (`_clip_coef` prefers the optimizer's own flush-and-clip;
spy test) · BUG-008 (stale `dW` casts removed at both sites; real `eps` plumbed through
`ctx.eps`; non-default-eps backward parity test) · PERF-003 (docstring now states the
N × V peak) · PERF-005 (`all_reduce_coalesced` with per-tensor fallback) · PERF-006
(Forge mirrors Trainer's `no_sync()` during accumulation; `evaluate` = one host read;
value-contract test).

**Rebutted with evidence** (no code change required):
BUG-005 `reshape(-1, n)` claim (reshape views-or-copies, never raises — direct test) and
the KL-stride claim (reshape gives the kernel contiguous rows) · PERF-006 "double batch
moves" (same-device `.to()` is a no-op) · BS-005's "misleading README" framing (README:238
/ docs:1038 scope `enable()` truthfully to RMSNorm+SwiGLU — the wiring gaps are real but
are opportunities, recorded below, not mislabels) · BUG-001's "defaults to clip=0"
(default is 1.0; the bug was in the documented `clip=0` path) · BUG-007 DDP divergence
narrowed: the default `skip_nonfinite` decision is driven by the synced grad-norm (rank-
identical after DDP allreduce); only the opt-in spike factor is rank-local.

**Documented honestly, no blind changes** (perf rewrites need the user's GPU):
BS-002 (§5.7: the chunk pipeline never engages in the Trainer flow — serialized D2H/clip/
Adam is the truth until a per-chunk buffer redesign is benchmarked) · BS-005 (wiring
inventory in §14: Router uses torch `softmax`/`topk`; GPT uses its own rope and never
`add_rms_norm`/`qk_norm_rope`; KL registers no `custom_op`) · PERF-001 (topk's ~(K+1)·N
serial passes) · PERF-002 (softmax three-pass design + output-dtype exp rounding) ·
PERF-004 (`dx == ddelta` stored twice; returning one aliased tensor for both grads would
be an autograd hazard) · PERF-005's per-expert `experts()` loop.

**Gates on this branch (all green):**
CPU `pytest tests/ -q` → **230 passed, 19 skipped** (218 + 12 new regression tests) ·
`TRITON_INTERPRET=1 pytest tests/ -q` → **248 passed, 1 skipped** (236 + 12) ·
`TRITON_INTERPRET=1 tests/kernel_interp_check.py` → **66/66** · `tools/tilelens_probe.py`
→ exit 0 · `pytest tests/test_docs.py -q` → **6/6** · `pyflakes forgemind/kernels/` →
clean (whole-package pyflakes: only the pre-existing `models/`/`parallel/` notes).

GPU speed claims remain deferred to the user's hardware: re-run
`kbench --ops rms_norm,add_rms_norm,swiglu,swiglu_split,qk_norm_rope --check` for the
"after" table (the qk row's 9.71× vs eager stands; its `nan` inductor cell now carries a
corrected attribution in `PERF.md` §13).

## Follow-up: kernel correctness and VRAM measurement

The earlier remediation counts above are historical, not the current suite totals.
A later check fixed the cross-entropy custom-op fake implementation for
multidimensional `reduction="none"` to match `targets.shape`, with an interpreter
fullgraph forward/backward regression test. Eager LayerNorm inference now omits two
fp32 per-row backward-statistic buffers, a logical `8 × rows` byte reduction per
call; training and the custom-op/compiled path are unchanged. This is not a
measured whole-model VRAM reduction.

`kbench --memory --no-compile` now reports isolated CUDA incremental peak
allocated bytes against eager torch, not Inductor. `bench --fused-ops` permits a
separate-process whole-model comparison against the unpatched default; subtract
`peak_mem_bytes` values for actual savings. See [PERF.md](PERF.md#vram-measuring-kernel-and-whole-model-impact)
for scope, commands and limitations. The comparison has not been run on a GPU here.
On the CPU/interpreter validation run preceding the final benchmark harness test,
pytest reported 252 passed / 36 skipped and 287 passed / 1 skipped respectively;
offline compilation passed 14/14 and interpreter kernel checks passed 109/109.
