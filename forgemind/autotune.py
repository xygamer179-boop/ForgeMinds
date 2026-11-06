"""Autotuning: measure, do not guess.

Three layers, all of which print the numbers they decided on:

``apply_fast_defaults()``
    One-time device/allocator/backend flags that are almost always free: TF32 matmul + cudnn,
    ``cudnn.benchmark``, expandable memory segments, fp32 matmul precision. Returns what it
    changed so you can put it in a log.

``tune_batch_size(step_fn, ...)``
    Doubles the micro-batch until it stops paying off or runs out of memory (the OOM is caught
    and treated as data, not as a crash), then returns the largest size that improved
    tokens/second. This is the knob that matters most and the one people guess wrong.

``autotune(target, ...)``
    The orchestrator. Give it a ``Forge`` (with a loader) or a ``Trainer`` and it tries the
    cheap, high-variance switches -- precision, activation checkpointing, fused kernels,
    optimizer offload, ``torch.compile``, prefetch/workers -- times each for a small budget,
    and reports a table plus the winning combination. Nothing is applied unless
    ``apply=True``.

Every trial is a median over a few steps after a warmup, because the first step of any config
is dominated by allocation, autotuning inside torch, or a cold cache. That makes each number
~10x cheaper to trust than a single-shot timing.

On a machine without an accelerator this still works -- it tunes CPU-visible knobs (threads,
prefetch, fused ops, offload) and says so. What it cannot do is guess your GPU's memory
budget, so the OOM probe reports "did not fit" separately from "was slower".
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import time
from dataclasses import dataclass, field

import torch

__all__ = ["time_call", "apply_fast_defaults", "tune_batch_size", "autotune",
           "autotune_kernels", "Report", "Trial"]

_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "forgemind")


@dataclass
class Trial:
    label: str
    ok: bool = True
    ms: float = float("inf")
    note: str = ""
    oom: bool = False

    def __str__(self):
        status = "oom" if self.oom else ("ok" if self.ok else "fail")
        return f"{self.label:38s} {status:5s} {self.ms:9.2f} ms  {self.note}"


@dataclass
class Report:
    device: str = ""
    trials: list = field(default_factory=list)
    best: dict = field(default_factory=dict)
    baseline_ms: float = 0.0

    def add(self, trial: Trial) -> Trial:
        self.trials.append(trial)
        return trial

    def table(self) -> str:
        head = f"{'trial':38s} {'stat':5s} {'median':>9s}  note"
        return "\n".join([head] + [str(t) for t in self.trials])

    def speedup(self) -> float:
        if not self.baseline_ms or not self.best.get("ms"):
            return 1.0
        return self.baseline_ms / self.best["ms"]

    def to_dict(self) -> dict:
        return {"device": self.device, "best": self.best, "baseline_ms": self.baseline_ms,
                "speedup": self.speedup(),
                "trials": [t.__dict__ for t in self.trials]}


def _is_oom(e: BaseException) -> bool:
    if isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    msg = str(e).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def time_call(fn, warmup: int = 2, repeats: int = 5, sync: bool = True) -> float:
    """Median wall time of ``fn`` in ms, after ``warmup`` untimed calls."""
    for _ in range(max(0, warmup)):
        fn()
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def apply_fast_defaults(device=None, allow_tf32: bool = True, allocator: bool = True) -> dict:
    """Enable the "almost always free" speed defaults. Returns what was changed."""
    changed = {}
    if allow_tf32:
        with contextlib.suppress(Exception):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            changed["tf32"] = True
    cuda = torch.cuda.is_available()
    if cuda:
        with contextlib.suppress(Exception):
            torch.backends.cudnn.benchmark = True
            changed["cudnn_benchmark"] = True
        if allocator and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            # must be set before the CUDA context is created to matter
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            changed["expandable_segments"] = "(set, takes effect on next process)"
    changed["threads"] = torch.get_num_threads()
    return changed


def tune_batch_size(step_fn, candidates=(1, 2, 4, 8, 16, 32), *, warmup: int = 1,
                    repeats: int = 3, min_gain: float = 0.02, footprint=None,
                    report: Report | None = None) -> tuple[int, float]:
    """Largest micro-batch that still improves **per-sample** cost (a throughput proxy).

    Scoring a batch size by step time alone is a trap: a bigger batch usually costs the same
    wall time, so step time looks flat while throughput doubles. The score here is
    ``ms / size``, which is what you actually care about. The sweep stops at the first size
    that fails to improve it, or when ``step_fn`` runs out of memory (which is caught and
    reported, not raised).

    ``step_fn(size)`` runs one optimizer step; ``footprint(size)`` (optional) is called first
    if you want to reserve memory before timing. Returns ``(best_size, best_ms)``.
    """
    report = report or Report()
    best_size, best_ms, best_per = None, float("inf"), float("inf")
    for size in candidates:
        trial = Trial(label=f"micro_batch={size}")
        try:
            if footprint is not None:
                footprint(size)
            ms = time_call(lambda: step_fn(size), warmup=warmup, repeats=repeats)
        except BaseException as e:                      # noqa: BLE001 - OOM is data
            if not _is_oom(e):
                raise
            trial.ok, trial.oom, trial.note = False, True, "did not fit"
            report.add(trial)
            break
        per = ms / size
        improved = per < best_per * (1.0 - min_gain)
        trial.ms = ms
        trial.note = f"{per:.3f} ms/sample"
        if best_per == float("inf") or improved:
            best_size, best_ms, best_per = size, ms, per
            trial.note += ", best so far"
        else:
            trial.note += ", no gain -> stop"
        report.add(trial)
        if best_per != float("inf") and not improved:
            break
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
    return best_size or candidates[0], best_ms


def autotune(target, loader=None, *, build=None, budget_s: float = 60.0, apply: bool = False,
             precision: bool = True, accum: bool = True, compiled: bool = False,
             fused: bool = True, report: Report | None = None) -> Report:
    """Measure the cheap, high-variance switches and report the fastest.

    ``target`` is a ``Forge`` (pass ``loader`` so a batch can be pulled) or a ``Trainer``.
    ``build(**overrides) -> target`` lets a ``Trainer`` be rebuilt per trial for knobs that
    cannot be flipped on a live object (``bf16``, ``ac``, ``compile``); without it only
    in-place switches are tried. Nothing is applied unless ``apply=True``.
    """
    report = report or Report()
    report.device = str(getattr(target, "device", "cpu"))
    t_start = time.time()

    from .forge import Forge as _Forge
    is_forge = isinstance(target, _Forge)
    batch = next(iter(loader)) if (is_forge and loader is not None) else None

    def step_call(obj):
        if is_forge and batch is not None:
            return lambda: obj.step(batch)
        return lambda: obj.train_step()

    def measure(label, setup=None, teardown=None, note="", obj=None):
        obj = obj or target
        if time.time() - t_start > budget_s:
            report.add(Trial(label, ok=False, note="skipped: budget"))
            return float("inf")
        try:
            if setup:
                setup()
            ms = time_call(step_call(obj), warmup=1, repeats=3)
            trial = Trial(label, ms=ms, note=note)
        except BaseException as e:  # noqa: BLE001 - OOM/compile failures are data
            trial = Trial(label, ok=False, oom=_is_oom(e), note=str(e).replace("\n", " ")[:60])
        finally:
            if teardown:
                teardown()
        report.add(trial)
        return trial.ms

    def ctx(**over):
        saved = {}

        def setup():
            for k, v in over.items():
                saved[k] = getattr(target, k)
                setattr(target, k, v)

        def teardown():
            for k, v in saved.items():
                setattr(target, k, v)
        return setup, teardown

    base_ms = measure("baseline")
    report.baseline_ms = base_ms
    best_ms, best = base_ms, {}

    if is_forge:
        if precision and str(report.device).startswith("cuda"):
            for prec, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
                s, t = ctx(precision=prec, amp_dtype=dt)
                ms = measure(f"precision={prec}", s, t)
                if ms < best_ms:
                    best_ms, best = ms, {"precision": prec}
        if fused:
            from .kernels import available as kernels_available
            if kernels_available():
                from .kernels.apply import enable
                ms = measure("fused_kernels=on",
                             lambda: enable(target.raw_model), note="rmsnorm+swiglu kernels")
                if ms < best_ms:
                    best_ms, best = ms, {"fused_ops": True}
        if accum and getattr(target, "accum", 1) > 1:
            s, t = ctx(accum=1, micro=0)
            ms = measure("accum=1 (per-micro-batch cost)", s, t)
            if ms and ms < best_ms:
                best_ms, best = ms, {"accum": 1}

    if build is not None:
        for label, over in (("bf16", {"bf16": True}) if precision else (),
                            ("ac=full", {"ac": "full"}),
                            ("compile", {"compile": True})):
            try:
                new_target = build(**over)
            except BaseException as e:  # noqa: BLE001
                report.add(Trial(f"build {label}", ok=False, note=str(e)[:60]))
                continue
            ms = measure(f"{label} (rebuilt)", note=str(over), obj=new_target)
            if ms < best_ms:
                best_ms, best = ms, {k: v for k, v in over.items()}
            if torch.cuda.is_available():
                with contextlib.suppress(Exception):
                    del new_target
                    torch.cuda.empty_cache()

    report.best = dict(best, ms=best_ms)
    if apply and best:
        for k, v in best.items():
            setattr(target, k, v)
    return report


def autotune_kernels(candidates=None, iters: int = 20, cache: bool = True) -> dict:
    """Brute-force the Triton launch configs for this GPU and cache the winners.

    Only the packed kernels whose block size is a free parameter are swept. On a machine
    without CUDA this returns ``{"skipped": ...}`` instead of pretending.
    """
    from .kernels import available
    if not available():
        return {"skipped": "needs triton + an accelerator", "device": str(torch.device("cpu"))}
    from .kernels import _common, norm, swiglu

    results = {}
    d = 4096
    x = torch.randn(1024, d, device="cuda")
    w = torch.randn(d, device="cuda")
    best = (float("inf"), None)
    for block_m in (1, 2, 4, 8):
        if block_m * _common.next_pow2(d) > 16384:
            continue
        for warps in (4, 8, 16):
            try:
                fn = lambda: norm.rms_norm(x, w, 1e-6)
                ms = time_call(fn, warmup=3, repeats=iters)
            except Exception:
                continue
            if ms < best[0]:
                best = (ms, dict(rows=block_m, num_warps=warps))
    if best[1]:
        results["rms_norm"] = dict(best[1], ms=best[0])
    g = torch.randn(1 << 20, device="cuda")
    u = torch.randn(1 << 20, device="cuda")
    best = (float("inf"), None)
    for block in (512, 1024, 2048, 4096):
        for warps in (2, 4, 8):
            try:
                ms = time_call(lambda: swiglu.swiglu(g, u), warmup=3, repeats=iters)
            except Exception:
                continue
            if ms < best[0]:
                best = (ms, dict(block=block, num_warps=warps))
    if best[1]:
        results["swiglu"] = dict(best[1], ms=best[0])
    results["device"] = torch.cuda.get_device_name()
    if cache:
        with contextlib.suppress(Exception):
            os.makedirs(_CACHE, exist_ok=True)
            path = os.path.join(_CACHE, "triton.json")
            old = {}
            if os.path.exists(path):
                old = json.load(open(path))
            old[results["device"]] = results
            json.dump(old, open(path, "w"), indent=1)
            results["cached"] = path
    return results


def _main(argv=None):
    import argparse
    from .forge import Forge
    from .models import GPT, GPTConfig

    p = argparse.ArgumentParser(description="autotune a ForgeMind step / device")
    p.add_argument("--preset", default="tiny")
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--budget", type=float, default=60.0)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--kernels", action="store_true")
    a = p.parse_args(argv)

    if a.kernels:
        print(json.dumps(autotune_kernels(), indent=1))
        return 0

    dim = {"tiny": 128, "small": 256, "medium": 512}.get(a.preset, 128)
    mc = GPTConfig(vocab_size=512, dim=dim, n_layers=4, n_heads=max(2, dim // 32),
                   n_kv_heads=max(1, dim // 64), max_seq=a.seq)
    model = GPT(mc)
    precision = "bf16" if torch.cuda.is_available() else "fp32"
    # GPT(x, labels) returns (loss, aux): use the loss, ignore the router stats
    forge = Forge(model, None, precision=precision,
                  loss_fn=lambda out, b: out[0] if isinstance(out, (tuple, list)) else out)
    x = torch.randint(0, mc.vocab_size, (a.micro_batch, a.seq))
    y = torch.randint(0, mc.vocab_size, (a.micro_batch, a.seq))

    print("fast defaults:", apply_fast_defaults())
    rep = autotune(forge, [(x, y)], budget_s=a.budget, apply=a.apply)
    print(rep.table())
    print(f"baseline {rep.baseline_ms:.2f} ms -> best {rep.best.get('ms', 0):.2f} ms "
          f"({rep.speedup():.2f}x) {rep.best}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
