"""Offline compile check: catches kernel compile crashes WITHOUT a GPU.

``triton.compile`` runs the full AST->TTIR->PTX pipeline for a fixed target, so type
errors that only appear at real launch (loop-carried dtype flips, bare python globals
inside ``@triton.jit`` bodies) are caught right here on a CPU-only box:

  * topk  - the ``Loop-carried variable last ... fp32 ... bf16`` crash users hit with
            bf16 inputs (both the threshold and the select kernel)
  * softmax - full-graph bf16 path never exercised on GPU in this repo's tests
  * kl    - same class, also never run on GPU here

Exit 0 = every kernel compiles; 1 = at least one fails (message printed).
"""
import sys
from pathlib import Path

# direct invocation puts tools/ (not the repo root) on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

# importlib: the package exposes functions named topk/softmax that SHADOW the
# submodules, so plain `import forgemind.kernels.softmax as ...` would bind the fn
import importlib

_topk = importlib.import_module("forgemind.kernels.topk")
sm = importlib.import_module("forgemind.kernels.softmax")
kl = importlib.import_module("forgemind.kernels.kl")

TARGET = GPUTarget(backend="cuda", arch=80, warp_size=32)  # sm_80: generic modern NVIDIA

CASES = [
    # (label, function, signature, constexprs)
    ("topk.threshold(bf16)", _topk._topk_threshold_kernel,
     {"T": "*fp32", "X": "*bf16", "N": "i32", "K": "i32",
      "M": "i32", "STRIDE": "i32", "BLOCK": "i32"},
     {"BLOCK": 512}),
    ("topk.select(bf16)", _topk._topk_select_kernel,
     {"V": "*bf16", "I": "*i64", "X": "*bf16", "T": "*fp32",
      "N": "i32", "K": "i32", "BLOCK": "i32"},
     {"BLOCK": 1024}),
    ("softmax(bf16)", sm._softmax_kernel,
     {"O": "*bf16", "X": "*bf16", "N": "i32", "BLOCK": "i32", "EVEN": "i32"},
     {"BLOCK": 512, "EVEN": 1}),
    ("kl(bf16)", kl._kl_fwd_bwd_kernel,
     {"S_ptr": "*bf16", "s_stride": "i32", "T_ptr": "*bf16", "t_stride": "i32",
      "G_ptr": "*bf16", "g_stride": "i32", "Loss_ptr": "*fp32", "V": "i32",
      "inv_N": "fp32", "BLOCK_V": "i32", "WRITE_GRAD": "i32", "SINGLE": "i32"},
     {"BLOCK_V": 32, "WRITE_GRAD": 1, "SINGLE": 0}),
]


def main() -> int:
    failed = 0
    for label, fn, signature, constexprs in CASES:
        try:
            triton.compile(ASTSource(fn=fn, signature=signature, constexprs=constexprs),
                           target=TARGET)
            print(f"OK    {label}")
        except Exception as exc:  # noqa: BLE001 - the message is the result
            failed += 1
            print(f"FAIL  {label}: {type(exc).__name__}: {exc}")
    print(f"{len(CASES) - failed}/{len(CASES)} kernels compile offline")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
