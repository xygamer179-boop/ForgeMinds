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
ln = importlib.import_module("forgemind.kernels.layernorm")
ce = importlib.import_module("forgemind.kernels.cross_entropy")
ge = importlib.import_module("forgemind.kernels.geglu")
r2 = importlib.import_module("forgemind.kernels.relu2")

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
    # ---- new ports: LayerNorm, cross-entropy, GEGLU, ReLU^2 (bf16 inputs, the
    # crash-class dtype for every loop-carried / scalar-argument rule) ----
    ("layernorm.fwd(bf16)", ln._ln_fwd,
     {"Y": "*bf16", "X": "*bf16", "W": "*bf16", "B": "*bf16", "Mu": "*fp32",
      "R": "*fp32", "Y_stride": "i32", "X_stride": "i32", "n_cols": "i32",
      "eps": "fp32", "BLOCK": "i32", "HAS_B": "i32", "SAVE_STATS": "i32"},
     {"n_cols": 768, "eps": 1e-5, "BLOCK": 1024, "HAS_B": 1, "SAVE_STATS": 1}),
    ("layernorm.fwd_inference(bf16)", ln._ln_fwd,
     {"Y": "*bf16", "X": "*bf16", "W": "*bf16", "B": "*bf16", "Mu": "*bf16",
      "R": "*bf16", "Y_stride": "i32", "X_stride": "i32", "n_cols": "i32",
      "eps": "fp32", "BLOCK": "i32", "HAS_B": "i32", "SAVE_STATS": "i32"},
     {"n_cols": 768, "eps": 1e-5, "BLOCK": 1024, "HAS_B": 1, "SAVE_STATS": 0}),
    ("layernorm.bwd_dw(bf16)", ln._ln_bwd_dw,
     {"dY": "*bf16", "dY_stride": "i32", "X": "*bf16", "X_stride": "i32",
      "W": "*bf16", "Mu": "*fp32", "R": "*fp32", "dX": "*bf16", "dX_stride": "i32",
      "dW": "*fp32", "dB": "*fp32", "n_rows": "i32", "rows_per_program": "i32",
      "n_cols": "i32", "HAS_B": "i32", "BLOCK": "i32",
      "TRAIN_W": "i32", "TRAIN_B": "i32"},
     {"n_cols": 768, "HAS_B": 1, "BLOCK": 1024, "TRAIN_W": 1, "TRAIN_B": 1}),
    ("layernorm.sum_partials", ln._ln_sum_partials,
     {"partials": "*fp32", "out": "*fp32", "n_programs": "i32", "n_cols": "i32",
      "BLOCK_ROWS": "i32", "BLOCK_COLS": "i32"},
     {"BLOCK_ROWS": 64, "BLOCK_COLS": 64}),
    ("cross_entropy.fwd(bf16)", ce._ce_fwd,
     {"Loss": "*fp32", "X": "*bf16", "X_stride": "i32", "T": "*i64",
      "n_cols": "i32", "ignore_index": "i32", "LS": "fp32", "BLOCK": "i32"},
     {"ignore_index": -100, "LS": 0.0, "BLOCK": 4096}),
    ("cross_entropy.bwd(bf16)", ce._ce_bwd,
     {"DX": "*bf16", "X": "*bf16", "X_stride": "i32", "T": "*i64",
      "Scale": "*fp32", "scale_stride": "i32", "n_cols": "i32",
      "ignore_index": "i32", "LS": "fp32", "BLOCK": "i32"},
     {"ignore_index": -100, "LS": 0.1, "BLOCK": 4096}),
    ("geglu.fwd(bf16)", ge._geglu_fwd,
     {"G": "*bf16", "U": "*bf16", "Out": "*bf16", "N": "i32",
      "BLOCK": "i32", "EVEN": "i32"},
     {"BLOCK": 1024, "EVEN": 1}),
    ("geglu.bwd(bf16)", ge._geglu_bwd,
     {"DY": "*bf16", "G": "*bf16", "U": "*bf16", "DG": "*bf16", "DU": "*bf16",
      "N": "i32", "BLOCK": "i32", "EVEN": "i32"},
     {"BLOCK": 1024, "EVEN": 0}),
    ("relu_squared.fwd(bf16)", r2._relu2_fwd,
     {"X": "*bf16", "Out": "*bf16", "N": "i32", "BLOCK": "i32", "EVEN": "i32"},
     {"BLOCK": 1024, "EVEN": 1}),
    ("relu_squared.bwd(bf16)", r2._relu2_bwd,
     {"DY": "*bf16", "X": "*bf16", "DX": "*bf16", "N": "i32",
      "BLOCK": "i32", "EVEN": "i32"},
     {"BLOCK": 1024, "EVEN": 0}),
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
