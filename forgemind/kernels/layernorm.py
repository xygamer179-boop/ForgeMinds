"""LayerNorm (mean-centering + variance normalisation) with weight, bias and backward.

Drop-in for ``F.layer_norm(x, (d,), w, b, eps)``: same signature, same numbers, one
kernel family. The forward saves fp32 ``mean``/``rstd`` per row for the backward; the
backward writes ``dX`` per row and reduces fp32 ``dW``/``dB`` partials in a fixed tile
order (device-deterministic -- the same scheme ``norm.py`` uses for RMSNorm).

Ported from Unsloth's ``layernorm.py`` (Apache-2.0, Karpathy/llm.c lineage for the row
math; the dW/dB partial scheme follows flash-linear-attention's ``layer_norm_bwd_kernel_row``).
Adaptations -- every one of them a place where the port would otherwise import a problem:

* **no eager-launch fast path**: Unsloth's ``triton_launch.py`` is AGPL-3.0 and is not
  copied; launches go through standard Triton, as every kernel here does.
* **no ``enable_fp_fusion`` / device plumbing**: ``torch_gpu_device``, NPU/XPU/HIP
  branches and Unsloth's compile-time settings cache are dropped; launch metadata comes
  from ``_common.tune`` (no-op under the interpreter).
* **``eps`` is a ``tl.constexpr`` and is widened with ``tl.full``**, never with
  ``eps.to(...)`` -- calling a method on a scalar kernel argument is exactly what breaks
  under ``torch.compile`` (Liger issue #1601: eager gets a Triton scalar, generate_ttir
  passes a plain Python float). ``tests/test_kernel_hygiene.py`` bans the pattern
  repo-wide.
* **int64 indexing is gated, not switched**: the package refuses ``numel >= 2**31``
  in ``_ok``, so row offsets stay in int32 (Unsloth's ``LONG_INDEXING`` constexpr
  becomes unnecessary).
* **the dX-only backward keeps its own launch** (no weight/bias grad requested), and the
  partial reduction reuses this repo's interpreter-safe ``while`` loop instead of
  ``tl.range(..., num_stages=)`` (the CPU interpreter cannot index a runtime bound).
* **HF-module patching, Gemma variants and the multi-row self-check are out of scope**:
  this file exposes the function; ``apply.py`` wires modules.
"""

# Portions of this file are adapted from Unsloth's ``unsloth/kernels/layernorm.py``:
#
#   Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#   Copyright 2024-present Andrej Karpathy & the llm.c team. All rights reserved.
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# The dW/dB partial accumulation additionally follows flash-linear-attention's
# ``layer_norm_bwd_kernel_row`` (MIT), as credited in Unsloth's source.

from __future__ import annotations

import typing

import torch

from . import _fallback as fb
from . import _ops
from ._common import (MAX_DIM, fast_ok, next_pow2, nprog_for,
                      tl, triton, triton_available, tune)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


if triton_available():

    @triton.jit
    def _ln_fwd(
        Y, X, W, B, Mu, R,
        Y_stride: tl.constexpr,   # row stride of the output
        X_stride: tl.constexpr,   # row stride of the input
        n_cols: tl.constexpr,     # hidden size
        eps: tl.constexpr,        # norm epsilon, per model
        BLOCK: tl.constexpr,      # n_cols rounded to a power of two
        HAS_B: tl.constexpr,      # bias present
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols

        x = tl.load(X + row * X_stride + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / n_cols
        # mask invalid lanes: subtraction turns the padded zeros into -mean.
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / n_cols
        # `tl.full`, not `eps.to(tl.float32)`: under torch.compile a scalar argument
        # arrives as a plain Python float (Liger #1601) and has no `.to`.
        rstd = tl.math.rsqrt(var + tl.full((), eps, tl.float32))
        tl.store(Mu + row, mean)
        tl.store(R + row, rstd)

        y = xc * rstd
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = y * w
        if HAS_B:
            b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
            y = y + b
        tl.store(Y + row * Y_stride + cols, y.to(Y.dtype.element_ty), mask=mask)

    @triton.jit
    def _ln_bwd_row(
        dY, dY_stride, X, X_stride, W, Mu, R, dX, dX_stride,
        row, cols, mask,
        n_cols: tl.constexpr,
        HAS_B: tl.constexpr,
        dW_acc, dB_acc,
        TRAIN_W: tl.constexpr,
        TRAIN_B: tl.constexpr,
    ):
        """dX for one row; accumulates this row into the (optional) fp32 partials."""
        dy = tl.load(dY + row * dY_stride + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + row * X_stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        mu = tl.load(Mu + row)
        r = tl.load(R + row)

        normed = (x - mu) * r
        dy_w = dy * w
        # masked lanes contribute 0 to both sums (dy was loaded with other=0).
        m1 = tl.sum(dy_w, axis=0) / n_cols
        m2 = tl.sum(dy_w * normed, axis=0) / n_cols
        dx = r * (dy_w - m1 - normed * m2)
        tl.store(dX + row * dX_stride + cols, dx, mask=mask)

        if TRAIN_W:
            dW_acc = dW_acc + dy * normed
        if TRAIN_B:
            dB_acc = dB_acc + dy
        return dW_acc, dB_acc

    @triton.jit(do_not_specialize=("n_rows", "rows_per_program"))
    def _ln_bwd_dw(
        dY, dY_stride, X, X_stride, W, Mu, R, dX, dX_stride,
        dW, dB,
        n_rows, rows_per_program,
        n_cols: tl.constexpr,
        HAS_B: tl.constexpr,
        BLOCK: tl.constexpr,
        TRAIN_W: tl.constexpr,
        TRAIN_B: tl.constexpr,
    ):
        """One program per partial slot: writes fp32 dW/dB partials, as in fla's row kernel."""
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        dW_acc = tl.zeros((BLOCK,), dtype=tl.float32)
        dB_acc = tl.zeros((BLOCK,), dtype=tl.float32)
        row_start = pid * rows_per_program
        # while, not range(): the CPU interpreter cannot __index__ a runtime bound.
        row_end = tl.minimum(row_start + rows_per_program, n_rows)
        row = row_start
        while row < row_end:
            dW_acc, dB_acc = _ln_bwd_row(
                dY, dY_stride, X, X_stride, W, Mu, R, dX, dX_stride,
                row, cols, mask, n_cols, HAS_B, dW_acc, dB_acc, TRAIN_W, TRAIN_B,
            )
            row = row + 1
        if TRAIN_W:
            tl.store(dW + pid * n_cols + cols, dW_acc, mask=mask)
        if TRAIN_B:
            tl.store(dB + pid * n_cols + cols, dB_acc, mask=mask)

    @triton.jit(do_not_specialize=("n_programs",))
    def _ln_sum_partials(
        partials, out, n_programs, n_cols,
        BLOCK_ROWS: tl.constexpr,   # partial rows per step
        BLOCK_COLS: tl.constexpr,   # columns per program
    ):
        # Fixed tile order, so eager and torch.compile give the same bytes.
        cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        col_mask = cols < n_cols
        acc = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)
        start = 0
        while start < n_programs:
            rows = start + tl.arange(0, BLOCK_ROWS)
            mask = (rows < n_programs)[:, None] & col_mask[None, :]
            acc += tl.load(partials + rows[:, None] * n_cols + cols[None, :],
                           mask=mask, other=0)
            start = start + BLOCK_ROWS
        tl.store(out + cols, tl.sum(acc, axis=0).to(out.dtype.element_ty), mask=col_mask)

    _ln_fwd = tune(_ln_fwd, key=["n_cols"])
    _ln_bwd_dw = tune(_ln_bwd_dw, key=["n_cols"])


def _ok(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> bool:
    # `numel < 2**31` keeps row offsets in int32; `<= MAX_DIM` bounds the constexpr tile.
    d = x.shape[-1] if x.dim() else 0
    if not (w is not None and 0 < d == w.numel() <= MAX_DIM and x.numel() > 0
            and x.numel() < 2 ** 31
            and w.dtype == x.dtype and w.is_contiguous()
            and fast_ok(x, w)):
        return False
    if b is None:
        return True
    return (b.numel() == d and b.dtype == x.dtype and b.is_contiguous() and fast_ok(b))


def _shape2(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    d = x.shape[-1]
    return x.contiguous().view(-1, d), tuple(x.shape)


def _launch_fwd(x, w, b, eps):
    """Returns ``(Y (x.shape), mu [n] fp32, r [n] fp32)`` -- Y is viewed back to the
    input's shape: a [B,T,D] activation must not come back as [B*T, D] (the fake impl
    promises x.shape, so a mismatch corrupts a compiled graph)."""
    x2, shape = _shape2(x)
    w = w.contiguous()
    n_rows, n_cols = x2.shape
    has_b = b is not None
    b = b.contiguous() if has_b else w          # placeholder pointer; never read (HAS_B=0)
    y = torch.empty((n_rows, n_cols), dtype=x.dtype, device=x.device)
    mu = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    r = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    block = next_pow2(n_cols)
    # no num_warps here: tune() owns launch metadata on real GPUs (see _common.tune)
    _ln_fwd[(n_rows,)](y, x2, w, b, mu, r, n_cols, n_cols, n_cols, eps, block, has_b)
    return y.view(shape), mu, r


def _launch_bwd(x, w, b, mu, r, dy, train_w: bool, train_b: bool):
    """Returns ``(dx (x.shape), dw or None, db or None)`` -- dx is viewed back for the
    same reason as the forward."""
    x2, shape = _shape2(x)
    dy2, _ = _shape2(dy)
    w = w.contiguous()
    n_rows, n_cols = x2.shape
    has_b = b is not None
    b = b.contiguous() if has_b else w
    dx = torch.empty_like(x2)
    block = next_pow2(n_cols)
    if not (train_w or train_b):
        _ln_bwd_row_launch(dy2, dx, x2, w, b, mu, r, n_rows, n_cols, block, has_b,
                           train_w=False, train_b=False)
        return dx.view(shape), None, None
    programs = nprog_for(n_rows)
    rows_per_program = max(1, _cdiv(n_rows, programs))
    programs = max(1, _cdiv(n_rows, rows_per_program))
    dwp = torch.empty((programs, n_cols), dtype=torch.float32, device=x.device) if train_w else None
    dbp = (torch.empty((programs, n_cols), dtype=torch.float32, device=x.device)
           if (train_b and has_b) else None)
    _ln_bwd_dw[(programs,)](
        dy2, n_cols, x2, n_cols, w, mu, r, dx, n_cols,
        dwp if dwp is not None else dx,          # placeholder; not read when TRAIN_W=0
        dbp if dbp is not None else dx,          # placeholder; not read when TRAIN_B=0
        n_rows, rows_per_program, n_cols, has_b, block, train_w, train_b and has_b,
    )
    dw = _sum_partials(dwp, w) if train_w else None
    db = _sum_partials(dbp, b) if (train_b and has_b) else None
    return dx.view(shape), dw, db


def _ln_bwd_row_launch(dy2, dx, x2, w, b, mu, r, n_rows, n_cols, block, has_b,
                       train_w, train_b):
    """dX-only: one row per program, no partials (TRAIN_W/TRAIN_B are both false)."""
    _ln_bwd_dw[(n_rows,)](
        dy2, n_cols, x2, n_cols, w, mu, r, dx, n_cols,
        dx, dx,                            # placeholders; not read (TRAIN_W=TRAIN_B=0)
        n_rows, 1, n_cols, has_b, block, False, False,
    )


def _sum_partials(partials, ref):
    programs, n_cols = partials.shape
    out = torch.empty(n_cols, dtype=ref.dtype, device=partials.device)
    _ln_sum_partials[(_cdiv(n_cols, 64),)](partials, out, programs, n_cols, 64, 64,
                                           enable_fp_fusion=True)
    return out


# ------------------------------------------------------------------ eager: autograd.Function
class _LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, eps):
        y, mu, r = _launch_fwd(x, w, b, eps)
        ctx.eps = float(eps)
        ctx.save_for_backward(x, w, b, mu, r)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, mu, r = ctx.saved_tensors
        train_w = ctx.needs_input_grad[1]
        train_b = b is not None and ctx.needs_input_grad[2]
        dx, dw, db = _launch_bwd(x, w, b, mu, r, dy, train_w, train_b)
        return dx, dw, db, None


# ------------------------------------------------------- traced: torch.library.custom_op
def _ln_fake(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, eps: float):
    n = x.numel() // x.shape[-1] if x.dim() and x.shape[-1] else 0
    return (torch.empty_like(x),
            torch.empty(n, dtype=torch.float32, device=x.device),
            torch.empty(n, dtype=torch.float32, device=x.device))


def _ln_setup(ctx, inputs, output):
    x, w, b, eps = inputs
    ctx.save_for_backward(x, w, b, output[1], output[2])   # mu/r travel through the graph


def _ln_bwd_op(ctx, dy, _dmu, _dr):
    x, w, b, mu, r = ctx.saved_tensors
    dx, dw, db = _LN_BWD_OP(x, w, b, mu, r, dy)
    # a missing bias is a None input: its grad must be None, not the op's placeholder
    return dx, dw, (db if b is not None else None), None


def _ln_bwd_fake(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None,
                 mu: torch.Tensor, r: torch.Tensor, dy: torch.Tensor):
    db = torch.empty_like(b) if b is not None else torch.empty(0, device=x.device)
    return torch.empty_like(x), torch.empty_like(w), db


def _ln_bwd_op_fwd(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None,
                   mu: torch.Tensor, r: torch.Tensor,
                   dy: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor]:
    dx, dw, db = _launch_bwd(x, w, b, mu, r, dy, train_w=True,
                             train_b=b is not None)
    if db is None:                    # no bias: emit a placeholder for the fixed schema
        db = torch.empty(0, device=x.device)
    return dx, dw if dw is not None else torch.empty_like(w), db


def _ln_op_fwd(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None,
               eps: float) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _launch_fwd(x, w, b, eps)


_LN_BWD_OP = _ops.define_backward("layer_norm", _ln_bwd_op_fwd, _ln_bwd_fake)
_LN_OP = _ops.define("layer_norm", _ln_op_fwd, _ln_fake, _ln_setup, _ln_bwd_op)


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None,
               eps: float = 1e-5) -> torch.Tensor:
    """``F.layer_norm(x, (x.shape[-1],), w, b, eps)`` with a fused kernel; falls back to torch.

    ``w=None`` (``elementwise_affine=False``) is supported by the fallback only -- the
    kernel path requires a weight.
    """
    if not _ok(x, w, b):
        return fb.layer_norm(x, w, b, eps)
    if _LN_OP is not None and _ops.wanted():
        return _LN_OP(x, w, b, eps)[0]
    return _LayerNormFn.apply(x, w, b, eps)
