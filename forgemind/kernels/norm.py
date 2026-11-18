"""Packed RMSNorm and fused residual-add + RMSNorm (+ backward).

Drop-in for ``F.rms_norm(x, (d,), w, eps)``: same signature, same numbers, one kernel.

The RMSNorm pair is ported from Unsloth's ``rms_layernorm.py`` (Apache-2.0, see the
attribution block below): one-row and multi-row forward/backward launches, the
per-architecture ``num_warps`` tables, SM-capped weight-gradient partials reduced
in a fixed Triton tile order, and int64 index guards for very large tensors. The
fused add-RMSNorm pair is this repo's own kernel (FlashAttention-style pre-norm-
residual fusion): ``add_rms_norm(x, delta)`` writes ``x + delta`` once and
normalizes it in the same pass instead of materializing the sum and reading it
back.

What this file still owns around the ported kernels: the public API
(``rms_norm`` / ``add_rms_norm``), the ``torch.autograd.Function`` +
``torch.library.custom_op`` layer that keeps both kernels inside
``torch.compile`` graphs, the ``_ok`` device/dtype gate with its silent torch
fallback, the add-RMSNorm kernels themselves, and launch autotuning for the
fused pair (``_common.tune``); the ported RMSNorm launches use Unsloth's
measured settings tables instead.

Lessons from this repo's first drafts, kept in the add-RMSNorm pair: the output
dtype follows ``x`` (never a module-level global), ``W`` is masked on non-pow2
widths, row/col tails are constexpr-folded when even, and the weight-gradient
partials are fp32 and device-deterministic.
"""

# Portions of this file are adapted from Unsloth's ``unsloth/kernels/rms_layernorm.py``:
#
#   Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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
# Adaptations: the eager-launch fast path through unsloth's triton_launch.py
# (AGPL-3.0-only) was replaced with Triton's standard launch; the Gemma variant
# dispatch, ``torch.library.triton_op`` / compile-cache layer, HF module patching
# and save-output / reuse-grad options were dropped (this repo wires kernels
# through its own custom_op layer); ``calculate_settings``, ``norm_num_warps``,
# ``torch_gpu_device`` and the int64-index predicate were reimplemented here from
# their published behaviour. The multi-row path stays CUDA-only: it replays GPU
# reduction order and self-checks bit parity on device.

from __future__ import annotations

import contextlib
import functools
import os
import typing
import warnings

import torch

from . import _fallback as fb
from . import _ops
from ._common import (MAX_DIM, block_m_for, fast_ok, next_pow2, nprog_for,
                      tl, triton, triton_available, tune)

if triton_available():

    # ---- launch settings and device helpers (ported / reimplemented) -------------
    next_power_of_2 = triton.next_power_of_2
    MAX_FUSED_SIZE: int = 65536

    def calculate_settings(n: int) -> (int, int):
        BLOCK_SIZE: int = next_power_of_2(n)
        if BLOCK_SIZE > MAX_FUSED_SIZE:
            raise RuntimeError(
                f"Cannot launch Triton kernel since n = {n} exceeds "
                f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
            )
        num_warps: int = 4
        if BLOCK_SIZE >= 32768:
            num_warps = 32
        elif BLOCK_SIZE >= 8192:
            num_warps = 16
        elif BLOCK_SIZE >= 2048:
            num_warps = 8
        return BLOCK_SIZE, num_warps

    # UNSLOTH_NORM_ARCH_WARPS=0 keeps calculate_settings' warps in the RMSNorm kernels.
    _NORM_ARCH_WARPS = os.environ.get("UNSLOTH_NORM_ARCH_WARPS", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )

    def _cuda_capability(index):
        return tuple(torch.cuda.get_device_capability(index))

    def norm_num_warps(table, n_cols, num_warps, device):
        """num_warps from table[(capability, n_cols)] for CUDA devices, else the default."""
        if not _NORM_ARCH_WARPS or device is None or device.type != "cuda" or torch.version.hip:
            return num_warps
        index = device.index if device.index is not None else torch.cuda.current_device()
        if torch.compiler.is_compiling():
            capability = tuple(torch.cuda.get_device_capability(index))
        else:
            capability = _cuda_capability(index)
        if type(n_cols) is not int:
            # A traced op's fake run passes a symbolic width, which cannot be hashed: match by
            # equality.
            for (table_capability, width), warps in table.items():
                if table_capability == capability and width == n_cols:
                    return warps
            return num_warps
        return table.get((capability, n_cols), num_warps)

    def torch_gpu_device(device):
        # Reimplemented: unsloth dispatches across cuda/xpu/npu/mlx; only CUDA and the
        # no-op (CPU interpreter / TileLens) cases exist in this repo.
        if device is not None and getattr(device, "type", None) == "cuda":
            return torch.cuda.device(device)
        return contextlib.nullcontext()

    def long_indexing(*tensors, block=0):
        # int64 offsets are needed when any tensor's reachable extent plus the margin can
        # pass 2**31; numel() alone misses strided views.
        for t in tensors:
            if t.is_contiguous():
                extent = t.numel()
            else:
                extent = (sum(sh * st for sh, st in zip(t.shape, t.stride()))
                          - sum(t.stride()) + 1)
            if extent + block > 2 ** 31:
                return True
        return False
    @triton.jit
    def _rms_layernorm_forward(
        Y,
        X,
        W,
        r,
        Y_row_stride: tl.constexpr,  # Y row stride (hidden)
        X_row_stride: tl.constexpr,  # X row stride (hidden)
        W_row_stride: tl.constexpr,  # W row stride
        r_row_stride: tl.constexpr,  # rstd row stride
        n_cols: tl.constexpr,  # hidden size (or hd)
        eps: tl.constexpr,  # norm epsilon, per model
        LONG_INDEXING: tl.constexpr,  # int64 offsets past 2**31
        BLOCK_SIZE: tl.constexpr,  # n_cols next pow2
    ):
        """
        Fast RMS Layernorm kernel
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
        """
        row_idx = tl.program_id(0)
        if LONG_INDEXING:
            row_idx = row_idx.to(tl.int64)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        Y += row_idx * Y_row_stride
        X += row_idx * X_row_stride
        r += row_idx * r_row_stride

        X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
        W_row = tl.load(W + col_offsets, mask = mask, other = 0)

        row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
        # Explicit float32 scalar to ensure correct type promotion on HIP/ROCm.
        eps_f32 = tl.full((), eps, tl.float32)
        inv_var = tl.math.rsqrt(row_var + eps_f32)
        tl.store(r, inv_var)
        normed = X_row * inv_var
        # Weight multiply in fp32 with a single rounding: matches torch's opmath path, and
        # the original bf16 x bf16 step (an HF-parity detail) cannot run in the CPU
        # interpreter (numpy's bf16 multiply is broken there).
        output = (normed * W_row.to(tl.float32)).to(Y.dtype.element_ty)
        tl.store(Y + col_offsets, output, mask = mask)


    @triton.jit
    def _rms_backward_row(
        dY,
        dY_row_stride: tl.constexpr,  # dY row stride (hidden)
        dX,
        dX_row_stride: tl.constexpr,  # dX row stride (hidden)
        X,
        X_row_stride: tl.constexpr,  # X row stride (hidden)
        W,
        r,
        r_row_stride: tl.constexpr,  # rstd row stride
        row_idx,
        col_offsets,
        mask,
        n_cols: tl.constexpr,  # hidden size (or hd)
        GEMMA: tl.constexpr,  # gemma (1 + w) weight
        dW_acc,
        TRAIN_W: tl.constexpr,  # weight grad partials
        FROM_Y: tl.constexpr,  # recompute normed from saved Y
    ):
        # with FROM_Y, X holds Y and normed = Y / W; r still scales dX.
        dY += row_idx * dY_row_stride
        X += row_idx * X_row_stride
        r += row_idx * r_row_stride
        dX += row_idx * dX_row_stride

        dY_row = tl.load(dY + col_offsets, mask = mask, other = 0).to(tl.float32)
        X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
        W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)

        inv_var = tl.load(r).to(tl.float32)
        if FROM_Y:
            # masked lanes would divide 0 by 0.
            normed = tl.where(mask, X_row / (W_row + 1.0 if GEMMA else W_row), 0.0)
        else:
            normed = X_row * inv_var
        if TRAIN_W:
            # accumulate dW before the row sum to free the fp32 dY row and keep all programs resident.
            if GEMMA:
                dW_acc += dY_row * normed
            else:
                # the forward weight-multiply is fp32 with a single rounding to Y's dtype;
                # dW must integrate the same fp32 `normed` (no W.dtype round in between).
                dW_acc += dY_row * normed

        if GEMMA:
            dY_W = dY_row * (W_row + 1.0)
        else:
            dY_W = dY_row * W_row

        rowsum_dY_normed = tl.sum(dY_W * normed, axis = 0)
        output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
        tl.store(dX + col_offsets, output, mask = mask)
        return dW_acc


    def _rms_layernorm_backward(
        dY,
        dX,
        X,
        W,
        r,
        dW,
        n_rows,
        rows_per_program,
        dY_row_stride: tl.constexpr,  # dY row stride (hidden)
        dX_row_stride: tl.constexpr,  # dX row stride (hidden)
        X_row_stride: tl.constexpr,  # X row stride (hidden)
        W_row_stride: tl.constexpr,  # W row stride
        r_row_stride: tl.constexpr,  # rstd row stride
        n_cols: tl.constexpr,  # hidden size (or hd)
        eps: tl.constexpr,  # norm epsilon, per model
        GEMMA: tl.constexpr,  # gemma (1 + w) weight
        LONG_INDEXING: tl.constexpr,  # int64 offsets past 2**31
        BLOCK_SIZE: tl.constexpr,  # n_cols next pow2
        TRAIN_W: tl.constexpr,  # weight grad partials
        FROM_Y: tl.constexpr,  # recompute normed from saved Y
    ):
        """Fast RMS layernorm backward kernel, inspired by the Triton layer-norm tutorial; TRAIN_W
        writes per-program fp32 dW partials, as in fla-org/flash-linear-attention
        layer_norm_bwd_kernel_row."""
        pid = tl.program_id(0)
        if LONG_INDEXING:
            pid = pid.to(tl.int64)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        if TRAIN_W:
            dW_acc = tl.zeros((BLOCK_SIZE,), dtype = tl.float32)
            row_start = pid * rows_per_program
            # while, not range(): triton's CPU interpreter cannot call __index__ on the
            # tl.scalar bound (same iteration order and bounds as the original).
            row_end = tl.minimum(row_start + rows_per_program, n_rows)
            row_idx = row_start
            while row_idx < row_end:
                dW_acc = _rms_backward_row(
                    dY,
                    dY_row_stride,
                    dX,
                    dX_row_stride,
                    X,
                    X_row_stride,
                    W,
                    r,
                    r_row_stride,
                    row_idx,
                    col_offsets,
                    mask,
                    n_cols,
                    GEMMA,
                    dW_acc,
                    True,
                    FROM_Y,
                )
                row_idx = row_idx + 1
            tl.store(dW + pid * n_cols + col_offsets, dW_acc, mask = mask)
        else:
            _rms_backward_row(
                dY,
                dY_row_stride,
                dX,
                dX_row_stride,
                X,
                X_row_stride,
                W,
                r,
                r_row_stride,
                pid,
                col_offsets,
                mask,
                n_cols,
                GEMMA,
                0.0,
                False,
                FROM_Y,
            )


    _rms_layernorm_backward = triton.jit(
        _rms_layernorm_backward, do_not_specialize = ("n_rows", "rows_per_program")
    )
    @triton.jit
    def _fold_lanes(
        x,
        ROWS: tl.constexpr,  # rows per program
        WARPS: tl.constexpr,  # num_warps of the launch
        HALF: tl.constexpr,  # half of a lane pair
    ):
        return tl.sum(tl.reshape(x, (ROWS, WARPS, 2, HALF)), axis = 2)


    # ROCm: Triton's tl.sum reduces each 16-lane row of a wavefront with DPP row_shr 8, 4, 2, 1,
    # then joins the rows (permlanex16 on RDNA, row_bcast 15 / 31 on CDNA), not CUDA's xor 16 .. 1.
    _HIP_ROW_ORDER = tl.constexpr(bool(torch.version.hip))


    @triton.jit
    def _row_dot_in_hip_row_order(
        a,
        b,
        ROWS: tl.constexpr,  # rows per program
        WARPS: tl.constexpr,  # num_warps of the launch
    ):
        # Bit-exact replay of the one-row kernels' tl.sum on AMD (WARPS = BLOCK_SIZE // wavefront):
        # fma(a_l, b_l, a_(l-8) * b_(l-8)), row_shr 4, 2, 1, 16-lane rows by lane bit 4 then 5, warps.
        LANES: tl.constexpr = a.shape[1] // WARPS
        ROWS16: tl.constexpr = LANES // 16
        a = tl.permute(tl.reshape(a, (ROWS, WARPS, ROWS16, 2, 8)), (0, 1, 2, 4, 3))
        b = tl.permute(tl.reshape(b, (ROWS, WARPS, ROWS16, 2, 8)), (0, 1, 2, 4, 3))
        a_lo, a_hi = tl.split(a)
        b_lo, b_hi = tl.split(b)
        acc = tl.fma(a_hi, b_hi, a_lo * b_lo)
        acc = tl.sum(tl.reshape(acc, (ROWS, WARPS, ROWS16, 2, 4)), axis = 3)
        acc = tl.sum(tl.reshape(acc, (ROWS, WARPS, ROWS16, 2, 2)), axis = 3)
        acc = tl.sum(acc, axis = 3)
        if ROWS16 == 4:
            acc = tl.sum(tl.reshape(acc, (ROWS, WARPS, 2, 2)), axis = 3)
        acc = tl.sum(acc, axis = 2)
        if WARPS == 4:
            acc = tl.sum(tl.reshape(acc, (ROWS, 2, 2)), axis = 1)
        return tl.sum(acc, axis = 1)


    @triton.jit
    def _row_dot_in_cuda_row_order(
        a,
        b,
        ROWS: tl.constexpr,  # rows per program
        WARPS: tl.constexpr,  # num_warps of the launch
    ):
        # Bit-exact replay of the one-row kernels' tl.sum (4 warps, 64 <= BLOCK_SIZE <= 128):
        # fma(a_l, b_l, a_(l+16) * b_(l+16)), lane xor 8, 4, 2, 1, warp xor 2, 1; explicit steps only.
        a = tl.permute(tl.reshape(a, (ROWS, WARPS, 2, 16)), (0, 1, 3, 2))
        b = tl.permute(tl.reshape(b, (ROWS, WARPS, 2, 16)), (0, 1, 3, 2))
        a_lo, a_hi = tl.split(a)
        b_lo, b_hi = tl.split(b)
        acc = tl.fma(a_lo, b_lo, a_hi * b_hi)
        acc = _fold_lanes(acc, ROWS, WARPS, 8)
        acc = _fold_lanes(acc, ROWS, WARPS, 4)
        acc = _fold_lanes(acc, ROWS, WARPS, 2)
        acc = _fold_lanes(acc, ROWS, WARPS, 1)
        acc = tl.reshape(acc, (ROWS, WARPS))
        if WARPS == 4:
            acc = tl.sum(tl.reshape(acc, (ROWS, 2, 2)), axis = 1)
        return tl.sum(acc, axis = 1)


    @triton.jit
    def _row_dot_in_row_kernel_order(
        a,
        b,
        ROWS: tl.constexpr,  # rows per program
        WARPS: tl.constexpr,  # num_warps of the launch
    ):
        if _HIP_ROW_ORDER:
            total = _row_dot_in_hip_row_order(a, b, ROWS, WARPS)
        else:
            total = _row_dot_in_cuda_row_order(a, b, ROWS, WARPS)
        return total


    @triton.jit(do_not_specialize = ("n_rows",))
    def _rms_layernorm_forward_rows(
        Y,
        X,
        W,
        r,
        n_rows,
        Y_row_stride: tl.constexpr,  # Y row stride (hidden)
        X_row_stride: tl.constexpr,  # X row stride (hidden)
        n_cols: tl.constexpr,  # hidden size (or hd)
        eps: tl.constexpr,  # norm epsilon, per model
        GEMMA: tl.constexpr,  # gemma (1 + w) weight
        BLOCK_SIZE: tl.constexpr,  # n_cols next pow2
        ROWS: tl.constexpr,  # rows per program
        WARPS: tl.constexpr,  # num_warps of the launch
        LONG_INDEXING: tl.constexpr,  # int64 offsets past 2**31
    ):
        pid = tl.program_id(0)
        if LONG_INDEXING:
            pid = pid.to(tl.int64)
        rows = pid * ROWS + tl.arange(0, ROWS)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        row_mask = rows < n_rows
        col_mask = col_offsets < n_cols
        mask = row_mask[:, None] & col_mask[None, :]

        X_rows = tl.load(
            X + rows[:, None] * X_row_stride + col_offsets[None, :], mask = mask, other = 0
        ).to(tl.float32)
        if GEMMA:
            W_row = tl.load(W + col_offsets, mask = col_mask, other = 0).to(tl.float32)
        else:
            W_row = tl.load(W + col_offsets, mask = col_mask, other = 0)

        row_var = _row_dot_in_row_kernel_order(X_rows, X_rows, ROWS, WARPS) / n_cols
        eps_f32 = tl.full((), eps, tl.float32)
        inv_var = tl.math.rsqrt(row_var + eps_f32)
        tl.store(r + rows, inv_var, mask = row_mask)
        normed = X_rows * inv_var[:, None]
        if GEMMA:
            output = normed * (W_row[None, :] + 1.0)
        else:
            # fp32 weight multiply with one rounding, as in _rms_layernorm_forward.
            output = (normed * W_row[None, :].to(tl.float32)).to(Y.dtype.element_ty)
        tl.store(Y + rows[:, None] * Y_row_stride + col_offsets[None, :], output, mask = mask)


    @triton.jit
    def _rms_backward_rows_block(
        dY,
        dY_row_stride: tl.constexpr,  # dY row stride (hidden)
        dX,
        dX_row_stride: tl.constexpr,  # dX row stride (hidden)
        X,
        X_row_stride: tl.constexpr,  # X row stride (hidden)
        W,
        r,
        rows,
        n_rows,
        n_cols: tl.constexpr,  # hidden size (or hd)
        GEMMA: tl.constexpr,  # gemma (1 + w) weight
        BLOCK_SIZE: tl.constexpr,  # n_cols next pow2
        ROWS: tl.constexpr,  # rows per program
        WARPS: tl.constexpr,  # num_warps of the launch
        FROM_Y: tl.constexpr,  # recompute normed from saved Y
    ):
        col_offsets = tl.arange(0, BLOCK_SIZE)
        row_mask = rows < n_rows
        col_mask = col_offsets < n_cols
        mask = row_mask[:, None] & col_mask[None, :]

        dY_rows = tl.load(
            dY + rows[:, None] * dY_row_stride + col_offsets[None, :], mask = mask, other = 0
        ).to(tl.float32)
        X_rows = tl.load(
            X + rows[:, None] * X_row_stride + col_offsets[None, :], mask = mask, other = 0
        ).to(tl.float32)
        W_row = tl.load(W + col_offsets, mask = col_mask, other = 0).to(tl.float32)

        inv_var = tl.load(r + rows, mask = row_mask, other = 0).to(tl.float32)
        if FROM_Y:
            normed = X_rows / (W_row[None, :] + 1.0 if GEMMA else W_row[None, :])
            normed = tl.where(mask, normed, 0.0)
        else:
            normed = X_rows * inv_var[:, None]

        if GEMMA:
            dY_W = dY_rows * (W_row[None, :] + 1.0)
        else:
            dY_W = dY_rows * W_row[None, :]

        rowsum_dY_normed = _row_dot_in_row_kernel_order(dY_W, normed, ROWS, WARPS)
        # match one-row arithmetic because its fused multiply-add turns -0 into +0 on sm75.
        output = (inv_var / n_cols)[:, None] * (n_cols * dY_W - normed * rowsum_dY_normed[:, None])
        tl.store(dX + rows[:, None] * dX_row_stride + col_offsets[None, :], output, mask = mask)
        # masked rows contribute zero; this dW path is dead unless TRAIN_W. `normed` stays
        # fp32, matching the forward's fp32 weight multiply.
        return tl.sum(dY_rows * normed, axis = 0)


    @triton.jit(do_not_specialize = ("n_rows", "blocks_per_program"))
    def _rms_layernorm_backward_rows(
        dY,
        dX,
        X,
        W,
        r,
        dW,
        n_rows,
        blocks_per_program,
        dY_row_stride: tl.constexpr,  # dY row stride (hidden)
        dX_row_stride: tl.constexpr,  # dX row stride (hidden)
        X_row_stride: tl.constexpr,  # X row stride (hidden)
        n_cols: tl.constexpr,  # hidden size (or hd)
        eps: tl.constexpr,  # norm epsilon, per model
        GEMMA: tl.constexpr,  # gemma (1 + w) weight
        BLOCK_SIZE: tl.constexpr,  # n_cols next pow2
        ROWS: tl.constexpr,  # rows per program
        WARPS: tl.constexpr,  # num_warps of the launch
        LONG_INDEXING: tl.constexpr,  # int64 offsets past 2**31
        TRAIN_W: tl.constexpr,  # weight grad partials
        FROM_Y: tl.constexpr,  # recompute normed from saved Y
    ):
        pid = tl.program_id(0)
        if LONG_INDEXING:
            pid = pid.to(tl.int64)
        if TRAIN_W:
            dW_acc = tl.zeros((BLOCK_SIZE,), dtype = tl.float32)
            block_start = pid * blocks_per_program
            block_end = tl.minimum(block_start + blocks_per_program, tl.cdiv(n_rows, ROWS))
            for block in range(block_start, block_end):
                dW_acc += _rms_backward_rows_block(
                    dY,
                    dY_row_stride,
                    dX,
                    dX_row_stride,
                    X,
                    X_row_stride,
                    W,
                    r,
                    block * ROWS + tl.arange(0, ROWS),
                    n_rows,
                    n_cols,
                    GEMMA,
                    BLOCK_SIZE,
                    ROWS,
                    WARPS,
                    FROM_Y,
                )
            col_offsets = tl.arange(0, BLOCK_SIZE)
            tl.store(dW + pid * n_cols + col_offsets, dW_acc, mask = col_offsets < n_cols)
        else:
            _rms_backward_rows_block(
                dY,
                dY_row_stride,
                dX,
                dX_row_stride,
                X,
                X_row_stride,
                W,
                r,
                pid * ROWS + tl.arange(0, ROWS),
                n_rows,
                n_cols,
                GEMMA,
                BLOCK_SIZE,
                ROWS,
                WARPS,
                FROM_Y,
            )


    @triton.jit(do_not_specialize = ("n_programs",))
    def _rms_sum_dw_partials(
        partials,
        dW,
        n_programs,
        n_cols,
        BLOCK_ROWS: tl.constexpr,  # partial rows per step
        BLOCK_COLS: tl.constexpr,  # columns per program
    ):
        # Fixed tile order, so eager and torch.compile give the same bytes.
        cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        col_mask = cols < n_cols
        acc = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype = tl.float32)
        # while, not tl.range(...): triton's CPU interpreter cannot call __index__ on a
        # runtime bound. Same fixed tile order (eager and compiled give the same bytes);
        # the num_stages=3 hint goes with it (pipelining only, not correctness).
        start = 0
        while start < n_programs:
            rows = start + tl.arange(0, BLOCK_ROWS)
            mask = (rows < n_programs)[:, None] & col_mask[None, :]
            acc += tl.load(partials + rows[:, None] * n_cols + cols[None, :], mask = mask, other = 0)
            start = start + BLOCK_ROWS
        tl.store(dW + cols, tl.sum(acc, axis = 0).to(dW.dtype.element_ty), mask = col_mask)


    def _cdiv(a, b):
        # triton.cdiv's formula without its call overhead (a ConstexprFunction on recent Triton).
        return (a + b - 1) // b


    # num_warps per (compute capability, n_cols), replacing calculate_settings' where a sweep (fp16 + bf16 on
    # sm_80+, fp16 on T4; 2k and 16k rows) found a width faster outside the run's A/A band in some cell and never
    # slower in any, at least 3% on the geometric mean, with max error vs fp64 no worse than the default's; every
    # entry then won again against main on torch 2.14.1 (same GPU tier, interleaved, outside A/A, never slower).
    # _NUM_WARPS: one-row forward + frozen-W backward; _DW_NUM_WARPS: the dW backward.
    _NUM_WARPS = {
        ((7, 5), 1024): 8,
        ((7, 5), 2048): 16,
        ((7, 5), 2560): 16,
        ((7, 5), 4096): 32,
        ((7, 5), 5120): 32,
        ((7, 5), 8192): 32,
        ((9, 0), 5120): 4,
    }
    _DW_NUM_WARPS = {
        ((8, 9), 2560): 4,
        ((8, 9), 4096): 2,
        ((12, 0), 1024): 8,
        ((12, 0), 2560): 4,
        ((12, 0), 5120): 8,
        ((12, 0), 8192): 4,
    }


    def _arch_settings(n_cols, device, dw):
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        table = _DW_NUM_WARPS if dw else _NUM_WARPS
        return BLOCK_SIZE, norm_num_warps(table, n_cols, num_warps, device)


    _cached_arch_settings = functools.lru_cache(maxsize = None)(_arch_settings)


    def _settings(
        n_cols,
        device = None,
        dw = False,
    ):
        # Cached per (width, device) for eager calls; tracing and symbolic (unhashable) widths skip it.
        if torch.compiler.is_compiling() or type(n_cols) is not int:
            return _arch_settings(n_cols, device, dw)
        return _cached_arch_settings(n_cols, device, dw)


    def _launch(wrap, kernel, grid, args, n_tensors, constexprs, device, **options):
        # Adapted: the original forwards eager CUDA launches through triton_launch.py
        # (AGPL-3.0-only, deliberately not ported); use Triton's standard launch path.
        wrap(kernel)[grid](*args, **constexprs, **options)

    def _triton_at_least(major, minor):
        try:
            return tuple(int(v) for v in triton.__version__.split(".")[:2]) >= (major, minor)
        except Exception:
            return False


    _MULTIROW = os.environ.get("UNSLOTH_RMSNORM_MULTIROW", "1") != "0" and _triton_at_least(3, 6)
    _MULTIROW_ELEMENTS = 4096
    _MULTIROW_NUM_WARPS = 4


    @functools.lru_cache(maxsize = None)
    def _hip_wavefront(index):
        # Lanes per wavefront the one-row kernels reduce over: 32 on RDNA, 64 on CDNA; None if unknown.
        try:
            with torch_gpu_device(torch.device("cuda", index)):
                lanes = int(triton.runtime.driver.active.get_current_target().warp_size)
        except Exception:
            try:
                lanes = int(torch.cuda.get_device_properties(index).warp_size)
            except Exception:
                return None
        return lanes if lanes in (32, 64) else None


    def _multirow_settings(n_cols, device = None):
        # Only widths 33..128 (one element per lane) can replay the one-row order; others stay one-row.
        if not _MULTIROW:
            return None
        BLOCK_SIZE, num_warps = _settings(n_cols)
        if num_warps != 4 or not (64 <= BLOCK_SIZE <= 128):
            return None
        if torch.version.hip:
            index = device.index if device is not None and device.index is not None else None
            lanes = _hip_wavefront(torch.cuda.current_device() if index is None else index)
            if lanes is None:
                return None
            return (
                BLOCK_SIZE,
                _MULTIROW_ELEMENTS // BLOCK_SIZE,
                BLOCK_SIZE // lanes,
                _MULTIROW_NUM_WARPS,
            )
        return BLOCK_SIZE, _MULTIROW_ELEMENTS // BLOCK_SIZE, BLOCK_SIZE // 32, _MULTIROW_NUM_WARPS


    _MULTIROW_CHECKED = {}


    def _multirow_disable(reason):
        global _MULTIROW
        if _MULTIROW:
            _MULTIROW = False
            warnings.warn(
                f"Unsloth: narrow-row RMSNorm kernels disabled ({reason}); using one row per program."
            )


    def _multirow_must_raise(error, wrap):
        # Never fall back while torch.compile traces, nor on OOM or dynamo / inductor errors.
        return (
            wrap is not _eager_kernel
            or isinstance(error, torch.cuda.OutOfMemoryError)
            or type(error).__module__.startswith(("torch._dynamo", "torch._inductor"))
            or torch.compiler.is_compiling()
        )


    def _bits(t):
        return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


    def _multirow_self_check(device, dtype, W_dtype, n_cols, eps, gemma, multirow):
        from torch.utils._python_dispatch import _disable_current_modes
        with _disable_current_modes(), torch.no_grad(), torch_gpu_device(device):
            g = torch.Generator(device = device).manual_seed(3407)
            shape = (2 * multirow[1] + 3, n_cols)
            X = torch.randn(shape, device = device, generator = g)
            X = X * torch.exp2(torch.randint(-12, 13, shape, device = device, generator = g).float())
            dY = torch.randn(shape, device = device, generator = g)
            X[0], dY[0, ::2] = 0, 0  # all-zero row: signed zeros in dX
            X, dY = X.to(dtype), dY.to(dtype)
            W = torch.randn(n_cols, device = device, generator = g).to(W_dtype)
            results = []
            for rows in (False, multirow):
                Y, r = _rms_forward(X, W, eps, gemma, _eager_kernel, rows)
                dX = torch.empty_like(dY)
                _rms_backward(dY, dX, X, W, r, eps, gemma, _eager_kernel, rows)
                results.append((Y, r, dX))
            return all(torch.equal(_bits(a), _bits(b)) for a, b in zip(*results))


    def _multirow_checked(X, W, eps, gemma):
        if X.device.type != "cuda":
            return None  # CUDA-only: the replay kernels and their self-check target GPU order.
        n_cols = X.shape[1]
        multirow = _multirow_settings(n_cols, X.device)
        if multirow is None:
            return None
        key = (X.device, X.dtype, W.dtype, int(n_cols), float(eps), gemma)
        verdict = _MULTIROW_CHECKED.get(key)
        if verdict is None:
            if X.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                return None  # check on a later uncaptured call
            # Real self-check launches need concrete dimensions, even during dynamic tracing.
            n_cols, multirow = int(n_cols), tuple(int(v) for v in multirow)
            try:
                verdict = _multirow_self_check(X.device, X.dtype, W.dtype, n_cols, eps, gemma, multirow)
                reason = f"self-check mismatch for {key}"
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as error:
                # Its launches re-raise while torch.compile traces; this call then runs one row.
                verdict, reason = False, f"self-check failed for {key}: {error!r}"
            _MULTIROW_CHECKED[key] = verdict
            if not verdict:
                _multirow_disable(reason)
        return multirow if verdict and _MULTIROW else None


    # Covers masked lanes past the last row: a block of columns, or ROWS rows in the multirow kernels.
    _INDEX_MARGIN = 1 << 20


    def _rms_forward(
        X,
        W,
        eps,
        gemma,
        wrap,
        multirow = None,
    ):
        # multirow: None picks (self-checked), False forces one row, settings force multi-row.
        n_rows, n_cols = X.shape
        device = X.device
        Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = device)
        r = torch.empty(n_rows, dtype = torch.float32, device = device)
        if multirow is None:
            multirow = _multirow_checked(X, W, eps, gemma)
        long = long_indexing(X, Y, block = _INDEX_MARGIN)
        if multirow:
            BLOCK_SIZE, ROWS, WARPS, num_warps = multirow
            try:
                _launch(
                    wrap,
                    _rms_layernorm_forward_rows,
                    ((n_rows + ROWS - 1) // ROWS,),
                    (Y, X, W, r, n_rows),
                    4,
                    dict(
                        Y_row_stride = Y.stride(0),
                        X_row_stride = X.stride(0),
                        n_cols = n_cols,
                        eps = eps,
                        GEMMA = gemma,
                        BLOCK_SIZE = BLOCK_SIZE,
                        ROWS = ROWS,
                        WARPS = WARPS,
                        # int32 rows measured slower only for the Gemma multi-row forward: keep main's int64.
                        LONG_INDEXING = long or gemma,
                    ),
                    device,
                    num_warps = num_warps,
                    enable_fp_fusion = True,
                )
                return Y, r
            except Exception as error:
                if _multirow_must_raise(error, wrap):
                    raise
                _multirow_disable(f"launch failed: {error!r}")
        BLOCK_SIZE, num_warps = _settings(n_cols, device)
        _launch(
            wrap,
            _rms_layernorm_forward,
            (n_rows,),
            (Y, X, W, r),
            4,
            dict(
                Y_row_stride = Y.stride(0),
                X_row_stride = X.stride(0),
                W_row_stride = W.stride(0),
                r_row_stride = r.stride(0),
                n_cols = n_cols,
                eps = eps,
                LONG_INDEXING = long,
                BLOCK_SIZE = BLOCK_SIZE,
            ),
            device,
            num_warps = num_warps,
            enable_fp_fusion = True,
        )
        return Y, r


    _DW_PROGRAMS_PER_SM = 4


    @functools.lru_cache(maxsize = None)
    def _sm_count(device):
        try:
            return int(torch.cuda.get_device_properties(device).multi_processor_count)
        except Exception:
            return 128


    @functools.lru_cache(maxsize = None)
    def _max_sm_threads(device):
        try:
            return int(torch.cuda.get_device_properties(device).max_threads_per_multi_processor)
        except Exception:
            return 0


    def _dw_split(units, max_programs):
        per_program = max(1, _cdiv(units, max(1, min(units, max_programs))))
        return max(1, _cdiv(units, per_program)), per_program


    def _dw_launch(units, device):
        return _dw_split(units, _sm_count(device) * _DW_PROGRAMS_PER_SM)


    @functools.lru_cache(maxsize = None)
    def _dw_row_config(device, n_cols):
        # Cached (host bound). 64-warp SMs, <= 16 elems/lane: cap registers at 64 so all programs stay resident.
        BLOCK_SIZE, num_warps = _settings(n_cols, device, dw = True)
        per_sm, options = _DW_PROGRAMS_PER_SM, {}
        if (
            device.type == "cuda"
            and not torch.version.hip
            and BLOCK_SIZE <= 512 * num_warps
            and _max_sm_threads(device) >= 2048
        ):
            per_sm, options = max(1, 32 // num_warps), {"maxnreg": 64}
        return BLOCK_SIZE, num_warps, _sm_count(device) * per_sm, options


    def _rms_backward(
        dY,
        dX,
        X,
        W,
        r,
        eps,
        gemma,
        wrap,
        multirow = None,
        train_w = False,
        from_y = False,
    ):
        n_rows, n_cols = dY.shape
        device = dY.device
        if multirow is None:
            multirow = _multirow_checked(X, W, eps, gemma)
        long = long_indexing(dY, dX, X, block = _INDEX_MARGIN)
        partials = None
        if multirow:
            BLOCK_SIZE, ROWS, WARPS, num_warps = multirow
            programs, per_program = _cdiv(n_rows, ROWS), 1
            if train_w:
                programs, per_program = _dw_launch(programs, device)
                partials = torch.empty((programs, n_cols), dtype = torch.float32, device = device)
            try:
                _launch(
                    wrap,
                    _rms_layernorm_backward_rows,
                    (programs,),
                    (dY, dX, X, W, r, r if partials is None else partials, n_rows, per_program),
                    6,
                    dict(
                        dY_row_stride = dY.stride(0),
                        dX_row_stride = dX.stride(0),
                        X_row_stride = X.stride(0),
                        n_cols = n_cols,
                        eps = eps,
                        GEMMA = gemma,
                        BLOCK_SIZE = BLOCK_SIZE,
                        ROWS = ROWS,
                        WARPS = WARPS,
                        LONG_INDEXING = long,
                        TRAIN_W = train_w,
                        FROM_Y = from_y,
                    ),
                    device,
                    num_warps = num_warps,
                    enable_fp_fusion = True,
                )
            except Exception as error:
                if _multirow_must_raise(error, wrap):
                    raise
                _multirow_disable(f"launch failed: {error!r}")
            else:
                return _rms_sum_partials(partials, W, wrap)
        programs, per_program, options = n_rows, 1, {}
        if train_w:
            BLOCK_SIZE, num_warps, max_programs, options = _dw_row_config(device, int(n_cols))
            programs, per_program = _dw_split(n_rows, max_programs)
            partials = torch.empty((programs, n_cols), dtype = torch.float32, device = device)
        else:
            BLOCK_SIZE, num_warps = _settings(n_cols, device)
        _launch(
            wrap,
            _rms_layernorm_backward,
            (programs,),
            (dY, dX, X, W, r, r if partials is None else partials, n_rows, per_program),
            6,
            dict(
                dY_row_stride = dY.stride(0),
                dX_row_stride = dX.stride(0),
                X_row_stride = X.stride(0),
                W_row_stride = W.stride(0),
                r_row_stride = r.stride(0),
                n_cols = n_cols,
                eps = eps,
                GEMMA = gemma,
                LONG_INDEXING = long,
                BLOCK_SIZE = BLOCK_SIZE,
                TRAIN_W = train_w,
                FROM_Y = from_y,
            ),
            device,
            num_warps = num_warps,
            **options,
            enable_fp_fusion = True,
        )
        return _rms_sum_partials(partials, W, wrap)


    def _rms_sum_partials(partials, W, wrap):
        if partials is None:
            return None
        programs, n_cols = partials.shape
        dW = torch.empty(n_cols, dtype = W.dtype, device = partials.device)
        _launch(
            wrap,
            _rms_sum_dw_partials,
            (_cdiv(n_cols, 64),),
            (partials, dW, programs, n_cols),
            2,
            dict(BLOCK_ROWS = 64, BLOCK_COLS = 64),
            partials.device,
            enable_fp_fusion = True,
        )
        return dW


    _eager_kernel = lambda kernel: kernel


    @triton.jit
    def _addrms_fwd(X, Dl, W, XN, Y, Rstd, N, eps,
                    d: tl.constexpr, ROWS: tl.constexpr, BLOCK_D: tl.constexpr,
                    EVEN_COL: tl.constexpr, EVEN_ROW: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK_D)
        off = rows[:, None] * d + cols[None, :]     # int32, see _rmsnorm_fwd
        cm = cols < d
        rm = rows < N
        if EVEN_COL:
            w = tl.load(W + cols).to(tl.float32)
        else:
            w = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)
        if EVEN_COL and EVEN_ROW:
            x = tl.load(X + off).to(tl.float32)
            dl = tl.load(Dl + off).to(tl.float32)
        elif EVEN_ROW:
            x = tl.load(X + off, mask=cm[None, :], other=0.0).to(tl.float32)
            dl = tl.load(Dl + off, mask=cm[None, :], other=0.0).to(tl.float32)
        elif EVEN_COL:
            x = tl.load(X + off, mask=rm[:, None], other=0.0).to(tl.float32)
            dl = tl.load(Dl + off, mask=rm[:, None], other=0.0).to(tl.float32)
        else:
            m = rm[:, None] & cm[None, :]
            x = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
            dl = tl.load(Dl + off, mask=m, other=0.0).to(tl.float32)
        xn = x + dl
        r = 1.0 / tl.sqrt(tl.sum(xn * xn, axis=1) / d + eps)
        y = (xn * r[:, None] * w[None, :]).to(Y.dtype.element_ty)
        xnc = xn.to(XN.dtype.element_ty)
        if EVEN_COL and EVEN_ROW:
            tl.store(XN + off, xnc)
            tl.store(Y + off, y)
            tl.store(Rstd + rows, r)
        elif EVEN_ROW:
            tl.store(XN + off, xnc, mask=cm[None, :])
            tl.store(Y + off, y, mask=cm[None, :])
            tl.store(Rstd + rows, r)
        elif EVEN_COL:
            tl.store(XN + off, xnc, mask=rm[:, None])
            tl.store(Y + off, y, mask=rm[:, None])
            tl.store(Rstd + rows, r, mask=rm)
        else:
            m2 = rm[:, None] & cm[None, :]
            tl.store(XN + off, xnc, mask=m2)
            tl.store(Y + off, y, mask=m2)
            tl.store(Rstd + rows, r, mask=rm)

    @triton.jit
    def _addrms_bwd(XN, W, DY, GXN, Rstd, DX, DD, DWP, N,
                    d: tl.constexpr, NPROG: tl.constexpr, BLOCK_D: tl.constexpr,
                    EVEN_COL: tl.constexpr):
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        cm = cols < d
        if EVEN_COL:
            w = tl.load(W + cols).to(tl.float32)
        else:
            w = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)
        dw = tl.zeros([BLOCK_D], dtype=tl.float32)
        row = pid
        while row < N:
            off = row * d + cols
            if EVEN_COL:
                xn = tl.load(XN + off).to(tl.float32)
                dy = tl.load(DY + off).to(tl.float32)
                gx = tl.load(GXN + off).to(tl.float32)
            else:
                xn = tl.load(XN + off, mask=cm, other=0.0).to(tl.float32)
                dy = tl.load(DY + off, mask=cm, other=0.0).to(tl.float32)
                gx = tl.load(GXN + off, mask=cm, other=0.0).to(tl.float32)
            r = tl.load(Rstd + row)
            xh = xn * r
            wdy = dy * w
            c = tl.sum(xh * wdy, axis=0) / d
            # d/dx == d/d(delta) because xn = x + delta -> compute once, store twice
            dx = ((wdy - xh * c) * r + gx).to(DX.dtype.element_ty)
            if EVEN_COL:
                tl.store(DX + off, dx)
                tl.store(DD + off, dx)
            else:
                tl.store(DX + off, dx, mask=cm)
                tl.store(DD + off, dx, mask=cm)
            dw += dy * xh
            row += NPROG
        tl.store(DWP + pid * d + cols, dw, mask=None if EVEN_COL else cm)


if triton_available():
    # Launch metadata (num_warps/num_stages) is chosen per shape on first GPU use.
    _addrms_fwd = tune(_addrms_fwd, key=["d"])
    _addrms_bwd = tune(_addrms_bwd, key=["d"])


def _launch_rms_fwd(x, w, eps):
    """Launch the ported RMSNorm forward; returns ``(y, rstd)`` (``rstd`` is fp32 inv-std, ``[N]``)."""
    d = x.shape[-1]
    x2 = x.contiguous().view(-1, d)
    w = w.contiguous()          # multi-row kernels index W at unit stride
    y2, rstd = _rms_forward(x2, w, eps, False, _eager_kernel)
    return y2.view(x.shape), rstd


def _launch_rms_bwd(x, w, rstd, dy, eps=1e-6):
    """Launch the ported RMSNorm backward; returns ``(dx, dw)``."""
    d = x.shape[-1]
    x2 = x.contiguous().view(-1, d)
    dy2 = dy.contiguous().view(-1, d)
    w = w.contiguous()          # multi-row kernels index W at unit stride
    dx2 = torch.empty_like(x2)
    # eps is unused by the backward math (rstd carries it); it only seeds the multi-row
    # self-check key, so passing the caller's real eps avoids a duplicate GPU self-check
    # (and cache key) when eps != 1e-6.
    dw = _rms_backward(dy2, dx2, x2, w, rstd, eps, False, _eager_kernel, train_w=True)
    return dx2.view(x.shape), dw


def _launch_addrms_fwd(x, delta, w, eps):
    """Fused ``x + delta`` / RMSNorm; returns ``(xn, y, rstd)``."""
    d = x.shape[-1]
    x2 = x.contiguous().view(-1, d)
    d2 = delta.contiguous().view(-1, d)
    n = x2.shape[0]
    xn = torch.empty_like(x2)
    y = torch.empty_like(x2)
    block_d = next_pow2(d)
    rows = block_m_for(block_d)
    rstd = torch.empty(n, device=x.device, dtype=torch.float32)
    _addrms_fwd[(triton.cdiv(n, rows),)](
        x2, d2, w, xn, y, rstd, n, eps, d, rows, block_d, d == block_d, n % rows == 0)
    return xn.view(x.shape), y.view(x.shape), rstd


def _launch_addrms_bwd(xn, w, rstd, gxn, gy):
    """Returns ``(dx, ddelta, dw)``; ``dx`` and ``ddelta`` are the same tensor value."""
    d = xn.shape[-1]
    xn2 = xn.contiguous().view(-1, d)
    n = xn2.shape[0]
    gxn2 = gxn.contiguous().view(n, d)
    gy2 = gy.contiguous().view(n, d)
    dx = torch.empty_like(xn2)
    dd = torch.empty_like(xn2)
    npg = nprog_for(n)
    dwp = torch.empty(npg, d, device=xn2.device, dtype=torch.float32)
    block_d = next_pow2(d)
    _addrms_bwd[(npg,)](xn2, w, gy2, gxn2, rstd, dx, dd, dwp, n, d, npg, block_d, d == block_d)
    return (dx.view(xn.shape), dd.view(xn.shape),
            dwp.sum(0, dtype=torch.float32).to(w.dtype))


# ------------------------------------------------------------------ eager: autograd.Function
class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        y, rstd = _launch_rms_fwd(x, w, eps)
        ctx.eps = float(eps)    # feeds the backward's self-check key (BUG-008)
        ctx.save_for_backward(x, w, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, rstd = ctx.saved_tensors
        dx, dw = _launch_rms_bwd(x, w, rstd, dy, getattr(ctx, "eps", 1e-6))
        return dx, dw, None


class _AddRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, w, eps):
        xn, y, rstd = _launch_addrms_fwd(x, delta, w, eps)
        ctx.save_for_backward(xn, w, rstd)
        return xn, y

    @staticmethod
    def backward(ctx, gxn, gy):
        xn, w, rstd = ctx.saved_tensors
        dx, dd, dw = _launch_addrms_bwd(xn, w, rstd, gxn, gy)
        return dx, dd, dw, None


# ------------------------------------------------------- traced: torch.library.custom_op
def _rms_fake(x: torch.Tensor, w: torch.Tensor, eps: float):
    return torch.empty_like(x), torch.empty(x.numel() // x.shape[-1], device=x.device,
                                            dtype=torch.float32)


def _rms_setup(ctx, inputs, output):
    x, w, eps = inputs
    ctx.save_for_backward(x, w, output[1])       # rstd travels through the graph


def _rms_bwd_op(ctx, dy, _drstd):
    x, w, rstd = ctx.saved_tensors
    dx, dw = _RMS_BWD_OP(x, w, rstd, dy)     # an op, not a raw launch: see _ops docstring
    return dx, dw, None


def _addrms_fake(x: torch.Tensor, delta: torch.Tensor, w: torch.Tensor, eps: float):
    return torch.empty_like(x), torch.empty_like(x), torch.empty(
        x.numel() // x.shape[-1], device=x.device, dtype=torch.float32)


def _addrms_setup(ctx, inputs, output):
    xn, y, rstd = output
    ctx.save_for_backward(xn, inputs[2], rstd)


def _addrms_bwd_op(ctx, gxn, gy, _grstd):
    xn, w, rstd = ctx.saved_tensors
    dx, dd, dw = _ADDRMS_BWD_OP(xn, w, rstd, gxn, gy)
    return dx, dd, dw, None


def _rms_bwd_fake(x: torch.Tensor, w: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor):
    return torch.empty_like(x), torch.empty_like(w)


def _addrms_bwd_fake(xn: torch.Tensor, w: torch.Tensor, rstd: torch.Tensor, gxn: torch.Tensor,
                     gy: torch.Tensor):
    return torch.empty_like(xn), torch.empty_like(xn), torch.empty_like(w)


def _rms_bwd_op_fwd(x: torch.Tensor, w: torch.Tensor, rstd: torch.Tensor,
                    dy: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    return _launch_rms_bwd(x, w, rstd, dy)


def _addrms_bwd_op_fwd(xn: torch.Tensor, w: torch.Tensor, rstd: torch.Tensor, gxn: torch.Tensor,
                       gy: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor,
                                                          torch.Tensor]:
    return _launch_addrms_bwd(xn, w, rstd, gxn, gy)


def _rms_op_fwd(x: torch.Tensor, w: torch.Tensor,
                eps: float) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    return _launch_rms_fwd(x, w, eps)


def _addrms_op_fwd(x: torch.Tensor, delta: torch.Tensor, w: torch.Tensor,
                   eps: float) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _launch_addrms_fwd(x, delta, w, eps)


# Registered at import time; each `_ops.define*` returns None when this torch has no
# custom_op API (the eager path then stays in charge, which is the pre-2.4 behaviour).
_RMS_BWD_OP = _ops.define_backward("rms_norm", _rms_bwd_op_fwd, _rms_bwd_fake)
_ADDRMS_BWD_OP = _ops.define_backward("add_rms_norm", _addrms_bwd_op_fwd, _addrms_bwd_fake)
_RMS_OP = _ops.define("rms_norm", _rms_op_fwd, _rms_fake, _rms_setup, _rms_bwd_op)
_ADDRMS_OP = _ops.define("add_rms_norm", _addrms_op_fwd, _addrms_fake, _addrms_setup,
                         _addrms_bwd_op)


def _ok(x, w):
    # `n * d < 2**31` keeps the row-offset arithmetic in int32 (the fast option); past that the
    # torch path is used. 2G elements per 2D matrix is ~4 GB of bf16.
    return (w is not None and x.shape[-1] == w.numel() and 0 < x.shape[-1] <= MAX_DIM
            and x.numel() < 2 ** 31 and w.dtype == x.dtype and w.is_contiguous()
            and fast_ok(x, w))


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``F.rms_norm(x, (x.shape[-1],), w, eps)`` with a fused kernel; falls back to torch."""
    if not _ok(x, w):
        return fb.rms_norm(x, w, eps)
    if _RMS_OP is not None and _ops.wanted():
        return _RMS_OP(x, w, eps)[0]
    return _RMSNormFn.apply(x, w, eps)


def add_rms_norm(x: torch.Tensor, delta: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    """Return ``(x + delta, rms_norm(x + delta, w, eps))`` writing the sum once."""
    if not _ok(x, w) or delta.shape != x.shape:
        return fb.add_rms_norm(x, delta, w, eps)
    if _ADDRMS_OP is not None and _ops.wanted():
        xn, y, _rstd = _ADDRMS_OP(x, delta, w, eps)
        return xn, y
    return _AddRMSNormFn.apply(x, delta, w, eps)
