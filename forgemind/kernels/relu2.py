"""ReLU-squared activation: ``relu(x) ** 2`` with its backward, one flat pass each way.

Ported from Liger-Kernel's ``relu_squared`` elementwise pair; the package's flat-kernel
pattern (int32 offsets behind a ``numel < 2**31`` gate, EVEN/mask split, single fp32
rounding on store) replaces Liger's row-grid ``calculate_settings`` launches, and the
kernel is wired through this repo's ``autograd.Function`` + ``torch.library.custom_op``
contract so it stays inside ``torch.compile`` graphs (Liger's own wrappers are the source
of open ``torch.compile`` exceptions; see the compile contract in ``_ops.py``).

Ported pieces: the forward ``max(0, x)`` then square, and the backward
``dx = dy * 2 * relu(x)`` -- the whole of Liger's two kernels, minus their row strides
(the flat layout covers the same shapes) and ``ensure_contiguous`` (the launch makes its
inputs contiguous explicitly, as every flat kernel here does).
"""

# Portions of this file are adapted from Liger-Kernel's ``relu_squared.py``:
#
#   Copyright (c) 2024-present Liger Kernel contributors. All rights reserved.
#   Licensed under the BSD 2-Clause "Simplified" License (SPDX: BSD-2-Clause);
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       https://github.com/linkedin/Liger-Kernel/blob/main/LICENSE
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import annotations

import torch

from . import _fallback as fb
from . import _ops
from ._common import fast_ok, tl, triton, triton_available, tune

_RELU2_BLOCK = 1024              # elements per program (host-side; passed as an argument)


if triton_available():

    @triton.jit
    def _relu2_fwd(X, Out, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        # int32 offsets: `_ok` refuses tensors with >= 2**31 elements (see swiglu).
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(X + off).to(tl.float32)
            r = tl.maximum(x, 0.0)
            tl.store(Out + off, (r * r).to(Out.dtype.element_ty))
        else:
            m = off < N
            x = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
            r = tl.maximum(x, 0.0)
            tl.store(Out + off, (r * r).to(Out.dtype.element_ty), mask=m)

    @triton.jit
    def _relu2_bwd(DY, X, DX, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            dy = tl.load(DY + off).to(tl.float32)
            x = tl.load(X + off).to(tl.float32)
            r = tl.maximum(x, 0.0)
            tl.store(DX + off, (dy * 2.0 * r).to(DX.dtype.element_ty))
        else:
            m = off < N
            dy = tl.load(DY + off, mask=m, other=0.0).to(tl.float32)
            x = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
            r = tl.maximum(x, 0.0)
            tl.store(DX + off, (dy * 2.0 * r).to(DX.dtype.element_ty), mask=m)

    _relu2_fwd = tune(_relu2_fwd, key=[])
    _relu2_bwd = tune(_relu2_bwd, key=[])


def _ok(x: torch.Tensor) -> bool:
    return (x.is_contiguous() and x.numel() > 0 and x.numel() < 2 ** 31 and fast_ok(x))


def _launch_fwd(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    block = _RELU2_BLOCK
    _relu2_fwd[(triton.cdiv(n, block),)](x, out, n, block, n % block == 0)
    return out


def _launch_bwd(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    dy = dy.contiguous()
    dx = torch.empty_like(dy)
    n = dy.numel()
    block = _RELU2_BLOCK
    _relu2_bwd[(triton.cdiv(n, block),)](dy, x, dx, n, block, n % block == 0)
    return dx


class _ReLUSquaredFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = _launch_fwd(x)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        return _launch_bwd(x, dy)


def _relu2_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _relu2_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


def _relu2_bwd_op(ctx, dy):
    (x,) = ctx.saved_tensors
    return _RELU2_BWD_OP(x, dy)


def _relu2_bwd_fake(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(dy)


def _relu2_bwd_op_fwd(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    return _launch_bwd(x, dy)


def _relu2_op_fwd(x: torch.Tensor) -> torch.Tensor:
    return _launch_fwd(x)


_RELU2_BWD_OP = _ops.define_backward("relu_squared", _relu2_bwd_op_fwd, _relu2_bwd_fake)
_RELU2_OP = _ops.define("relu_squared", _relu2_op_fwd, _relu2_fake, _relu2_setup,
                        _relu2_bwd_op)


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """``relu(x) ** 2`` -- drop-in for ``torch.relu(x) ** 2``."""
    if not _ok(x):
        return fb.relu_squared(x)
    if _RELU2_OP is not None and _ops.wanted():
        return _RELU2_OP(x)
    return _ReLUSquaredFn.apply(x)
