"""Exact-erf GEGLU: ``gelu(gate) * up`` with its backward, one flat pass each way.

``gelu_exact`` is HF's ``GEGLU`` activation (erf form, not the tanh approximation):
``f = 1/2 * e * (1 + erf(e / sqrt(2)))``, ``h = f * up``. The backward is the exact
derivative ``df/de = 1/2 * (1 + erf(e/sqrt2)) + e/sqrt(2pi) * exp(-e^2/2)``, reused so
one erf evaluation serves both ``f`` and ``df``.

Ported from Unsloth's ``geglu.py``. Two of their choices do not survive the port, and
both are deliberate:

* **the backward wrote its gradients into the gate/up buffers in place** (``tl.store``
  into ``e``/``g``). That mutates tensors autograd has saved, and under
  ``torch.compile`` it either trips functionalization or corrupts the graph -- the
  in-place half of the compiler issues this port exists to avoid. This port writes
  ``dg``/``du`` to fresh buffers and only reads the inputs.
* Unsloth's ``LONG_INDEXING`` constexpr (int64 offsets past 2**31) is replaced by this
  package's standard gate: flat kernels refuse tensors with ``numel >= 2**31``, so the
  int32 ``pid * BLOCK + lane`` cannot wrap.

Also dropped: the tanh-approximation variant (``approx`` -- HF's GEGLU is the erf form),
``calculate_settings``/``torch_gpu_device``/``enable_fp_fusion`` (the package's
``_common.tune`` and launch defaults own that decision), and the module-level
``BLOCK_SIZE`` (passed as an argument instead of read as a global inside the kernel --
bare globals inside ``@triton.jit`` bodies do not compile on triton >= 3.2).
"""

# Portions of this file are adapted from Unsloth's ``unsloth/kernels/geglu.py``:
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

from __future__ import annotations

import torch

from . import _fallback as fb
from . import _ops
from ._common import fast_ok, tl, triton, triton_available, tune

_GEGLU_BLOCK = 1024             # elements per program (host-side; passed as an argument)


if triton_available():

    @triton.jit
    def _geglu_fwd(G, U, Out, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)   # int32: N < 2**31, see _ok
        if EVEN:
            e = tl.load(G + off).to(tl.float32)
            u = tl.load(U + off).to(tl.float32)
            f = 0.5 * e * (tl.math.erf(e * 0.7071067811865476) + 1.0)
            tl.store(Out + off, (f * u).to(Out.dtype.element_ty))
        else:
            m = off < N
            e = tl.load(G + off, mask=m, other=0.0).to(tl.float32)
            u = tl.load(U + off, mask=m, other=0.0).to(tl.float32)
            f = 0.5 * e * (tl.math.erf(e * 0.7071067811865476) + 1.0)
            tl.store(Out + off, (f * u).to(Out.dtype.element_ty), mask=m)

    @triton.jit
    def _geglu_bwd(DY, G, U, DG, DU, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        # f = 0.5*e*(erf(e/sqrt2)+1); df/de = 0.5*(erf+1) + e/sqrt(2pi)*exp(-e^2/2).
        # One erf serves both: f_partial = 0.5*(erf+1), f = f_partial * e.
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            dy = tl.load(DY + off).to(tl.float32)
            e = tl.load(G + off).to(tl.float32)
            u = tl.load(U + off).to(tl.float32)
            fp = 0.5 * (tl.math.erf(e * 0.7071067811865476) + 1.0)
            f = fp * e
            df = fp + e * 0.3989422804014327 * tl.exp(-0.5 * e * e)
            tl.store(DU + off, (dy * f).to(DU.dtype.element_ty))
            tl.store(DG + off, (dy * u * df).to(DG.dtype.element_ty))
        else:
            m = off < N
            dy = tl.load(DY + off, mask=m, other=0.0).to(tl.float32)
            e = tl.load(G + off, mask=m, other=0.0).to(tl.float32)
            u = tl.load(U + off, mask=m, other=0.0).to(tl.float32)
            fp = 0.5 * (tl.math.erf(e * 0.7071067811865476) + 1.0)
            f = fp * e
            df = fp + e * 0.3989422804014327 * tl.exp(-0.5 * e * e)
            tl.store(DU + off, (dy * f).to(DU.dtype.element_ty), mask=m)
            tl.store(DG + off, (dy * u * df).to(DG.dtype.element_ty), mask=m)

    _geglu_fwd = tune(_geglu_fwd, key=[])
    _geglu_bwd = tune(_geglu_bwd, key=[])


def _ok(gate, up):
    return (gate.shape == up.shape and gate.is_contiguous() and up.is_contiguous()
            and gate.numel() > 0 and gate.numel() < 2 ** 31 and fast_ok(gate, up))


def _launch_fwd(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate = gate.contiguous()
    up = up.contiguous()
    out = torch.empty_like(gate)
    n = gate.numel()
    block = _GEGLU_BLOCK
    _geglu_fwd[(triton.cdiv(n, block),)](gate, up, out, n, block, n % block == 0)
    return out


def _launch_bwd(gate: torch.Tensor, up: torch.Tensor, dy: torch.Tensor):
    gate = gate.contiguous()
    up = up.contiguous()
    dy = dy.contiguous()
    dg = torch.empty_like(gate)
    du = torch.empty_like(up)
    n = gate.numel()
    block = _GEGLU_BLOCK
    _geglu_bwd[(triton.cdiv(n, block),)](dy, gate, up, dg, du, n, block, n % block == 0)
    return dg, du


class _GEGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        out = _launch_fwd(gate, up)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, dy):
        gate, up = ctx.saved_tensors
        return _launch_bwd(gate, up, dy)


def _geglu_fake(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(gate)


def _geglu_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1])


def _geglu_bwd_op(ctx, dy):
    gate, up = ctx.saved_tensors
    return _GEGLU_BWD_OP(gate, up, dy)


def _geglu_bwd_fake(gate: torch.Tensor, up: torch.Tensor, dy: torch.Tensor):
    return torch.empty_like(gate), torch.empty_like(up)


def _geglu_bwd_op_fwd(gate: torch.Tensor, up: torch.Tensor,
                      dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _launch_bwd(gate, up, dy)


def _geglu_op_fwd(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return _launch_fwd(gate, up)


_GEGLU_BWD_OP = _ops.define_backward("geglu", _geglu_bwd_op_fwd, _geglu_bwd_fake)
_GEGLU_OP = _ops.define("geglu", _geglu_op_fwd, _geglu_fake, _geglu_setup, _geglu_bwd_op)


def geglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``gelu(gate, approximate='none') * up`` -- HF's GEGLU activation."""
    if not _ok(gate, up):
        return fb.geglu(gate, up)
    if _GEGLU_OP is not None and _ops.wanted():
        return _GEGLU_OP(gate, up)
    return _GEGLUFn.apply(gate, up)
