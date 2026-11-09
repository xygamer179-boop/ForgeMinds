"""Newton-Schulz orthogonalization: the polar-factor step of a Muon update.

``newton_schulz(g, steps=5)`` maps a matrix (or a batch of matrices) ``(..., m, n)`` to a
matrix of the same shape whose singular values are driven toward 1 by the quintic iteration

    X <- a*X + (b*A + c*A@A) @ X,    A = X @ X.mT,    (a, b, c) = (3.4445, -4.7750, 2.0315)

after a spectral-norm pre-scale. That is the iteration Muon uses instead of an SVD: ~5 steps
of matmuls buy an orthogonal factor, which is all ``optimizer.step`` needs.

Why the elementwise parts are fused
-----------------------------------
Per iteration the math is 2 GEMMs plus 5 elementwise kernels in eager torch
(``b*A``, ``c*A2``, their add, ``a*X``, the final add). The GEMMs are cuBLAS -- the same
backend inductor would pick -- so they stay torch; the elementwise chain becomes **two**
Triton kernels (the quintic polynomial, and ``a*X + (B@X)``). Same FLOPs, same order of
operations, strictly fewer launches and one fewer round-trip of each intermediate through
HBM: structurally not slower than the torch path by construction, which is the only kind of
"no regression" claim that can be made without a GPU (the real gate is
``python -m forgemind.cli kbench --check``, which includes this op).

Dispatch (deliberately *not* a ``torch.library`` op)
----------------------------------------------------
* eager + CUDA (or ``TRITON_INTERPRET=1``) + backend not ``fallback`` -> fused Triton;
* **while ``torch.compile`` traces** -> the pure-torch path, so dynamo sees ordinary tensor
  ops, ``fullgraph=True`` holds, and inductor fuses the elementwise chain itself (it would
  generate the same two kernels, from the other side);
* ``fused=False`` -> the pure-torch path, for A/B measurement.

The iteration runs under ``torch.no_grad()``: Muon calls it inside ``Optimizer.step`` and
differentiating through an orthogonalization is not a supported operation (there is no
autograd formula, on either path).

Numerics: both paths do the GEMMs in torch (batched vs the reference's per-matrix
``matmul`` -- bit-identical accumulation on CPU/interpreter) and the elementwise math in fp32
with one rounding at the store. Two deliberate re-associations keep the kernel at two
elementwise launches per iteration:
* the quintic is ``b*A + c*(A@A)`` where the reference's source line ``b*A + c*A @ A`` parses
  (left-to-right) as ``b*A + (c*A)@A`` -- mathematically identical, but the pre-scaled
  operand changes bf16 rounding; measured worst case ~0.04 absolute on the NS output after
  five iterations (a few bf16 ulps), not bit-for-bit;
* the fp32-inner/single-rounding elementwise kernels versus torch's per-op bf16 rounding
  (~1 ulp).
The two *paths* agree within that same rounding; ``tests/test_muon.py`` pins both against
the reference with dtype-appropriate tolerances (0.05 bf16 / 1e-5 fp32).

Reference semantics kept from the canonical implementation (Keller Jordan / modded-nanogpt,
as shipped in this task's attachment): bf16 compute for half inputs, the Gram built on the
*shorter* side (the input is transposed when ``m > n``), the ``+ 1e-7`` in the pre-scale, and
the aspect-ratio ``sqrt(max(1, m/n))`` rescale applied by the caller.
"""

from __future__ import annotations

import torch

from . import _common
from ._common import fast_ok, num_warps_for, tl, triton, triton_available

#: quintic Newton-Schulz coefficients (spectral norm mapped to ~[0, 1] in five steps)
NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315

_BLOCK = 1024

if triton_available():

    @triton.jit
    def _ns_quintic_kernel(O, A, A2, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        """``O = NS_B * A + NS_C * A2`` -- the polynomial half of one NS iteration.

        int32 offsets: the caller refuses ``numel >= 2**31`` (``_ok``), so ``pid * BLOCK``
        cannot wrap; fp32 math with one rounding at the store matches torch's opmath
        behaviour on bf16/fp16 inputs.
        """
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            a = tl.load(A + off).to(tl.float32)
            a2 = tl.load(A2 + off).to(tl.float32)
            tl.store(O + off, (NS_B * a + NS_C * a2).to(O.dtype.element_ty))
        else:
            m = off < N
            a = tl.load(A + off, mask=m, other=0.0).to(tl.float32)
            a2 = tl.load(A2 + off, mask=m, other=0.0).to(tl.float32)
            tl.store(O + off, (NS_B * a + NS_C * a2).to(O.dtype.element_ty), mask=m)

    @triton.jit
    def _ns_update_kernel(O, X, T, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        """``O = NS_A * X + T`` where ``T`` is the already-computed ``B @ X`` GEMM result."""
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(X + off).to(tl.float32)
            t = tl.load(T + off).to(tl.float32)
            tl.store(O + off, (NS_A * x + t).to(O.dtype.element_ty))
        else:
            m = off < N
            x = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
            t = tl.load(T + off, mask=m, other=0.0).to(tl.float32)
            tl.store(O + off, (NS_A * x + t).to(O.dtype.element_ty), mask=m)


def _ok(g: torch.Tensor) -> bool:
    return (g.ndim >= 2 and 0 < g.numel() < 2 ** 31 and fast_ok(g))


def _compute_dtype(g: torch.Tensor, dtype) -> torch.dtype:
    if dtype is not None:
        return dtype
    # bf16/fp16 inputs compute in bf16 (the canonical Muon choice); fp32 stays fp32.
    return torch.float32 if g.dtype == torch.float32 else torch.bfloat16


def _prepare(g: torch.Tensor, dtype):
    """``(..., m, n)`` -> ``(B, short, long)`` on the short-side Gram, plus bookkeeping."""
    m, n = g.shape[-2], g.shape[-1]
    x = g.reshape(-1, m, n)
    transposed = m > n                       # build the Gram on the shorter side
    if transposed:
        x = x.mT
    x = x.to(_compute_dtype(g, dtype))
    # pre-scale: spectral norm ~1 (torch computes the norm in fp32 on both paths)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    return x, transposed


def _finish(x: torch.Tensor, transposed: bool, shape) -> torch.Tensor:
    if transposed:
        x = x.mT
    return x.reshape(shape)


def _ns_torch(g: torch.Tensor, steps: int, dtype) -> torch.Tensor:
    """Pure-torch implementation: also the ``torch.compile`` path and the A/B baseline."""
    shape = g.shape
    x, transposed = _prepare(g, dtype)
    for _ in range(steps):
        a_mat = x @ x.mT
        x = NS_A * x + (NS_B * a_mat + NS_C * (a_mat @ a_mat)) @ x
    return _finish(x, transposed, shape)


def _launch_quintic(out: torch.Tensor, a_mat: torch.Tensor, a2: torch.Tensor) -> None:
    n = out.numel()
    even = n % _BLOCK == 0
    _ns_quintic_kernel[(triton.cdiv(n, _BLOCK),)](
        out, a_mat, a2, n, _BLOCK, even, num_warps=num_warps_for(_BLOCK))


def _launch_update(out: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> None:
    n = out.numel()
    even = n % _BLOCK == 0
    _ns_update_kernel[(triton.cdiv(n, _BLOCK),)](
        out, x, t, n, _BLOCK, even, num_warps=num_warps_for(_BLOCK))


def _ns_fused(g: torch.Tensor, steps: int, dtype) -> torch.Tensor:
    """GEMMs in torch (cuBLAS), the elementwise chain in two fused Triton kernels."""
    shape = g.shape
    x, transposed = _prepare(g, dtype)
    x = x.contiguous()                       # after mT: flat-pointer kernels need dense input
    for _ in range(steps):
        a_mat = x @ x.mT                       # GEMM 1 (torch / cuBLAS)
        a2 = a_mat @ a_mat                     # GEMM 2 (torch / cuBLAS)
        quintic = torch.empty_like(a_mat)
        _launch_quintic(quintic, a_mat, a2)    # fused: b*A + c*A@A
        t = quintic @ x                        # GEMM 3 (torch / cuBLAS)
        new_x = torch.empty_like(x)
        _launch_update(new_x, x, t)            # fused: a*X + t
        x = new_x
    return _finish(x, transposed, shape)


def newton_schulz(g: torch.Tensor, steps: int = 5, dtype: torch.dtype | None = None,
                  fused: bool = True) -> torch.Tensor:
    """Orthogonalize ``g`` (``..., m, n``) toward ``U @ V`` of its SVD via quintic NS.

    Parameters
    ----------
    g : matrix or batch of matrices, ``ndim >= 2``, fp32/fp16/bf16, ``numel < 2**31``
    steps : NS iterations (5 is the canonical value; more converges harder, costs more)
    dtype : compute dtype; default ``fp32`` for fp32 inputs, ``bf16`` for half inputs.
        The result has the compute dtype (pass ``g.dtype`` back if you need the input's).
    fused : ``True`` uses the fused Triton elementwise kernels when the backend allows it;
        ``False`` forces the pure-torch path (the A/B baseline).

    Differentiation through this function is not supported (Muon runs it under
    ``no_grad``); while ``torch.compile`` traces, the pure-torch path is used so the
    call stays in one graph.
    """
    if g.ndim < 2:
        raise ValueError(f"newton_schulz needs ndim >= 2 (a matrix), got shape {tuple(g.shape)}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if dtype is not None and dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"dtype must be float32/float16/bfloat16, got {dtype}")
    with torch.no_grad():
        if fused and _ok(g) and not _common.tracing():
            return _ns_fused(g, steps, dtype)
        return _ns_torch(g, steps, dtype)


__all__ = ["newton_schulz", "NS_A", "NS_B", "NS_C"]
