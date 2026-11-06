"""Composable building blocks: use one fused op, or one block, inside *your* architecture.

This package is an extension, not a framework. You do not have to adopt :class:`GPT`,
`Trainer` or `Forge` to get the packed kernels: every piece below is an ordinary
``nn.Module`` that you can drop into your own model, and every one of them **is a normal
torch module when Triton is unavailable** (CPU, MPS, CI), with the same numbers to the last
bit of the accumulation order.

One op, no module
-----------------
The functional kernels are the lowest level and need nothing from this package::

    import forgemind as fm

    y = fm.kernels.rms_norm(x, w, eps=1e-6)          # == F.rms_norm(x, (d,), w, eps)
    y = fm.kernels.swiglu(gate, up)                  # == F.silu(gate) * up
    y = fm.kernels.swiglu_split(x)                   # == silu(x[..., :h]) * x[..., h:]
    xn, y = fm.kernels.add_rms_norm(x, delta, w)     # residual + norm in one pass
    y = fm.kernels.qk_norm_rope(q, w, cos, sin)      # per-head norm + RoPE

One block
---------
Two ready-made modules, each kernel-backed where it can be::

    from forgemind import nn as fmnn

    class Block(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.norm = fmnn.RMSNorm(dim)            # weights, eps, state_dict like nn.RMSNorm
            self.mlp = fmnn.SwiGLU(dim, hidden=4 * dim)

        def forward(self, x):
            x = x + self.norm(x)
            return x + self.mlp(x)

``RMSNorm`` keeps ``nn.RMSNorm``'s parameter name and shape, so an existing checkpoint loads
straight into it (and back out). ``SwiGLU`` is ``w2(silu(w1(x)) * w3(x))`` with the elementwise
part fused; ``SwiGLU(..., split=True)`` instead concatenates the gate and up projections into one
GEMM and uses the fused gate|up epilogue.

Your own module, fused
----------------------
Already have a norm or an MLP class you like? Fuse it in place, one module at a time::

    fm.kernels.patch(model.norm)          # name looks like a norm + a 1-D `weight` + `eps`
    fm.kernels.patch(model.mlp)           # any module with w1/w2/w3 Linears
    fm.kernels.patch(model.scale, kind="norm")   # my class name says nothing: be explicit
    fm.kernels.patch(model.mlp, split=True)

    fm.kernels.enable(model)              # ...or the whole tree, which is what the trainer does

``patch`` returns ``True`` when it recognised the module, so you can assert on it instead of
hoping. ``fm.kernels.set_backend("fallback")`` forces the torch path for A/B numbers, and
``fm.kernels.stats()`` reports which path actually ran.

Notes
-----
* Everything here works in fp32/bf16/fp16 and on CPU/CUDA/MPS. The kernels need CUDA + Triton;
  elsewhere you get the torch expression, not an error.
* Norms are over the **last** dimension (that is what the kernel computes); ``RMSNorm(dim)``
  checks ``x.shape[-1] == dim`` for you unless you pass ``strict=False``.
* Gradients are exact: ``tests/test_nn_blocks.py`` runs ``torch.autograd.gradcheck`` in float64 on
  both blocks and the functional ops, not just a forward comparison.
* Nothing here assumes the ForgeMind ``GPT``. ``Trainer`` only needs
  ``model(tokens, labels=None) -> (ce, aux)``; DDP/FSDP2, accumulation, AC, ``torch.compile``,
  offload and checkpointing all work on a hand-written model (TP/PP/EP still need this repo's
  model, because they rewrite its module names).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .kernels import add_rms_norm, rms_norm, swiglu, swiglu_split

__all__ = ["RMSNorm", "SwiGLU"]


class RMSNorm(nn.Module):
    """``x / sqrt(mean(x**2) + eps) * weight`` over the last dimension, fused when possible.

    Drop-in for :class:`torch.nn.RMSNorm` (same ``weight`` parameter, same numbers), with two
    additions: a fused residual variant, and an explicit fallback path so it is safe to use in
    code that must also run on CPU.

    >>> import torch
    >>> norm = RMSNorm(64)
    >>> x = torch.randn(2, 8, 64)
    >>> norm(x).shape                                 # [..., 64] -> [..., 64]
    torch.Size([2, 8, 64])
    >>> normed, new_residual = norm.forward_with_residual(x, x)   # pre-norm residual fusion
    >>> normed.shape == new_residual.shape == x.shape
    True

    ``forward_with_residual(x, residual)`` returns ``(normed, x + residual)``: the residual sum
    is written once and normalized in the same pass, instead of being materialized and read
    back. Passing ``residual=None`` starts from zero.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True,
                 strict: bool = True, device=None, dtype=None):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.strict = bool(strict)
        self.elementwise_affine = bool(elementwise_affine)
        if elementwise_affine:
            # same name/shape as nn.RMSNorm -> state_dict compatible in both directions
            self.weight = nn.Parameter(torch.ones(self.dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"

    def _w(self, x: torch.Tensor):
        w = self.weight
        if w is None:
            return torch.ones(self.dim, device=x.device, dtype=x.dtype)
        return w if w.dtype == x.dtype else w.to(x.dtype)

    def _check(self, x: torch.Tensor) -> None:
        if self.strict and x.shape[-1] != self.dim:
            raise ValueError(f"RMSNorm({self.dim}) got a tensor with last dim {x.shape[-1]}; "
                             f"norms run over the last dimension of contiguous [..., {self.dim}]")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        return rms_norm(x, self._w(x), self.eps)

    def forward_with_residual(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        """Return ``(normed, x + residual)`` -- one pass, one write of the residual stream."""
        self._check(x)
        w = self._w(x)
        if residual is None:
            residual = torch.zeros_like(x)
        new_residual, normed = add_rms_norm(x, residual, w, self.eps)
        return normed, new_residual


class SwiGLU(nn.Module):
    """``w2(silu(w1(x)) * w3(x))`` -- the usual gated MLP with the elementwise part fused.

    ``split=True`` fuses the two projections as well: ``w1`` and ``w3`` become one
    ``gate_up`` Linear of shape ``[2 * hidden, dim]`` (gate first, then up) and the epilogue is
    :func:`forgemind.kernels.swiglu_split`. That is one GEMM with one weight stream instead of
    two, which is where most of the win is on a real GPU -- but it changes the parameter layout,
    so ``split`` defaults to ``False`` and both variants are numerically identical.

    >>> import torch
    >>> mlp = SwiGLU(512, hidden=2048)
    >>> mlp(torch.randn(4, 512)).shape
    torch.Size([4, 512])
    """

    def __init__(self, dim: int, hidden: int | None = None, bias: bool = False,
                 split: bool = False, device=None, dtype=None):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden if hidden is not None else 4 * dim)
        self.split = bool(split)
        kw = dict(bias=bias, device=device, dtype=dtype)
        if self.split:
            self.gate_up = nn.Linear(self.dim, 2 * self.hidden, **kw)
        else:
            self.w1 = nn.Linear(self.dim, self.hidden, **kw)      # gate
            self.w3 = nn.Linear(self.dim, self.hidden, **kw)      # up
        self.w2 = nn.Linear(self.hidden, self.dim, **kw)          # down

    def extra_repr(self) -> str:
        return f"{self.dim} -> {self.hidden} -> {self.dim}, split={self.split}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.split:
            return self.w2(swiglu_split(self.gate_up(x)))
        return self.w2(swiglu(self.w1(x), self.w3(x)))

    @torch.no_grad()
    def to_split(self, inplace: bool = True) -> "SwiGLU":
        """Concatenate ``w1|w3`` into one ``gate_up`` Linear (same output, one GEMM, new keys).

        ``w2`` is carried over unchanged, so the module computes exactly what it did before --
        only the parameter names change (``w1``/``w3`` -> ``gate_up``), which is why this is not
        done for you by default.
        """
        if self.split:
            return self
        if (self.w1.bias is None) != (self.w3.bias is None):
            raise ValueError("to_split() needs both gate/up projections to have (or both to lack) a "
                             "bias; concatenating would drop one of them")
        out = self
        if not inplace:
            out = SwiGLU(self.dim, self.hidden, bias=self.w2.bias is not None, split=False,
                         device=self.w2.weight.device, dtype=self.w2.weight.dtype)
            out.w2.weight.copy_(self.w2.weight)
            if out.w2.bias is not None:
                out.w2.bias.copy_(self.w2.bias)
        fused = nn.Linear(self.dim, 2 * self.hidden, bias=self.w1.bias is not None,
                          device=self.w1.weight.device, dtype=self.w1.weight.dtype)
        fused.weight.copy_(torch.cat([self.w1.weight, self.w3.weight], 0))
        if fused.bias is not None:
            fused.bias.copy_(torch.cat([self.w1.bias, self.w3.bias], 0))
        del out.w1, out.w3
        out.gate_up = fused
        out.split = True
        return out
