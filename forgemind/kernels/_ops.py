"""The bridge between the Triton kernels and ``torch.library``, plus the dispatch rule.

Why two paths exist
-------------------
The same Triton launch is reachable two ways:

* **eager** -> ``torch.autograd.Function.apply``
* **traced** (``torch.compile``, ``torch.export``, ``torch.func``) ->
  ``torch.library.custom_op`` with a fake implementation and a registered backward.

That is not hedging, it is a measured trade. On this machine (CPU, torch 2.9, 20k timed calls
of a trivial 64x64 op):

===================  ==========
``plain torch``      11.7 us
``Function.apply``   24.2 us
``custom_op``        59.0 us
===================  ==========

A custom op is the *slowest* thing to call from Python, because every call re-enters the
dispatcher through a boxed Python kernel. In eager mode that overhead is per launch and would
make the kernels lose to plain torch on small tensors -- exactly the "no speed decrease"
property these kernels are not allowed to break.

Under ``torch.compile`` the calculus inverts: the op is captured **once**, becomes a node in
the graph, and afterwards the whole graph is replayed from C++ without touching Python. Then
the custom op costs nothing per call and buys three things ``autograd.Function`` cannot give:

1. a **fake implementation**, so dynamo/inductor/export propagate shapes without running the
   kernel (``autograd.Function`` needs a real trace of ``forward`` to learn the output meta);
2. a **registered backward** that AOTAutograd places in the backward graph like any other op,
   including under activation checkpointing;
3. ``torch.export.export`` compatibility, which is what ahead-of-time inductor (AOTI) serving
   stacks consume -- an ``autograd.Function`` is a black box to them.

So: ``Function`` in eager, ``custom_op`` while tracing. Same launch, same numbers.
``fm.kernels.use_custom_ops(True)`` forces the op path in eager for tests and exporters, and
``FORGEMIND_KERNEL_OPS=ops|eager`` does the same from the environment.

What each kernel must provide
-----------------------------
Two ops, a forward and a backward, both with fake implementations:

``forward`` (launch), ``fake`` (shapes/dtypes only, no kernel), ``setup_context``
(``ctx.save_for_backward(...)`` on tensors the backward needs) and ``backward`` returning one
gradient per *input*. Values the backward needs but the caller does not (the ``1/rms`` scale,
the residual sum) are returned as extra outputs of the op, so they travel through the graph
instead of being recomputed or stashed in a global.

**The backward must be an op too**, not a bare launch. A registered backward is ordinary Python
that dynamo traces -- including inside AOTAutograd's *joint* trace, where every tensor is a
FakeTensor. A raw Triton launch inside it would be executed with fake tensors the moment the
graph is compiled (``Attempted to access the data pointer on an invalid python storage``). So
each kernel registers ``forgemind::<name>`` (autograd) *and* ``forgemind::<name>_backward``
(fake impl, no autograd), and the Python backward is a single op call. Second derivatives are
not supported, exactly as before: the Triton backward never had a grad_fn.
"""

from __future__ import annotations

import os

import torch

try:  # torch >= 2.4 for custom_op, >= 2.5 for a registered backward
    from torch.library import CustomOpDef

    _HAS_AUTOGRAD = hasattr(CustomOpDef, "register_autograd")
except Exception:  # pragma: no cover - older torch
    _HAS_AUTOGRAD = False

AVAILABLE = bool(getattr(torch.library, "custom_op", None)) and _HAS_AUTOGRAD
_ENV = os.environ.get("FORGEMIND_KERNEL_OPS", "auto").strip().lower()
_FORCED: list = [None]          # None = automatic (on while tracing)


def available() -> bool:
    """Is the custom-op path usable on this torch build?"""
    return AVAILABLE


def use_custom_ops(flag: "bool | None" = None) -> bool:
    """Force the custom-op path on (``True``) / off (``False``) / automatic (``None``).

    Returns the resulting state. Automatic means: on while ``torch.compile`` traces, off in
    eager -- see the module docstring for the measurements behind that rule.
    """
    if flag and not AVAILABLE:  # pragma: no cover - torch < 2.5
        raise RuntimeError("torch.library.custom_op with a registered backward is not "
                           f"available on torch {torch.__version__}")
    _FORCED[0] = flag
    return wanted()


def wanted() -> bool:
    """Should the next kernel call go through ``torch.ops.forgemind``?"""
    if not AVAILABLE:
        return False
    if _FORCED[0] is not None:
        return bool(_FORCED[0])
    if _ENV == "ops":
        return True
    if _ENV == "eager":
        return False
    try:
        return torch.compiler.is_compiling()
    except Exception:  # pragma: no cover - torch without the API
        return False


def _register(name: str, forward, fake):
    """``torch.library.custom_op`` + fake impl; returns the callable, reusing an existing op."""
    try:
        op = torch.library.custom_op(f"forgemind::{name}", mutates_args=())(forward)
        op.register_fake(fake)
        return op
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - duplicate registration
        existing = getattr(getattr(torch.ops, "forgemind", None), name, None)
        if existing is None or ("already exists" not in str(exc)
                                and "overriding" not in str(exc)):
            raise
        return existing.default


def define(name: str, forward, fake, setup_context, backward):
    """Register the differentiable op ``forgemind::<name>`` (fake impl + backward)."""
    if not AVAILABLE:  # pragma: no cover
        return None
    op = _register(name, forward, fake)
    op.register_autograd(backward, setup_context=setup_context)
    return op


def define_backward(name: str, forward, fake):
    """Register ``forgemind::<name>_backward``: an op with a fake impl and no autograd.

    Called from inside a registered backward, so it must itself be an op -- see the module
    docstring. Registration at import time is duplicate-safe (module reload, two imports).
    """
    if not AVAILABLE:  # pragma: no cover
        return None
    return _register(f"{name}_backward", forward, fake)


__all__ = ["AVAILABLE", "available", "use_custom_ops", "wanted", "define", "define_backward"]
