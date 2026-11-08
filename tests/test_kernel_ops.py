"""The ``torch.library.custom_op`` path: same numbers as the eager path, and traceable.

The kernels launch either through ``torch.autograd.Function`` (eager) or through
``torch.ops.forgemind.*`` (while tracing). These tests pin both halves of that contract:

* the two paths produce **identical** values and gradients, because they call the same launch;
* the op path survives ``torch.compile(fullgraph=True)`` *and* ``torch.export``, which is the
  reason for the op path to exist at all (see ``forgemind/kernels/_ops.py``);
* the fake implementations agree with the real ones on shape and dtype, otherwise inductor
  would allocate the wrong things.

Everything here runs in Triton's interpreter (``TRITON_INTERPRET=1``), so it works without a
GPU. Set ``FORGEMIND_TEST_INDUCTOR=1`` to also run the real inductor backend.

Run:  TRITON_INTERPRET=1 python tests/test_kernel_ops.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch

import forgemind as fm
from forgemind.kernels import _common, _ops

BACKEND = os.environ.get("FORGEMIND_TEST_BACKEND", "aot_eager")
HAS_KERNELS = _common.INTERPRET or torch.cuda.is_available()

if not _common.INTERPRET and torch.cuda.is_available():  # pragma: no cover
    _common.set_backend("triton")


def needs_kernels(f):
    """Runs only where a real kernel runs (the interpreter or a GPU); the CPU-only tests below
    deliberately do not carry this, so the default suite still exercises the dispatch rules."""
    return pytest.mark.skipif(not HAS_KERNELS,
                              reason="needs TRITON_INTERPRET=1 or CUDA")(f)


def needs_ops(f):
    """torch without the ``custom_op`` API (pre-2.4) has no op path to test."""
    return pytest.mark.skipif(not _ops.AVAILABLE,
                              reason="this torch has no torch.library.custom_op")(f)


@pytest.fixture(autouse=True)
def auto_mode():
    """Every test starts and ends in the default (automatic) dispatch mode."""
    _ops.use_custom_ops(None)
    yield
    _ops.use_custom_ops(None)


def _cases():
    """(name, callable taking a mode-independent call, inputs) for every wrapped kernel."""
    torch.manual_seed(0)
    x = torch.randn(6, 16)
    w = torch.randn(16)
    gate, up = torch.randn(6, 32), torch.randn(6, 32)
    split = torch.randn(6, 64)
    xr = torch.randn(2, 4, 2, 8)
    wr = torch.randn(8)
    cos = torch.randn(4, 4)
    sin = torch.randn(4, 4)

    def rms(t):
        return fm.kernels.rms_norm(t[0], t[1], 1e-6)

    def addrms(t):
        a, b = fm.kernels.add_rms_norm(t[0], t[1], t[2], 1e-6)
        return a + b

    def swig(t):
        return fm.kernels.swiglu(t[0], t[1])

    def swigs(t):
        return fm.kernels.swiglu_split(t[0])

    def rope(t):
        return fm.kernels.qk_norm_rope(t[0], t[1], t[2], t[3], 1e-6)

    return [
        ("rms_norm", rms, [x, w]),
        ("add_rms_norm", addrms, [x, x.clone(), w]),
        ("swiglu", swig, [gate, up]),
        ("swiglu_split", swigs, [split]),
        ("qk_norm_rope", rope, [xr, wr, cos, sin]),
    ]


@pytest.mark.parametrize("name,fn,inputs", _cases(), ids=lambda v: v if isinstance(v, str) else "")
@needs_kernels
def test_op_path_is_numerically_identical_to_the_eager_path(name, fn, inputs):
    def run(mode):
        _ops.use_custom_ops(mode)
        ins = [t.clone().requires_grad_(t.is_floating_point()) for t in inputs]
        out = fn(ins)
        out.square().sum().backward()
        return out.detach().clone(), [t.grad.clone() for t in ins if t.grad is not None]

    y_eager, g_eager = run(False)
    y_op, g_op = run(True)
    assert torch.equal(y_eager, y_op), f"{name}: forward differs between the two paths"
    assert len(g_eager) == len(g_op)
    for a, b in zip(g_eager, g_op):
        assert torch.equal(a, b), f"{name}: gradient differs between the two paths"


@needs_ops
def test_all_ops_are_registered():
    names = {n for n in dir(torch.ops.forgemind) if not n.startswith("_")}
    for op in ("rms_norm", "add_rms_norm", "swiglu", "swiglu_split", "qk_norm_rope"):
        assert op in names, op
        assert f"{op}_backward" in names, f"{op}_backward"


@pytest.mark.parametrize("name,fn,inputs", _cases(), ids=lambda v: v if isinstance(v, str) else "")
@needs_kernels
def test_compiled_forward_and_backward_match_eager(name, fn, inputs):
    """fullgraph=True: the op is *in* the graph, and the compiled backward is the kernel."""
    def run(mode, compile_it):
        _ops.use_custom_ops(mode)
        ins = [t.clone().requires_grad_(t.is_floating_point()) for t in inputs]
        f = torch.compile(fn, backend=BACKEND, fullgraph=True) if compile_it else fn
        out = f(ins)
        out.square().sum().backward()
        return out.detach().clone(), [t.grad.clone() for t in ins if t.grad is not None]

    y_ref, g_ref = run(False, False)
    y, g = run(None, True)
    assert torch.allclose(y, y_ref, atol=2e-6, rtol=1e-5), name
    for a, b in zip(g, g_ref):
        assert torch.allclose(a, b, atol=2e-6, rtol=1e-5), name


@needs_kernels
def test_export_keeps_the_kernel_as_one_node():
    """``torch.export`` is what AOTInductor serving consumes; an autograd.Function is a black
    box to it, a registered custom op is not."""
    x, w = torch.randn(6, 16), torch.randn(16)

    class M(torch.nn.Module):
        def forward(self, t):
            return fm.kernels.rms_norm(t, w, 1e-6)

    _ops.use_custom_ops(None)          # automatic: export traces, so the op path is taken
    ep = torch.export.export(M(), (x,))
    targets = [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]
    assert "forgemind.rms_norm.default" in targets, targets
    assert ep.module()(x).shape == fm.kernels.rms_norm(x, w, 1e-6).shape


@pytest.mark.parametrize("name,fn,inputs", _cases(), ids=lambda v: v if isinstance(v, str) else "")
@needs_kernels
def test_fake_impls_agree_with_the_real_ones(name, fn, inputs):
    """A fake that disagrees on shape or dtype silently corrupts a compiled graph."""
    _ops.use_custom_ops(True)
    real = fn([t.clone() for t in inputs])
    with torch._subclasses.fake_tensor.FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fake = fn([mode.from_tensor(t) for t in inputs])
    assert tuple(fake.shape) == tuple(real.shape), name
    assert fake.dtype == real.dtype, name


@needs_ops
def test_dispatch_flag_and_environment():
    assert _ops.wanted() is False                      # eager: the Function path
    assert _ops.use_custom_ops(True) is True
    assert _ops.wanted() is True
    assert _ops.use_custom_ops(False) is False
    assert _ops.use_custom_ops(None) is False           # back to automatic
    assert _ops.available() is True


@needs_kernels
def test_inference_path_does_not_need_autograd():
    x, w = torch.randn(4, 8), torch.randn(8)
    _ops.use_custom_ops(True)
    with torch.no_grad():
        y = fm.kernels.rms_norm(x, w, 1e-6)
    assert y.shape == x.shape and not y.requires_grad


def test_custom_ops_are_not_used_when_the_kernels_cannot_run():
    """On a CPU without the interpreter the op path must never be entered: it would freeze the
    torch expression inside an opaque op and block inductor's fusion."""
    saved = _common.get_backend()
    _common.set_backend("fallback")
    try:
        _ops.use_custom_ops(True)
        x, w = torch.randn(4, 8), torch.randn(8)
        y = fm.kernels.rms_norm(x, w, 1e-6)
        assert torch.allclose(y, torch.nn.functional.rms_norm(x, (8,), w, 1e-6))
        assert fm.kernels.stats()["fallback"] >= 1
    finally:
        _common.set_backend(saved)
