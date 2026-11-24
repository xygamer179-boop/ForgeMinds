"""Static bans for two kernel bug classes that only surface on a real GPU compile.

Both rules came from crashes this repo has actually taken (or inherited from a port):

1. **method calls on scalar kernel arguments** -- ``eps.to(tl.float32)`` works in an
   eager launch (the argument is a Triton scalar there) but ``torch.compile``'s
   ``generate_ttir`` passes plain Python floats, so the same kernel dies with
   ``'float' object has no attribute 'to'`` (Liger-Kernel #1601: wrong output on
   torch 2.8, illegal memory access under FSDP). The fix is ``tl.full((), eps,
   tl.float32)`` or plain arithmetic -- never a method on the parameter.
2. **bare Python module globals inside ``@triton.jit`` bodies** -- triton >= 3.2
   rejects them at compile time (``Cannot access global variable ... from within
   @jit'ed function``); this repo hit it with topk's ``_E_BLOCK``. Constants that a
   kernel body needs must be ``tl.constexpr``-instantiated, passed as arguments, or
   inlined as literals.

The rules are AST-based so they run on a CPU-only box with no triton at all, and they
cover every module in ``forgemind/kernels/`` -- ports included, because the source
kernels are exactly where these patterns come from.

Run:  python -m pytest tests/test_kernel_hygiene.py -q
"""

from __future__ import annotations

import ast
import pathlib

KERNELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "forgemind" / "kernels"
BANNED_METHODS = {"to", "cast", "float", "half", "int", "long", "bool"}


def _jit_functions(tree: ast.Module) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Every function decorated with ``triton.jit`` (bare or with arguments)."""
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if (isinstance(target, ast.Attribute) and target.attr == "jit"
                    and isinstance(target.value, ast.Name) and target.value.id == "triton"):
                out.append((node.name, node))
                break
    return out


def _final_module_constants(tree: ast.Module) -> dict[str, object]:
    """Names whose FINAL module-level assignment is a plain constant.

    A later ``x = tl.constexpr(x)`` re-binding (the sanctioned fix) removes the name
    from the ban -- only an unconverted constant counts.
    """
    consts: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if not isinstance(value, ast.Constant):
            for t in targets:
                if isinstance(t, ast.Name):
                    consts.pop(t.id, None)
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                consts[t.id] = value.value
    return consts


def _kernel_files() -> list[pathlib.Path]:
    return sorted(p for p in KERNELS_DIR.glob("*.py") if p.name != "__init__.py")


def test_no_method_calls_on_scalar_kernel_arguments():
    """Liger #1601: ``param.to(...)`` dies under torch.compile's generate_ttir."""
    violations = []
    for path in _kernel_files():
        tree = ast.parse(path.read_text())
        for name, fn in _jit_functions(tree):
            params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (isinstance(func, ast.Attribute) and func.attr in BANNED_METHODS):
                    continue
                if (isinstance(func.value, ast.Name) and func.value.id in params):
                    violations.append(f"{path.name}:{node.lineno} {name}(): "
                                      f"{func.value.id}.{func.attr}(...)")
    assert not violations, ("method call on a kernel argument breaks torch.compile "
                            "(use tl.full((), x, tl.float32) or plain arithmetic):\n"
                            + "\n".join(violations))


def test_no_bare_python_globals_inside_jit_bodies():
    """triton >= 3.2: module constants must be constexpr-instantiated or passed in."""
    violations = []
    for path in _kernel_files():
        tree = ast.parse(path.read_text())
        consts = _final_module_constants(tree)
        if not consts:
            continue
        for name, fn in _jit_functions(tree):
            for node in ast.walk(fn):
                if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                        and node.id in consts):
                    violations.append(f"{path.name}:{node.lineno} {name}(): "
                                      f"reads module constant {node.id!r} = {consts[node.id]!r}")
    assert not violations, ("bare python global inside a @triton.jit body does not compile "
                            "on triton >= 3.2 (make it tl.constexpr, pass it as an argument, "
                            "or inline the literal):\n" + "\n".join(violations))
