"""The documentation is part of the interface: it must not describe code that does not exist.

Two failure modes are worth a test, and both have actually happened in this repo:

* a public helper was documented (``fm.async_loader``) *before* it was exported from the package,
  so every example in the docs raised ``AttributeError``;
* a table row was edited and a cross-reference anchor stopped resolving.

These tests read ``README.md`` and ``documentation.md`` and check both. They are cheap (< 100 ms),
need no accelerator, and fail loudly with the offending name.
"""

from __future__ import annotations

import dataclasses
import importlib
import re
from pathlib import Path

import pytest

import forgemind as fm

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", ROOT / "documentation.md"]

def _docs():
    for p in DOCS:
        assert p.exists(), f"{p} is missing"
        yield p, p.read_text(encoding="utf-8")


def _resolve(dotted: str):
    """Resolve ``fm.a.b`` through attributes, falling back to importing the submodule."""
    parts = dotted.split(".")
    assert parts[0] == "fm"
    obj = fm
    for i, part in enumerate(parts[1:], start=1):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            module = "forgemind." + ".".join(parts[1:])
            try:
                obj = importlib.import_module(module)
            except ImportError:
                return None, part
    return obj, None


@pytest.mark.parametrize("path", [p.name for p in DOCS])
def test_every_documented_attribute_exists(path):
    """`fm.something` in the docs must resolve on the installed package."""
    text = (ROOT / path).read_text(encoding="utf-8")
    missing = []
    for dotted in sorted(set(re.findall(r"\bfm(?:\.[A-Za-z_][A-Za-z0-9_]*)+", text))):
        obj, bad = _resolve(dotted)
        if obj is None:
            missing.append(f"{dotted} (no attribute {bad!r})")
    assert not missing, f"{path} documents names that do not exist: {missing}"


def _slug(heading: str) -> str:
    h = heading.strip().lower().replace("`", "")
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def _headings(path: Path) -> set[str]:
    return {_slug(m.group(2)) for m in re.finditer(r"^(#{1,6}) (.+)$", path.read_text(encoding="utf-8"), re.M)}


@pytest.mark.parametrize("path", [p.name for p in DOCS])
def test_internal_links_resolve(path):
    """`[text](#anchor)` and `[text](documentation.md#anchor)` must point at a real heading."""
    text = (ROOT / path).read_text(encoding="utf-8")
    targets = {"README.md": _headings(ROOT / "README.md"),
               "documentation.md": _headings(ROOT / "documentation.md")}
    broken = []
    for target_file, anchor in re.findall(r"\]\((README\.md|documentation\.md)?#([^)\s]+)\)", text):
        target = target_file or path
        if anchor not in targets[target]:
            broken.append(f"#{anchor} -> {target}")
    assert not broken, f"{path} has broken anchors: {broken}"


def test_public_api_is_documented():
    """Every name in `fm.__all__` should be mentioned in the reference at least once."""
    doc = (ROOT / "documentation.md").read_text(encoding="utf-8")
    undocumented = [n for n in fm.__all__ if n not in doc]
    assert not undocumented, f"public names missing from documentation.md: {undocumented}"


def test_every_config_field_is_documented():
    """A new knob must be documented, otherwise the reference silently rots."""
    doc = (ROOT / "documentation.md").read_text(encoding="utf-8")
    classes = (fm.ParallelConfig, fm.QuantConfig, fm.OffloadConfig, fm.TrainConfig, fm.GPTConfig)
    missing = [f"{cls.__name__}.{f.name}"
               for cls in classes for f in dataclasses.fields(cls) if f.name not in doc]
    assert not missing, f"config fields missing from documentation.md: {missing}"
