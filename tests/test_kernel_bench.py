"""``python -m forgemind.cli kbench``: the harness that decides whether a kernel is worth it.

The numbers are only meaningful on a GPU (the CPU run exercises the tool, not the kernels), but
the *plumbing* is testable everywhere: the table renders, the JSON lands, and ``--check``
refuses to pretend on a machine where no kernel runs.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch

from forgemind.cli import kbench

TINY = ["--tokens", "64", "--dim", "64", "--hidden", "96", "--seq", "16", "--heads", "2",
        "--head-dim", "8", "--batch", "2", "--vocab", "64", "--iters", "2", "--warmup", "1",
        "--no-compile"]


def test_kbench_runs_and_writes_json(tmp_path):
    out = tmp_path / "k.json"
    rc = kbench.main(["--ops", "rms_norm,swiglu,swiglu_split,qk_norm_rope", *TINY, "--allow-cpu",
                      "--json", str(out)])
    assert rc == 0
    rows = json.loads(out.read_text())
    assert {r["op"] for r in rows} == {"rms_norm", "swiglu", "swiglu_split", "qk_norm_rope"}
    for r in rows:
        assert r["kernel"] > 0 and r["eager"] > 0


def test_check_refuses_to_run_on_cpu_without_asking():
    """On CPU the kernels fall back, so every column measures the same thing: a pass would be
    a lie, and the whole point of --check is to be a gate on real hardware."""
    if torch.cuda.is_available():  # pragma: no cover
        pytest.skip("CUDA present: the refusal does not apply")
    assert kbench.main(["--ops", "rms_norm", *TINY, "--check"]) == 2


def test_kbench_is_reachable_as_a_subcommand():
    r = subprocess.run([sys.executable, "-m", "forgemind.cli", "kbench", "--help"],
                       capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert r.returncode == 0, r.stderr[-500:]
    assert "--tolerance" in r.stdout
