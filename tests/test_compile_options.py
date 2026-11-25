"""Per-compile Inductor policy; never changes the global config or other backends."""
import pytest
import torch

from forgemind._compile import compile_options


def test_inductor_options_preserve_mode_and_do_not_mutate_global_config():
    if "triton.use_block_ptr" not in torch._inductor.list_options():
        pytest.skip("block pointer option unavailable in this PyTorch")
    with torch._inductor.config.patch({"triton.use_block_ptr": True}):
        for mode in ("default", "reduce-overhead", "max-autotune"):
            kw = compile_options("inductor", mode)
            expected = dict(torch._inductor.list_mode_options(mode=mode))
            expected["triton.use_block_ptr"] = False
            assert kw == {"options": expected}
            assert torch._inductor.config.triton.use_block_ptr is True
            # Ensure PyTorch accepts the combination without running a compiler.
            torch.compile(lambda x: x, backend="inductor", **kw)


def test_other_backend_keeps_mode():
    assert compile_options("aot_eager", "default") == {"mode": "default"}
    assert compile_options(lambda gm, example_inputs: gm, "default") == {"mode": "default"}


def test_forge_and_trainer_pass_inductor_options(monkeypatch):
    from types import SimpleNamespace
    from forgemind import Forge
    from forgemind.trainer import Trainer

    if "triton.use_block_ptr" not in torch._inductor.list_options():
        pytest.skip("block pointer option unavailable in this PyTorch")
    calls = []

    def fake_compile(model, **kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setenv("FORGEMIND_COMPILE_CPU", "1")
    monkeypatch.setattr(torch, "compile", fake_compile)
    Forge(torch.nn.Linear(2, 2), compile=True, precision="fp32", quiet=True)
    assert calls[-1]["options"]["triton.use_block_ptr"] is False
    assert calls[-1]["backend"] == "inductor"

    class Model:
        layers = []

        def compile(self, **kwargs):
            calls.append(kwargs)

    config = SimpleNamespace(compile_mode="reduce-overhead", compile_fullgraph=False,
                             compile_backend="inductor")
    Trainer._compile_blocks(None, Model(), config)
    assert calls[-1]["options"]["triton.use_block_ptr"] is False
    assert calls[-1]["options"]["triton.cudagraphs"] is True

    config.compile_backend = "aot_eager"
    Trainer._compile_blocks(None, Model(), config)
    assert "options" not in calls[-1] and calls[-1]["mode"] == "reduce-overhead"


def test_removed_inductor_option_is_not_passed(monkeypatch):
    monkeypatch.setattr(torch._inductor, "list_options", lambda: [])
    assert compile_options("inductor", "reduce-overhead") == {"mode": "reduce-overhead"}
