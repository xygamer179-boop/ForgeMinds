"""torch.compile conformance: zero graph breaks on the dense path, compile == eager numerically.

`fullgraph=True` is the strictest possible check -- dynamo raises instead of falling back, so a passing
test here means the model really is one fused graph. The MoE test asserts that the *only* graph break left
is the deliberate one around the data-dependent expert dispatch (which is unavoidable without
`torch._grouped_mm`, and which the CUDA+bf16 path does not have at all).

Uses the `aot_eager` backend so the suite runs without a C++ toolchain; set FORGEMIND_TEST_INDUCTOR=1 to
also run the real inductor backend.

Run:  python tests/test_compile.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from forgemind import ForgeConfig, GPT, GPTConfig, Trainer, SyntheticLM
from forgemind.parallel.ac import apply_activation_checkpointing

BACKEND = os.environ.get("FORGEMIND_TEST_BACKEND", "aot_eager")
MC = dict(vocab_size=64, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq=32)


def _model(**kw):
    m = GPT(GPTConfig(**MC, **kw))
    m.prepare(device=torch.device("cpu"), dtype=torch.float32, seq=MC["max_seq"])
    return m


def _batch(bs=2, t=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 64, (bs, t), generator=g), torch.randint(0, 64, (bs, t), generator=g)


def test_dense_is_one_graph():
    m, (x, y) = _model(), _batch()
    e = torch._dynamo.explain(lambda t, l: m(t, labels=l))(x, y)
    assert e.graph_break_count == 0, [str(b.reason)[:200] for b in e.break_reasons]
    assert e.graph_count == 1, e.graph_count
    print(f"dense: {e.graph_count} graph, {e.graph_break_count} breaks")


def test_activation_checkpointing_is_one_graph():
    for mode in ("full", "selective"):
        m, (x, y) = _model(), _batch()
        apply_activation_checkpointing(m, mode)
        e = torch._dynamo.explain(lambda t, l: m(t, labels=l))(x, y)
        assert e.graph_break_count == 0, (mode, [str(b.reason)[:200] for b in e.break_reasons])
        print(f"ac={mode}: {e.graph_count} graph, {e.graph_break_count} breaks")


def test_fullgraph_matches_eager():
    """fullgraph=True forward+backward, gradients identical to eager."""
    for mode in (None, "full", "selective"):
        torch.manual_seed(0)
        m = _model()
        torch.manual_seed(0)
        ref = _model()          # same weights, independent object (deepcopy would share the AC closures)
        if mode:
            apply_activation_checkpointing(m, mode)
            apply_activation_checkpointing(ref, mode)
        x, y = _batch()
        ce_e, aux_e = m(x, labels=y)
        (ce_e + (aux_e if aux_e is not None else 0)).backward()
        eager_grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

        cm = torch.compile(lambda t, l: ref(t, labels=l), dynamic=False, fullgraph=True, backend=BACKEND)
        ce_c, aux_c = cm(x, y)
        (ce_c + (aux_c if aux_c is not None else 0)).backward()
        assert torch.allclose(ce_e, ce_c, atol=1e-5), (ce_e, ce_c)
        for n, p in ref.named_parameters():
            if n in eager_grads:
                assert torch.allclose(p.grad, eager_grads[n], atol=1e-5), n
        print(f"fullgraph ac={mode}: loss + all grads match eager")


def test_moe_only_breaks_on_the_dispatcher():
    m, (x, y) = _model(n_experts=4, top_k=2), _batch()
    e = torch._dynamo.explain(lambda t, l: m(t, labels=l))(x, y)
    reasons = " ".join(str(b.reason) for b in e.break_reasons)
    assert "bincount" not in reasons.lower(), "bincount came back into the traced region"
    assert "data-dependent" not in reasons.lower() and "guard on data-dependent" not in reasons.lower(), reasons[:400]
    bad = [r for r in (str(b.reason) for b in e.break_reasons) if "compiler.disable" not in r]
    assert not bad, f"unexpected MoE graph breaks: {bad[:3]}"
    n_moe = sum(1 for b in m.layers.values() if b.is_moe)
    print(f"moe: {e.graph_count} graphs, {e.graph_break_count} breaks "
          f"({n_moe} deliberate eager dispatches, torch._grouped_mm unavailable on CPU)")


def test_rope_cache_is_not_rebuilt_inside_the_graph():
    m, (x, y) = _model(), _batch()
    assert not m.layers["0"].attn._cos.numel() == 0, "prepare() did not fill the RoPE buffers"
    before = m.layers["0"].attn._cos
    torch.compile(lambda t, l: m(t, labels=l), fullgraph=True, backend=BACKEND)(x, y)
    assert m.layers["0"].attn._cos is before, "RoPE tables were rebuilt (and would break the graph)"


def test_trainer_compiles_end_to_end():
    cfg = ForgeConfig()
    cfg.train.__dict__.update(max_steps=4, warmup_steps=1, bf16=False, resume=False, log_every=10 ** 9,
                              compile=True, compile_backend=BACKEND, compile_fullgraph=True)
    ds = SyntheticLM(64, 32, n=128)
    tr = Trainer(lambda c: GPT(GPTConfig(**MC)), cfg, ds, micro_batch=8)
    h = tr.fit()
    assert h[-1]["loss"] < h[0]["loss"], [round(x["loss"], 3) for x in h]
    print("trainer compile=True/fullgraph=True:", [round(x["loss"], 3) for x in h])


def test_trainer_compiles_with_ac_and_accum():
    cfg = ForgeConfig()
    cfg.train.__dict__.update(max_steps=3, warmup_steps=1, bf16=False, resume=False, log_every=10 ** 9,
                              compile=True, compile_backend=BACKEND, compile_fullgraph=True,
                              ac="selective", grad_accum=2)
    ds = SyntheticLM(64, 32, n=128)
    tr = Trainer(lambda c: GPT(GPTConfig(**MC)), cfg, ds, micro_batch=8)
    h = tr.fit()
    assert h[-1]["loss"] < h[0]["loss"]
    print("trainer compile + selective AC + accum: OK")


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            print("==", k)
            f()
    print("ALL COMPILE TESTS PASSED")
