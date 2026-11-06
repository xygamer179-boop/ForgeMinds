import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from forgemind import ForgeConfig
from forgemind.parallel.deepspeed_backend import build_ds_config

def test_zero3_offload_tp():
    c = ForgeConfig(); c.parallel.zero_stage = 3; c.parallel.zero_offload_optimizer = True
    c.parallel.zero_offload_param = True; c.parallel.tp = 2; c.train.grad_accum = 4
    d = build_ds_config(c, world=8, micro_batch=2)
    z = d["zero_optimization"]
    assert z["stage"] == 3 and z["offload_optimizer"]["device"] == "cpu" and z["offload_param"]["device"] == "cpu"
    assert d["tensor_parallel"]["autotp_size"] == 2 and d["gradient_accumulation_steps"] == 4 and d["bf16"]["enabled"]

def test_pp_rejects_zero2():
    c = ForgeConfig(); c.parallel.pp = 2; c.parallel.zero_stage = 2
    try: build_ds_config(c, 4, 1)
    except ValueError: return
    raise AssertionError("expected ValueError")

if __name__ == "__main__":
    test_zero3_offload_tp(); test_pp_rejects_zero2(); print("DS CONFIG TESTS PASSED")
