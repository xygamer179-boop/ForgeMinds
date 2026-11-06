"""DeepSpeed backend: ZeRO-1/2/3 (+CPU/NVMe offload), AutoTP, DeepSpeed pipeline engine => 3D parallelism.
NOTE: written against the public DeepSpeed API but NOT exercised in CI (needs GPUs); config builder is unit-tested."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..config import ForgeConfig


def build_ds_config(cfg: ForgeConfig, world: int, micro_batch: int) -> dict:
    pc, tc, oc = cfg.parallel, cfg.train, cfg.offload
    dp = world // (pc.tp * pc.pp)
    if pc.pp > 1 and pc.zero_stage > 1:
        raise ValueError("DeepSpeed pipeline parallelism supports ZeRO stage <= 1 only")
    zero = {"stage": pc.zero_stage, "overlap_comm": True, "contiguous_gradients": True, "reduce_bucket_size": 5e8}
    if pc.zero_offload_optimizer or oc.optimizer:
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    if pc.zero_offload_param and pc.zero_stage == 3:
        zero["offload_param"] = {"device": "cpu", "pin_memory": True}
    if pc.zero_stage == 3:
        zero.update({"stage3_prefetch_bucket_size": "auto", "stage3_param_persistence_threshold": "auto",
                     "stage3_gather_16bit_weights_on_model_save": True})
    ds = {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": tc.grad_accum,
        "gradient_clipping": tc.grad_clip,
        "bf16": {"enabled": tc.bf16},
        "zero_optimization": zero,
        "steps_per_print": 10 ** 9,
        "wall_clock_breakdown": False,
    }
    if pc.tp > 1:
        ds["tensor_parallel"] = {"autotp_size": pc.tp}
    ds.update(pc.ds_extra)
    return ds


def to_pipeline_module(model, num_stages: int, loss_fn, topology=None):
    """Flatten the reference GPT into a DeepSpeed PipelineModule (embed -> blocks -> norm+head)."""
    from deepspeed.pipe import PipelineModule

    class Head(nn.Module):
        def __init__(self, norm, head):
            super().__init__(); self.norm, self.head = norm, head
        def forward(self, h):
            return self.head(self.norm(h))

    layers = [model.tok_emb] + [model.layers[k] for k in sorted(model.layers, key=int)] + [Head(model.norm, model.lm_head)]
    return PipelineModule(layers=layers, num_stages=num_stages, loss_fn=loss_fn, topology=topology,
                          partition_method="parameters")


def init_deepspeed(model, param_groups, cfg: ForgeConfig, world: int, micro_batch: int):
    import deepspeed
    ds_cfg = build_ds_config(cfg, world, micro_batch)
    engine, optim, _, _ = deepspeed.initialize(model=model, model_parameters=param_groups, config=ds_cfg,
                                               optimizer=_ds_optimizer(cfg, param_groups))
    return engine, optim


def _ds_optimizer(cfg, groups):
    tc = cfg.train
    if cfg.parallel.zero_offload_optimizer or cfg.offload.optimizer:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        return DeepSpeedCPUAdam(groups, lr=tc.lr, betas=tc.betas, eps=tc.eps, weight_decay=tc.weight_decay)
    from deepspeed.ops.adam import FusedAdam
    return FusedAdam(groups, lr=tc.lr, betas=tc.betas, eps=tc.eps, weight_decay=tc.weight_decay)
