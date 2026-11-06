from __future__ import annotations
from dataclasses import dataclass, field, asdict


@dataclass
class ParallelConfig:
    """Mesh layout. world_size = pp * dp_replicate * dp_shard * tp.
    dp_shard=-1 -> fill whatever is left. ep carves expert-parallel groups out of the DP ranks."""
    backend: str = "native"          # "native" (torch FSDP2/DDP/TP/PP) | "deepspeed" (ZeRO + AutoTP/PP)
    dp_replicate: int = 1            # DDP degree (or HSDP replicate dim when dp_shard > 1)
    dp_shard: int = 1                # FSDP2 shard degree
    tp: int = 1
    pp: int = 1
    ep: int = 1                      # must divide dp_replicate*dp_shard
    pp_microbatches: int = 4
    pp_schedule: str = "1f1b"        # "1f1b" | "gpipe"
    fsdp_reshard_after_forward: bool = True
    fsdp_cpu_offload: bool = False
    # DDP: let the reducer bucket unused params instead of scanning the graph every iteration
    ddp_static_graph: bool = False
    init_timeout_s: int = 1800
    # DeepSpeed only
    zero_stage: int = 2
    zero_offload_optimizer: bool = False
    zero_offload_param: bool = False
    ds_extra: dict = field(default_factory=dict)


@dataclass
class QuantConfig:
    mode: str = "none"               # "none" | "fp8" | "int8" (mixed-precision training) | "int8_wo" (frozen weight-only)
    backend: str = "auto"            # "auto" | "te" | "torchao"
    exclude: tuple = ("lm_head", "gate", "router", "tok_emb")
    optimizer_8bit: bool = False     # torchao AdamW8bit


@dataclass
class OffloadConfig:
    optimizer: bool = False          # fp32 master + Adam states on CPU, grads streamed GPU->pinned CPU
    smart: bool = True               # only stream big grads / only when GPU memory is under pressure
    min_numel: int = 1 << 18
    gpu_watermark: float = 0.0       # 0 = always offload eligible grads; 0.8 = only when >80% of GPU mem used
    activations: bool = False
    act_min_bytes: int = 1 << 20
    act_layers: float = 1.0          # fraction of blocks (from the front) whose activations are offloaded
    act_pool: bool = True            # recycle pinned staging buffers instead of allocating per saved tensor
    experts: bool = False            # keep MoE expert weights in pinned CPU RAM, stream only the active experts


@dataclass
class TrainConfig:
    max_steps: int = 1000
    grad_accum: int = 1
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    bf16: bool = True
    compile: bool = False
    compile_mode: str = "default"    # torch.compile mode: "default" | "reduce-overhead" | "max-autotune"
    compile_backend: str = "inductor"  # torch.compile backend ("aot_eager" for tests without a C++ toolchain)
    compile_fullgraph: bool = False  # raises instead of falling back on a graph break
    ac: str = "none"                 # "none" | "full" | "selective"
    ac_preserve_rng: bool = False    # True if the model uses dropout/random ops inside a checkpointed block
    seed: int = 1234
    log_every: int = 10
    eval_every: int = 0
    eval_steps: int = 20
    ckpt_every: int = 0
    ckpt_dir: str = "checkpoints"
    resume: bool = True
    skip_spike_factor: float = 0.0   # >0: skip update if loss > factor * EMA(loss)
    skip_nonfinite: bool = True
    # data pipeline: prefetch overlaps the host->device copy of batch N+1 with step N
    prefetch: bool = True
    workers: int = 0


@dataclass
class ForgeConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)
