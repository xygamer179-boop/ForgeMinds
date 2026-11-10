from .config import ForgeConfig, ParallelConfig, QuantConfig, OffloadConfig, TrainConfig
from .distributed import (init_distributed, DistContext, cleanup,
                         all_reduce, all_gather, broadcast, barrier)
from .trainer import Trainer, moe_kwargs
from .models import GPT, GPTConfig
from .moe import MoE
from .offload import OffloadedAdamW
from .data import (SyntheticLM, TokenFile, BatchStream, ResumableSampler, build_loader,
                   async_loader, to_device)
from .perf import count_host_syncs, host_syncs
from .forge import Forge, launch, seed_everything

__version__ = "0.9.0"


def __getattr__(name):
    """Lazy submodule access: ``fm.kernels`` only imports triton when you touch it.

    ``fm.nn`` (the composable ``RMSNorm`` / ``SwiGLU`` blocks) is lazy for the same reason: it
    pulls in the kernels package, and ``import forgemind`` should stay cheap for people who only
    want the trainer.
    """
    if name in ("kernels", "forge", "autotune", "nn", "muon"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    if name == "Muon":
        from .muon import Muon
        return Muon
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
__all__ = [
    "ForgeConfig", "ParallelConfig", "QuantConfig", "OffloadConfig", "TrainConfig",
    "init_distributed", "DistContext", "cleanup",
    "Trainer", "moe_kwargs",
    "GPT", "GPTConfig", "MoE", "OffloadedAdamW",
    "SyntheticLM", "TokenFile", "BatchStream", "ResumableSampler", "build_loader",
    "async_loader", "to_device",
    "count_host_syncs", "host_syncs",
    "Forge", "launch", "seed_everything", "Muon",
    "all_reduce", "all_gather", "broadcast", "barrier",
]
