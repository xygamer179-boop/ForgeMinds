from .config import ForgeConfig, ParallelConfig, QuantConfig, OffloadConfig, TrainConfig
from .distributed import init_distributed, DistContext, cleanup
from .trainer import Trainer, moe_kwargs
from .models import GPT, GPTConfig
from .moe import MoE
from .offload import OffloadedAdamW
from .data import SyntheticLM, TokenFile, BatchStream, ResumableSampler, build_loader
from .perf import count_host_syncs, host_syncs
from .forge import Forge, launch

__version__ = "0.4.0"


def __getattr__(name):
    """Lazy submodule access: ``fm.kernels`` only imports triton when you touch it."""
    if name in ("kernels", "forge", "autotune"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
__all__ = [
    "ForgeConfig", "ParallelConfig", "QuantConfig", "OffloadConfig", "TrainConfig",
    "init_distributed", "DistContext", "cleanup",
    "Trainer", "moe_kwargs",
    "GPT", "GPTConfig", "MoE", "OffloadedAdamW",
    "SyntheticLM", "TokenFile", "BatchStream", "ResumableSampler", "build_loader",
    "count_host_syncs", "host_syncs",
    "Forge", "launch",
]
