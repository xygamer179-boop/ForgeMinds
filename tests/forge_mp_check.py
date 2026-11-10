"""Multi-process check for ``forgemind.Forge`` / ``forgemind.launch``.

    python tests/forge_mp_check.py            # spawns 2 processes via fm.launch
    torchrun --standalone --nproc_per_node=2 tests/forge_mp_check.py   # already distributed

Both paths must leave every rank with bit-identical parameters (DDP averages gradients, so
ranks that start from the same weights and step together stay in sync), report the same
gradient norm, and agree on a **rank-0 checkpoint** (``Forge.save`` writes from rank 0 only;
``fm.all_gather``/``fm.barrier`` are the embedded collectives).
"""

import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import forgemind as fm  # noqa: E402


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(),
                                     torch.nn.Linear(16, 1))

    def forward(self, x, y=None):
        return self.f(x)


def main():
    torch.manual_seed(0)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    model = Net()
    forge = fm.Forge(model, torch.optim.AdamW(model.parameters(), lr=1e-2), precision="fp32",
                     quiet=True, callbacks=[])
    # every rank uses the same data so the losses are directly comparable
    g = torch.Generator().manual_seed(7)
    x = torch.randn(32, 8, generator=g)
    y = torch.randn(32, 1, generator=g)
    loss_fn = lambda out, b: torch.nn.functional.mse_loss(out, b[1])
    forge.loss_fn = loss_fn
    losses = []
    for _ in range(4):
        st = None
        while st is None:
            st = forge.step((x, y))
        losses.append(st["loss"])

    flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    gathered = fm.all_gather(flat.clone())      # world 1: as-is; else [rank0 | rank1 | ...]
    n = flat.numel()
    same = all(torch.equal(gathered[:n], gathered[i * n:(i + 1) * n]) for i in range(world))
    if not same:
        print(f"RESULT FAIL rank {rank}: replicas diverged")
        sys.exit(1)

    # rank-0 checkpoint round-trip (shared path: mkdtemp would give each process its own dir)
    path = os.path.join(tempfile.gettempdir(), "forge_mp_check_shared.pt")
    if rank == 0 and os.path.exists(path):
        os.remove(path)
    fm.barrier()
    forge.save(path)                            # rank 0 writes; the others wait at the barrier
    net2 = Net()
    forge2 = fm.Forge(net2, torch.optim.AdamW(net2.parameters(), lr=1e-2), precision="fp32",
                      quiet=True, callbacks=[])
    forge2.load(path)
    ckpt_ok = (forge2.step_count == forge.step_count
               and all(torch.equal(a, b)
                       for a, b in zip(model.parameters(), forge2.raw_model.parameters())))
    if not ckpt_ok:
        print(f"RESULT FAIL rank {rank}: checkpoint mismatch")
        sys.exit(1)
    print(f"RESULT forge_mp rank {rank}/{world} loss {losses[0]:.6f} -> {losses[-1]:.6f} "
          f"grad_norm {st['grad_norm']:.6f} in_sync=yes ckpt_ok=yes")


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        fm.launch(nproc=1)(main)()          # already under torchrun
    else:
        fm.launch(nproc=2)(main)()
