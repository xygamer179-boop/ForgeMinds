"""Multi-process check for ``forgemind.Fabric``: DDP setup, collectives, rank-0 checkpoints.

    python tests/fabric_mp_check.py            # spawns 2 processes via Fabric.launch

Every rank trains on identical data, so DDP's averaged gradients must leave the replicas
bit-identical; ``save`` must write from rank 0 only and ``load`` must restore the same bytes
on both ranks.
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import forgemind as fm  # noqa: E402


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.f(x)


def main():
    # constructed per process: the launcher's object only exists to spawn us
    fabric = fm.Fabric(accelerator="cpu", devices=2, strategy="ddp", precision="32-true")
    assert fabric.world_size == 2, fabric.world_size

    torch.manual_seed(0)
    net = Net()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    model, optimizer = fabric.setup(net, optimizer)

    # identical data on every rank, so losses are directly comparable
    g = torch.Generator().manual_seed(7)
    x = torch.randn(32, 8, generator=g)
    y = torch.randn(32, 1, generator=g)

    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        with fabric.autocast():
            loss = nn.functional.mse_loss(model(x), y)
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=1.0)
        optimizer.step()
        losses.append(float(fabric.all_reduce(loss.detach().clone()).item()))

    flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    gathered = fabric.all_gather(flat.clone())      # cat'd along dim 0: [rank0 | rank1]
    n = flat.numel()
    in_sync = all(torch.equal(gathered[:n], gathered[i * n:(i + 1) * n])
                  for i in range(fabric.world_size))

    # one path for every rank (mkdtemp would give each process its own directory)
    path = os.path.join(tempfile.gettempdir(), "fabric_mp_check_shared.pt")
    if fabric.is_global_zero and os.path.exists(path):
        os.remove(path)
    fabric.barrier()
    state = {"model": model, "optimizer": optimizer, "loss": losses[-1]}
    fabric.save(path, state)                     # rank 0 writes; then barrier
    net2, opt2 = Net(), None
    opt2 = torch.optim.SGD(net2.parameters(), lr=0.1)
    restored = {"model": net2, "optimizer": opt2, "loss": 0.0}
    fabric.load(path, restored)
    same_ckpt = (restored["loss"] == losses[-1]           # plain floats round-trip exactly
                 and all(torch.equal(a, b)
                         for a, b in zip(model.parameters(), net2.parameters())))

    if not in_sync or not same_ckpt:
        print(f"RESULT FAIL rank {fabric.global_rank}: in_sync={in_sync} ckpt={same_ckpt}")
        sys.exit(1)
    print(f"RESULT fabric_mp rank {fabric.global_rank}/{fabric.world_size} "
          f"loss {losses[0]:.6f} -> {losses[-1]:.6f} in_sync=yes ckpt_ok=yes")


if __name__ == "__main__":
    fabric = fm.Fabric(devices=2)                # only used to drive the launcher
    fabric.launch(main)
