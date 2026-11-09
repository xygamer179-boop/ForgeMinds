"""Multi-process check for ``forgemind.Muon`` under torch.distributed (gloo/CPU).

    python tests/muon_mp_check.py                            # spawns 2 processes
    torchrun --standalone --nproc_per_node=2 tests/muon_mp_check.py   # already distributed

Two ranks DDP-train a small model with ``fm.Muon`` on identical data. Every shape bucket
must be sharded across ranks and recombined with a single ``all_reduce`` over disjoint
slices, so the replicas stay bit-identical **and** the result must equal a plain
single-process ``fm.Muon`` run of the same steps (no dist call, world 1).
"""

import os
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import forgemind as fm  # noqa: E402


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4))

    def forward(self, x):
        return self.f(x)


def _train(net, opt, x, y, steps=4):
    model = net if not hasattr(net, "module") else net
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def _reference_run():
    """The same training with world 1: what a non-distributed Muon produces."""
    torch.manual_seed(0)
    ref_net = Net()
    ref_opt = fm.Muon(ref_net.parameters(), lr=0.05, weight_decay=0.01)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(32, 8, generator=g)
    y = torch.randn(32, 4, generator=g)
    _train(ref_net, ref_opt, x, y)
    return torch.cat([p.detach().reshape(-1) for p in ref_net.parameters()])


def run(rank, world, port):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)

    torch.manual_seed(0)
    net = Net()
    opt = fm.Muon(net.parameters(), lr=0.05, weight_decay=0.01)
    model = nn.parallel.DistributedDataParallel(net)

    # identical data on every rank: DDP averages identical grads (exact in fp32),
    # so the replicas must stay bit-identical and match the single-process reference
    g = torch.Generator().manual_seed(7)
    x = torch.randn(32, 8, generator=g)
    y = torch.randn(32, 4, generator=g)
    losses = _train(model, opt, x, y)

    flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    gathered = [torch.empty_like(flat) for _ in range(world)]
    dist.all_gather(gathered, flat)
    in_sync = all(torch.equal(gathered[0], t) for t in gathered)

    ref_flat = _reference_run()
    ref_ok = torch.equal(flat, ref_flat)

    if not in_sync or not ref_ok:
        d_sync = max((gathered[0] - t).abs().max().item() for t in gathered)
        d_ref = (flat - ref_flat).abs().max().item()
        print(f"RESULT FAIL rank {rank}: in_sync={in_sync}({d_sync}) ref_ok={ref_ok}({d_ref})",
              flush=True)
        dist.barrier()
        sys.exit(1)
    print(f"RESULT muon_mp rank {rank}/{world} loss {losses[0]:.6f} -> {losses[-1]:.6f} "
          f"in_sync=yes ref_ok=yes", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        run(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), 0)
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(run, args=(2, port), nprocs=2, join=True)


if __name__ == "__main__":
    main()
