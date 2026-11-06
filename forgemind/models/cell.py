"""GatedCell: a GRU-style recurrent cell as a torch.nn.Module.
Input x_t [B, D], state h [B, H] -> new state [B, H]. `GatedCell.scan(x)` unrolls over time [B, T, D] -> [B, T, H]."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.x2g = nn.Linear(input_dim, 3 * hidden_dim, bias=bias)   # update z, reset r, candidate n
        self.h2g = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        std = self.hidden_dim ** -0.5
        for w in self.parameters():
            nn.init.uniform_(w, -std, std)

    def forward(self, x, h=None):
        if h is None:
            h = x.new_zeros(x.size(0), self.hidden_dim)
        xz, xr, xn = self.x2g(x).chunk(3, -1)
        hz, hr, hn = self.h2g(h).chunk(3, -1)
        z = torch.sigmoid(xz + hz)
        r = torch.sigmoid(xr + hr)
        n = torch.tanh(xn + r * hn)
        return (1 - z) * n + z * h

    @torch._dynamo.disable
    def scan(self, x, h=None):
        """Sequential scan over time. Deliberately eager: the loop is a data dependency chain that
        torch.compile would unroll T times (2048 blocks for a 2k context) for zero gain."""
        outs = []
        for t in range(x.size(1)):
            h = self(x[:, t], h)
            outs.append(h)
        return torch.stack(outs, 1), h


class CellMixer(nn.Module):
    """Drop-in sequence mixer built on GatedCell; same [B,T,D] -> [B,T,D] contract as attention."""
    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        self.cell = GatedCell(dim, hidden or dim)
        self.out = nn.Linear(hidden or dim, dim, bias=False)

    def forward(self, x):
        y, _ = self.cell.scan(x)
        return self.out(y)
