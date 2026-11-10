"""``fm.kernels.topk``: exact, deterministic top-k selection for the last dimension.

Structure follows FlashTopK (Tencent Hunyuan, Apache-2.0 -- see the attribution below),
translated from CuTe DSL + PTX to portable Triton:

* **strided-sample threshold** (their *prologue*): row ``pid``'s k-th largest is estimated
  from ``m = min(n, max(2k, 256))`` evenly strided samples. For any subset with >= k
  elements, its k-th largest is <= the full row's, so the estimate is never *too high* --
  that one-sided invariant is what makes the rest exact;
* **exact pivot** (their *selection*): if more than k elements sit above the estimate, the
  true k-th lives inside ``{x > t}`` and is found there by repeated masked argmax (their
  four radix passes over the pivot -- ours k maxes, same job: locate the pivot);
* **canonical order** (their counting sort): output is sorted by (value descending, index
  ascending) -- ties at the k-th score keep the smallest indices, so the same input gives
  the same bits every run, which is the property FlashTopK advertises and ``torch.topk``
  does not promise.

Two launches per call (threshold, select), both one program per row. NaN is treated as the
largest value (as ``torch.topk`` does) and the *original* NaN is what lands in the output.

Autograd: values participate in the graph (the indices do not) -- backward is a scatter,
the same as ``torch.topk``'s, so a router can backprop through its gating weights.

Dispatch (like ``newton_schulz``): eager + CUDA (or ``TRITON_INTERPRET=1``) takes the fused
path; while ``torch.compile`` traces, plain ``torch.topk`` is used (dynamo sees one aten
node, ``fullgraph=True`` holds); ``k > 512``, unsupported dtypes, ``numel >= 2**31``,
``dim != -1`` or a fallback backend all go to torch and are counted in
``fm.kernels.stats()``.

Attribution: the algorithm (sample threshold -> exact selection with index tie-breaks)
follows FlashTopK (https://github.com/Tencent-Hunyuan/FlashTopK, Apache-2.0), whose
selection stage in turn follows LiteTopK. This file is an independent Triton
implementation of those ideas -- no FlashTopK code is copied.
"""

from __future__ import annotations

import torch

from . import _common
from ._common import fast_ok, num_warps_for, tl, triton, triton_available

#: fused path is used for k up to this; larger k falls back to torch (counted in stats)
MAX_K = 512

_K1_BLOCK = 512     # sample capacity: m = min(n, max(2k, 256)) <= 512
_BLOCK = 1024       # row chunk for the vectorised passes
_E_BLOCK = 512      # slot block for the final k-way canonicalisation (>= MAX_K)
_HUGE = 1 << 62     # int64 sentinel for "not selected" (never a real index)


if triton_available():

    @triton.jit
    def _topk_threshold_kernel(T, X, N, K, M, STRIDE, BLOCK: tl.constexpr):
        """``T[pid]`` = k-th largest of row ``pid``'s strided sample (never above the true k-th)."""
        pid = tl.program_id(0).to(tl.int32)
        row = pid * N
        lanes = tl.arange(0, BLOCK)
        sidx = lanes * STRIDE
        valid = (lanes < M) & (sidx < N)
        sv = tl.load(X + row + sidx, mask=valid, other=float("-inf"))
        sv = tl.where(sv == sv, sv, float("inf"))        # NaN -> +inf for comparisons
        alive = valid
        last = float("-inf")
        it = 0
        while it < K:                                    # dynamic bound: no unroll blow-up
            masked = tl.where(alive, sv, float("-inf"))
            pos = tl.argmax(masked, 0)                      # any max lane is fine for a value
            val = tl.sum(tl.where(lanes == pos, sv, 0), axis=0)  # the value AT that lane (ties safe)
            alive = alive & (lanes != pos)               # kill exactly one sample element
            last = val
            it += 1
        tl.store(T + pid, last)

    @triton.jit
    def _topk_select_kernel(V, I, X, T, N, K, BLOCK: tl.constexpr):
        """Exact top-k of row ``pid`` given threshold ``T[pid]``; canonical (value desc, idx asc)."""
        pid = tl.program_id(0).to(tl.int32)
        row = pid * N
        out_v = V + pid * K
        out_i = I + pid * K
        lanes = tl.arange(0, BLOCK)
        t = tl.load(T + pid)

        # ---- pass 1: how many sit strictly above the estimate -----------------------
        c = 0
        start = 0
        while start < N:
            off = start + lanes
            m = off < N
            xr = tl.load(X + row + off, mask=m, other=0.0)
            xv = tl.where(xr == xr, xr, float("inf"))     # NaN -> +inf
            c += tl.sum((((xv > t) & m)).to(tl.float32), axis=0)
            start += BLOCK

        pivot = t
        q = c                                             # strictly-above count
        if c >= K:
            # Estimate too low: the true k-th lives in {x > t} -- k masked compound maxes
            # find it. Kills happen in descending (value, index) order, so "alive" is simply
            # *strictly worse than the last kill* -- one scalar pair (m_k, p_k) carries the
            # whole kill history (a single killed-position scalar would resurrect the rest).
            m_k = float("inf")                            # last kill: alive = worse than (m_k, p_k)
            p_k = -1
            last = float("-inf")
            it = 0
            while it < K:
                bm = float("-inf")
                bp = -1
                start = 0
                while start < N:
                    off = start + lanes
                    m = off < N
                    xr = tl.load(X + row + off, mask=m, other=0.0)
                    xv = tl.where(xr == xr, xr, float("inf"))
                    worse = (xv < m_k) | ((xv == m_k) & (off > p_k))
                    sub = m & (xv > t) & worse
                    cm = tl.max(tl.where(sub, xv, float("-inf")), axis=0)
                    cand = start + tl.argmin(tl.where(sub & (xv == cm), off, N + 1), 0)
                    if cm > bm:
                        bm = cm
                        bp = cand
                    elif (cm == bm) and (cand < bp):
                        bp = cand                         # cross-chunk tie: keep the smaller index
                    start += BLOCK
                last = bm
                m_k = bm
                p_k = bp
                it += 1
            pivot = last
            # recount against the exact pivot (q < K always holds: pivot IS the row's k-th)
            q = 0
            start = 0
            while start < N:
                off = start + lanes
                m = off < N
                xr = tl.load(X + row + off, mask=m, other=0.0)
                xv = tl.where(xr == xr, xr, float("inf"))
                q += tl.sum((((xv > pivot) & m)).to(tl.float32), axis=0)
                start += BLOCK

        # ---- gather the strictly-above set into out[0:q] (row order; canonicalised later)
        qi = q.to(tl.int32)                              # counts are fp32; offsets must be int
        written = 0.0
        start = 0
        while start < N:
            off = start + lanes
            m = off < N
            xr = tl.load(X + row + off, mask=m, other=0.0)
            xv = tl.where(xr == xr, xr, float("inf"))
            gt = (xv > pivot) & m
            gtf = gt.to(tl.float32)
            pos = (written + tl.cumsum(gtf, axis=0) - 1).to(tl.int32)
            keep = gt & (pos < K)
            tl.store(out_v + pos, xr, mask=keep)          # original values (NaN preserved)
            tl.store(out_i + pos, off.to(tl.int64), mask=keep)
            written += tl.sum(gtf, axis=0)
            start += BLOCK

        # ---- tie band: the (K - q) smallest indices among {x == pivot} ---------------
        qp = K - q
        prev = -1
        i = 0
        while i < qp:
            best = N + 1                                  # sentinel larger than any index
            start = 0
            while start < N:
                off = start + lanes
                m = off < N
                xr = tl.load(X + row + off, mask=m, other=0.0)
                xv = tl.where(xr == xr, xr, float("inf"))
                eq = (xv == pivot) & m & (off > prev)
                best = tl.minimum(best, tl.min(tl.where(eq, off, N + 1), axis=0))
                start += BLOCK
            # fetch the ORIGINAL value at `best` (NaN rows: the pivot is mapped, the value isn't)
            orig = 0.0
            start = 0
            while start < N:
                off = start + lanes
                m = off < N
                xr = tl.load(X + row + off, mask=m, other=0.0)
                orig += tl.sum(tl.where((off == best) & m, xr, 0), axis=0)
                start += BLOCK
            tl.store(out_v + qi + i, orig, mask=best <= N)
            tl.store(out_i + qi + i, best.to(tl.int64), mask=best <= N)
            prev = best
            i += 1

        # ---- canonicalise out[0:K]: k selection-sort passes, compound (value, index) --
        i = 0
        while i < K:
            s = tl.arange(0, _E_BLOCK)
            valid = (s >= i) & (s < K)
            vv = tl.load(out_v + s, mask=valid, other=float("-inf"))
            ii = tl.load(out_i + s, mask=valid, other=0)
            vm = tl.where(vv == vv, vv, float("inf"))      # NaN -> +inf for the compare
            mv = tl.max(tl.where(valid, vm, float("-inf")), axis=0)
            sel = valid & (vm == mv)
            pos = tl.argmin(tl.where(sel, ii, _HUGE), 0)      # smallest index among the maxes
            # swap slots i and pos (original values: NaN stays NaN in the output)
            vi = tl.sum(tl.where(s == i, vv, 0), axis=0)
            ii_ = tl.sum(tl.where(s == i, ii, 0), axis=0).to(tl.int64)
            vp = tl.sum(tl.where(s == pos, vv, 0), axis=0)
            ip = tl.sum(tl.where(s == pos, ii, 0), axis=0).to(tl.int64)
            tl.store(out_v + i, vp)
            tl.store(out_i + i, ip)
            tl.store(out_v + pos, vi)
            tl.store(out_i + pos, ii_)
            i += 1


def _ok(x: torch.Tensor, k: int) -> bool:
    if x.shape[-1] > 2 ** 24:                            # float32 counts stay exact below this
        _common._note("topk n>2**24")
        return False
    if x.numel() >= 2 ** 31:
        _common._note("topk numel>=2**31")
        return False
    if k > MAX_K:
        _common._note(f"topk k>{MAX_K}")
        return False
    return fast_ok(x)


def _fused(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(values, indices) for ``x`` of shape ``(..., n)`` through the two Triton kernels."""
    n = x.shape[-1]
    rows = x.reshape(-1, n)
    r = rows.shape[0]
    m = min(n, max(2 * k, 256))
    stride = max(1, n // m)
    t = torch.empty(r, dtype=rows.dtype, device=rows.device)
    _topk_threshold_kernel[(r,)](t, rows, n, k, m, stride, _K1_BLOCK,
                                 num_warps=num_warps_for(_K1_BLOCK))
    values = torch.empty(r, k, dtype=rows.dtype, device=rows.device)
    indices = torch.empty(r, k, dtype=torch.int64, device=rows.device)
    _topk_select_kernel[(r,)](values, indices, rows, t, n, k, _BLOCK,
                              num_warps=num_warps_for(_BLOCK))
    shape = x.shape[:-1] + (k,)
    return values.reshape(shape), indices.reshape(shape)


class _TopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        with torch.no_grad():
            values, indices = _fused(x, k)
        ctx.shape = x.shape
        ctx.save_for_backward(indices)
        return values, indices

    @staticmethod
    def backward(ctx, grad_values, grad_indices):
        (indices,) = ctx.saved_tensors
        if grad_values is None:
            return None, None
        grad = grad_values.new_zeros(ctx.shape)
        grad.scatter_(-1, indices, grad_values)
        return grad, None


def topk(x: torch.Tensor, k: int, dim: int = -1, sorted: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``k`` values and indices along the last dimension (``torch.topk`` drop-in for ``dim=-1``).

    * values come back in descending order; ties at the k-th value keep the **smallest
      indices** (deterministic -- ``torch.topk`` promises no tie order);
    * NaN is treated as the largest value and is passed through as-is;
    * gradients flow to the values only (scatter backward, same as torch);
    * ``sorted=False`` still returns sorted order (allowed by torch's contract: the result
      then merely *happens* to be sorted).

    Falls back to ``torch.topk`` for ``k > 512``, non-float dtypes, ``numel >= 2**31``,
    ``dim`` other than the last, or while ``torch.compile`` traces (one aten node in the
    graph); the fallback reasons show up in ``fm.kernels.stats()``.
    """
    if x.dim() < 1:
        raise ValueError(f"topk needs ndim >= 1, got shape {tuple(x.shape)}")
    if dim not in (-1, x.dim() - 1):
        # the fused kernels address the last dim only; everything else is torch's problem
        return torch.topk(x, k, dim=dim, sorted=True)
    n = x.shape[-1]
    if k < 1 or k > n:
        raise RuntimeError(f"selected index k out of range (got {k} vs {n})")
    if _common.tracing() or not _ok(x, k):
        return torch.topk(x, k, dim=-1, sorted=True)
    return _TopK.apply(x, k)


__all__ = ["topk", "MAX_K"]
