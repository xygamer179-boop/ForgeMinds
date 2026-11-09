"""Muon for ForgeMind: one optimizer for the whole model.

Matrices (``ndim >= 2``, conv weights included) are updated with **Muon** -- momentum in
the param's space, orthogonalized by quintic Newton-Schulz (``forgemind.kernels.ns``),
which turns the step into steepest descent under the spectral norm. Everything else
(biases, norms, embeddings' rows stay with Adam when you say so) is updated with a
decoupled **AdamW**, so a single ``fm.Muon(model.parameters())`` can train the model.

    model = torch.nn.Linear(64, 64)
    opt = fm.Muon(model.parameters(), lr=0.02, weight_decay=0.0)

    # explicit control (param-group dicts, like any torch optimizer):
    mats = [p for p in model.parameters() if p.ndim >= 2]
    scalars = [p for p in model.parameters() if p.ndim < 2]
    opt = fm.Muon([
        dict(params=mats, lr=0.05, use_muon=True),
        dict(params=scalars, lr=0.04, use_muon=False, betas=(0.8, 0.95)),
    ])

Design notes (and bugs this implementation fixes vs. the reference Muon file)
-----------------------------------------------------------------------------
1. **Nesterov does not touch ``p.grad``.** The reference does
   ``grad.lerp_(momentum, beta)`` -- an in-place mutation of the user's gradient. With
   ``zero_grad(set_to_none=False)`` (or any code that reads grads after ``step``), the
   nesterov blend silently corrupts accumulation: the next backward adds to a grad that
   already contains momentum. Here the blend is ``g.lerp(buf, beta)`` (out-of-place).
2. **``nesterov=False`` does not scale the momentum buffer.** The reference returns the
   *buffer itself* as ``update`` and then does ``update *= sqrt(max(1, rows/cols))`` --
   re-seating an aspect-ratio factor into the state on **every** step. On a tall matrix
   (``rows > cols``) that compounds as ``(m/n)**(t/2)`` and the buffer explodes; on wide
   matrices it silently shrinks. Here the scale is applied only to the NS *output*, never
   to state.
3. **No ``dist`` calls at world 1.** The reference ``Muon`` calls ``dist.get_world_size()``
   unconditionally (``AssertionError`` in a plain single-process run) and its pad count
   ``world - len(params) % world`` adds a whole extra pad block when ``len % world == 0``.
   Here: ``dist`` is consulted only when initialized with world > 1, there is no padding
   (each bucket's shares are disjoint and summed with one ``all_reduce``), and world 1
   skips collectives entirely.
4. **Any parameter iterable works.** The reference asserts ``isinstance(params[0],
   Parameter)`` (generators and param-group dicts die) and runs the NS ``ndim >= 2``
   assert on the first bias of an unsplit model. Here the constructor auto-splits
   ``model.parameters()`` into Muon/AdamW groups, accepts generators, honors an explicit
   ``use_muon`` flag per group, and rejects a ``use_muon=True`` group containing a vector
   with a message that says what to do instead.
5. **No strict key-set asserts.** The reference asserts
   ``set(group.keys()) == {...}``, so any extra key (``ns_steps``, ``betas`` on a muon
   group, a scheduler's leftover) explodes at construction. Here unknown/extra group keys
   pass through untouched.
6. **``requires_grad=False`` params are skipped** (no state, no update), and a ``grad is
   None`` param contributes a *local* zero tensor -- the reference writes
   ``p.grad = zeros`` into the model (visible to the user and to other code after step).
7. **Same-shape matrices are orthogonalized as one batch.** ``torch.stack`` + a batched
   NS call replaces per-param launches; multi-GPU splits each bucket across ranks
   (``rank::world``) and combines with a single ``all_reduce`` over disjoint slices.

Per-group ``lr`` / ``weight_decay`` / ``momentum`` / ``nesterov`` / ``ns_steps`` /
``ns_dtype`` (Muon) and ``betas`` / ``eps`` (AdamW) are honored. Weight decay is decoupled
on both paths (``p *= 1 - lr*wd`` before the step), matching AdamW/Muon practice.

See ``tests/test_muon.py`` for the regression tests named after bugs 1-6, and
``PERF.md`` (round six) for the full critique of the reference file.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor

from .kernels.ns import newton_schulz


def _adam_update(g: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor, step: int,
                 betas: tuple[float, float], eps: float) -> Tensor:
    """AdamW update with bias correction (decoupled decay is applied to the param)."""
    exp_avg.lerp_(g, 1 - betas[0])
    exp_avg_sq.lerp_(g.square(), 1 - betas[1])
    m_hat = exp_avg / (1 - betas[0] ** step)
    v_hat = exp_avg_sq / (1 - betas[1] ** step)
    return m_hat / (v_hat.sqrt() + eps)


def _split_group(gd: dict) -> list[dict]:
    """One declared group -> concrete groups (auto-split when ``use_muon`` is absent)."""
    params = gd.pop("params")
    if isinstance(params, Tensor):
        params = [params]
    else:
        params = list(params)
    if not params:
        return []
    use_muon = gd.pop("use_muon", None)
    if use_muon is None:
        out = []
        mats = [p for p in params if p.ndim >= 2]
        vecs = [p for p in params if p.ndim < 2]
        if mats:
            out.append(dict(gd, params=mats, use_muon=True))
        if vecs:
            out.append(dict(gd, params=vecs, use_muon=False))
        return out
    if use_muon:
        bad = [tuple(p.shape) for p in params if p.ndim < 2]
        if bad:
            raise ValueError(
                f"use_muon=True group contains {len(bad)} vector parameter(s) e.g. shape "
                f"{bad[0]}: Muon needs ndim >= 2. Drop the use_muon flag to auto-split, "
                f"or move vector params into a use_muon=False group.")
        return [dict(gd, params=params, use_muon=True)]
    return [dict(gd, params=params, use_muon=False)]


def _as_groups(params, lr: float) -> list[dict]:
    """Any supported ``params`` spec -> concrete param-group dicts (generators included)."""
    if isinstance(params, Tensor):
        items = [params]
    else:
        items = list(params)              # generators/iterables: materialize, no [0] peeking
    if not items:
        raise ValueError("Muon received an empty parameter iterable")
    if isinstance(items[0], dict):
        out: list[dict] = []
        for gd in items:
            out.extend(_split_group(dict(gd)))
        if not out:
            raise ValueError("Muon received param groups but no parameters")
        return out
    if not isinstance(items[0], Tensor):
        raise TypeError(f"Muon params must be Tensors or param-group dicts, got "
                        f"{type(items[0]).__name__}")
    mats = [p for p in items if p.ndim >= 2]
    vecs = [p for p in items if p.ndim < 2]
    out = []
    if mats:
        out.append(dict(params=mats, use_muon=True))
    if vecs:
        out.append(dict(params=vecs, use_muon=False, lr=lr))
    return out


class Muon(torch.optim.Optimizer):
    """Muon for matrices + AdamW for vectors, in one optimizer.

    Parameters
    ----------
    params : iterable of Tensors/Parameters (auto-split by ``ndim >= 2``) or iterable of
        param-group dicts. A group with ``use_muon=True`` is Muon, ``False`` is AdamW,
        and a group without the flag is split by ``ndim`` keeping its hyperparameters.
    lr : learning rate (default for both paths; the AdamW group gets ``adam_lr`` when set)
    weight_decay : decoupled decay, applied as ``p *= 1 - lr * wd`` on both paths
    momentum : Muon momentum buffer factor (``buf.lerp_(grad, 1 - momentum)``)
    nesterov : ``True`` blends the grad with the updated buffer before NS (out-of-place)
    ns_steps : Newton-Schulz iterations (5 is the canonical value)
    ns_dtype : compute dtype inside NS (default bf16, the canonical fast choice; pass
        ``torch.float32`` for maximum accuracy -- the kernels accept either)
    betas, eps : AdamW parameters for the vector groups
    adam_lr : learning rate for the auto-created AdamW groups (default: ``lr``)
    """

    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                 ns_dtype: torch.dtype | None = torch.bfloat16,
                 betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-10,
                 adam_lr: float | None = None):
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")
        groups = _as_groups(params, lr if adam_lr is None else adam_lr)
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps, ns_dtype=ns_dtype,
                        betas=betas, eps=eps)
        super().__init__(groups, defaults)

    def add_param_group(self, param_group: dict) -> None:
        """Add a group with the same auto-split/validation as the constructor."""
        for g in _split_group(dict(param_group)):
            super().add_param_group(g)

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if distributed else 0
        world = dist.get_world_size() if distributed else 1
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_group(group, rank, world, distributed)
            else:
                self._adam_group(group)
        return loss

    def _muon_group(self, group: dict, rank: int, world: int, distributed: bool) -> None:
        lr = group["lr"]
        wd = group["weight_decay"]
        beta = group["momentum"]
        nesterov = group["nesterov"]
        params = [p for p in group["params"] if p.requires_grad]
        if not params:
            return

        buckets: dict[tuple, list[Tensor]] = {}       # flattened 2D shape -> params
        for p in params:
            if p.ndim < 2:                            # construction validates; keep the guard
                raise RuntimeError(
                    f"Muon group received a {p.ndim}-D parameter (shape {tuple(p.shape)}); "
                    f"vector params must be in a use_muon=False group.")
            key = (p.shape[0], p[0].numel()) if p.ndim > 2 else tuple(p.shape)
            buckets.setdefault(key, []).append(p)

        for key, plist in buckets.items():
            mats = []
            for p in plist:
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(p)
                g = p.grad
                if g is None:
                    g = torch.zeros_like(p)           # local zeros: p.grad stays None
                buf.lerp_(g, 1 - beta)                # buf = beta*buf + (1-beta)*g
                # out-of-place nesterov (or the raw buffer); NEVER mutate g or buf here
                eff = g.lerp(buf, beta) if nesterov else buf
                mats.append(eff.reshape(key).to(torch.float32))   # fp32 stack: mixed-dtype safe
            G = torch.stack(mats)                     # (N, r, c)
            ns_dtype = group["ns_dtype"]

            if distributed:
                # disjoint shares -> one all_reduce sums them into the full result;
                # no padding, and every rank's bucket list is identical (same model).
                out = torch.zeros(G.shape, dtype=G.dtype if ns_dtype is None else ns_dtype,
                                  device=G.device)
                own = list(range(rank, G.shape[0], world))
                if own:
                    out[own] = newton_schulz(G[own], steps=group["ns_steps"], dtype=ns_dtype)
                dist.all_reduce(out)
            else:
                out = newton_schulz(G, steps=group["ns_steps"], dtype=ns_dtype)

            scale = max(1.0, out.size(-2) / out.size(-1)) ** 0.5
            for i, p in enumerate(plist):
                upd = out[i].reshape(p.shape).to(p.dtype)
                p.mul_(1 - lr * wd)                   # decoupled weight decay
                p.add_(upd, alpha=-lr * scale)

    def _adam_group(self, group: dict) -> None:
        lr = group["lr"]
        wd = group["weight_decay"]
        for p in group["params"]:
            if not p.requires_grad:
                continue
            g = p.grad
            if g is None:
                g = torch.zeros_like(p)               # local zeros: p.grad stays None
            state = self.state[p]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] = 0
            state["step"] += 1
            update = _adam_update(g, state["exp_avg"], state["exp_avg_sq"],
                                  state["step"], group["betas"], group["eps"])
            p.mul_(1 - lr * wd)                       # decoupled weight decay
            p.add_(update, alpha=-lr)


__all__ = ["Muon"]
