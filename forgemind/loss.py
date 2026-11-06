"""Cross entropy for vocabularies that do not fit in memory as a single [N, vocab] tensor.

The original implementation up-cast every chunk of logits to fp32 before the loss
(``(hc @ w.t()).float()``), which for vocab=32k and chunk=4096 materialises a 512 MB fp32 temp per
chunk, and it re-cast the *whole* lm_head weight to the activation dtype once per chunk inside the
checkpointed region. Both of those are gone:

  * the weight cast happens once per call (a single cast node, outside the recompute region);
  * the logits stay in the compute dtype -- ``log_softmax``/``nll_loss`` already accumulate in fp32
    for half/bf16 inputs on CUDA, so up-casting first bought nothing but bandwidth;
  * only the scalar reduction is fp32;
  * no checkpoint is installed when we are not recording gradients (eval / inference).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ["chunked_cross_entropy", "logits_cross_entropy"]


def chunked_cross_entropy(h, weight, labels, chunk: int = 4096, ignore_index: int = -100):
    """CE over ``h @ weight.t()`` without ever materialising the full [N, vocab] logits.

    Each chunk is recomputed in the backward pass, so peak memory is one chunk of logits regardless
    of sequence length. Returns a 0-dim fp32 tensor (mean over non-ignored tokens).
    """
    h = h.reshape(-1, h.size(-1))
    y = labels.reshape(-1)
    n = (y != ignore_index).sum().clamp(min=1).to(torch.float32)

    use_ckpt = h.requires_grad or weight.requires_grad
    w = weight.to(h.dtype) if weight.dtype != h.dtype else weight

    def piece(hc, yc):
        ce = F.cross_entropy(F.linear(hc, w), yc, ignore_index=ignore_index, reduction="sum")
        return ce.float()

    total = None
    n_tok = h.size(0)
    for i in range(0, n_tok, chunk):
        hc, yc = h[i:i + chunk], y[i:i + chunk]
        out = checkpoint(piece, hc, yc, use_reentrant=False) if use_ckpt else piece(hc, yc)
        total = out if total is None else total + out
    if total is None:                                    # empty batch
        total = h.new_zeros((), dtype=torch.float32)
        if use_ckpt:
            total.requires_grad_(True)
        return total
    return total / n


def logits_cross_entropy(logits, labels, ignore_index: int = -100, chunk: int = 4096):
    """CE when the logits already exist (the pipeline runner's last stage).

    Reduced in token chunks: `F.cross_entropy` on the whole [N, vocab] tensor materialises a second
    full-size fp32 log-softmax buffer, which for a 32k vocab is hundreds of MB of pure bandwidth.
    """
    N = logits.size(0) * logits.size(1) if logits.dim() == 3 else logits.size(0)
    lg = logits.reshape(N, -1)
    y = labels.reshape(N)
    n = (y != ignore_index).sum().clamp(min=1).to(torch.float32)
    if N <= chunk:
        return F.cross_entropy(lg, y, ignore_index=ignore_index, reduction="sum").float() / n
    total = None
    for i in range(0, N, chunk):
        ce = F.cross_entropy(lg[i:i + chunk], y[i:i + chunk], ignore_index=ignore_index, reduction="sum")
        total = ce.float() if total is None else total + ce.float()
    return total / n
