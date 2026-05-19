"""BinFlash: Flash attention with binary block-mask skipping.

Public API:

    from binflash import binflash_attention

    out = binflash_attention(q, k, v, mask, sm_scale=None)

``q``, ``k``, ``v`` are ``(B, H, N, D)`` tensors with ``D in {16, 32, 64, 128}``.
``mask`` is an ``(N, N)`` bool tensor where ``True`` means attend. The function
supports autograd; use inside ``torch.no_grad()`` for inference if you want to
skip ctx saves.

See ``binflash.masks`` for ready-made mask generators (causal, sliding window,
longformer, log-tree, block-diagonal, etc.).
"""

from .binflash_attention import binflash_attention

__all__ = ["binflash_attention"]
