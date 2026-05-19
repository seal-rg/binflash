"""Reference math-mode attention for correctness verification."""

import torch  # type: ignore


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Standard scaled dot-product attention in pure PyTorch math.

    Args:
        q: (B, H, N, D)
        k: (B, H, N, D)
        v: (B, H, N, D)
        mask: (N, N) bool tensor — True means "attend", False means "mask out"
        sm_scale: softmax scale factor, defaults to 1/sqrt(D)

    Returns:
        output: (B, H, N, D), same dtype as q
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    B, H, N, D = q.shape

    # Whole batch fits in the budget.
    if B * H * N * N * q.element_size() <= 2e9:
        scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        return torch.matmul(attn, v)

    # Chunked fallback — process one (b, h) pair at a time. No casting.
    out = torch.empty_like(q)
    for b in range(B):
        for h in range(H):
            scores = torch.matmul(q[b, h], k[b, h].transpose(-2, -1)) * sm_scale
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1).nan_to_num(0.0)
            out[b, h] = torch.matmul(attn, v[b, h])
    return out
