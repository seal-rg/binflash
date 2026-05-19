"""Mask generation utilities for attention benchmarking."""

import random

import torch  # type: ignore


def causal_mask(seq_len: int, device: str = "cuda") -> torch.Tensor:
    """Lower-triangular causal mask. Shape: (seq_len, seq_len), dtype bool."""
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def sliding_window_mask(seq_len: int, window_size: int, device: str = "cuda") -> torch.Tensor:
    """Symmetric sliding window mask. Each token attends to +-window_size neighbors."""
    idx = torch.arange(seq_len, device=device)
    return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= window_size


def causal_sliding_window_mask(seq_len: int, window_size: int, device: str = "cuda") -> torch.Tensor:
    """Causal + sliding window: attend to previous window_size tokens only."""
    idx = torch.arange(seq_len, device=device)
    diff = idx.unsqueeze(0) - idx.unsqueeze(1)  # q - kv
    return (diff >= 0) & (diff <= window_size)


def block_diagonal_mask(seq_len: int, block_size: int, device: str = "cuda") -> torch.Tensor:
    """Block diagonal mask — independent segments of block_size tokens."""
    idx = torch.arange(seq_len, device=device)
    return (idx.unsqueeze(0) // block_size) == (idx.unsqueeze(1) // block_size)


def prefix_lm_mask(seq_len: int, prefix_len: int, device: str = "cuda") -> torch.Tensor:
    """Prefix LM: prefix tokens attend bidirectionally, suffix is causal."""
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    # Prefix attends to all prefix tokens (bidirectional)
    mask[:prefix_len, :prefix_len] = True
    # All tokens attend to prefix
    mask[:, :prefix_len] = True
    # Suffix is causal
    suffix_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    mask = mask | suffix_mask
    return mask


def random_sparse_mask(seq_len: int, density: float, device: str = "cuda") -> torch.Tensor:
    """Random sparse mask with given density, symmetrized, with diagonal always on."""
    mask = torch.rand(seq_len, seq_len, device=device) < density
    mask = mask | mask.T  # symmetrize
    mask.fill_diagonal_(True)
    return mask


def longformer_mask(seq_len: int, window_size: int, num_global: int, device: str = "cuda") -> torch.Tensor:
    """LongFormer-style: sliding window + first num_global tokens are global."""
    mask = sliding_window_mask(seq_len, window_size, device)
    # Global tokens attend to and are attended by all
    mask[:num_global, :] = True
    mask[:, :num_global] = True
    return mask


def log_tree_mask(seq_len: int, device: str = "cuda") -> torch.Tensor:
    """Hierarchical log-tree attention pattern.

    Partitions seq_len into log2(seq_len/min_stream) streams of geometrically
    decreasing size. Stream 0 is the base (largest), stream k has size N/2^(k+1).
    Each stream attends to itself (causal) and to all streams below it.

    Example for N=1024:
      Stream 0: tokens [0, 512)     — 512 tokens, attend only within stream 0 (causal)
      Stream 1: tokens [512, 768)   — 256 tokens, attend to stream 0 + stream 1 (causal)
      Stream 2: tokens [768, 896)   — 128 tokens, attend to streams 0-2
      Stream 3: tokens [896, 960)   — 64 tokens, attend to streams 0-3
      Stream 4: tokens [960, 1024)  — 64 tokens (min), attend to everything
    """
    min_stream = 64  # minimum stream size
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    # Build stream boundaries
    boundaries = [0]
    remaining = seq_len
    while remaining > min_stream * 2:
        half = remaining // 2
        boundaries.append(boundaries[-1] + half)
        remaining -= half
    boundaries.append(seq_len)

    num_streams = len(boundaries) - 1

    # Each stream k attends to its own stream (causal) + the stream directly below it
    for k in range(num_streams):
        q_start, q_end = boundaries[k], boundaries[k + 1]
        # Attend within own stream (causal)
        stream_size = q_end - q_start
        for i in range(stream_size):
            mask[q_start + i, q_start : q_start + i + 1] = True
        # Attend to the stream directly below (k-1), full attention
        if k > 0:
            kv_start, kv_end = boundaries[k - 1], boundaries[k]
            mask[q_start:q_end, kv_start:kv_end] = True

    return mask


def medusa_tree_mask(seq_len: int, num_heads: int = 4, candidates_per_head: int = 3, device: str = "cuda") -> torch.Tensor:
    """MEDUSA-style tree mask for speculative decoding verification.

    Creates a mask where: one "base" token attends to all prior context (causal),
    and `num_heads * candidates_per_head` speculative tokens each attend to the
    base token and their respective head's candidates.
    Padded to seq_len with identity blocks.
    """
    tree_size = 1 + num_heads * candidates_per_head  # base + candidates
    # Build tree mask for one speculation step
    tree = torch.zeros(tree_size, tree_size, dtype=torch.bool, device=device)
    tree[0, 0] = True  # base attends to itself
    for h in range(num_heads):
        for c in range(candidates_per_head):
            idx = 1 + h * candidates_per_head + c
            tree[idx, 0] = True  # all candidates attend to base
            tree[idx, idx] = True  # attend to self
            # Candidates within same head attend to each other (causal within head)
            for c2 in range(c):
                tree[idx, 1 + h * candidates_per_head + c2] = True

    # Tile the tree mask along the diagonal to fill seq_len
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for start in range(0, seq_len - tree_size + 1, tree_size):
        mask[start : start + tree_size, start : start + tree_size] = tree
        # Each tree block also attends to all prior blocks (causal at tree level)
        if start > 0:
            mask[start : start + tree_size, :start] = True
    # Fill remaining diagonal
    remaining = seq_len % tree_size
    if remaining > 0:
        start = seq_len - remaining
        mask[start:, start:] = True
        if start > 0:
            mask[start:, :start] = True
    return mask


def causal_document_mask(seq_len: int, doc_lens: list[int], device: str = "cuda") -> torch.Tensor:
    """Causal + document masking: causal within each document, no cross-document attention.

    Common in training with packed sequences where multiple documents are
    concatenated into one sequence.

    Args:
        seq_len: total sequence length
        doc_lens: list of document lengths that sum to seq_len
    """
    if doc_lens is None:
        doc_lens = _random_doc_lens(seq_len)
    assert sum(doc_lens) == seq_len
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    start = 0
    for doc_len in doc_lens:
        end = start + doc_len
        # Causal within document
        doc_mask = torch.ones(doc_len, doc_len, dtype=torch.bool, device=device).tril()
        mask[start:end, start:end] = doc_mask
        start = end
    return mask


def causal_doc_window_sinks(
    seq_len: int,
    doc_lens: list[int],
    window_size: int = 128,
    num_sinks: int = 4,
    device: str = "cuda",
) -> torch.Tensor:
    """Causal + document masking + sliding window + attention sinks.

    Each token attends to:
    - The first num_sinks tokens in its document (sinks)
    - The previous window_size tokens (sliding window)
    - Only within its own document (document boundary)
    All causal (no future tokens).
    """
    if doc_lens is None:
        doc_lens = _random_doc_lens(seq_len)
    assert sum(doc_lens) == seq_len
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    doc_start = 0
    for doc_len in doc_lens:
        doc_end = doc_start + doc_len
        for i in range(doc_len):
            qi = doc_start + i
            # Sinks: first num_sinks tokens in this document
            sink_end = min(doc_start + num_sinks, doc_end)
            mask[qi, doc_start:sink_end] = True
            # Sliding window: previous window_size tokens (causal)
            win_start = max(doc_start, qi - window_size)
            mask[qi, win_start : qi + 1] = True
        doc_start = doc_end
    return mask


def make_block_mask(mask: torch.Tensor, block_m: int = 128, block_n: int = 128) -> torch.Tensor:
    """Convert a full (seq_len, seq_len) bool mask to a binary block mask.

    Returns a (num_blocks_m, num_blocks_n) bool tensor where entry (i,j) is True
    if the corresponding block contains any True value.
    """
    seq_len_q, seq_len_k = mask.shape
    assert seq_len_q % block_m == 0 and seq_len_k % block_n == 0
    # Reshape into blocks and check if any element is nonzero
    return mask.view(seq_len_q // block_m, block_m, seq_len_k // block_n, block_n).any(dim=(1, 3))


def mask_sparsity(mask: torch.Tensor) -> float:
    """Return the fraction of mask elements that are zero (skippable)."""
    return 1.0 - (mask != 0).float().mean().item()


def _random_doc_lens(seq_len, num_docs=7, seed=42):
    """Pseudorandom document lengths summing to seq_len.

    Uses exponential distribution for realistic variation (some long, some short).
    Deterministic: same seed gives same proportions for any seq_len.
    """
    rng = random.Random(seed)
    raw = [rng.expovariate(1.0) for _ in range(num_docs)]
    total = sum(raw)
    lengths = [max(1, round(r / total * seq_len)) for r in raw]
    # Fix rounding error on the largest segment
    lengths[lengths.index(max(lengths))] += seq_len - sum(lengths)
    return lengths
