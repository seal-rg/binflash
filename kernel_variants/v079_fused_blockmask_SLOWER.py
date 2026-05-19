"""Binary Block Masked Flash Attention (BinFlash) — v079: FUSED block mask.

RESULT: CATASTROPHICALLY SLOWER for sparse patterns (0.37-0.40x).
Loading the full mask tile (8KB) for every block to check emptiness is far
more expensive than the precomputed 1-byte block mask scalar check.
Sparse patterns: sliding 0.40x, blkdiag 0.37x, longformer 0.52x.
Dense patterns: causal 0.87x (also slower due to mask tile load overhead).

REVERTED — this approach is a dead end.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _binflash_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr, Mask_block_ptr,
    qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
):
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_raw = tl.load(Mask_block_ptr)
        any_true = tl.max(tl.max(mask_raw.to(tl.int8), 1), 0) > 0
        if any_true:
            all_true = tl.min(tl.min(mask_raw.to(tl.int8), 1), 0) > 0
            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)
            qk = tl.dot(q, k) * qk_scale
            if not all_true:
                mask = mask_raw != 0
                qk += tl.where(mask, 0.0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            p = p.to(v.dtype)
            acc = tl.dot(p, v, acc)
            m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        Mask_block_ptr = tl.advance(Mask_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i
