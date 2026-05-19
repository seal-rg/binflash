"""Binary Block Masked Flash Attention (BinFlash).

Core idea from "Efficiently Dispatching Flash Attention For Partially Filled
Attention Masks" (Sharma & Geiping, NeurIPS ENLSP 2024), with optimizations:

1. Precompute a coarse binary block mask → compact index list of non-zero blocks.
2. Separate "full" blocks (all mask entries True) from "partial" blocks — skip mask
   load entirely for full blocks.
3. Use pre-computed pointer grids with scalar offsets (Phi-3 style) for efficient
   memory access in the index-based inner loop.
"""

import torch
import triton
import triton.language as tl

from masks import make_block_mask


# ────────────────────────── Forward kernel ──────────────────────────


@triton.jit
def _binflash_fwd_inner(
    acc, l_i, m_i, q,
    k_grid, v_grid, mask_grid,
    kv_indices_ptr, full_kv_indices_ptr,
    num_partial, num_full,
    stride_kn, stride_vk, stride_mask_n,
    qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    # Phase 1: partial blocks (need mask load)
    for idx in range(num_partial):
        col_idx = tl.load(kv_indices_ptr + idx)
        kv_offset = col_idx * BLOCK_N

        # Load K as (BLOCK_N, HEAD_DIM), transpose for dot product
        k = tl.load(k_grid + kv_offset * stride_kn)  # (BLOCK_N, HEAD_DIM)
        qk = tl.dot(q, tl.trans(k)) * qk_scale  # (BLOCK_M, BLOCK_N)

        # Apply fine-grained mask for this partial block
        mask = tl.load(mask_grid + kv_offset * stride_mask_n) != 0
        qk += tl.where(mask, 0.0, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(v_grid + kv_offset * stride_vk)  # (BLOCK_N, HEAD_DIM)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

    # Phase 2: full blocks (skip mask load — all entries are True)
    for idx in range(num_full):
        col_idx = tl.load(full_kv_indices_ptr + idx)
        kv_offset = col_idx * BLOCK_N

        k = tl.load(k_grid + kv_offset * stride_kn)
        qk = tl.dot(q, tl.trans(k)) * qk_scale

        # No mask needed — all entries attend
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(v_grid + kv_offset * stride_vk)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

    return acc, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [2, 3, 4]
        for w in [4, 8]
    ],
    key=["N_CTX", "HEAD_DIM"],
)
@triton.jit
def _binflash_fwd(
    Q, K, V, sm_scale, Out, Mask,
    KV_partial_indices, KV_partial_counts,
    KV_full_indices, KV_full_counts,
    LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_mask_m, stride_mask_n,
    stride_pidx_row, stride_pidx_col,
    stride_fidx_row, stride_fidx_col,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # Load Q block using block pointer (loaded once, stays in SRAM)
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )

    # Pre-compute pointer grids for K, V, Mask
    # K: load as (BLOCK_N, HEAD_DIM) — coalesced in D dimension (stride_kk=1)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_bm = tl.arange(0, BLOCK_M)
    k_grid = K + qvk_offset + offs_bn[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_grid = V + qvk_offset + offs_bn[:, None] * stride_vk + offs_d[None, :] * stride_vn
    mask_grid = Mask + (start_m * BLOCK_M + offs_bm[:, None]) * stride_mask_m + offs_bn[None, :] * stride_mask_n

    # Counts for this Q-row
    num_partial = tl.load(KV_partial_counts + start_m)
    num_full = tl.load(KV_full_counts + start_m)
    partial_ptr = KV_partial_indices + start_m * stride_pidx_row
    full_ptr = KV_full_indices + start_m * stride_fidx_row

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)

    acc, l_i, m_i = _binflash_fwd_inner(
        acc, l_i, m_i, q,
        k_grid, v_grid, mask_grid,
        partial_ptr, full_ptr,
        num_partial, num_full,
        stride_kn, stride_vk, stride_mask_n,
        qk_scale, BLOCK_M, BLOCK_N, HEAD_DIM,
    )

    acc = acc / l_i[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + off_hz * N_CTX + offs_m
    tl.store(lse_ptrs, m_i + tl.math.log2(l_i))
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


# ────────────────────────── Preprocessing ──────────────────────────


def _build_dispatch_indices(mask: torch.Tensor, block_mask: torch.Tensor, block_m: int, block_n: int):
    """Build separate index lists for full blocks and partial blocks.

    Full blocks: all mask entries are True → skip mask load in kernel.
    Partial blocks: some True, some False → must load mask.

    Returns:
        partial_indices: (num_block_rows, max_cols) int32
        partial_counts: (num_block_rows,) int32
        full_indices: (num_block_rows, max_cols) int32
        full_counts: (num_block_rows,) int32
    """
    num_rows, num_cols = block_mask.shape

    # Compute block sums to identify full blocks
    block_sums = mask.view(num_rows, block_m, num_cols, block_n).sum(dim=(1, 3))
    is_full = block_sums == (block_m * block_n)
    is_partial = block_mask & ~is_full

    # Build sorted index arrays (non-zero indices packed to front)
    def _pack_indices(bm):
        counts = bm.sum(dim=1).to(torch.int32)
        col_indices = torch.arange(num_cols, device=bm.device).expand(num_rows, -1)
        order = bm.int().mul(-1).argsort(dim=1, stable=True)
        indices = col_indices.gather(1, order).to(torch.int32)
        return indices, counts

    partial_indices, partial_counts = _pack_indices(is_partial)
    full_indices, full_counts = _pack_indices(is_full)

    return partial_indices, partial_counts, full_indices, full_counts


# ────────────────────────── Python wrapper ──────────────────────────


def binflash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    sm_scale: float | None = None,
    block_m: int = 128,
    block_n: int = 64,
    block_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary-block-masked flash attention with index-based dispatch.

    Args:
        q, k, v: (B, H, N, D) in float16/bfloat16
        mask: (N, N) bool tensor
        sm_scale: softmax scale, defaults to 1/sqrt(D)
        block_m, block_n: tile sizes (must divide N, block_n <= D)
        block_mask: precomputed (N//block_m, N//block_n) bool tensor (optional)
    """
    B, H, N, D = q.shape
    assert N % block_m == 0 and N % block_n == 0
    assert block_n <= D, f"BLOCK_N ({block_n}) must be <= HEAD_DIM ({D})"
    assert D in {16, 32, 64, 128}
    if sm_scale is None:
        sm_scale = D ** -0.5

    if block_mask is None:
        block_mask = make_block_mask(mask, block_m, block_n)

    # Build dispatch structures: separate full and partial blocks
    partial_idx, partial_cnt, full_idx, full_cnt = _build_dispatch_indices(
        mask, block_mask, block_m, block_n
    )

    o = torch.empty_like(q)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    mask_int = mask.to(torch.int8)

    grid = lambda META: (triton.cdiv(N, block_m), B * H)
    _binflash_fwd[grid](
        q, k, v, sm_scale, o, mask_int,
        partial_idx, partial_cnt,
        full_idx, full_cnt,
        lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        mask_int.stride(0), mask_int.stride(1),
        partial_idx.stride(0), partial_idx.stride(1),
        full_idx.stride(0), full_idx.stride(1),
        B, H, N,
        HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
    )
    return o
