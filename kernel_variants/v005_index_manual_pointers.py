"""Binary Block Masked Flash Attention (BinFlash).

Core idea from "Efficiently Dispatching Flash Attention For Partially Filled
Attention Masks" (Sharma & Geiping, NeurIPS ENLSP 2024):

1. Precompute a coarse binary block mask: for each (BLOCK_M x BLOCK_N) tile of the
   full attention mask, store 1 if any entry is nonzero, 0 otherwise.
2. During the flash-attention inner loop, only iterate over non-zero blocks using
   a precomputed index list (not scanning all positions).

This avoids HBM reads for K, V, and the fine-grained mask for all-zero blocks,
giving large speedups on sparse masks.
"""

import torch
import triton
import triton.language as tl

from masks import make_block_mask


# ────────────────────────── Forward kernel ──────────────────────────


@triton.jit
def _binflash_fwd_inner(
    acc, l_i, m_i, q,
    K_base, V_base, Mask_base,
    kv_indices_ptr, num_blocks,
    stride_kn, stride_kk,
    stride_vk, stride_vn,
    stride_mask_m, stride_mask_n,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    FULL_BLOCKS: tl.constexpr,
):
    offs_k = tl.arange(0, HEAD_DIM)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_bm = tl.arange(0, BLOCK_M)

    for idx in range(num_blocks):
        # Load the column index for this non-zero block
        col_idx = tl.load(kv_indices_ptr + idx)

        # Compute K, V offsets from column index
        kv_offset = col_idx * BLOCK_N

        # Load K block: shape (HEAD_DIM, BLOCK_N)
        k_ptrs = K_base + offs_k[:, None] * stride_kk + (kv_offset + offs_bn[None, :]) * stride_kn
        k = tl.load(k_ptrs)
        qk = tl.dot(q, k) * qk_scale

        if FULL_BLOCKS == 0:
            # Load and apply fine-grained mask
            mask_ptrs = Mask_base + (start_m * BLOCK_M + offs_bm[:, None]) * stride_mask_m + (kv_offset + offs_bn[None, :]) * stride_mask_n
            mask = tl.load(mask_ptrs) != 0
            qk += tl.where(mask, 0.0, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # Load V block: shape (BLOCK_N, HEAD_DIM)
        v_ptrs = V_base + (kv_offset + offs_bn[:, None]) * stride_vk + offs_k[None, :] * stride_vn
        v = tl.load(v_ptrs)
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
    Q, K, V, sm_scale, Out, Mask, KV_indices, KV_num_blocks, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_mask_m, stride_mask_n,
    stride_idx_row, stride_idx_col,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FULL_BLOCKS: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # Load Q block
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )

    # Output block pointer
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )

    # How many non-zero KV blocks for this Q-row
    num_blocks = tl.load(KV_num_blocks + start_m)
    # Pointer to the index list for this Q-row
    kv_indices_ptr = KV_indices + start_m * stride_idx_row

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # sm_scale / ln(2)
    q = tl.load(Q_block_ptr)

    # Base pointers for K, V, Mask (not block ptrs — we index dynamically)
    K_base = K + qvk_offset
    V_base = V + qvk_offset
    Mask_base = Mask

    acc, l_i, m_i = _binflash_fwd_inner(
        acc, l_i, m_i, q,
        K_base, V_base, Mask_base,
        kv_indices_ptr, num_blocks,
        stride_kn, stride_kk,
        stride_vk, stride_vn,
        stride_mask_m, stride_mask_n,
        start_m,
        qk_scale, BLOCK_M, BLOCK_N, HEAD_DIM,
        FULL_BLOCKS,
    )

    acc = acc / l_i[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + off_hz * N_CTX + offs_m
    tl.store(lse_ptrs, m_i + tl.math.log2(l_i))
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


# ────────────────────────── Preprocessing ──────────────────────────


def _build_kv_indices(block_mask: torch.Tensor):
    """Convert a binary block mask to (kv_num_blocks, kv_indices) for index-based dispatch.

    Args:
        block_mask: (num_rows, num_cols) bool tensor

    Returns:
        kv_num_blocks: (num_rows,) int32 — number of non-zero blocks per row
        kv_indices: (num_rows, num_cols) int32 — column indices of non-zero blocks,
                    padded with 0s (ignored since num_blocks bounds the loop)
    """
    num_rows, num_cols = block_mask.shape
    kv_num_blocks = block_mask.sum(dim=1).to(torch.int32)
    # Build sorted indices per row
    kv_indices = torch.zeros(num_rows, num_cols, dtype=torch.int32, device=block_mask.device)
    for i in range(num_rows):
        nz = block_mask[i].nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
        kv_indices[i, : nz.shape[0]] = nz
    return kv_num_blocks, kv_indices


def _build_kv_indices_fast(block_mask: torch.Tensor):
    """Vectorized version of _build_kv_indices — no Python loops."""
    num_rows, num_cols = block_mask.shape
    kv_num_blocks = block_mask.sum(dim=1).to(torch.int32)

    # Sort each row so True values come first, and their original column indices are preserved
    # argsort with stable=True on the negated mask puts True (1) positions first
    col_indices = torch.arange(num_cols, device=block_mask.device).expand(num_rows, -1)
    # Use the mask to scatter: for each row, pack non-zero indices to the left
    sorted_indices = block_mask.int().mul(-1).argsort(dim=1, stable=True)
    kv_indices = col_indices.gather(1, sorted_indices).to(torch.int32)

    return kv_num_blocks, kv_indices


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

    # Check if all non-zero blocks are fully ones (no partial masking needed)
    num_blocks_m, num_blocks_n = N // block_m, N // block_n
    # A block is "full" if every element is True. Check by comparing block sums.
    block_sums = mask.view(num_blocks_m, block_m, num_blocks_n, block_n).sum(dim=(1, 3))
    full_block_count = (block_sums == block_m * block_n).sum().item()
    nonzero_count = block_mask.sum().item()
    all_full = full_block_count == nonzero_count

    # Build index-based dispatch structures
    kv_num_blocks, kv_indices = _build_kv_indices_fast(block_mask)

    o = torch.empty_like(q)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    mask_int = mask.to(torch.int8)

    grid = lambda META: (triton.cdiv(N, block_m), B * H)
    _binflash_fwd[grid](
        q, k, v, sm_scale, o, mask_int, kv_indices, kv_num_blocks, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        mask_int.stride(0), mask_int.stride(1),
        kv_indices.stride(0), kv_indices.stride(1),
        B, H, N,
        HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
        FULL_BLOCKS=1 if all_full else 0,
    )
    return o
