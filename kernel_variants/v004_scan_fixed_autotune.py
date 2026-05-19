"""Binary Block Masked Flash Attention (BinFlash).

Core idea from "Efficiently Dispatching Flash Attention For Partially Filled
Attention Masks" (Sharma & Geiping, NeurIPS ENLSP 2024):

1. Precompute a coarse binary block mask: for each (BLOCK_M x BLOCK_N) tile of the
   full attention mask, store 1 if any entry is nonzero, 0 otherwise.
2. During the flash-attention inner loop, check the block mask before loading K/V/mask
   data. Skip the entire tile when the block mask is 0.

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
    K_block_ptr, V_block_ptr, Mask_block_ptr,
    block_mask_ptr, bm_stride_row, bm_stride_col, start_m,
    qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
):
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        col_idx = start_n // BLOCK_N
        # Load the coarse block mask value for this (row-block, col-block)
        bm_val = tl.load(block_mask_ptr + start_m * bm_stride_row + col_idx * bm_stride_col)

        if bm_val != 0:
            k = tl.load(K_block_ptr)
            qk = tl.dot(q, k) * qk_scale

            # Apply fine-grained mask within this non-empty block
            mask = tl.load(Mask_block_ptr) != 0
            qk += tl.where(mask, 0.0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)

            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            v = tl.load(V_block_ptr)
            p = p.to(v.dtype)
            acc = tl.dot(p, v, acc)
            m_i = m_ij

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        Mask_block_ptr = tl.advance(Mask_block_ptr, (0, BLOCK_N))

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
    Q, K, V, sm_scale, Out, Mask, BlockMask, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_mask_m, stride_mask_n,
    stride_bm_row, stride_bm_col,
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

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset, shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn), offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn), offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    Mask_block_ptr = tl.make_block_ptr(
        base=Mask, shape=(N_CTX, N_CTX),
        strides=(stride_mask_m, stride_mask_n), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # sm_scale / ln(2)
    q = tl.load(Q_block_ptr)

    acc, l_i, m_i = _binflash_fwd_inner(
        acc, l_i, m_i, q,
        K_block_ptr, V_block_ptr, Mask_block_ptr,
        BlockMask, stride_bm_row, stride_bm_col, start_m,
        qk_scale, BLOCK_M, BLOCK_N, HEAD_DIM, N_CTX,
    )

    acc = acc / l_i[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + off_hz * N_CTX + offs_m
    tl.store(lse_ptrs, m_i + tl.math.log2(l_i))
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


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
    """Binary-block-masked flash attention.

    Args:
        q, k, v: (B, H, N, D) in float16/bfloat16
        mask: (N, N) bool tensor
        sm_scale: softmax scale, defaults to 1/sqrt(D)
        block_m, block_n: tile sizes (must divide N, block_n <= D)
        block_mask: precomputed (N//block_m, N//block_n) bool tensor (optional,
                    computed from mask if not provided)
    """
    B, H, N, D = q.shape
    assert N % block_m == 0 and N % block_n == 0
    assert block_n <= D, f"BLOCK_N ({block_n}) must be <= HEAD_DIM ({D})"
    assert D in {16, 32, 64, 128}
    if sm_scale is None:
        sm_scale = D ** -0.5

    if block_mask is None:
        block_mask = make_block_mask(mask, block_m, block_n)

    o = torch.empty_like(q)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)

    mask_int = mask.to(torch.int8)
    bm_int = block_mask.to(torch.int8)

    grid = lambda META: (triton.cdiv(N, block_m), B * H)
    _binflash_fwd[grid](
        q, k, v, sm_scale, o, mask_int, bm_int, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        mask_int.stride(0), mask_int.stride(1),
        bm_int.stride(0), bm_int.stride(1),
        B, H, N,
        HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
    )
    return o
