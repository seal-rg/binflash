"""Binary Block Masked Flash Attention (BinFlash).

Scan-and-skip with bit-packed masks: the N×N bool mask is packed into
N × (N/64) int64 values (8x compression). Inside the kernel, each row's
64-bit value is unpacked to reconstruct the bool mask for that block.
This reduces mask bandwidth from 8KB to 1KB per block (29% of total loads).
"""

import torch
import triton
import triton.language as tl

from masks import make_block_mask


@triton.jit
def _binflash_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    packed_mask_ptr, stride_pm_m, stride_pm_n,
    block_mask_ptr, bm_stride_row, bm_stride_col, start_m,
    qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
):
    # Bit indices for unpacking: [0, 1, 2, ..., BLOCK_N-1]
    bit_indices = tl.arange(0, BLOCK_N).to(tl.int64)
    # Row offsets for packed mask loading
    pm_row_offs = (start_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)

    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        col_idx = start_n // BLOCK_N
        bm_val = tl.load(block_mask_ptr + start_m * bm_stride_row + col_idx * bm_stride_col)
        if bm_val != 0:
            k = tl.load(K_block_ptr)
            qk = tl.dot(q, k) * qk_scale

            # Load packed mask: BLOCK_M int64 values (one per Q-row)
            packed = tl.load(packed_mask_ptr + pm_row_offs * stride_pm_m + col_idx * stride_pm_n)  # (BLOCK_M,)
            # Unpack: shift right by bit index, check LSB
            mask = ((packed[:, None] >> bit_indices[None, :]) & 1) != 0  # (BLOCK_M, BLOCK_N)
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
    return acc, l_i, m_i


@triton.autotune(
    configs=[triton.Config({}, num_stages=s, num_warps=w)
             for s in [1, 2, 3, 4, 5] for w in [4, 8]],
    key=["N_CTX", "HEAD_DIM"],
)
@triton.jit
def _binflash_fwd(
    Q, K, V, sm_scale, Out, PackedMask, BlockMask, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_pm_m, stride_pm_n,
    stride_bm_row, stride_bm_col,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tl.static_assert(BLOCK_N == 64)  # bit-packing requires BLOCK_N=64 for int64
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh
    Q_bp = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    K_bp = tl.make_block_ptr(base=K + qvk_offset, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    V_bp = tl.make_block_ptr(base=V + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vk, stride_vn), offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    O_bp = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_bp)
    acc, l_i, m_i = _binflash_fwd_inner(acc, l_i, m_i, q, K_bp, V_bp, PackedMask, stride_pm_m, stride_pm_n, BlockMask, stride_bm_row, stride_bm_col, start_m, sm_scale * 1.44269504, BLOCK_M, BLOCK_N, HEAD_DIM, N_CTX)
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(LSE + (off_hz * N_CTX) + offs_m, m_i + tl.math.log2(l_i))
    tl.store(O_bp, acc.to(Out.type.element_ty))


def _pack_mask_bits(mask: torch.Tensor) -> torch.Tensor:
    """Pack bool mask (N, N) into int64 (N, N//64). Each int64 packs 64 consecutive KV bits."""
    N = mask.shape[0]
    assert N % 64 == 0
    # Reshape to (N, N//64, 64), convert to int, multiply by powers of 2, sum
    bits = mask.view(N, N // 64, 64).to(torch.int64)
    powers = (1 << torch.arange(64, device=mask.device, dtype=torch.int64))
    packed = (bits * powers).sum(dim=2)  # (N, N//64) int64
    return packed


def binflash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: torch.Tensor, sm_scale: float | None = None,
    block_m: int = 128, block_n: int = 64,
    block_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary-block-masked flash attention with bit-packed masks."""
    B, H, N, D = q.shape
    if block_n != 64:
        raise ValueError("Bit-packed mask requires block_n=64")
    if N % block_m != 0 or N % block_n != 0:
        raise ValueError(f"N ({N}) must be divisible by block_m ({block_m}) and block_n ({block_n})")
    if block_n > D:
        raise ValueError(f"block_n ({block_n}) must be <= HEAD_DIM ({D})")
    if D not in {16, 32, 64, 128}:
        raise ValueError(f"HEAD_DIM ({D}) must be one of {{16, 32, 64, 128}}")
    if sm_scale is None:
        sm_scale = D ** -0.5
    if block_mask is None:
        block_mask = make_block_mask(mask, block_m, block_n)

    # Pack the mask: (N, N) bool → (N, N//64) int64
    packed_mask = _pack_mask_bits(mask)

    o = torch.empty_like(q)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    bm_int = block_mask.to(torch.int8)
    grid = lambda META: (triton.cdiv(N, block_m), B * H)

    _binflash_fwd[grid](
        q, k, v, sm_scale, o, packed_mask, bm_int, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        packed_mask.stride(0), packed_mask.stride(1),
        bm_int.stride(0), bm_int.stride(1),
        B, H, N, HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
    )
    return o
