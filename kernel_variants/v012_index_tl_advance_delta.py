"""Binary Block Masked Flash Attention (BinFlash).

Index-based dispatch with tl.advance dynamic offsets:
- Precompute non-zero block column indices per Q-row
- Iterate only over non-zero blocks
- Use tl.advance with dynamic deltas to jump between blocks (keeps hardware-
  optimized block pointer loads, no per-iteration make_block_ptr overhead)
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
    kv_indices_ptr, num_blocks,
    qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    prev_col = tl.zeros([], dtype=tl.int32)

    for idx in range(num_blocks):
        col_idx = tl.load(kv_indices_ptr + idx)
        # Jump block pointers from previous position to current
        delta = (col_idx - prev_col) * BLOCK_N
        K_block_ptr = tl.advance(K_block_ptr, (0, delta))
        V_block_ptr = tl.advance(V_block_ptr, (delta, 0))
        Mask_block_ptr = tl.advance(Mask_block_ptr, (0, delta))

        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k) * qk_scale

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

        prev_col = col_idx

    return acc, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [1, 2, 3, 4]
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
    # K, V, Mask block pointers start at column 0 — we'll advance dynamically
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

    num_blocks = tl.load(KV_num_blocks + start_m)
    kv_indices_ptr = KV_indices + start_m * stride_idx_row

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)

    acc, l_i, m_i = _binflash_fwd_inner(
        acc, l_i, m_i, q,
        K_block_ptr, V_block_ptr, Mask_block_ptr,
        kv_indices_ptr, num_blocks,
        qk_scale, BLOCK_M, BLOCK_N, HEAD_DIM,
    )

    acc = acc / l_i[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + off_hz * N_CTX + offs_m
    tl.store(lse_ptrs, m_i + tl.math.log2(l_i))
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


# ────────────────────────── Preprocessing ──────────────────────────


def _build_kv_indices(block_mask: torch.Tensor):
    """Convert binary block mask to (kv_num_blocks, kv_indices) for index dispatch."""
    num_rows, num_cols = block_mask.shape
    kv_num_blocks = block_mask.sum(dim=1).to(torch.int32)
    col_indices = torch.arange(num_cols, device=block_mask.device).expand(num_rows, -1)
    sorted_order = block_mask.int().mul(-1).argsort(dim=1, stable=True)
    kv_indices = col_indices.gather(1, sorted_order).to(torch.int32)
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

    kv_num_blocks, kv_indices = _build_kv_indices(block_mask)

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
    )
    return o
