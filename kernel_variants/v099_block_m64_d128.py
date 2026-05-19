"""Binary Block Masked Flash Attention — v099: BLOCK_M=64 at D=128.

At D=128, BLOCK_M=128 forces warps=8, stages=1 (dead end #17). But
BLOCK_M=64 halves register pressure (acc: 32KB vs 64KB) and shared memory
per stage, potentially enabling stages=2 pipelining.

Tradeoff: 2x more programs (2x K/V loads) vs pipeline overlap.
Autotune BLOCK_M in {64, 128} so the compiler picks the winner per config.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _binflash_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr, Mask_block_ptr,
    block_mask_ptr, bm_stride_row, bm_stride_col, start_m,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
):
    # ── Pass 1: full blocks (bm_val==2) — K/V only, no mask ──
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        col_idx = start_n // BLOCK_N
        bm_val = tl.load(block_mask_ptr + start_m * bm_stride_row + col_idx * bm_stride_col)
        if bm_val == 2:
            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)
            qk = tl.dot(q, k)
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

    # ── Pass 2: partial blocks (bm_val==1) — K/V + mask ──
    K_block_ptr = tl.advance(K_block_ptr, (0, -N_CTX))
    V_block_ptr = tl.advance(V_block_ptr, (-N_CTX, 0))
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        col_idx = start_n // BLOCK_N
        bm_val = tl.load(block_mask_ptr + start_m * bm_stride_row + col_idx * bm_stride_col)
        if bm_val == 1:
            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)
            mask = tl.load(Mask_block_ptr) != 0
            qk = tl.dot(q, k)
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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm}, num_stages=s, num_warps=w)
        for bm in [64, 128]
        for s in [1, 2, 3, 4, 5]
        for w in [4, 8]
    ],
    key=["N_CTX", "HEAD_DIM", "Z_TIMES_H"],
)
@triton.jit
def _binflash_fwd(
    Q, K, V, sm_scale, Out, Mask, BlockMask64, BlockMask128, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_mask_m, stride_mask_n,
    stride_bm64_row, stride_bm64_col,
    stride_bm128_row, stride_bm128_col,
    Z, H, N_CTX, Z_TIMES_H,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh
    Q_bp = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    K_bp = tl.make_block_ptr(base=K + qvk_offset, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    V_bp = tl.make_block_ptr(base=V + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vk, stride_vn), offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    O_bp = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    M_bp = tl.make_block_ptr(base=Mask, shape=(N_CTX, N_CTX), strides=(stride_mask_m, stride_mask_n), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))

    # Select block mask matching BLOCK_M
    if BLOCK_M == 64:
        bm_ptr = BlockMask64
        bm_stride_row = stride_bm64_row
        bm_stride_col = stride_bm64_col
    else:
        bm_ptr = BlockMask128
        bm_stride_row = stride_bm128_row
        bm_stride_col = stride_bm128_col

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_bp)
    q = (q * (sm_scale * 1.44269504)).to(q.dtype)
    acc, l_i, m_i = _binflash_fwd_inner(acc, l_i, m_i, q, K_bp, V_bp, M_bp, bm_ptr, bm_stride_row, bm_stride_col, start_m, BLOCK_M, BLOCK_N, HEAD_DIM, N_CTX)
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(LSE + (off_hz * N_CTX) + offs_m, m_i + tl.math.log2(l_i))
    tl.store(O_bp, acc.to(Out.type.element_ty))


_bm_cache_key = None
_bm_cache_val = None


def binflash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: torch.Tensor, sm_scale: float | None = None,
    block_m: int = 128, block_n: int | None = None,
    block_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary-block-masked flash attention with autotuned BLOCK_M."""
    global _bm_cache_key, _bm_cache_val
    B, H, N, D = q.shape
    if block_n is None:
        block_n = min(D, 64)
    if N % block_m != 0 or N % block_n != 0:
        raise ValueError(f"N ({N}) must be divisible by block_m ({block_m}) and block_n ({block_n})")
    if block_n > D:
        raise ValueError(f"block_n ({block_n}) must be <= HEAD_DIM ({D})")
    if D not in {16, 32, 64, 128}:
        raise ValueError(f"HEAD_DIM ({D}) must be one of {{16, 32, 64, 128}}")
    if sm_scale is None:
        sm_scale = D ** -0.5

    # Precompute block masks for both BLOCK_M values (autotuner picks)
    key = (mask.data_ptr(), mask.shape[0], mask.shape[1], block_n)
    if key == _bm_cache_key:
        bm_64, bm_128 = _bm_cache_val
    else:
        def _make_bm(bm):
            if N % bm != 0:
                return torch.zeros(1, N // block_n, dtype=torch.int8, device=mask.device)
            reshaped = mask.view(N // bm, bm, N // block_n, block_n)
            has_any = reshaped.any(dim=(1, 3))
            has_all = reshaped.all(dim=(1, 3))
            return has_any.to(torch.int8) + has_all.to(torch.int8)

        bm_64 = _make_bm(64)
        bm_128 = _make_bm(128)
        _bm_cache_key = key
        _bm_cache_val = (bm_64, bm_128)

    o = torch.empty_like(q)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    # Grid uses max BLOCK_M since autotuner may pick either; Triton adjusts
    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), B * H)

    _binflash_fwd[grid](
        q, k, v, sm_scale, o, mask, bm_64, bm_128, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        mask.stride(0), mask.stride(1),
        bm_64.stride(0), bm_64.stride(1),
        bm_128.stride(0), bm_128.stride(1),
        B, H, N, B * H, HEAD_DIM=D, BLOCK_N=block_n,
    )
    return o
