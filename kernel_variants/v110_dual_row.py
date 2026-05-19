"""Binary Block Masked Flash Attention — v110: dual Q-row per program.

Process 2 adjacent Q-block rows in a single program, sharing K/V loads
between them. Adjacent Q-rows attend to overlapping K/V ranges (especially
for causal/local patterns), so sharing avoids redundant loads.

At BM=64 (halved from 128): register pressure for 2 rows × (Q + acc) is
manageable. Grid is halved: (N / (2*BM), B*H).

Key: the UNION of non-empty blocks for both rows is scanned together.
Each K/V load is used for both Q tiles.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _compute_block_mask(
    Mask, BlockMask,
    stride_mask_m, stride_mask_n, nkb,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs_m = row * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = col * BLOCK_N + tl.arange(0, BLOCK_N)
    tile_ptrs = Mask + offs_m[:, None] * stride_mask_m + offs_n[None, :] * stride_mask_n
    tile = tl.load(tile_ptrs)
    tile_sum = tl.sum(tile.to(tl.int32))
    has_any = tile_sum > 0
    has_all = tile_sum == (BLOCK_M * BLOCK_N)
    bm_val = has_any.to(tl.int8) + has_all.to(tl.int8)
    tl.store(BlockMask + row * nkb + col, bm_val)


def _fused_preproc(mask, block_m, block_n):
    """Compute block_mask + union-bounds for paired rows.

    For dual-row processing, we need:
    - fine block_mask: (nqb, nkb) as before
    - paired first_nz/last_nz: for each PAIR of rows, the union bounds
    """
    N = mask.shape[0]
    nqb = N // block_m
    nkb = N // block_n
    block_mask = torch.empty(nqb, nkb, dtype=torch.int8, device=mask.device)
    _compute_block_mask[(nqb, nkb)](
        mask, block_mask,
        mask.stride(0), mask.stride(1), nkb,
        BLOCK_M=block_m, BLOCK_N=block_n,
    )

    # Paired bounds: for rows (2i, 2i+1), compute union of non-empty blocks
    nz = block_mask > 0
    # Reshape to pairs: (nqb/2, 2, nkb)
    paired_nz = nz.view(nqb // 2, 2, nkb).any(dim=1)  # (nqb/2, nkb)

    col_indices = torch.arange(nkb, device=mask.device)
    has_any_in_pair = paired_nz.any(dim=1)
    last_nz = torch.where(
        has_any_in_pair,
        (paired_nz.to(torch.int32) * col_indices).max(dim=1).values,
        torch.tensor(-1, device=mask.device),
    ).to(torch.int32)
    first_nz = paired_nz.to(torch.int32).argmax(dim=1).to(torch.int32)
    first_nz = torch.where(has_any_in_pair, first_nz, torch.tensor(0, device=mask.device)).to(torch.int32)

    return block_mask, first_nz, last_nz


@triton.jit
def _binflash_dual_inner(
    acc1, l_i1, m_i1, q1,
    acc2, l_i2, m_i2, q2,
    K_base, V_base, Mask_base,
    stride_kk, stride_kn, stride_vk, stride_vn,
    stride_mask_m, stride_mask_n,
    block_mask_ptr, bm_stride_row, bm_stride_col,
    row1, row2,
    first_nz_col, last_nz_col,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
):
    offs_d = tl.arange(0, HEAD_DIM)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_bm1 = row1 * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bm2 = row2 * BLOCK_M + tl.arange(0, BLOCK_M)

    # ── Pass 1: process when EITHER row has a full block ──
    col_idx = first_nz_col
    while col_idx <= last_nz_col:
        bm_val1 = tl.load(block_mask_ptr + row1 * bm_stride_row + col_idx * bm_stride_col)
        bm_val2 = tl.load(block_mask_ptr + row2 * bm_stride_row + col_idx * bm_stride_col)
        if (bm_val1 == 2) or (bm_val2 == 2):
            start_n = col_idx * BLOCK_N
            k_ptrs = K_base + offs_d[:, None] * stride_kk + (start_n + offs_bn)[None, :] * stride_kn
            k = tl.load(k_ptrs)
            v_ptrs = V_base + (start_n + offs_bn)[:, None] * stride_vk + offs_d[None, :] * stride_vn
            v = tl.load(v_ptrs)

            if bm_val1 == 2:
                qk1 = tl.dot(q1, k)
                m_ij1 = tl.maximum(m_i1, tl.max(qk1, 1))
                qk1 -= m_ij1[:, None]
                p1 = tl.math.exp2(qk1)
                l_ij1 = tl.sum(p1, 1)
                alpha1 = tl.math.exp2(m_i1 - m_ij1)
                l_i1 = l_i1 * alpha1 + l_ij1
                acc1 = acc1 * alpha1[:, None]
                p1 = p1.to(v.dtype)
                acc1 = tl.dot(p1, v, acc1)
                m_i1 = m_ij1

            if bm_val2 == 2:
                qk2 = tl.dot(q2, k)
                m_ij2 = tl.maximum(m_i2, tl.max(qk2, 1))
                qk2 -= m_ij2[:, None]
                p2 = tl.math.exp2(qk2)
                l_ij2 = tl.sum(p2, 1)
                alpha2 = tl.math.exp2(m_i2 - m_ij2)
                l_i2 = l_i2 * alpha2 + l_ij2
                acc2 = acc2 * alpha2[:, None]
                p2 = p2.to(v.dtype)
                acc2 = tl.dot(p2, v, acc2)
                m_i2 = m_ij2
        col_idx += 1

    # ── Pass 2: process when EITHER row has a partial block ──
    col_idx = first_nz_col
    while col_idx <= last_nz_col:
        bm_val1 = tl.load(block_mask_ptr + row1 * bm_stride_row + col_idx * bm_stride_col)
        bm_val2 = tl.load(block_mask_ptr + row2 * bm_stride_row + col_idx * bm_stride_col)
        if (bm_val1 == 1) or (bm_val2 == 1):
            start_n = col_idx * BLOCK_N
            k_ptrs = K_base + offs_d[:, None] * stride_kk + (start_n + offs_bn)[None, :] * stride_kn
            k = tl.load(k_ptrs)
            v_ptrs = V_base + (start_n + offs_bn)[:, None] * stride_vk + offs_d[None, :] * stride_vn
            v = tl.load(v_ptrs)

            if bm_val1 == 1:
                mask_ptrs1 = Mask_base + offs_bm1[:, None] * stride_mask_m + (start_n + offs_bn)[None, :] * stride_mask_n
                mask1 = tl.load(mask_ptrs1) != 0
                qk1 = tl.dot(q1, k)
                qk1 += tl.where(mask1, 0.0, -1.0e6)
                m_ij1 = tl.maximum(m_i1, tl.max(qk1, 1))
                qk1 -= m_ij1[:, None]
                p1 = tl.math.exp2(qk1)
                l_ij1 = tl.sum(p1, 1)
                alpha1 = tl.math.exp2(m_i1 - m_ij1)
                l_i1 = l_i1 * alpha1 + l_ij1
                acc1 = acc1 * alpha1[:, None]
                p1 = p1.to(v.dtype)
                acc1 = tl.dot(p1, v, acc1)
                m_i1 = m_ij1

            if bm_val2 == 1:
                mask_ptrs2 = Mask_base + offs_bm2[:, None] * stride_mask_m + (start_n + offs_bn)[None, :] * stride_mask_n
                mask2 = tl.load(mask_ptrs2) != 0
                qk2 = tl.dot(q2, k)
                qk2 += tl.where(mask2, 0.0, -1.0e6)
                m_ij2 = tl.maximum(m_i2, tl.max(qk2, 1))
                qk2 -= m_ij2[:, None]
                p2 = tl.math.exp2(qk2)
                l_ij2 = tl.sum(p2, 1)
                alpha2 = tl.math.exp2(m_i2 - m_ij2)
                l_i2 = l_i2 * alpha2 + l_ij2
                acc2 = acc2 * alpha2[:, None]
                p2 = p2.to(v.dtype)
                acc2 = tl.dot(p2, v, acc2)
                m_i2 = m_ij2
        col_idx += 1

    return acc1, l_i1, m_i1, acc2, l_i2, m_i2


@triton.autotune(
    configs=[triton.Config({}, num_stages=s, num_warps=w)
             for s in [1, 2, 3, 4] for w in [4, 8]],
    key=["N_CTX", "HEAD_DIM", "Z_TIMES_H"],
)
@triton.jit
def _binflash_fwd(
    Q, K, V, sm_scale, Out, Mask, BlockMask, FirstNz, LastNz, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_mask_m, stride_mask_n,
    stride_bm_row, stride_bm_col,
    Z, H, N_CTX, Z_TIMES_H,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    pair_idx = tl.program_id(0)  # pair index (N / (2*BM))
    off_hz = tl.program_id(1)
    qvk_offset = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh

    row1 = pair_idx * 2
    row2 = pair_idx * 2 + 1

    # Two Q block ptrs
    Q1_bp = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk), offsets=(row1 * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    Q2_bp = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk), offsets=(row2 * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    O1_bp = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on), offsets=(row1 * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    O2_bp = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on), offsets=(row2 * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    K_base = K + qvk_offset
    V_base = V + qvk_offset

    m_i1 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i1 = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc1 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i2 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i2 = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc2 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q1 = tl.load(Q1_bp)
    q1 = (q1 * (sm_scale * 1.44269504)).to(q1.dtype)
    q2 = tl.load(Q2_bp)
    q2 = (q2 * (sm_scale * 1.44269504)).to(q2.dtype)

    # Shared bounds (union over both rows)
    first_nz = tl.load(FirstNz + pair_idx)
    last_nz = tl.load(LastNz + pair_idx)

    acc1, l_i1, m_i1, acc2, l_i2, m_i2 = _binflash_dual_inner(
        acc1, l_i1, m_i1, q1,
        acc2, l_i2, m_i2, q2,
        K_base, V_base, Mask,
        stride_kk, stride_kn, stride_vk, stride_vn,
        stride_mask_m, stride_mask_n,
        BlockMask, stride_bm_row, stride_bm_col,
        row1, row2, first_nz, last_nz,
        BLOCK_M, BLOCK_N, HEAD_DIM, N_CTX,
    )

    l_i1 = tl.where(l_i1 == 0.0, 1.0, l_i1)
    l_i2 = tl.where(l_i2 == 0.0, 1.0, l_i2)
    acc1 = acc1 / l_i1[:, None]
    acc2 = acc2 / l_i2[:, None]

    offs_m1 = row1 * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m2 = row2 * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(LSE + (off_hz * N_CTX) + offs_m1, m_i1 + tl.math.log2(l_i1))
    tl.store(LSE + (off_hz * N_CTX) + offs_m2, m_i2 + tl.math.log2(l_i2))
    tl.store(O1_bp, acc1.to(Out.type.element_ty))
    tl.store(O2_bp, acc2.to(Out.type.element_ty))


_bm_cache_key = None
_bm_cache_val = None


def binflash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: torch.Tensor, sm_scale: float | None = None,
    block_m: int = 64,  # halved so that 2 rows = 128 total processed per program
    block_n: int | None = None,
    block_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary-block-masked flash attention with dual Q-row processing."""
    global _bm_cache_key, _bm_cache_val
    B, H, N, D = q.shape
    if block_n is None:
        block_n = min(D, 64)
    if N % (2 * block_m) != 0 or N % block_n != 0:
        raise ValueError(f"N must be divisible by 2*block_m={2*block_m} and block_n={block_n}")
    if block_n > D:
        raise ValueError(f"block_n must be <= HEAD_DIM")
    if D not in {16, 32, 64, 128}:
        raise ValueError(f"HEAD_DIM must be one of {{16, 32, 64, 128}}")
    if sm_scale is None:
        sm_scale = D ** -0.5

    key = (mask.data_ptr(), mask.shape[0], mask.shape[1], block_m, block_n)
    if key == _bm_cache_key:
        block_mask, first_nz, last_nz = _bm_cache_val
    else:
        block_mask, first_nz, last_nz = _fused_preproc(mask, block_m, block_n)
        _bm_cache_key = key
        _bm_cache_val = (block_mask, first_nz, last_nz)

    o = torch.empty_like(q)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    num_pairs = N // (2 * block_m)
    grid = (num_pairs, B * H)

    _binflash_fwd[grid](
        q, k, v, sm_scale, o, mask, block_mask, first_nz, last_nz, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        mask.stride(0), mask.stride(1),
        block_mask.stride(0), block_mask.stride(1),
        B, H, N, B * H, HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
    )
    return o
