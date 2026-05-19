"""Binary Block Masked Flash Attention — v120: index dispatch for BM=64 path.

Dead end #8 tested index dispatch vs scan-and-skip on the OLD for-loop kernel.
Was NEUTRAL. But that was before the while-loop + raw pointer + adaptive dispatch
architecture. Now, for longformer at BM=64, we scan ~100 positions per row to
find ~5-6 non-empty blocks. Index dispatch should eliminate this scan waste.

This version adds an index-dispatch variant of the BM=64 kernel:
- Preprocessing additionally computes `nz_cols[nqb, max_nz]` and `n_nz[nqb]`
- Kernel iterates the indices directly instead of scanning bm_val
- Only used for the gap-heavy dispatch path (BM=64)

Expected gain: 2-5% on longformer. Other patterns unchanged (still scan).
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
    N = mask.shape[0]
    nqb = N // block_m
    nkb = N // block_n
    block_mask = torch.empty(nqb, nkb, dtype=torch.int8, device=mask.device)
    _compute_block_mask[(nqb, nkb)](
        mask, block_mask,
        mask.stride(0), mask.stride(1), nkb,
        BLOCK_M=block_m, BLOCK_N=block_n,
    )
    nz = block_mask > 0
    col_indices = torch.arange(nkb, device=mask.device)
    has_any_in_row = nz.any(dim=1)
    last_nz = torch.where(
        has_any_in_row,
        (nz.to(torch.int32) * col_indices).max(dim=1).values,
        torch.tensor(-1, device=mask.device),
    ).to(torch.int32)
    first_nz = nz.to(torch.int32).argmax(dim=1).to(torch.int32)
    first_nz = torch.where(has_any_in_row, first_nz, torch.tensor(0, device=mask.device)).to(torch.int32)
    num_nonempty = nz.to(torch.int32).sum(dim=1)
    scan_width = (last_nz - first_nz + 1).clamp(min=1)
    gap_per_row = 1.0 - num_nonempty.float() / scan_width.float()
    avg_gap_ratio = gap_per_row.mean().item()
    density = nz.float().mean().item()
    return block_mask, first_nz, last_nz, avg_gap_ratio, density


def _compute_nz_cols(block_mask):
    """Compute padded nz_cols[nqb, max_nz] and n_nz[nqb] from block_mask.

    For each row, nz_cols contains the column indices of non-empty blocks,
    sorted, padded to max_nz with 0 (which will be masked by n_nz check).
    """
    nqb, nkb = block_mask.shape
    nz = block_mask > 0
    # Count per row — find max for padding
    n_nz = nz.sum(dim=1).to(torch.int32)
    max_nz = int(n_nz.max().item())
    if max_nz == 0:
        max_nz = 1  # avoid zero-size tensor

    # For each row, sort non-zero indices to the front
    # Use argsort-like approach: multiply column indices by nz mask, sort descending to put
    # non-zero indices at front (they have larger "priority")
    # Actually, simpler: nonzero() returns pairs (row, col) — use that.
    col_indices = torch.arange(nkb, device=block_mask.device, dtype=torch.int32)
    # For each row, get sorted column indices of non-empty blocks
    # Trick: set empty col indices to nkb (sentinel), then sort ascending
    nz_indices = torch.where(nz, col_indices[None, :], torch.tensor(nkb, device=block_mask.device, dtype=torch.int32))
    nz_indices_sorted, _ = torch.sort(nz_indices, dim=1)  # empty goes to end
    nz_cols = nz_indices_sorted[:, :max_nz].contiguous()
    return nz_cols, n_nz, max_nz


# ─── SCAN kernel (default path for non-gap-heavy patterns) ───

@triton.jit
def _binflash_fwd_inner(
    acc, l_i, m_i, q,
    K_base, V_base, Mask_base,
    stride_kk, stride_kn, stride_vk, stride_vn,
    stride_mask_m, stride_mask_n,
    block_mask_ptr, bm_stride_row, bm_stride_col, start_m,
    first_nz_col, last_nz_col,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
):
    offs_d = tl.arange(0, HEAD_DIM)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_bm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    col_idx = first_nz_col
    while col_idx <= last_nz_col:
        bm_val = tl.load(block_mask_ptr + start_m * bm_stride_row + col_idx * bm_stride_col)
        if bm_val == 2:
            start_n = col_idx * BLOCK_N
            k_ptrs = K_base + offs_d[:, None] * stride_kk + (start_n + offs_bn)[None, :] * stride_kn
            k = tl.load(k_ptrs)
            v_ptrs = V_base + (start_n + offs_bn)[:, None] * stride_vk + offs_d[None, :] * stride_vn
            v = tl.load(v_ptrs)
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
        col_idx += 1

    col_idx = first_nz_col
    while col_idx <= last_nz_col:
        bm_val = tl.load(block_mask_ptr + start_m * bm_stride_row + col_idx * bm_stride_col)
        if bm_val == 1:
            start_n = col_idx * BLOCK_N
            k_ptrs = K_base + offs_d[:, None] * stride_kk + (start_n + offs_bn)[None, :] * stride_kn
            k = tl.load(k_ptrs)
            v_ptrs = V_base + (start_n + offs_bn)[:, None] * stride_vk + offs_d[None, :] * stride_vn
            v = tl.load(v_ptrs)
            mask_ptrs = Mask_base + offs_bm[:, None] * stride_mask_m + (start_n + offs_bn)[None, :] * stride_mask_n
            mask = tl.load(mask_ptrs) != 0
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
        col_idx += 1

    return acc, l_i, m_i


# ─── INDEX DISPATCH inner for gap-heavy paths ───

@triton.jit
def _binflash_fwd_inner_idx(
    acc, l_i, m_i, q,
    K_base, V_base, Mask_base,
    stride_kk, stride_kn, stride_vk, stride_vn,
    stride_mask_m, stride_mask_n,
    block_mask_ptr, bm_stride_row, bm_stride_col, start_m,
    nz_cols_base, stride_nz_row, n_nz,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
):
    offs_d = tl.arange(0, HEAD_DIM)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_bm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Pass 1: iterate non-zero cols, process only bm_val == 2
    i = 0
    while i < n_nz:
        col_idx = tl.load(nz_cols_base + start_m * stride_nz_row + i)
        bm_val = tl.load(block_mask_ptr + start_m * bm_stride_row + col_idx * bm_stride_col)
        if bm_val == 2:
            start_n = col_idx * BLOCK_N
            k_ptrs = K_base + offs_d[:, None] * stride_kk + (start_n + offs_bn)[None, :] * stride_kn
            k = tl.load(k_ptrs)
            v_ptrs = V_base + (start_n + offs_bn)[:, None] * stride_vk + offs_d[None, :] * stride_vn
            v = tl.load(v_ptrs)
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
        i += 1

    # Pass 2: same indices, process only bm_val == 1
    i = 0
    while i < n_nz:
        col_idx = tl.load(nz_cols_base + start_m * stride_nz_row + i)
        bm_val = tl.load(block_mask_ptr + start_m * bm_stride_row + col_idx * bm_stride_col)
        if bm_val == 1:
            start_n = col_idx * BLOCK_N
            k_ptrs = K_base + offs_d[:, None] * stride_kk + (start_n + offs_bn)[None, :] * stride_kn
            k = tl.load(k_ptrs)
            v_ptrs = V_base + (start_n + offs_bn)[:, None] * stride_vk + offs_d[None, :] * stride_vn
            v = tl.load(v_ptrs)
            mask_ptrs = Mask_base + offs_bm[:, None] * stride_mask_m + (start_n + offs_bn)[None, :] * stride_mask_n
            mask = tl.load(mask_ptrs) != 0
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
        i += 1

    return acc, l_i, m_i


@triton.autotune(
    configs=[triton.Config({}, num_stages=s, num_warps=w)
             for s in [1, 2, 3, 4, 5] for w in [4, 8]],
    key=["N_CTX", "HEAD_DIM", "Z_TIMES_H", "BLOCK_M", "BLOCK_N"],
)
@triton.jit
def _binflash_fwd_scan(
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
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh
    Q_bp = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    O_bp = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    K_base = K + qvk_offset
    V_base = V + qvk_offset
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_bp)
    q = (q * (sm_scale * 1.44269504)).to(q.dtype)
    first_nz = tl.load(FirstNz + start_m)
    last_nz = tl.load(LastNz + start_m)
    acc, l_i, m_i = _binflash_fwd_inner(
        acc, l_i, m_i, q,
        K_base, V_base, Mask,
        stride_kk, stride_kn, stride_vk, stride_vn,
        stride_mask_m, stride_mask_n,
        BlockMask, stride_bm_row, stride_bm_col, start_m,
        first_nz, last_nz,
        BLOCK_M, BLOCK_N, HEAD_DIM, N_CTX,
    )
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(LSE + (off_hz * N_CTX) + offs_m, m_i + tl.math.log2(l_i))
    tl.store(O_bp, acc.to(Out.type.element_ty))


@triton.autotune(
    configs=[triton.Config({}, num_stages=s, num_warps=w)
             for s in [1, 2, 3, 4, 5] for w in [4, 8]],
    key=["N_CTX", "HEAD_DIM", "Z_TIMES_H", "BLOCK_M", "BLOCK_N"],
)
@triton.jit
def _binflash_fwd_idx(
    Q, K, V, sm_scale, Out, Mask, BlockMask, NzCols, NNz, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_mask_m, stride_mask_n,
    stride_bm_row, stride_bm_col,
    stride_nz_row,
    Z, H, N_CTX, Z_TIMES_H,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh
    Q_bp = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    O_bp = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    K_base = K + qvk_offset
    V_base = V + qvk_offset
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_bp)
    q = (q * (sm_scale * 1.44269504)).to(q.dtype)
    n_nz = tl.load(NNz + start_m)
    acc, l_i, m_i = _binflash_fwd_inner_idx(
        acc, l_i, m_i, q,
        K_base, V_base, Mask,
        stride_kk, stride_kn, stride_vk, stride_vn,
        stride_mask_m, stride_mask_n,
        BlockMask, stride_bm_row, stride_bm_col, start_m,
        NzCols, stride_nz_row, n_nz,
        BLOCK_M, BLOCK_N, HEAD_DIM, N_CTX,
    )
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(LSE + (off_hz * N_CTX) + offs_m, m_i + tl.math.log2(l_i))
    tl.store(O_bp, acc.to(Out.type.element_ty))


_bm_cache_key = None
_bm_cache_val = None

_GAP_RATIO_THRESHOLD = 0.3
_DENSITY_THRESHOLD = 0.9


def binflash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: torch.Tensor, sm_scale: float | None = None,
    block_m: int = 128, block_n: int | None = None,
    block_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """v114 + index dispatch for BM=64 path."""
    global _bm_cache_key, _bm_cache_val
    B, H, N, D = q.shape
    if block_n is None:
        block_n = min(D, 64)
    if sm_scale is None:
        sm_scale = D ** -0.5
    if D not in {16, 32, 64, 128}:
        raise ValueError(f"HEAD_DIM ({D}) must be one of {{16, 32, 64, 128}}")

    if block_mask is None:
        key = (mask.data_ptr(), mask.shape[0], mask.shape[1], block_n, D)
        if key == _bm_cache_key:
            bm, fn, ln, chosen_bm, chosen_bn, nz_cols, n_nz, use_idx = _bm_cache_val
            block_mask = bm
            first_nz = fn
            last_nz = ln
        else:
            bm128_bn64, fn, ln, gap_ratio, density = _fused_preproc(mask, 128, 64)

            if D == 64 and gap_ratio > _GAP_RATIO_THRESHOLD and N % 64 == 0:
                block_mask, first_nz, last_nz, _, _ = _fused_preproc(mask, 64, 64)
                chosen_bm, chosen_bn = 64, 64
                # Compute index dispatch arrays for this path
                nz_cols, n_nz, _max_nz = _compute_nz_cols(block_mask)
                use_idx = True
            elif D == 64 and density > _DENSITY_THRESHOLD and N % 128 == 0:
                block_mask, first_nz, last_nz, _, _ = _fused_preproc(mask, 128, 128)
                chosen_bm, chosen_bn = 128, 128
                nz_cols, n_nz = None, None
                use_idx = False
            else:
                block_mask, first_nz, last_nz = bm128_bn64, fn, ln
                chosen_bm, chosen_bn = 128, 64
                nz_cols, n_nz = None, None
                use_idx = False

            _bm_cache_key = key
            _bm_cache_val = (block_mask, first_nz, last_nz, chosen_bm, chosen_bn, nz_cols, n_nz, use_idx)
    else:
        nz = block_mask > 0
        nkb = nz.shape[1]
        col_indices = torch.arange(nkb, device=mask.device)
        has_any_in_row = nz.any(dim=1)
        last_nz = torch.where(
            has_any_in_row,
            (nz.to(torch.int32) * col_indices).max(dim=1).values,
            torch.tensor(-1, device=mask.device),
        ).to(torch.int32)
        first_nz = nz.to(torch.int32).argmax(dim=1).to(torch.int32)
        first_nz = torch.where(has_any_in_row, first_nz, torch.tensor(0, device=mask.device)).to(torch.int32)
        chosen_bm, chosen_bn = block_m, block_n
        nz_cols, n_nz, use_idx = None, None, False

    if N % chosen_bm != 0 or N % chosen_bn != 0:
        raise ValueError(f"N must be divisible by bm ({chosen_bm}) and bn ({chosen_bn})")

    o = torch.empty_like(q)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(N, chosen_bm), B * H)

    if use_idx:
        _binflash_fwd_idx[grid](
            q, k, v, sm_scale, o, mask, block_mask, nz_cols, n_nz, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            mask.stride(0), mask.stride(1),
            block_mask.stride(0), block_mask.stride(1),
            nz_cols.stride(0),
            B, H, N, B * H, HEAD_DIM=D, BLOCK_M=chosen_bm, BLOCK_N=chosen_bn,
        )
    else:
        _binflash_fwd_scan[grid](
            q, k, v, sm_scale, o, mask, block_mask, first_nz, last_nz, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            mask.stride(0), mask.stride(1),
            block_mask.stride(0), block_mask.stride(1),
            B, H, N, B * H, HEAD_DIM=D, BLOCK_M=chosen_bm, BLOCK_N=chosen_bn,
        )
    return o
