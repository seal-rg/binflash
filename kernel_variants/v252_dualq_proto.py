"""V193: V192 + fused Triton kernel for col_indices preprocessing.

Same gathered-dispatch kernel as v192. The only difference is that
`_build_gathered_indices` now uses a single Triton kernel instead of
PyTorch's `argsort`, which saves ~0.2ms of preproc cost per call.

Kernel body is IDENTICAL to v192. Only the preproc path differs.
"""

import torch  # type: ignore
import triton  # type: ignore
import triton.language as tl  # type: ignore

# ─── Fused preprocessing (unchanged from production) ───


@triton.jit
def _reduce_block_mask(
    BlockMask,
    FirstNz,
    LastNz,
    GlobalSums,
    GlobalGapFloat,
    stride_bm_row,
    stride_bm_col,
    nkb,
    NKB_POW2: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, NKB_POW2)
    valid = offs < nkb
    bm_row = tl.load(
        BlockMask + row * stride_bm_row + offs * stride_bm_col,
        mask=valid,
        other=0,
    )
    is_nonempty = bm_row > 0
    is_partial = bm_row == 1
    has_any = tl.sum(is_nonempty.to(tl.int32)) > 0
    ranked_last = tl.where(is_nonempty, offs, -1)
    last_nz = tl.max(ranked_last, axis=0)
    ranked_first = tl.where(is_nonempty, offs, nkb)
    first_nz_tmp = tl.min(ranked_first, axis=0)
    first_nz = tl.where(has_any, first_nz_tmp, 0)
    num_nonempty = tl.sum(is_nonempty.to(tl.int32))
    num_partial = tl.sum(is_partial.to(tl.int32))
    scan_width = tl.where(has_any, last_nz - first_nz + 1, 1)
    gap_for_row = 1.0 - num_nonempty.to(tl.float32) / scan_width.to(tl.float32)
    tl.store(FirstNz + row, first_nz.to(tl.int32))
    tl.store(LastNz + row, last_nz.to(tl.int32))
    tl.atomic_add(GlobalSums + 0, num_nonempty)
    tl.atomic_add(GlobalSums + 1, num_partial)
    tl.atomic_add(GlobalGapFloat, gap_for_row)


@triton.jit
def _compute_and_pack_mask_multi(
    Mask,
    BlockMaskCoarse,
    BlockMaskFine,
    BlockMaskBN128,
    Packed,
    stride_mask_m,
    stride_mask_n,
    stride_packed_m,
    stride_packed_n,
    nkb,
    nkb_bn128,
    BM_COARSE: tl.constexpr,
    BM_FINE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BN_CHUNKS: tl.constexpr,
):
    row_coarse = tl.program_id(0)
    col_chunk = tl.program_id(1)
    offs_m = row_coarse * BM_COARSE + tl.arange(0, BM_COARSE)
    offs_n_base = col_chunk * (BN_CHUNKS * BLOCK_N)
    offs_n = offs_n_base + tl.arange(0, BN_CHUNKS * BLOCK_N)
    tile_ptrs = Mask + offs_m[:, None] * stride_mask_m + offs_n[None, :] * stride_mask_n
    tile = tl.load(tile_ptrs)

    tile_3d_bm = tl.reshape(tile, (BM_COARSE, BN_CHUNKS, BLOCK_N))
    tile_i32 = tile_3d_bm.to(tl.int32)
    sum_col = tl.sum(tile_i32, axis=0)
    coarse_sums = tl.sum(sum_col, axis=1)
    has_any_c = coarse_sums > 0
    has_all_c = coarse_sums == (BM_COARSE * BLOCK_N)
    bm_coarse = has_any_c.to(tl.int8) + has_all_c.to(tl.int8)
    col_start = col_chunk * BN_CHUNKS + tl.arange(0, BN_CHUNKS)
    tl.store(BlockMaskCoarse + row_coarse * nkb + col_start, bm_coarse)

    TILE_SIZE: tl.constexpr = BM_COARSE * BN_CHUNKS * BLOCK_N
    bn128_total = tl.sum(coarse_sums)
    has_any_128 = bn128_total > 0
    has_all_128 = bn128_total == TILE_SIZE
    bm_bn128 = has_any_128.to(tl.int8) + has_all_128.to(tl.int8)
    tl.store(BlockMaskBN128 + row_coarse * nkb_bn128 + col_chunk, bm_bn128)

    tile_fine_4d = tl.reshape(tile_i32, (2, BM_FINE, BN_CHUNKS, BLOCK_N))
    sum_bm_f = tl.sum(tile_fine_4d, axis=1)
    fine_sums = tl.sum(sum_bm_f, axis=2)
    has_any_f = fine_sums > 0
    has_all_f = fine_sums == (BM_FINE * BLOCK_N)
    bm_fine = has_any_f.to(tl.int8) + has_all_f.to(tl.int8)
    pair_idx = tl.arange(0, 2)
    fine_row = 2 * row_coarse + pair_idx
    fine_ptrs = BlockMaskFine + fine_row[:, None] * nkb + col_start[None, :]
    tl.store(fine_ptrs, bm_fine)

    WORDS: tl.constexpr = (BN_CHUNKS * BLOCK_N) // 32
    tile_3d_pk = tl.reshape(tile, (BM_COARSE, WORDS, 32))
    bits_i32 = tile_3d_pk.to(tl.int32)
    bit_weights = 1 << tl.arange(0, 32)
    weighted = bits_i32 * bit_weights[None, None, :]
    packed = tl.sum(weighted, axis=2)
    word_start = col_chunk * WORDS + tl.arange(0, WORDS)
    packed_ptrs = Packed + offs_m[:, None] * stride_packed_m + word_start[None, :] * stride_packed_n
    tl.store(packed_ptrs, packed)


def _pick_bn_chunks(block_m, block_n, nkb, target_tile_bytes=65536, start=8):
    max_by_budget = max(1, target_tile_bytes // max(1, block_m * block_n))
    bn_chunks = min(start, max_by_budget)
    while nkb % bn_chunks != 0 and bn_chunks > 1:
        bn_chunks //= 2
    return bn_chunks


def _fused_preproc_multigrain(mask, block_n):
    N = mask.shape[0]
    assert block_n == 64
    assert N % 128 == 0 and N % 64 == 0
    nkb = N // block_n
    nkb_bn128 = N // 128
    nqb_coarse = N // 128
    nqb_fine = N // 64

    block_mask_coarse = torch.empty(nqb_coarse, nkb, dtype=torch.int8, device=mask.device)
    block_mask_fine = torch.empty(nqb_fine, nkb, dtype=torch.int8, device=mask.device)
    block_mask_bn128 = torch.empty(nqb_coarse, nkb_bn128, dtype=torch.int8, device=mask.device)
    mask_packed = torch.empty((N, N // 32), dtype=torch.int32, device=mask.device)

    bn_chunks = _pick_bn_chunks(128, block_n, nkb, start=2)
    assert bn_chunks * block_n == 128
    _compute_and_pack_mask_multi[(nqb_coarse, nkb // bn_chunks)](
        mask,
        block_mask_coarse,
        block_mask_fine,
        block_mask_bn128,
        mask_packed,
        mask.stride(0),
        mask.stride(1),
        mask_packed.stride(0),
        mask_packed.stride(1),
        nkb,
        nkb_bn128,
        BM_COARSE=128,
        BM_FINE=64,
        BLOCK_N=block_n,
        BN_CHUNKS=bn_chunks,
    )

    first_nz_c = torch.empty(nqb_coarse, dtype=torch.int32, device=mask.device)
    last_nz_c = torch.empty(nqb_coarse, dtype=torch.int32, device=mask.device)
    global_sums = torch.zeros(2, dtype=torch.int32, device=mask.device)
    global_gap = torch.zeros(1, dtype=torch.float32, device=mask.device)
    nkb_pow2 = 1
    while nkb_pow2 < nkb:
        nkb_pow2 *= 2
    _reduce_block_mask[(nqb_coarse,)](
        block_mask_coarse,
        first_nz_c,
        last_nz_c,
        global_sums,
        global_gap,
        block_mask_coarse.stride(0),
        block_mask_coarse.stride(1),
        nkb,
        NKB_POW2=nkb_pow2,
    )
    metrics = torch.cat([global_sums.to(torch.float32), global_gap])
    num_nonempty_total, num_partial_total, global_gap_sum = metrics.tolist()
    density = num_nonempty_total / (nqb_coarse * nkb)
    partial_fraction = num_partial_total / max(num_nonempty_total, 1.0)
    avg_gap_ratio = global_gap_sum / nqb_coarse
    return {
        "block_mask_coarse": block_mask_coarse,
        "block_mask_fine": block_mask_fine,
        "block_mask_bn128": block_mask_bn128,
        "mask_packed": mask_packed,
        "first_nz_coarse": first_nz_c,
        "last_nz_coarse": last_nz_c,
        "gap_ratio": avg_gap_ratio,
        "density": density,
        "partial_fraction": partial_fraction,
    }


# ─── Gathered indices preprocessing (V193: fused Triton kernel) ───


@triton.jit
def _build_col_indices_kernel(
    BlockMask,  # (nqb, nkb) int8
    ColIndices,  # (nqb, max_nnz) int32 — output
    NFull,  # (nqb,) int32 — output
    NTotal,  # (nqb,) int32 — output
    stride_bm_row,
    stride_bm_col,
    stride_ci_row,
    nkb,
    NKB_POW2: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, NKB_POW2)
    valid = offs < nkb
    bm_row = tl.load(
        BlockMask + row * stride_bm_row + offs * stride_bm_col,
        mask=valid,
        other=0,
    )
    is_full = (bm_row == 2) & valid
    is_partial = (bm_row == 1) & valid

    # Compute per-row counts (scalar)
    n_full = tl.sum(is_full.to(tl.int32))
    n_partial = tl.sum(is_partial.to(tl.int32))
    n_total = n_full + n_partial
    tl.store(NFull + row, n_full)
    tl.store(NTotal + row, n_total)

    # Compute target rank for each non-empty position:
    #   full blocks get rank [0, n_full)
    #   partial blocks get rank [n_full, n_total)
    # Use cumsum to rank within each category.
    full_rank = tl.cumsum(is_full.to(tl.int32), axis=0) - 1  # cumsum is 1-based
    partial_rank = tl.cumsum(is_partial.to(tl.int32), axis=0) - 1 + n_full
    target_rank = tl.where(is_full, full_rank, partial_rank)

    # Scatter column indices to their target ranks.
    is_nonempty = is_full | is_partial
    ci_ptrs = ColIndices + row * stride_ci_row + target_rank
    tl.store(ci_ptrs, offs.to(tl.int32), mask=is_nonempty)


def _build_gathered_indices(block_mask):
    """Fused Triton kernel version of v192's _build_gathered_indices.

    Returns:
        col_indices: (nqb, max_nnz) int32 — sorted column indices
        n_full: (nqb,) int32 — count of full blocks per row
        n_total: (nqb,) int32 — count of all non-empty blocks per row
    """
    nqb, nkb = block_mask.shape
    nkb_pow2 = 1
    while nkb_pow2 < nkb:
        nkb_pow2 *= 2

    n_full = torch.empty(nqb, dtype=torch.int32, device=block_mask.device)
    n_total = torch.empty(nqb, dtype=torch.int32, device=block_mask.device)
    # Over-allocate to nkb so we never exceed. Actual max_nnz ≤ nkb.
    col_indices = torch.empty(nqb, nkb, dtype=torch.int32, device=block_mask.device)

    _build_col_indices_kernel[(nqb,)](
        block_mask,
        col_indices,
        n_full,
        n_total,
        block_mask.stride(0),
        block_mask.stride(1),
        col_indices.stride(0),
        nkb,
        NKB_POW2=nkb_pow2,
    )

    return col_indices, n_full, n_total


# ─── Dual-Q preproc (V252) ───


@triton.jit
def _build_dual_col_indices_kernel(
    BlockMask,  # (nqb, nkb) int8
    DualColIndices,  # (dual_nqb, nkb) int32 — output
    NDualFull,  # (dual_nqb,) int32 — count where BOTH Q-blocks are full
    NDualTotal,  # (dual_nqb,) int32 — count where either is non-empty
    stride_bm_row,
    stride_bm_col,
    stride_dci_row,
    nkb,
    NKB_POW2: tl.constexpr,
):
    pair = tl.program_id(0)
    offs = tl.arange(0, NKB_POW2)
    valid = offs < nkb
    bm1 = tl.load(BlockMask + (2 * pair) * stride_bm_row + offs * stride_bm_col, mask=valid, other=0)
    bm2 = tl.load(BlockMask + (2 * pair + 1) * stride_bm_row + offs * stride_bm_col, mask=valid, other=0)
    both_full = (bm1 == 2) & (bm2 == 2) & valid
    either_nonempty = ((bm1 > 0) | (bm2 > 0)) & valid
    mixed = either_nonempty & ~both_full
    n_full = tl.sum(both_full.to(tl.int32))
    n_mixed = tl.sum(mixed.to(tl.int32))
    n_total = n_full + n_mixed
    tl.store(NDualFull + pair, n_full)
    tl.store(NDualTotal + pair, n_total)
    # Ranks: both_full first [0, n_full), mixed after [n_full, n_total)
    full_rank = tl.cumsum(both_full.to(tl.int32), axis=0) - 1
    mixed_rank = tl.cumsum(mixed.to(tl.int32), axis=0) - 1 + n_full
    target_rank = tl.where(both_full, full_rank, mixed_rank)
    is_needed = both_full | mixed
    ci_ptrs = DualColIndices + pair * stride_dci_row + target_rank
    tl.store(ci_ptrs, offs.to(tl.int32), mask=is_needed)


def _build_dual_gathered_indices(block_mask):
    """Build dual-Q col indices: union of adjacent Q-block pairs.

    Returns:
        dual_col_indices: (dual_nqb, nkb) int32 — sorted [both_full | mixed]
        n_dual_full: (dual_nqb,) int32
        n_dual_total: (dual_nqb,) int32
    """
    nqb, nkb = block_mask.shape
    assert nqb % 2 == 0
    dual_nqb = nqb // 2
    nkb_pow2 = 1
    while nkb_pow2 < nkb:
        nkb_pow2 *= 2
    n_dual_full = torch.empty(dual_nqb, dtype=torch.int32, device=block_mask.device)
    n_dual_total = torch.empty(dual_nqb, dtype=torch.int32, device=block_mask.device)
    dual_col_indices = torch.empty(dual_nqb, nkb, dtype=torch.int32, device=block_mask.device)
    _build_dual_col_indices_kernel[(dual_nqb,)](
        block_mask,
        dual_col_indices,
        n_dual_full,
        n_dual_total,
        block_mask.stride(0),
        block_mask.stride(1),
        dual_col_indices.stride(0),
        nkb,
        NKB_POW2=nkb_pow2,
    )
    return dual_col_indices, n_dual_full, n_dual_total


# ─── Dual-Q inner kernel (V252) ───


@triton.jit
def _dualq_fwd_inner(
    acc1, l_i1, m_i1,
    acc2, l_i2, m_i2,
    q1, q2,
    K_base, V_base, MaskPacked_base,
    stride_kk, stride_kn, stride_vk, stride_vn, stride_mp_m, stride_mp_n,
    col_indices_ptr,
    n_full, n_total,
    pair_idx,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """Dual-Q inner: share K, V loads across 2 adjacent Q-blocks (BM each)."""
    offs_d = tl.arange(0, HEAD_DIM)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_bm1 = (2 * pair_idx) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bm2 = (2 * pair_idx + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
    BN_PACKED: tl.constexpr = BLOCK_N // 32
    offs_bn_packed = tl.arange(0, BN_PACKED)
    bit_offs = tl.arange(0, 32)

    # Pass 1: BOTH full — no mask load
    for idx in tl.range(0, n_full, num_stages=2):
        col_idx = tl.load(col_indices_ptr + idx)
        start_n = col_idx * BLOCK_N
        k_ptrs = K_base + offs_d[:, None] * stride_kk + (start_n + offs_bn)[None, :] * stride_kn
        k = tl.load(k_ptrs)
        v_ptrs = V_base + (start_n + offs_bn)[:, None] * stride_vk + offs_d[None, :] * stride_vn
        v = tl.load(v_ptrs)
        # Q1
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
        # Q2
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

    # Pass 2: mixed — load mask for both (empty Q-block → mask all-zero → qk gets -1e6)
    for idx in tl.range(n_full, n_total, num_stages=2):
        col_idx = tl.load(col_indices_ptr + idx)
        start_n = col_idx * BLOCK_N
        k_ptrs = K_base + offs_d[:, None] * stride_kk + (start_n + offs_bn)[None, :] * stride_kn
        k = tl.load(k_ptrs)
        v_ptrs = V_base + (start_n + offs_bn)[:, None] * stride_vk + offs_d[None, :] * stride_vn
        v = tl.load(v_ptrs)
        start_n_packed = (col_idx * BLOCK_N) // 32
        mp1_ptrs = MaskPacked_base + offs_bm1[:, None] * stride_mp_m + (start_n_packed + offs_bn_packed)[None, :] * stride_mp_n
        mp2_ptrs = MaskPacked_base + offs_bm2[:, None] * stride_mp_m + (start_n_packed + offs_bn_packed)[None, :] * stride_mp_n
        packed1 = tl.load(mp1_ptrs)
        packed2 = tl.load(mp2_ptrs)
        bits1 = (packed1[:, :, None] >> bit_offs[None, None, :]) & 1
        bits2 = (packed2[:, :, None] >> bit_offs[None, None, :]) & 1
        mask1 = tl.reshape(bits1, (BLOCK_M, BLOCK_N)) != 0
        mask2 = tl.reshape(bits2, (BLOCK_M, BLOCK_N)) != 0
        # Q1
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
        # Q2
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

    return acc1, l_i1, m_i1, acc2, l_i2, m_i2


# ─── Main attention kernel ───


@triton.jit
def _gathered_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_base,
    V_base,
    Mask_base,
    MaskPacked_base,
    stride_kk,
    stride_kn,
    stride_vk,
    stride_vn,
    stride_mask_m,
    stride_mask_n,
    stride_mp_m,
    stride_mp_n,
    col_indices_ptr,
    n_full,
    n_total,
    start_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
    USE_PACKED: tl.constexpr,
):
    offs_d = tl.arange(0, HEAD_DIM)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_bm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    BN_PACKED: tl.constexpr = BLOCK_N // 32
    offs_bn_packed = tl.arange(0, BN_PACKED)
    bit_offs = tl.arange(0, 32)

    # Pass 1: full blocks — iterate [0, n_full), no mask load needed.
    idx = n_full - n_full  # Triton scalar zero
    for idx in tl.range(0, n_full, num_stages=2):
        col_idx = tl.load(col_indices_ptr + idx)
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

    # Pass 2: partial blocks — iterate [n_full, n_total), with mask.
    for idx in tl.range(n_full, n_total, num_stages=2):
        col_idx = tl.load(col_indices_ptr + idx)
        start_n = col_idx * BLOCK_N
        k_ptrs = K_base + offs_d[:, None] * stride_kk + (start_n + offs_bn)[None, :] * stride_kn
        k = tl.load(k_ptrs)
        v_ptrs = V_base + (start_n + offs_bn)[:, None] * stride_vk + offs_d[None, :] * stride_vn
        v = tl.load(v_ptrs)
        if USE_PACKED:
            start_n_packed = (col_idx * BLOCK_N) // 32
            mp_ptrs = (
                MaskPacked_base + offs_bm[:, None] * stride_mp_m + (start_n_packed + offs_bn_packed)[None, :] * stride_mp_n
            )
            packed = tl.load(mp_ptrs)
            bits_3d = (packed[:, :, None] >> bit_offs[None, None, :]) & 1
            mask_ = tl.reshape(bits_3d, (BLOCK_M, BLOCK_N)) != 0
        else:
            mask_ptrs = Mask_base + offs_bm[:, None] * stride_mask_m + (start_n + offs_bn)[None, :] * stride_mask_n
            mask_ = tl.load(mask_ptrs) != 0
        qk = tl.dot(q, k)
        qk += tl.where(mask_, 0.0, -1.0e6)
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

    return acc, l_i, m_i


_autotune_configs = []
for _s in [1, 2, 3, 4, 5, 6, 7]:
    for _w in [4, 8]:
        for _mnr in [None, 168, 192]:
            _kw = {}
            if _mnr is not None:
                _kw["maxnreg"] = _mnr
            _autotune_configs.append(triton.Config({}, num_stages=_s, num_warps=_w, **_kw))


@triton.autotune(
    configs=_autotune_configs,
    key=["N_CTX", "HEAD_DIM", "Z_TIMES_H", "BLOCK_M", "BLOCK_N", "USE_PACKED"],
)
@triton.jit
def _gathered_fwd(
    Q,
    K,
    V,
    sm_scale,
    Out,
    Mask,
    MaskPacked,
    ColIndices,
    NFull,
    NTotal,
    LSE,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_mask_m,
    stride_mask_n,
    stride_mp_m,
    stride_mp_n,
    stride_ci_row,
    Z,
    H,
    N_CTX,
    Z_TIMES_H,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_PACKED: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh
    Q_bp = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    O_bp = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_base = K + qvk_offset
    V_base = V + qvk_offset
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_bp)
    q = (q * (sm_scale * 1.44269504)).to(q.dtype)

    # Load per-row gathered metadata
    n_full = tl.load(NFull + start_m)
    n_total = tl.load(NTotal + start_m)
    ci_base = ColIndices + start_m * stride_ci_row

    acc, l_i, m_i = _gathered_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_base,
        V_base,
        Mask,
        MaskPacked,
        stride_kk,
        stride_kn,
        stride_vk,
        stride_vn,
        stride_mask_m,
        stride_mask_n,
        stride_mp_m,
        stride_mp_n,
        ci_base,
        n_full,
        n_total,
        start_m,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        N_CTX,
        USE_PACKED,
    )
    # v188 epilogue: keep LSE store + tl.where as scheduling anchors.
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(LSE + (off_hz * N_CTX) + offs_m, m_i + tl.math.log2(l_i))
    tl.store(O_bp, acc.to(Out.type.element_ty))


# ─── Dual-Q main kernel (V252) ───


_dualq_autotune_configs = []
for _s in [1, 2, 3, 4]:
    for _w in [4, 8]:
        _dualq_autotune_configs.append(triton.Config({}, num_stages=_s, num_warps=_w))


@triton.autotune(
    configs=_dualq_autotune_configs,
    key=["N_CTX", "HEAD_DIM", "Z_TIMES_H", "BLOCK_M", "BLOCK_N"],
)
@triton.jit
def _dualq_fwd(
    Q, K, V, sm_scale, Out, MaskPacked,
    DualColIndices, NDualFull, NDualTotal, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_mp_m, stride_mp_n,
    stride_dci_row,
    Z, H, N_CTX, Z_TIMES_H,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,  # per-Q-block M
    BLOCK_N: tl.constexpr,
):
    pair_idx = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh
    Q1_bp = tl.make_block_ptr(
        base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk),
        offsets=(2 * pair_idx * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    Q2_bp = tl.make_block_ptr(
        base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk),
        offsets=((2 * pair_idx + 1) * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    O1_bp = tl.make_block_ptr(
        base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on),
        offsets=(2 * pair_idx * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    O2_bp = tl.make_block_ptr(
        base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on),
        offsets=((2 * pair_idx + 1) * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    K_base = K + qvk_offset
    V_base = V + qvk_offset
    m_i1 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i1 = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc1 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i2 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i2 = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc2 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q1 = tl.load(Q1_bp)
    q2 = tl.load(Q2_bp)
    q1 = (q1 * (sm_scale * 1.44269504)).to(q1.dtype)
    q2 = (q2 * (sm_scale * 1.44269504)).to(q2.dtype)

    n_full = tl.load(NDualFull + pair_idx)
    n_total = tl.load(NDualTotal + pair_idx)
    ci_base = DualColIndices + pair_idx * stride_dci_row

    acc1, l_i1, m_i1, acc2, l_i2, m_i2 = _dualq_fwd_inner(
        acc1, l_i1, m_i1, acc2, l_i2, m_i2, q1, q2,
        K_base, V_base, MaskPacked,
        stride_kk, stride_kn, stride_vk, stride_vn, stride_mp_m, stride_mp_n,
        ci_base, n_full, n_total, pair_idx,
        BLOCK_M, BLOCK_N, HEAD_DIM,
    )
    l_i1 = tl.where(l_i1 == 0.0, 1.0, l_i1)
    l_i2 = tl.where(l_i2 == 0.0, 1.0, l_i2)
    acc1 = acc1 / l_i1[:, None]
    acc2 = acc2 / l_i2[:, None]
    offs_m1 = 2 * pair_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m2 = (2 * pair_idx + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(LSE + (off_hz * N_CTX) + offs_m1, m_i1 + tl.math.log2(l_i1))
    tl.store(LSE + (off_hz * N_CTX) + offs_m2, m_i2 + tl.math.log2(l_i2))
    tl.store(O1_bp, acc1.to(Out.type.element_ty))
    tl.store(O2_bp, acc2.to(Out.type.element_ty))


# ─── Dispatch & cache ───

_bm_cache_key = None
_bm_cache_val = None

_GAP_RATIO_THRESHOLD = 0.3
_DENSITY_THRESHOLD_BN128 = 0.9
_DENSITY_THRESHOLD_PACKED = 0.8
_PARTIAL_FRACTION_THRESHOLD = 0.5


def _dispatch_and_preprocess(mask: torch.Tensor, D: int):
    N = mask.shape[0]
    out = _fused_preproc_multigrain(mask, 64)
    gap_ratio = out["gap_ratio"]
    density = out["density"]
    partial_fraction = out["partial_fraction"]
    mask_packed = out["mask_packed"]

    if D == 64 and gap_ratio > _GAP_RATIO_THRESHOLD and N % 64 == 0:
        # Path 1: gap-heavy (longformer, cdw_sinks) → BM=64 BN=64
        block_mask = out["block_mask_fine"]
        col_indices, n_full, n_total = _build_gathered_indices(block_mask)
        return (
            col_indices,
            n_full,
            n_total,
            mask_packed,
            64,
            64,
            1,  # use_packed always on path 1
        )

    if D == 64 and density > _DENSITY_THRESHOLD_BN128 and N % 128 == 0:
        # Path 2: dense BN=128
        block_mask = out["block_mask_bn128"]
        col_indices, n_full, n_total = _build_gathered_indices(block_mask)
        return col_indices, n_full, n_total, mask_packed, 128, 128, 1

    use_packed = (
        1
        if (density > _DENSITY_THRESHOLD_PACKED or gap_ratio > _GAP_RATIO_THRESHOLD or partial_fraction > _PARTIAL_FRACTION_THRESHOLD)
        else 0
    )
    if not use_packed:
        mask_packed = None
    block_mask = out["block_mask_coarse"]
    col_indices, n_full, n_total = _build_gathered_indices(block_mask)
    return (
        col_indices,
        n_full,
        n_total,
        mask_packed,
        128,
        64,
        use_packed,
    )


def preproc(mask: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float):
    """Generic preprocessing entry point for benchmarking."""
    D = q.shape[-1]
    return _dispatch_and_preprocess(mask, D)


_dual_cache_key = None
_dual_cache_val = None


def binflash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, sm_scale: float | None = None
) -> torch.Tensor:
    """V252 dual-Q prototype: always uses dual-Q kernel at BM=64 per Q-block."""
    global _dual_cache_key, _dual_cache_val
    B, H, N, D = q.shape
    BM = 64  # per-Q-block M (effective BM=128 per CTA)
    BN = 64
    if sm_scale is None:
        sm_scale = D**-0.5
    if D not in {16, 32, 64, 128}:
        raise ValueError(f"HEAD_DIM ({D}) must be one of {{16, 32, 64, 128}}")
    assert N % (2 * BM) == 0, f"N ({N}) must be divisible by 2*BM ({2*BM}) for dual-Q"

    cache_key = (mask.data_ptr(), mask.shape[0], BM, BN, D)
    if cache_key == _dual_cache_key:
        dual_col_indices, n_dual_full, n_dual_total, mask_packed = _dual_cache_val
    else:
        out_preproc = _fused_preproc_multigrain(mask, BN)
        mask_packed = out_preproc["mask_packed"]
        # Use block_mask_fine (BM=64) for dual-Q
        bm_fine = out_preproc["block_mask_fine"]
        dual_col_indices, n_dual_full, n_dual_total = _build_dual_gathered_indices(bm_fine)
        _dual_cache_key = cache_key
        _dual_cache_val = (dual_col_indices, n_dual_full, n_dual_total, mask_packed)

    o = torch.empty_like(q)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    dual_nqb = N // (2 * BM)
    grid = (dual_nqb, B * H)
    _dualq_fwd[grid](
        q, k, v, sm_scale, o, mask_packed,
        dual_col_indices, n_dual_full, n_dual_total, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        mask_packed.stride(0), mask_packed.stride(1),
        dual_col_indices.stride(0),
        B, H, N, B * H,
        HEAD_DIM=D, BLOCK_M=BM, BLOCK_N=BN,
    )
    return o


def _binflash_attention_single(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, sm_scale: float | None = None
) -> torch.Tensor:
    """Original single-Q dispatch (unused in V252 prototype)."""
    global _bm_cache_key, _bm_cache_val
    B, H, N, D = q.shape
    block_n = min(D, 64)
    if sm_scale is None:
        sm_scale = D**-0.5
    if D not in {16, 32, 64, 128}:
        raise ValueError(f"HEAD_DIM ({D}) must be one of {{16, 32, 64, 128}}")

    key = (mask.data_ptr(), mask.shape[0], mask.shape[1], block_n, D)
    if key == _bm_cache_key:
        (
            col_indices,
            n_full,
            n_total,
            mask_packed,
            chosen_bm,
            chosen_bn,
            use_packed,
        ) = _bm_cache_val
    else:
        cache_val = _dispatch_and_preprocess(mask, D)
        _bm_cache_key = key
        _bm_cache_val = cache_val
        (
            col_indices,
            n_full,
            n_total,
            mask_packed,
            chosen_bm,
            chosen_bn,
            use_packed,
        ) = cache_val

    if N % chosen_bm != 0 or N % chosen_bn != 0:
        raise ValueError(f"N ({N}) must be divisible by chosen_bm ({chosen_bm}) and chosen_bn ({chosen_bn})")

    o = torch.empty_like(q)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(N, chosen_bm), B * H)  # noqa: E731

    if mask_packed is None:
        mp_tensor = mask
        mp_stride_0, mp_stride_1 = 0, 0
    else:
        mp_tensor = mask_packed
        mp_stride_0, mp_stride_1 = mp_tensor.stride(0), mp_tensor.stride(1)

    _gathered_fwd[grid](
        q,
        k,
        v,
        sm_scale,
        o,
        mask,
        mp_tensor,
        col_indices,
        n_full,
        n_total,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        mask.stride(0),
        mask.stride(1),
        mp_stride_0,
        mp_stride_1,
        col_indices.stride(0),
        B,
        H,
        N,
        B * H,
        HEAD_DIM=D,
        BLOCK_M=chosen_bm,
        BLOCK_N=chosen_bn,
        USE_PACKED=use_packed,
    )
    return o
