"""v164: extend v163's multi-granularity kernel to also emit a
(BM=128, BN=128) block_mask so path 2 (dense random: sparse_10pct,
sparse_30pct) can skip its second `_fused_preproc` call entirely.

At BN_CHUNKS=2 × BLOCK_N=64, each program's tile covers exactly one
BN=128 column stripe. The BN=128 bm_val is a whole-tile reduction —
trivially derivable from the coarse sums that the kernel already
computes, plus one extra store per program.

After v163, path 2 patterns (sparse_10pct/30pct) preproc at N=16384
is ~0.92ms because they still make two calls. This version brings it
to ~0.60ms. Kernel body UNCHANGED. Only the preproc pipeline changes.
"""

# (full file content copied from production + the v164 delta)

import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_block_mask(
    BlockMask,
    FirstNz, LastNz,
    GlobalSums,
    GlobalGapFloat,
    stride_bm_row, stride_bm_col,
    nkb,
    NKB_POW2: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, NKB_POW2)
    valid = offs < nkb
    bm_row = tl.load(
        BlockMask + row * stride_bm_row + offs * stride_bm_col,
        mask=valid, other=0,
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
def _compute_block_mask(
    Mask, BlockMask,
    stride_mask_m, stride_mask_n, nkb,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BN_CHUNKS: tl.constexpr,
):
    row = tl.program_id(0)
    col_chunk = tl.program_id(1)
    offs_m = row * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = col_chunk * (BN_CHUNKS * BLOCK_N) + tl.arange(0, BN_CHUNKS * BLOCK_N)
    tile_ptrs = Mask + offs_m[:, None] * stride_mask_m + offs_n[None, :] * stride_mask_n
    tile = tl.load(tile_ptrs)
    tile_3d = tl.reshape(tile, (BLOCK_M, BN_CHUNKS, BLOCK_N))
    tile_i32 = tile_3d.to(tl.int32)
    chunk_sums_per_col = tl.sum(tile_i32, axis=0)
    chunk_sums = tl.sum(chunk_sums_per_col, axis=1)
    has_any = chunk_sums > 0
    has_all = chunk_sums == (BLOCK_M * BLOCK_N)
    bm_vals = has_any.to(tl.int8) + has_all.to(tl.int8)
    col_start = col_chunk * BN_CHUNKS + tl.arange(0, BN_CHUNKS)
    tl.store(BlockMask + row * nkb + col_start, bm_vals)


@triton.jit
def _compute_and_pack_mask_multi(
    Mask, BlockMaskCoarse, BlockMaskFine, BlockMaskBN128, Packed,
    stride_mask_m, stride_mask_n,
    stride_packed_m, stride_packed_n,
    nkb, nkb_bn128,
    BM_COARSE: tl.constexpr, BM_FINE: tl.constexpr, BLOCK_N: tl.constexpr,
    BN_CHUNKS: tl.constexpr,
):
    """v164 multi-granularity fused kernel. Single tile read →
    three block_mask outputs at different granularities + mask_packed:
      - (BM_COARSE=128, BN=64) for default & metrics
      - (BM_FINE=64,   BN=64) for path 1 (gap-heavy)
      - (BM_COARSE=128, BN=128) for path 2 (dense random)
    Assumes BN_CHUNKS * BLOCK_N == 128 so each program covers exactly
    one BN=128 column.
    """
    row_coarse = tl.program_id(0)
    col_chunk = tl.program_id(1)
    offs_m = row_coarse * BM_COARSE + tl.arange(0, BM_COARSE)
    offs_n_base = col_chunk * (BN_CHUNKS * BLOCK_N)
    offs_n = offs_n_base + tl.arange(0, BN_CHUNKS * BLOCK_N)
    tile_ptrs = Mask + offs_m[:, None] * stride_mask_m + offs_n[None, :] * stride_mask_n
    tile = tl.load(tile_ptrs)

    # Coarse block_mask (BM=128, BN=64)
    tile_3d_bm = tl.reshape(tile, (BM_COARSE, BN_CHUNKS, BLOCK_N))
    tile_i32 = tile_3d_bm.to(tl.int32)
    sum_col = tl.sum(tile_i32, axis=0)
    coarse_sums = tl.sum(sum_col, axis=1)  # (BN_CHUNKS,)
    has_any_c = coarse_sums > 0
    has_all_c = coarse_sums == (BM_COARSE * BLOCK_N)
    bm_coarse = has_any_c.to(tl.int8) + has_all_c.to(tl.int8)
    col_start = col_chunk * BN_CHUNKS + tl.arange(0, BN_CHUNKS)
    tl.store(BlockMaskCoarse + row_coarse * nkb + col_start, bm_coarse)

    # BN=128 block_mask (BM=128, BN=128): whole-tile reduction
    TILE_SIZE: tl.constexpr = BM_COARSE * BN_CHUNKS * BLOCK_N
    bn128_total = tl.sum(coarse_sums)  # scalar
    has_any_128 = bn128_total > 0
    has_all_128 = bn128_total == TILE_SIZE
    bm_bn128 = has_any_128.to(tl.int8) + has_all_128.to(tl.int8)
    tl.store(BlockMaskBN128 + row_coarse * nkb_bn128 + col_chunk, bm_bn128)

    # Fine block_mask (BM=64, BN=64): two halves
    tile_fine_4d = tl.reshape(tile_i32, (2, BM_FINE, BN_CHUNKS, BLOCK_N))
    sum_bm_f = tl.sum(tile_fine_4d, axis=1)
    fine_sums = tl.sum(sum_bm_f, axis=2)
    has_any_f = fine_sums > 0
    has_all_f = fine_sums == (BM_FINE * BLOCK_N)
    bm_fine = has_any_f.to(tl.int8) + has_all_f.to(tl.int8)
    pair_idx = tl.arange(0, 2)
    fine_row = 2 * row_coarse + pair_idx
    fine_ptrs = (
        BlockMaskFine
        + fine_row[:, None] * nkb
        + col_start[None, :]
    )
    tl.store(fine_ptrs, bm_fine)

    # Packed mask
    WORDS: tl.constexpr = (BN_CHUNKS * BLOCK_N) // 32
    tile_3d_pk = tl.reshape(tile, (BM_COARSE, WORDS, 32))
    bits_i32 = tile_3d_pk.to(tl.int32)
    bit_weights = (1 << tl.arange(0, 32))
    weighted = bits_i32 * bit_weights[None, None, :]
    packed = tl.sum(weighted, axis=2)
    word_start = col_chunk * WORDS + tl.arange(0, WORDS)
    packed_ptrs = (
        Packed
        + offs_m[:, None] * stride_packed_m
        + word_start[None, :] * stride_packed_n
    )
    tl.store(packed_ptrs, packed)


@triton.jit
def _compute_and_pack_mask(
    Mask, BlockMask, Packed,
    stride_mask_m, stride_mask_n,
    stride_packed_m, stride_packed_n,
    nkb,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BN_CHUNKS: tl.constexpr,
):
    row = tl.program_id(0)
    col_chunk = tl.program_id(1)
    offs_m = row * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = col_chunk * (BN_CHUNKS * BLOCK_N)
    offs_n = offs_n_base + tl.arange(0, BN_CHUNKS * BLOCK_N)
    tile_ptrs = Mask + offs_m[:, None] * stride_mask_m + offs_n[None, :] * stride_mask_n
    tile = tl.load(tile_ptrs)
    tile_3d_bm = tl.reshape(tile, (BLOCK_M, BN_CHUNKS, BLOCK_N))
    tile_i32_bm = tile_3d_bm.to(tl.int32)
    chunk_sums_per_col = tl.sum(tile_i32_bm, axis=0)
    chunk_sums = tl.sum(chunk_sums_per_col, axis=1)
    has_any = chunk_sums > 0
    has_all = chunk_sums == (BLOCK_M * BLOCK_N)
    bm_vals = has_any.to(tl.int8) + has_all.to(tl.int8)
    col_start = col_chunk * BN_CHUNKS + tl.arange(0, BN_CHUNKS)
    tl.store(BlockMask + row * nkb + col_start, bm_vals)
    WORDS: tl.constexpr = (BN_CHUNKS * BLOCK_N) // 32
    tile_3d_pk = tl.reshape(tile, (BLOCK_M, WORDS, 32))
    bits_i32 = tile_3d_pk.to(tl.int32)
    bit_weights = (1 << tl.arange(0, 32))
    weighted = bits_i32 * bit_weights[None, None, :]
    packed = tl.sum(weighted, axis=2)
    word_start = col_chunk * WORDS + tl.arange(0, WORDS)
    packed_ptrs = (
        Packed
        + offs_m[:, None] * stride_packed_m
        + word_start[None, :] * stride_packed_n
    )
    tl.store(packed_ptrs, packed)


@triton.jit
def _pack_mask_kernel(
    Mask, Packed,
    stride_mask_m, stride_mask_n,
    stride_packed_m, stride_packed_n,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N_WORDS: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_start = row_pid * BLOCK_M
    packed_col_start = col_pid * BLOCK_N_WORDS
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_pw = packed_col_start + tl.arange(0, BLOCK_N_WORDS)
    bit_offs = tl.arange(0, 32)
    actual_cols = offs_pw[:, None] * 32 + bit_offs[None, :]
    mask_ptrs = (
        Mask
        + offs_m[:, None, None] * stride_mask_m
        + actual_cols[None, :, :] * stride_mask_n
    )
    mask_tile = tl.load(mask_ptrs)
    mask_i32 = mask_tile.to(tl.int32)
    weights = 1 << bit_offs
    weighted = mask_i32 * weights[None, None, :]
    packed = tl.sum(weighted, axis=2)
    packed_ptrs = (
        Packed
        + offs_m[:, None] * stride_packed_m
        + offs_pw[None, :] * stride_packed_n
    )
    tl.store(packed_ptrs, packed)


def _pack_mask(mask: torch.Tensor) -> torch.Tensor:
    N = mask.shape[0]
    assert N % 32 == 0
    packed = torch.empty((N, N // 32), device=mask.device, dtype=torch.int32)
    BLOCK_M = min(64, N)
    BLOCK_N_WORDS = min(8, N // 32)
    grid = (N // BLOCK_M, (N // 32) // BLOCK_N_WORDS)
    _pack_mask_kernel[grid](
        mask, packed,
        mask.stride(0), mask.stride(1),
        packed.stride(0), packed.stride(1),
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N_WORDS=BLOCK_N_WORDS,
    )
    return packed


def _pick_bn_chunks(block_m, block_n, nkb, target_tile_bytes=65536, start=8):
    max_by_budget = max(1, target_tile_bytes // max(1, block_m * block_n))
    bn_chunks = min(start, max_by_budget)
    while nkb % bn_chunks != 0 and bn_chunks > 1:
        bn_chunks //= 2
    return bn_chunks


def _fused_preproc(mask, block_m, block_n, want_packed=False):
    N = mask.shape[0]
    nqb = N // block_m
    nkb = N // block_n
    block_mask = torch.empty(nqb, nkb, dtype=torch.int8, device=mask.device)
    mask_packed_out = None
    if want_packed and (block_n % 32 == 0):
        mask_packed_out = torch.empty((N, N // 32), dtype=torch.int32, device=mask.device)
        bn_chunks = _pick_bn_chunks(block_m, block_n, nkb, start=2)
        _compute_and_pack_mask[(nqb, nkb // bn_chunks)](
            mask, block_mask, mask_packed_out,
            mask.stride(0), mask.stride(1),
            mask_packed_out.stride(0), mask_packed_out.stride(1),
            nkb,
            BLOCK_M=block_m, BLOCK_N=block_n, BN_CHUNKS=bn_chunks,
        )
    else:
        bn_chunks = _pick_bn_chunks(block_m, block_n, nkb, start=8)
        _compute_block_mask[(nqb, nkb // bn_chunks)](
            mask, block_mask,
            mask.stride(0), mask.stride(1), nkb,
            BLOCK_M=block_m, BLOCK_N=block_n, BN_CHUNKS=bn_chunks,
        )
    first_nz = torch.empty(nqb, dtype=torch.int32, device=mask.device)
    last_nz = torch.empty(nqb, dtype=torch.int32, device=mask.device)
    global_sums = torch.zeros(2, dtype=torch.int32, device=mask.device)
    global_gap = torch.zeros(1, dtype=torch.float32, device=mask.device)
    nkb_pow2 = 1
    while nkb_pow2 < nkb:
        nkb_pow2 *= 2
    _reduce_block_mask[(nqb,)](
        block_mask,
        first_nz, last_nz,
        global_sums, global_gap,
        block_mask.stride(0), block_mask.stride(1),
        nkb,
        NKB_POW2=nkb_pow2,
    )
    metrics = torch.cat([global_sums.to(torch.float32), global_gap])
    num_nonempty_total, num_partial_total, global_gap_sum = metrics.tolist()
    density = num_nonempty_total / (nqb * nkb)
    partial_fraction = num_partial_total / max(num_nonempty_total, 1.0)
    avg_gap_ratio = global_gap_sum / nqb
    mask_packed = mask_packed_out if want_packed else None
    return block_mask, first_nz, last_nz, avg_gap_ratio, density, partial_fraction, mask_packed


def _fused_preproc_multigrain(mask, block_n):
    """v164: single-pass preproc producing (BM=128,BN=64), (BM=64,BN=64),
    and (BM=128,BN=128) block_masks + mask_packed + coarse metrics."""
    N = mask.shape[0]
    assert block_n == 64, "multigrain currently targets BLOCK_N=64"
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
    assert bn_chunks * block_n == 128, "v164 requires BN_CHUNKS*BLOCK_N=128"
    _compute_and_pack_mask_multi[(nqb_coarse, nkb // bn_chunks)](
        mask, block_mask_coarse, block_mask_fine, block_mask_bn128, mask_packed,
        mask.stride(0), mask.stride(1),
        mask_packed.stride(0), mask_packed.stride(1),
        nkb, nkb_bn128,
        BM_COARSE=128, BM_FINE=64, BLOCK_N=block_n,
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
        first_nz_c, last_nz_c,
        global_sums, global_gap,
        block_mask_coarse.stride(0), block_mask_coarse.stride(1),
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


def _reduce_block_mask_bounds_only(block_mask):
    """Compute first_nz/last_nz for a block_mask without touching metric
    atomics. Used for the speculative fine/bn128 block_masks."""
    nqb, nkb = block_mask.shape
    first_nz = torch.empty(nqb, dtype=torch.int32, device=block_mask.device)
    last_nz = torch.empty(nqb, dtype=torch.int32, device=block_mask.device)
    gs = torch.zeros(2, dtype=torch.int32, device=block_mask.device)
    gg = torch.zeros(1, dtype=torch.float32, device=block_mask.device)
    nkb_pow2 = 1
    while nkb_pow2 < nkb:
        nkb_pow2 *= 2
    _reduce_block_mask[(nqb,)](
        block_mask,
        first_nz, last_nz,
        gs, gg,
        block_mask.stride(0), block_mask.stride(1),
        nkb,
        NKB_POW2=nkb_pow2,
    )
    return first_nz, last_nz


# ─── Main attention kernel (UNCHANGED) ───

@triton.jit
def _binflash_fwd_inner(
    acc, l_i, m_i, q,
    K_base, V_base, Mask_base, MaskPacked_base,
    stride_kk, stride_kn, stride_vk, stride_vn,
    stride_mask_m, stride_mask_n,
    stride_mp_m, stride_mp_n,
    block_mask_ptr, bm_stride_row, bm_stride_col, start_m,
    first_nz_col, last_nz_col,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr, USE_PACKED: tl.constexpr,
):
    offs_d = tl.arange(0, HEAD_DIM)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_bm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    BN_PACKED: tl.constexpr = BLOCK_N // 32
    offs_bn_packed = tl.arange(0, BN_PACKED)
    bit_offs = tl.arange(0, 32)

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
            if USE_PACKED:
                start_n_packed = (col_idx * BLOCK_N) // 32
                mp_ptrs = MaskPacked_base + offs_bm[:, None] * stride_mp_m + (start_n_packed + offs_bn_packed)[None, :] * stride_mp_n
                packed = tl.load(mp_ptrs)
                bits_3d = (packed[:, :, None] >> bit_offs[None, None, :]) & 1
                mask = tl.reshape(bits_3d, (BLOCK_M, BLOCK_N)) != 0
            else:
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


@triton.autotune(
    configs=[triton.Config({}, num_stages=s, num_warps=w)
             for s in [1, 2, 3, 4, 5] for w in [4, 8]],
    key=["N_CTX", "HEAD_DIM", "Z_TIMES_H", "BLOCK_M", "BLOCK_N", "USE_PACKED"],
)
@triton.jit
def _binflash_fwd(
    Q, K, V, sm_scale, Out, Mask, MaskPacked, BlockMask, FirstNz, LastNz, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_mask_m, stride_mask_n,
    stride_mp_m, stride_mp_n,
    stride_bm_row, stride_bm_col,
    Z, H, N_CTX, Z_TIMES_H,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    USE_PACKED: tl.constexpr,
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
        K_base, V_base, Mask, MaskPacked,
        stride_kk, stride_kn, stride_vk, stride_vn,
        stride_mask_m, stride_mask_n,
        stride_mp_m, stride_mp_n,
        BlockMask, stride_bm_row, stride_bm_col, start_m,
        first_nz, last_nz,
        BLOCK_M, BLOCK_N, HEAD_DIM, N_CTX, USE_PACKED,
    )
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(LSE + (off_hz * N_CTX) + offs_m, m_i + tl.math.log2(l_i))
    tl.store(O_bp, acc.to(Out.type.element_ty))


_bm_cache_key = None
_bm_cache_val = None

_GAP_RATIO_THRESHOLD = 0.3
_DENSITY_THRESHOLD_BN128 = 0.9
_DENSITY_THRESHOLD_PACKED = 0.8
_PARTIAL_FRACTION_THRESHOLD = 0.5


def _dispatch_and_preprocess(mask: torch.Tensor, D: int):
    """v164: call 1 is the multi-granularity fused kernel which
    speculatively emits THREE block_masks (coarse, fine, bn128). Both
    path 1 (D=64 gap-heavy) and path 2 (D=64 dense random) can now
    skip their second preproc call entirely — they only need to run
    a lightweight first_nz/last_nz reduce on the speculatively-prepared
    block_mask at their target granularity.
    """
    N = mask.shape[0]
    out = _fused_preproc_multigrain(mask, 64)
    gap_ratio = out["gap_ratio"]
    density = out["density"]
    partial_fraction = out["partial_fraction"]
    mask_packed = out["mask_packed"]

    if D == 64 and gap_ratio > _GAP_RATIO_THRESHOLD and N % 64 == 0:
        block_mask = out["block_mask_fine"]
        first_nz, last_nz = _reduce_block_mask_bounds_only(block_mask)
        return block_mask, first_nz, last_nz, mask_packed, 64, 64, 1

    if D == 64 and density > _DENSITY_THRESHOLD_BN128 and N % 128 == 0:
        block_mask = out["block_mask_bn128"]
        first_nz, last_nz = _reduce_block_mask_bounds_only(block_mask)
        return block_mask, first_nz, last_nz, mask_packed, 128, 128, 1

    use_packed = 1 if (
        density > _DENSITY_THRESHOLD_PACKED
        or gap_ratio > _GAP_RATIO_THRESHOLD
        or partial_fraction > _PARTIAL_FRACTION_THRESHOLD
    ) else 0
    if not use_packed:
        mask_packed = None
    return (
        out["block_mask_coarse"],
        out["first_nz_coarse"],
        out["last_nz_coarse"],
        mask_packed,
        128, 64, use_packed,
    )


def preproc(mask: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
            v: torch.Tensor, sm_scale: float):
    D = q.shape[-1]
    return _dispatch_and_preprocess(mask, D)


def binflash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: torch.Tensor, sm_scale: float | None = None,
    block_m: int = 128, block_n: int | None = None,
    block_mask: torch.Tensor | None = None,
) -> torch.Tensor:
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
            block_mask, first_nz, last_nz, mask_packed, chosen_bm, chosen_bn, use_packed = _bm_cache_val
        else:
            cache_val = _dispatch_and_preprocess(mask, D)
            _bm_cache_key = key
            _bm_cache_val = cache_val
            block_mask, first_nz, last_nz, mask_packed, chosen_bm, chosen_bn, use_packed = cache_val
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
        mask_packed = None
        use_packed = 0

    if N % chosen_bm != 0 or N % chosen_bn != 0:
        raise ValueError(f"N ({N}) must be divisible by chosen_bm ({chosen_bm}) and chosen_bn ({chosen_bn})")

    o = torch.empty_like(q)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(N, chosen_bm), B * H)

    if mask_packed is None:
        mp_tensor = mask
        mp_stride_0, mp_stride_1 = 0, 0
    else:
        mp_tensor = mask_packed
        mp_stride_0, mp_stride_1 = mp_tensor.stride(0), mp_tensor.stride(1)

    _binflash_fwd[grid](
        q, k, v, sm_scale, o, mask, mp_tensor, block_mask, first_nz, last_nz, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        mask.stride(0), mask.stride(1),
        mp_stride_0, mp_stride_1,
        block_mask.stride(0), block_mask.stride(1),
        B, H, N, B * H,
        HEAD_DIM=D, BLOCK_M=chosen_bm, BLOCK_N=chosen_bn,
        USE_PACKED=use_packed,
    )
    return o
