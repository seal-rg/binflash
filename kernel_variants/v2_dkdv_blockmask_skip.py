"""v2 dkdv_blockmask_skip — self-contained. Both dQ and dK/dV kernels now
use block-mask skipping. dK/dV iterates a transposed col_indices (per K-block,
list of Q-blocks that attend to it)."""


import torch  # type: ignore
import triton  # type: ignore
import triton.language as tl  # type: ignore

# ─── Fused preprocessing ───


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
    COMPUTE_FINE: tl.constexpr,  # V266: conditionally skip unused outputs
    COMPUTE_BN128: tl.constexpr,
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

    if COMPUTE_BN128:
        TILE_SIZE: tl.constexpr = BM_COARSE * BN_CHUNKS * BLOCK_N
        bn128_total = tl.sum(coarse_sums)
        has_any_128 = bn128_total > 0
        has_all_128 = bn128_total == TILE_SIZE
        bm_bn128 = has_any_128.to(tl.int8) + has_all_128.to(tl.int8)
        tl.store(BlockMaskBN128 + row_coarse * nkb_bn128 + col_chunk, bm_bn128)

    if COMPUTE_FINE:
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


def _fused_preproc_multigrain(mask, block_n, compute_fine=False, compute_bn128=False):
    N = mask.shape[0]
    assert block_n == 64
    assert N % 128 == 0 and N % 64 == 0
    nkb = N // block_n
    nkb_bn128 = N // 128
    nqb_coarse = N // 128
    nqb_fine = N // 64

    block_mask_coarse = torch.empty(nqb_coarse, nkb, dtype=torch.int8, device=mask.device)
    # V266: allocate fine/bn128 only if requested
    if compute_fine:
        block_mask_fine = torch.empty(nqb_fine, nkb, dtype=torch.int8, device=mask.device)
    else:
        block_mask_fine = block_mask_coarse  # dummy, never written/read
    if compute_bn128:
        block_mask_bn128 = torch.empty(nqb_coarse, nkb_bn128, dtype=torch.int8, device=mask.device)
    else:
        block_mask_bn128 = block_mask_coarse  # dummy
    # V268: transposed physical layout for coalesced per-col access across BM rows
    mask_packed = torch.empty((N // 32, N), dtype=torch.int32, device=mask.device).T

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
        COMPUTE_FINE=compute_fine,
        COMPUTE_BN128=compute_bn128,
    )

    # V262: metrics will be computed inside _build_gathered_indices_merged (single kernel, no separate _reduce_block_mask launch)
    return {
        "block_mask_coarse": block_mask_coarse,
        "block_mask_fine": block_mask_fine,
        "block_mask_bn128": block_mask_bn128,
        "mask_packed": mask_packed,
        "nqb_coarse": nqb_coarse,
        "nkb": nkb,
    }


# ─── Gathered indices preprocessing


@triton.jit
def _build_col_indices_kernel(
    BlockMask,
    ColIndices,
    NFull,
    NTotal,
    GlobalSums,  # (2,) int32 — [num_nonempty, num_partial] (V262)
    GlobalGapFloat,  # (1,) float32 — sum of gap_for_row  (V262)
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
    is_nonempty = is_full | is_partial

    n_full = tl.sum(is_full.to(tl.int32))
    n_partial = tl.sum(is_partial.to(tl.int32))
    n_total = n_full + n_partial
    tl.store(NFull + row, n_full)
    tl.store(NTotal + row, n_total)

    # V262: compute metrics inline (fold _reduce_block_mask into here)
    has_any = n_total > 0
    ranked_last = tl.where(is_nonempty, offs, -1)
    last_nz = tl.max(ranked_last, axis=0)
    ranked_first = tl.where(is_nonempty, offs, nkb)
    first_nz = tl.where(has_any, tl.min(ranked_first, axis=0), 0)
    scan_width = tl.where(has_any, last_nz - first_nz + 1, 1)
    gap_for_row = 1.0 - n_total.to(tl.float32) / scan_width.to(tl.float32)
    tl.atomic_add(GlobalSums + 0, n_total)
    tl.atomic_add(GlobalSums + 1, n_partial)
    tl.atomic_add(GlobalGapFloat, gap_for_row)

    full_rank = tl.cumsum(is_full.to(tl.int32), axis=0) - 1
    partial_rank = tl.cumsum(is_partial.to(tl.int32), axis=0) - 1 + n_full
    target_rank = tl.where(is_full, full_rank, partial_rank)
    ci_ptrs = ColIndices + row * stride_ci_row + target_rank
    tl.store(ci_ptrs, offs.to(tl.int32), mask=is_nonempty)


def _build_gathered_indices_merged(block_mask):
    """V262 merged: returns col_indices, counts, AND metrics in one kernel launch."""
    nqb, nkb = block_mask.shape
    nkb_pow2 = 1
    while nkb_pow2 < nkb:
        nkb_pow2 *= 2

    n_full = torch.empty(nqb, dtype=torch.int32, device=block_mask.device)
    n_total = torch.empty(nqb, dtype=torch.int32, device=block_mask.device)
    col_indices = torch.empty(nqb, nkb, dtype=torch.int32, device=block_mask.device)
    global_sums = torch.zeros(2, dtype=torch.int32, device=block_mask.device)
    global_gap = torch.zeros(1, dtype=torch.float32, device=block_mask.device)

    _build_col_indices_kernel[(nqb,)](
        block_mask,
        col_indices,
        n_full,
        n_total,
        global_sums,
        global_gap,
        block_mask.stride(0),
        block_mask.stride(1),
        col_indices.stride(0),
        nkb,
        NKB_POW2=nkb_pow2,
    )
    return col_indices, n_full, n_total, global_sums, global_gap


def _build_gathered_indices(block_mask):
    col_indices, n_full, n_total, _, _ = _build_gathered_indices_merged(block_mask)
    return col_indices, n_full, n_total


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
    for idx in tl.range(0, n_full, num_stages=3):
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
    for idx in tl.range(n_full, n_total, num_stages=3):
        col_idx = tl.load(col_indices_ptr + idx)
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
    tl.assume(n_full >= 0)
    tl.assume(n_total >= n_full)
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


# ─── Dispatch & cache ───

_bm_cache_key = None
_bm_cache_val = None

_GAP_RATIO_THRESHOLD = 0.3
_DENSITY_THRESHOLD_BN128 = 0.9
_DENSITY_THRESHOLD_PACKED = 0.8
_PARTIAL_FRACTION_THRESHOLD = 0.5


def _dispatch_and_preprocess(mask: torch.Tensor, D: int):
    N = mask.shape[0]
    # V266: for D=128 workloads, don't compute fine/bn128 block_masks (unused).
    need_fine_bn128 = D == 64
    out = _fused_preproc_multigrain(mask, 64, compute_fine=need_fine_bn128, compute_bn128=need_fine_bn128)
    mask_packed = out["mask_packed"]
    nqb_coarse = out["nqb_coarse"]
    nkb = out["nkb"]

    if D == 128:
        block_mask = out["block_mask_coarse"]
        col_indices, n_full, n_total, _, _ = _build_gathered_indices_merged(block_mask)
        return col_indices, n_full, n_total, mask_packed, 128, 64, 1

    # D=64 path: need the metrics for Path 1/2 dispatch. Keep the old logic.
    block_mask = out["block_mask_coarse"]
    col_indices, n_full, n_total, global_sums, global_gap = _build_gathered_indices_merged(block_mask)
    metrics = torch.cat([global_sums.to(torch.float32), global_gap])
    num_nonempty_total, num_partial_total, global_gap_sum = metrics.tolist()
    density = num_nonempty_total / (nqb_coarse * nkb)
    partial_fraction = num_partial_total / max(num_nonempty_total, 1.0)
    gap_ratio = global_gap_sum / nqb_coarse

    if D == 64 and gap_ratio > _GAP_RATIO_THRESHOLD and N % 64 == 0:
        block_mask_fine = out["block_mask_fine"]
        col_indices, n_full, n_total = _build_gathered_indices(block_mask_fine)
        return col_indices, n_full, n_total, mask_packed, 64, 64, 1

    if D == 64 and density > _DENSITY_THRESHOLD_BN128 and N % 128 == 0:
        block_mask_bn128 = out["block_mask_bn128"]
        col_indices, n_full, n_total = _build_gathered_indices(block_mask_bn128)
        return col_indices, n_full, n_total, mask_packed, 128, 128, 1

    use_packed = (
        1
        if (density > _DENSITY_THRESHOLD_PACKED or gap_ratio > _GAP_RATIO_THRESHOLD or partial_fraction > _PARTIAL_FRACTION_THRESHOLD)
        else 0
    )
    if not use_packed:
        mask_packed = None
    return col_indices, n_full, n_total, mask_packed, 128, 64, use_packed


def preproc(mask: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float):
    """Generic preprocessing entry point for benchmarking."""
    D = q.shape[-1]
    return _dispatch_and_preprocess(mask, D)


def binflash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, sm_scale: float | None = None
) -> torch.Tensor:
    """V192: gathered block dispatch attention."""
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

# ─── BACKWARD PASS ───


import torch  # type: ignore
import triton  # type: ignore
import triton.language as tl  # type: ignore



# ─── Preprocessing: per-row delta = sum(dO * O) ───

@triton.jit
def _bwd_preprocess_kernel(
    Out,
    DO,
    Delta,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_doz, stride_doh, stride_dom, stride_don,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    o_ptrs = Out + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    do_ptrs = DO + off_z * stride_doz + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_don
    o = tl.load(o_ptrs).to(tl.float32)
    do = tl.load(do_ptrs).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + offs_m, delta)


# ─── dK, dV kernel (K-outer) ───

@triton.jit
def _bwd_dkdv_kernel(
    Q, K, V, sm_scale,
    DO, DK, DV,
    Mask,
    LSE, Delta,
    RowIndicesKT, NFullKT, NTotalKT,  # v2: transposed col_indices — per K-block, list of Q-blocks
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_doz, stride_doh, stride_dom, stride_don,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvk, stride_dvn,
    stride_mask_m, stride_mask_n,
    stride_ri_row,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    start_n = tl.program_id(0)  # which K-block
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qvk_z_off_q = off_z * stride_qz + off_h * stride_qh
    qvk_z_off_k = off_z * stride_kz + off_h * stride_kh
    qvk_z_off_v = off_z * stride_vz + off_h * stride_vh
    qvk_z_off_do = off_z * stride_doz + off_h * stride_doh
    qvk_z_off_dk = off_z * stride_dkz + off_h * stride_dkh
    qvk_z_off_dv = off_z * stride_dvz + off_h * stride_dvh

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    k_ptrs = K + qvk_z_off_k + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + qvk_z_off_v + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    n_full = tl.load(NFullKT + start_n)
    n_total = tl.load(NTotalKT + start_n)
    tl.assume(n_full >= 0)
    tl.assume(n_total >= n_full)
    ri_base = RowIndicesKT + start_n * stride_ri_row

    # Pass 1: full Q-blocks — no mask.
    for idx in range(0, n_full):
        row_idx = tl.load(ri_base + idx)
        start_m = row_idx * BLOCK_M
        m_cur = start_m + offs_m
        q_ptrs = Q + qvk_z_off_q + m_cur[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = DO + qvk_z_off_do + m_cur[:, None] * stride_dom + offs_d[None, :] * stride_don
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)

        lse_i = tl.load(LSE + off_hz * N_CTX + m_cur)
        d_i = tl.load(Delta + off_hz * N_CTX + m_cur)

        qk = tl.dot(q, tl.trans(k))
        qk_scaled = qk * (sm_scale * 1.44269504)
        p = tl.math.exp2(qk_scaled - lse_i[:, None])

        dv += tl.dot(tl.trans(p.to(do.dtype)), do)

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - d_i[:, None])
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * sm_scale

    # Pass 2: partial Q-blocks — inline fine mask.
    for idx in range(n_full, n_total):
        row_idx = tl.load(ri_base + idx)
        start_m = row_idx * BLOCK_M
        m_cur = start_m + offs_m
        q_ptrs = Q + qvk_z_off_q + m_cur[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = DO + qvk_z_off_do + m_cur[:, None] * stride_dom + offs_d[None, :] * stride_don
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)

        lse_i = tl.load(LSE + off_hz * N_CTX + m_cur)
        d_i = tl.load(Delta + off_hz * N_CTX + m_cur)

        qk = tl.dot(q, tl.trans(k))
        qk_scaled = qk * (sm_scale * 1.44269504)
        mask_ptrs = Mask + m_cur[:, None] * stride_mask_m + offs_n[None, :] * stride_mask_n
        mask_ = tl.load(mask_ptrs) != 0
        qk_scaled = tl.where(mask_, qk_scaled, -float("inf"))
        p = tl.math.exp2(qk_scaled - lse_i[:, None])

        dv += tl.dot(tl.trans(p.to(do.dtype)), do)

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - d_i[:, None])
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * sm_scale

    dk_ptrs = DK + qvk_z_off_dk + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk
    dv_ptrs = DV + qvk_z_off_dv + offs_n[:, None] * stride_dvk + offs_d[None, :] * stride_dvn
    tl.store(dk_ptrs, dk.to(DK.type.element_ty))
    tl.store(dv_ptrs, dv.to(DV.type.element_ty))


# ─── dQ kernel (Q-outer) ───

@triton.jit
def _bwd_dq_kernel(
    Q, K, V, sm_scale,
    DO, DQ,
    Mask,
    LSE, Delta,
    ColIndices, NFull, NTotal,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_doz, stride_doh, stride_dom, stride_don,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_mask_m, stride_mask_n,
    stride_ci_row,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qvk_z_off_q = off_z * stride_qz + off_h * stride_qh
    qvk_z_off_k = off_z * stride_kz + off_h * stride_kh
    qvk_z_off_v = off_z * stride_vz + off_h * stride_vh
    qvk_z_off_do = off_z * stride_doz + off_h * stride_doh
    qvk_z_off_dq = off_z * stride_dqz + off_h * stride_dqh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + qvk_z_off_q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    do_ptrs = DO + qvk_z_off_do + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_don
    q = tl.load(q_ptrs)
    do = tl.load(do_ptrs)

    lse_i = tl.load(LSE + off_hz * N_CTX + offs_m)
    d_i = tl.load(Delta + off_hz * N_CTX + offs_m)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    n_full = tl.load(NFull + start_m)
    n_total = tl.load(NTotal + start_m)
    tl.assume(n_full >= 0)
    tl.assume(n_total >= n_full)
    ci_base = ColIndices + start_m * stride_ci_row

    # Pass 1: full blocks — no mask needed.
    for idx in range(0, n_full):
        col_idx = tl.load(ci_base + idx)
        start_n = col_idx * BLOCK_N
        n_cur = start_n + offs_n
        k_ptrs = K + qvk_z_off_k + n_cur[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V + qvk_z_off_v + n_cur[:, None] * stride_vk + offs_d[None, :] * stride_vn
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        qk = tl.dot(q, tl.trans(k))
        qk_scaled = qk * (sm_scale * 1.44269504)
        p = tl.math.exp2(qk_scaled - lse_i[:, None])

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - d_i[:, None])
        dq += tl.dot(ds.to(k.dtype), k) * sm_scale

    # Pass 2: partial blocks — inline fine-mask check.
    for idx in range(n_full, n_total):
        col_idx = tl.load(ci_base + idx)
        start_n = col_idx * BLOCK_N
        n_cur = start_n + offs_n
        k_ptrs = K + qvk_z_off_k + n_cur[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V + qvk_z_off_v + n_cur[:, None] * stride_vk + offs_d[None, :] * stride_vn
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        qk = tl.dot(q, tl.trans(k))
        qk_scaled = qk * (sm_scale * 1.44269504)
        mask_ptrs = Mask + offs_m[:, None] * stride_mask_m + n_cur[None, :] * stride_mask_n
        mask_ = tl.load(mask_ptrs) != 0
        qk_scaled = tl.where(mask_, qk_scaled, -float("inf"))
        p = tl.math.exp2(qk_scaled - lse_i[:, None])

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - d_i[:, None])
        dq += tl.dot(ds.to(k.dtype), k) * sm_scale

    dq_ptrs = DQ + qvk_z_off_dq + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqk
    tl.store(dq_ptrs, dq.to(DQ.type.element_ty))


# ─── Fwd that returns LSE (for the autograd wrapper) ───

def _binflash_fwd_with_lse(q, k, v, mask, sm_scale):
    """Fwd pass that also returns LSE, reusing binflash_attention's internals."""
    B, H, N, D = q.shape
    block_n = min(D, 64)
    if sm_scale is None:
        sm_scale = D ** -0.5

    col_indices, n_full, n_total, mask_packed, chosen_bm, chosen_bn, use_packed = (
        _dispatch_and_preprocess(mask, D)
    )
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
        q, k, v, sm_scale, o, mask, mp_tensor,
        col_indices, n_full, n_total, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        mask.stride(0), mask.stride(1),
        mp_stride_0, mp_stride_1,
        col_indices.stride(0),
        B, H, N, B * H,
        HEAD_DIM=D, BLOCK_M=chosen_bm, BLOCK_N=chosen_bn,
        USE_PACKED=use_packed,
    )
    return o, lse


# ─── Bwd Python entry ───

_BWD_BLOCK_M = 64
_BWD_BLOCK_N = 64


def _compute_bwd_block_mask(mask, block_m, block_n):
    """Reduce (N, N) bool mask to (N/block_m, N/block_n) int8:
    0 = all masked, 1 = mixed, 2 = all attended."""
    N = mask.shape[0]
    nqb = N // block_m
    nkb = N // block_n
    m4d = mask.view(nqb, block_m, nkb, block_n)
    has_any = m4d.any(dim=1).any(dim=2)
    has_all = m4d.all(dim=1).all(dim=2)
    return has_any.to(torch.int8) + has_all.to(torch.int8)


def _binflash_bwd(do, q, k, v, o, lse, mask, sm_scale):
    B, H, N, D = q.shape
    assert N % _BWD_BLOCK_M == 0
    assert N % _BWD_BLOCK_N == 0
    do = do.contiguous()

    delta = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    # D=128 needs num_stages=1 to fit in SMEM with BLOCK_M=BLOCK_N=64.
    bwd_num_stages = 1 if D >= 128 else 2

    # v2: build bwd-grain (64, 64) block masks for BOTH dQ (Q-outer) and dK/dV (K-outer).
    bwd_block_mask = _compute_bwd_block_mask(mask, _BWD_BLOCK_M, _BWD_BLOCK_N)
    col_indices, n_full, n_total = _build_gathered_indices(bwd_block_mask)
    # Transpose → per-K-block row of Q-indices. .contiguous() because _build_gathered_indices
    # reads (nqb, nkb) with stride(0)=nkb.
    bwd_block_mask_kt = bwd_block_mask.T.contiguous()
    row_indices_kt, n_full_kt, n_total_kt = _build_gathered_indices(bwd_block_mask_kt)

    grid_pre = (triton.cdiv(N, _BWD_BLOCK_M), B * H)
    _bwd_preprocess_kernel[grid_pre](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        B, H, N,
        BLOCK_M=_BWD_BLOCK_M, HEAD_DIM=D,
    )

    grid_dkdv = (triton.cdiv(N, _BWD_BLOCK_N), B * H)
    _bwd_dkdv_kernel[grid_dkdv](
        q, k, v, sm_scale, do, dk, dv, mask, lse, delta,
        row_indices_kt, n_full_kt, n_total_kt,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        mask.stride(0), mask.stride(1),
        row_indices_kt.stride(0),
        B, H, N,
        BLOCK_M=_BWD_BLOCK_M, BLOCK_N=_BWD_BLOCK_N, HEAD_DIM=D,
        num_stages=bwd_num_stages,
    )

    grid_dq = (triton.cdiv(N, _BWD_BLOCK_M), B * H)
    _bwd_dq_kernel[grid_dq](
        q, k, v, sm_scale, do, dq, mask, lse, delta,
        col_indices, n_full, n_total,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        mask.stride(0), mask.stride(1),
        col_indices.stride(0),
        B, H, N,
        BLOCK_M=_BWD_BLOCK_M, BLOCK_N=_BWD_BLOCK_N, HEAD_DIM=D,
        num_stages=bwd_num_stages,
    )

    return dq, dk, dv


class _BinFlashFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, sm_scale):
        if sm_scale is None:
            sm_scale = q.shape[-1] ** -0.5
        o, lse = _binflash_fwd_with_lse(q, k, v, mask, sm_scale)
        ctx.save_for_backward(q, k, v, o, lse, mask)
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, mask = ctx.saved_tensors
        dq, dk, dv = _binflash_bwd(do, q, k, v, o, lse, mask, ctx.sm_scale)
        return dq, dk, dv, None, None


def binflash_attention_autograd(q, k, v, mask, sm_scale=None):
    """Drop-in replacement for binflash_attention with autograd support."""
    return _BinFlashFunction.apply(q, k, v, mask, sm_scale)
