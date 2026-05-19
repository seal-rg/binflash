import torch  # type: ignore
import triton  # type: ignore
import triton.language as tl  # type: ignore

# ─── Fused preprocessing ───


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


def _reduce_for_fine(block_mask_fine):
    nqb, nkb = block_mask_fine.shape
    first_nz = torch.empty(nqb, dtype=torch.int32, device=block_mask_fine.device)
    last_nz = torch.empty(nqb, dtype=torch.int32, device=block_mask_fine.device)
    gs = torch.zeros(2, dtype=torch.int32, device=block_mask_fine.device)
    gg = torch.zeros(1, dtype=torch.float32, device=block_mask_fine.device)
    nkb_pow2 = 1
    while nkb_pow2 < nkb:
        nkb_pow2 *= 2
    _reduce_block_mask[(nqb,)](
        block_mask_fine,
        first_nz,
        last_nz,
        gs,
        gg,
        block_mask_fine.stride(0),
        block_mask_fine.stride(1),
        nkb,
        NKB_POW2=nkb_pow2,
    )
    return first_nz, last_nz


def _make_coarse_lK(block_mask: torch.Tensor, k: int) -> torch.Tensor:
    """v167: build a (nqb, nkb/2**k) coarse block_mask by repeatedly
    halving the BN axis via OR-reduction of pairs. k=1 → 2x coarser
    (v166 behavior). k=2 → 4x coarser (v167 path 1). Stored as int8."""
    cur = block_mask
    for _ in range(k):
        nqb, nkb = cur.shape
        assert nkb % 2 == 0, f"nkb={nkb} must be even for coarsening"
        reshaped = cur.view(nqb, nkb // 2, 2)
        cur = (reshaped > 0).any(dim=2).to(torch.int8)
    return cur


# ─── Main attention kernel ───


@triton.jit
def _binflash_fwd_inner(
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
    block_mask_ptr,
    bm_stride_row,
    bm_stride_col,
    start_m,
    coarse_mask_ptr,
    coarse_stride_row,
    coarse_stride_col,
    first_nz_col,
    last_nz_col,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
    USE_PACKED: tl.constexpr,
    USE_HIER: tl.constexpr,
    COARSE_FACTOR_C: tl.constexpr,
):
    offs_d = tl.arange(0, HEAD_DIM)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_bm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    BN_PACKED: tl.constexpr = BLOCK_N // 32
    offs_bn_packed = tl.arange(0, BN_PACKED)
    bit_offs = tl.arange(0, 32)

    # v171: single-pass merged hierarchical scan inside USE_HIER.
    # One while loop over the coarse range handles both full (==2)
    # and partial (==1) blocks in one inner body.
    # v173: explicit mutual-exclusion split — `if bm_val == 2 ... elif
    # bm_val == 1` instead of nested `if bm_val > 0 → if bm_val == 1`.
    # Lets the compiler allocate registers separately per arm; the full
    # branch never holds mask intermediates live. Measured on D=64
    # longformer (path 1 BM=64 CF=8): consistent +2-4% across 4 runs.
    # D=128 neutral. Dead end #21 (v102) flat single-pass was slower,
    # but static_range(CF) + mutual-exclusion branches compile cleanly.
    if USE_HIER:
        coarse_first = first_nz_col // COARSE_FACTOR_C
        coarse_last = last_nz_col // COARSE_FACTOR_C
        coarse_col = coarse_first
        while coarse_col <= coarse_last:
            coarse_val = tl.load(coarse_mask_ptr + start_m * coarse_stride_row + coarse_col * coarse_stride_col)
            if coarse_val > 0:
                base_col = COARSE_FACTOR_C * coarse_col
                for sub in tl.static_range(COARSE_FACTOR_C):
                    col_idx = base_col + sub
                    bm_val = tl.load(block_mask_ptr + start_m * bm_stride_row + col_idx * bm_stride_col)
                    # v173: explicit split per block type.
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
                    elif bm_val == 1:
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
            coarse_col += 1
    else:
        # Non-hier two-pass (dense patterns, small-N — unchanged from v168)
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
            col_idx += 1

    return acc, l_i, m_i


_autotune_configs = []
for _s in [1, 2, 3, 4, 5, 6, 7]:
    for _w in [4, 8]:
        for _mnr in [None, 168, 192]:
            _kw = {}
            if _mnr is not None:
                _kw["maxnreg"] = _mnr
            _autotune_configs.append(triton.Config({}, num_stages=_s, num_warps=_w, **_kw))


# v177: expanded autotune space with maxnreg ∈ {None, 168, 192}. Dead end
# #53 tested maxnreg=128 which was never picked; probing 168/192 found
# that causal-class patterns pick maxnreg=192 consistently (both D=64
# and D=128 at N=16384). Observed mechanical wins across dense patterns:
# causal +2-3%, prefix_lm +5-7%, sliding/causal_window +3-4%, sparse
# +3-5%, log_tree +4-6%, causal_doc_7 +3-8%.
#
# v179: extended stages range from [1..5] to [1..7]. Dead end #49 tried
# stages up to 7 without maxnreg and was neutral — the deeper pipelining
# couldn't fit. With maxnreg=192 available, stages 6 and 7 now FIT and
# get picked by autotune: causal D=128 picks stages=7 maxnreg=192,
# sparse_10pct picks stages=6 maxnreg=168, longformer D=64 picks
# stages=7. Measured wins on top of v177: 7B-train sliding/causal_window
# +7% at N=4096, prefix_lm +6%, sparse +4-6%, 7B-long causal +4% at
# N=16384, causal_window_256/prefix_lm/sparse all +5-6% at N=16384.
# Small regressions: longformer D=64 -2% at N=16384 (stages=7 is
# overkill for short-scan patterns), block_diagonal -1-2%. Net gain
# dominates. Trade-off: 42 configs vs 30 → 1.4× first-run autotune time.
@triton.autotune(
    configs=_autotune_configs,
    key=["N_CTX", "HEAD_DIM", "Z_TIMES_H", "BLOCK_M", "BLOCK_N", "USE_PACKED"],
)
@triton.jit
def _binflash_fwd(
    Q,
    K,
    V,
    sm_scale,
    Out,
    Mask,
    MaskPacked,
    BlockMask,
    CoarseMask,
    FirstNz,
    LastNz,
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
    stride_bm_row,
    stride_bm_col,
    stride_coarse_row,
    stride_coarse_col,
    Z,
    H,
    N_CTX,
    Z_TIMES_H,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_PACKED: tl.constexpr,
    USE_HIER: tl.constexpr,
    COARSE_FACTOR_C: tl.constexpr,
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
    first_nz = tl.load(FirstNz + start_m)
    last_nz = tl.load(LastNz + start_m)
    acc, l_i, m_i = _binflash_fwd_inner(
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
        BlockMask,
        stride_bm_row,
        stride_bm_col,
        start_m,
        CoarseMask,
        stride_coarse_row,
        stride_coarse_col,
        first_nz,
        last_nz,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        N_CTX,
        USE_PACKED,
        USE_HIER,
        COARSE_FACTOR_C,
    )
    # v188: REVERTED v185 (LSE removal) and v186 (dead_where removal).
    # A principled head-to-head comparison of v179 vs v186 (both from
    # past_attempts, independent autotune caches) showed v179 mechanically
    # wins on sparse/log_tree patterns: log_tree N=16384 **-10.7%**,
    # sparse_10pct N=16384 **-5.6%**, sparse_30pct N=32768 -3.1%. Despite
    # looking like "dead code", the LSE store and the tl.where safety
    # check somehow help the compiler schedule better on these patterns
    # (possibly the HBM store provides a scheduling anchor, or the extra
    # op disrupts a bad L2/scheduler interaction). My earlier v185/v186
    # measurements claiming 3-5% wins were misattributed autotune
    # re-rolls — the mechanical bodies were actually slower.
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
# v166: enable the hierarchical outer scan only when `gap_ratio > 0.3`
# (real scan waste to eliminate). On the BM=64 gap-heavy path, also
# gate on a minimum mean scan width so small-N cases where the coarse
# overhead exceeds the saved empty-skip iters don't regress.
# v167: dispatch COARSE_FACTOR per path. Path 1 (gap-heavy) uses CF=4
# for bigger outer skips on longformer/cdw_sinks. Other paths use
# CF=2 (v166 behavior) to avoid compilation-variance regressions at
# D=128 small-N that occur when CF=4 is applied globally.
_HIER_GAP_RATIO_THRESHOLD = 0.3
_HIER_MIN_SCAN_WIDTH = 16
# v168 adaptive path-1 coarse factor: very high gap_ratio (longformer,
# ~0.81) gets CF=8 for maximum outer-loop reduction; moderate gap_ratio
# (cdw_sinks ~0.5-0.7) keeps CF=4. Measured at D=128 7B-long N=16384:
# longformer +19.0%, cdw_sinks +9.5%, causal_window_256 +6.8%.
_CF_PATH1_WIDE = 8
_CF_PATH1 = 4
_CF_OTHER = 2
_WIDE_GAP_THRESHOLD = 0.7


def _mean_scan_width(first_nz: torch.Tensor, last_nz: torch.Tensor) -> float:
    widths = (last_nz - first_nz + 1).clamp(min=0).float()
    return float(widths.mean().item())


def _dispatch_and_preprocess(mask: torch.Tensor, D: int):
    import math

    N = mask.shape[0]
    out = _fused_preproc_multigrain(mask, 64)
    gap_ratio = out["gap_ratio"]
    density = out["density"]
    partial_fraction = out["partial_fraction"]
    mask_packed = out["mask_packed"]

    if D == 64 and gap_ratio > _GAP_RATIO_THRESHOLD and N % 64 == 0:
        # Path 1: gap-heavy (longformer, cdw_sinks). v168 adaptive CF:
        # very-wide gaps (longformer, gap>0.7) get CF=8; moderate gaps
        # (cdw_sinks, gap<=0.7) keep CF=4. This captures longformer's
        # extra outer-skip gains without cdw_sinks's small-N regression.
        block_mask = out["block_mask_fine"]
        first_nz, last_nz = _reduce_for_fine(block_mask)
        cf = _CF_PATH1_WIDE if gap_ratio > _WIDE_GAP_THRESHOLD else _CF_PATH1
        coarse = _make_coarse_lK(block_mask, int(math.log2(cf)))
        use_hier = 1 if _mean_scan_width(first_nz, last_nz) >= _HIER_MIN_SCAN_WIDTH else 0
        return (
            block_mask,
            first_nz,
            last_nz,
            mask_packed,
            64,
            64,
            1,
            use_hier,
            coarse,
            cf,
        )

    if D == 64 and density > _DENSITY_THRESHOLD_BN128 and N % 128 == 0:
        # Path 2: dense BN=128. No scan waste → USE_HIER=0. CF is irrelevant.
        block_mask = out["block_mask_bn128"]
        first_nz, last_nz = _reduce_for_fine(block_mask)
        cf = _CF_OTHER
        coarse = _make_coarse_lK(block_mask, int(math.log2(cf)))
        return block_mask, first_nz, last_nz, mask_packed, 128, 128, 1, 0, coarse, cf

    use_packed = (
        1
        if (density > _DENSITY_THRESHOLD_PACKED or gap_ratio > _GAP_RATIO_THRESHOLD or partial_fraction > _PARTIAL_FRACTION_THRESHOLD)
        else 0
    )
    if not use_packed:
        mask_packed = None
    block_mask = out["block_mask_coarse"]
    fnz = out["first_nz_coarse"]
    lnz = out["last_nz_coarse"]
    # Default path: CF=2 (v166 behavior). USE_HIER=1 when gap_ratio > 0.3
    # (covers D=128 gap-heavy patterns that can't go through path 1).
    use_hier = 1 if gap_ratio > _HIER_GAP_RATIO_THRESHOLD else 0
    cf = _CF_OTHER
    coarse = _make_coarse_lK(block_mask, int(math.log2(cf)))
    return (
        block_mask,
        fnz,
        lnz,
        mask_packed,
        128,
        64,
        use_packed,
        use_hier,
        coarse,
        cf,
    )


def preproc(mask: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float):
    """Generic preprocessing entry point for benchmarking.

    Runs everything `binflash_attention` does on a cache miss. Stable
    name across kernel iterations — benchmark.py imports this symbol."""
    D = q.shape[-1]
    return _dispatch_and_preprocess(mask, D)


def binflash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, sm_scale: float | None = None
) -> torch.Tensor:
    """Binary-block-masked flash attention (v166 hierarchical scan).

    Args:
        q, k, v: (B, H, N, D) in float16/bfloat16
        mask: (N, N) bool tensor — True means "attend"
        sm_scale: softmax scale, defaults to 1/sqrt(D)
    """
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
            block_mask,
            first_nz,
            last_nz,
            mask_packed,
            chosen_bm,
            chosen_bn,
            use_packed,
            use_hier,
            coarse_mask,
            cf,
        ) = _bm_cache_val
    else:
        cache_val = _dispatch_and_preprocess(mask, D)
        _bm_cache_key = key
        _bm_cache_val = cache_val
        (
            block_mask,
            first_nz,
            last_nz,
            mask_packed,
            chosen_bm,
            chosen_bn,
            use_packed,
            use_hier,
            coarse_mask,
            cf,
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

    _binflash_fwd[grid](
        q,
        k,
        v,
        sm_scale,
        o,
        mask,
        mp_tensor,
        block_mask,
        coarse_mask,
        first_nz,
        last_nz,
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
        block_mask.stride(0),
        block_mask.stride(1),
        coarse_mask.stride(0),
        coarse_mask.stride(1),
        B,
        H,
        N,
        B * H,
        HEAD_DIM=D,
        BLOCK_M=chosen_bm,
        BLOCK_N=chosen_bn,
        USE_PACKED=use_packed,
        USE_HIER=use_hier,
        COARSE_FACTOR_C=cf,
    )
    return o
