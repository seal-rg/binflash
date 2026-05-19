"""v172: test whether the v171 single-pass merged hier body beats the
v168 non-hier two-pass body on ALL default-path patterns (not just
gap_ratio > 0.3).

Rationale: v171 confirmed the merged single-pass hier body is faster
than two-pass inside the hier branch — the `tl.static_range(CF)` gives
the compiler compile-time visibility over the merged full/partial body
that dead end #21 (v102 flat single-pass) lacked. For patterns with
gap_ratio ≤ 0.3 (causal, log_tree, sparse, block_diagonal, prefix_lm,
causal_doc_7) we're currently on the v168 two-pass flat path. With
CF=2, the hier branch's outer coarse loop has exactly as many inner
bm_val checks per block as the two-pass flat path, PLUS a cheap coarse
scalar load per 2 cols (always >0 for dense patterns). The merged body
may still win if the static_range visibility helps register allocation.

Risk: (a) extra coarse load overhead for dense patterns where
coarse_val is always >0; (b) forces fresh compiled specializations for
BM=128 BN=64 USE_HIER=1 at D=128 — may trigger autotune variance.

Only changes dispatch: `use_hier = 1` unconditionally on the default
path. Kernel functions imported from the v171 test file so the
compiled binaries are shared (pure A/B on dispatch flag).
"""
import importlib.util
import math
import os
import torch
import triton

# ── begin inlined legacy _prod helpers (frozen v167-era snapshot) ──
import torch  # noqa: E402,F401
import triton  # noqa: E402,F401
import triton.language as tl  # noqa: E402

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
def _compute_and_pack_mask_multi(
    Mask, BlockMaskCoarse, BlockMaskFine, BlockMaskBN128, Packed,
    stride_mask_m, stride_mask_n,
    stride_packed_m, stride_packed_n,
    nkb, nkb_bn128,
    BM_COARSE: tl.constexpr, BM_FINE: tl.constexpr, BLOCK_N: tl.constexpr,
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
    block_mask_coarse = torch.empty(nqb_coarse, nkb, dtype=torch.int8, device=mask.device)
    block_mask_fine = torch.empty(N // 64, nkb, dtype=torch.int8, device=mask.device)
    block_mask_bn128 = torch.empty(nqb_coarse, nkb_bn128, dtype=torch.int8, device=mask.device)
    mask_packed = torch.empty((N, N // 32), dtype=torch.int32, device=mask.device)
    bn_chunks = _pick_bn_chunks(128, block_n, nkb, start=2)
    assert bn_chunks * block_n == 128
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
        first_nz, last_nz,
        gs, gg,
        block_mask_fine.stride(0), block_mask_fine.stride(1),
        nkb,
        NKB_POW2=nkb_pow2,
    )
    return first_nz, last_nz


def _mean_scan_width(first_nz, last_nz):
    widths = (last_nz - first_nz + 1).clamp(min=0).float()
    return float(widths.mean().item())


def _make_coarse_lK(block_mask, k):
    cur = block_mask
    for _ in range(k):
        nqb, nkb = cur.shape
        assert nkb % 2 == 0, f"nkb={nkb} must be even for coarsening"
        reshaped = cur.view(nqb, nkb // 2, 2)
        cur = (reshaped > 0).any(dim=2).to(torch.int8)
    return cur


_GAP_RATIO_THRESHOLD = 0.3
_DENSITY_THRESHOLD_BN128 = 0.9
_DENSITY_THRESHOLD_PACKED = 0.8
_PARTIAL_FRACTION_THRESHOLD = 0.5
# ── end inlined legacy _prod helpers ──


_V171_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v171_singlepass_hier.py")
_spec = importlib.util.spec_from_file_location("_v171_singlepass_hier_mod", _V171_PATH)
_v171 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v171)

_binflash_fwd_v171 = _v171._binflash_fwd_v171
_HIER_GAP_RATIO_THRESHOLD = 0.3
_HIER_MIN_SCAN_WIDTH = 16
_CF_PATH1_WIDE = 8
_CF_PATH1 = 4
_CF_OTHER = 2
_WIDE_GAP_THRESHOLD = 0.7


_bm_cache_key = None
_bm_cache_val = None


def _dispatch_and_preprocess(mask: torch.Tensor, D: int):
    N = mask.shape[0]
    out = _fused_preproc_multigrain(mask, 64)
    gap_ratio = out["gap_ratio"]
    density = out["density"]
    partial_fraction = out["partial_fraction"]
    mask_packed = out["mask_packed"]

    if D == 64 and gap_ratio > _GAP_RATIO_THRESHOLD and N % 64 == 0:
        block_mask = out["block_mask_fine"]
        first_nz, last_nz = _reduce_for_fine(block_mask)
        cf = _CF_PATH1_WIDE if gap_ratio > _WIDE_GAP_THRESHOLD else _CF_PATH1
        coarse = _make_coarse_lK(block_mask, int(math.log2(cf)))
        use_hier = 1 if _mean_scan_width(first_nz, last_nz) >= _HIER_MIN_SCAN_WIDTH else 0
        return block_mask, first_nz, last_nz, mask_packed, 64, 64, 1, use_hier, coarse, cf

    if D == 64 and density > _DENSITY_THRESHOLD_BN128 and N % 128 == 0:
        block_mask = out["block_mask_bn128"]
        first_nz, last_nz = _reduce_for_fine(block_mask)
        cf = _CF_OTHER
        coarse = _make_coarse_lK(block_mask, int(math.log2(cf)))
        return block_mask, first_nz, last_nz, mask_packed, 128, 128, 1, 0, coarse, cf

    use_packed = 1 if (
        density > _DENSITY_THRESHOLD_PACKED
        or gap_ratio > _GAP_RATIO_THRESHOLD
        or partial_fraction > _PARTIAL_FRACTION_THRESHOLD
    ) else 0
    if not use_packed:
        mask_packed = None
    block_mask = out["block_mask_coarse"]
    fnz = out["first_nz_coarse"]
    lnz = out["last_nz_coarse"]
    # v172: force USE_HIER=1 unconditionally on the default path so all
    # patterns use the v171 merged single-pass body.
    use_hier = 1
    cf = _CF_OTHER
    coarse = _make_coarse_lK(block_mask, int(math.log2(cf)))
    return (block_mask, fnz, lnz, mask_packed, 128, 64, use_packed, use_hier, coarse, cf)


def preproc(mask, q, k, v, sm_scale):
    D = q.shape[-1]
    return _dispatch_and_preprocess(mask, D)


def binflash_attention(q, k, v, mask, sm_scale=None, block_m=128, block_n=None, block_mask=None):
    global _bm_cache_key, _bm_cache_val
    B, H, N, D = q.shape
    if block_n is None:
        block_n = min(D, 64)
    if sm_scale is None:
        sm_scale = D ** -0.5

    if block_mask is None:
        key = (mask.data_ptr(), mask.shape[0], mask.shape[1], block_n, D)
        if key == _bm_cache_key:
            (block_mask, first_nz, last_nz, mask_packed, chosen_bm, chosen_bn,
             use_packed, use_hier, coarse_mask, cf) = _bm_cache_val
        else:
            cache_val = _dispatch_and_preprocess(mask, D)
            _bm_cache_key = key
            _bm_cache_val = cache_val
            (block_mask, first_nz, last_nz, mask_packed, chosen_bm, chosen_bn,
             use_packed, use_hier, coarse_mask, cf) = cache_val
    else:
        raise NotImplementedError("external block_mask path not implemented")

    o = torch.empty_like(q)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(N, chosen_bm), B * H)

    if mask_packed is None:
        mp_tensor = mask
        mp_stride_0, mp_stride_1 = 0, 0
    else:
        mp_tensor = mask_packed
        mp_stride_0, mp_stride_1 = mp_tensor.stride(0), mp_tensor.stride(1)

    _binflash_fwd_v171[grid](
        q, k, v, sm_scale, o, mask, mp_tensor, block_mask, coarse_mask, first_nz, last_nz, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        mask.stride(0), mask.stride(1),
        mp_stride_0, mp_stride_1,
        block_mask.stride(0), block_mask.stride(1),
        coarse_mask.stride(0), coarse_mask.stride(1),
        B, H, N, B * H,
        HEAD_DIM=D, BLOCK_M=chosen_bm, BLOCK_N=chosen_bn,
        USE_PACKED=use_packed, USE_HIER=use_hier,
        COARSE_FACTOR_C=cf,
    )
    return o
