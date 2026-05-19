"""v160: fused _reduce_block_mask Triton kernel that replaces the PyTorch
nz/argmax/max/where/sum chain in _fused_preproc with one kernel pass.

After v159, profiling shows:
  _compute_block_mask (v159):    0.395ms
  nz+argmax+metrics chain:       0.378ms  ← this target
  _pack_mask (v158):             1.032ms
  FULL want_packed=False:        0.777ms

The PyTorch chain does ~8 separate CUDA launches (nz > 0, arange, .any,
.argmax, .where, .sum, metric stack, .tolist). Each launch has ~30-50μs
overhead → the chain time is mostly launch overhead, not compute.

v160 replaces this chain with a single Triton kernel:
  - Grid: (nqb,), one program per Q-row
  - Loads block_mask[start_m, :] as a (nkb,) vector
  - Computes first_nz, last_nz, num_nonempty, num_partial per row
  - Stores per-row first_nz/last_nz (needed by main kernel)
  - Atomically adds to (3,) global counter for gap_sum, nonempty_sum, partial_sum

Python then reads 3 scalars via one .tolist() and divides to get the dispatch
metrics. Cuts the chain from ~0.38ms to ~0.05-0.10ms.

Kernel body UNCHANGED from v149/v158/v159.
"""

import torch
import triton
import triton.language as tl


# ─── Fused preprocessing ───

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
def _reduce_block_mask(
    BlockMask,
    FirstNz, LastNz,
    GlobalSums,  # int64 (3,): [gap_scan_width_sum, nonempty_sum, partial_sum]
    GlobalGapFloat,  # fp32 scalar: sum of (1 - num_nonempty/scan_width) across rows
    stride_bm_row, stride_bm_col,
    nkb,
    NKB_POW2: tl.constexpr,
):
    """Grid (nqb,). One program per Q-row. Reads block_mask row, computes
    per-row metrics, stores first_nz/last_nz and atomically accumulates
    global counters."""
    row = tl.program_id(0)

    offs = tl.arange(0, NKB_POW2)
    valid = offs < nkb
    bm_row = tl.load(
        BlockMask + row * stride_bm_row + offs * stride_bm_col,
        mask=valid, other=0,
    )
    is_nonempty = bm_row > 0  # int1 (NKB_POW2,)
    is_partial = bm_row == 1

    # First / last nonempty indices. For empty rows, default 0 / -1.
    has_any = tl.sum(is_nonempty.to(tl.int32)) > 0
    # last_nz: max index among nonempty positions, else -1
    ranked = tl.where(is_nonempty, offs, -1)
    last_nz = tl.max(ranked, axis=0)
    # first_nz: min index among nonempty positions, else 0
    ranked_first = tl.where(is_nonempty, offs, nkb)  # nkb is max+1
    first_nz_tmp = tl.min(ranked_first, axis=0)
    first_nz = tl.where(has_any, first_nz_tmp, 0)

    num_nonempty = tl.sum(is_nonempty.to(tl.int32))
    num_partial = tl.sum(is_partial.to(tl.int32))

    # scan_width = max(1, last_nz - first_nz + 1) — careful for empty rows
    scan_width = tl.where(has_any, last_nz - first_nz + 1, 1)
    gap_for_row = 1.0 - num_nonempty.to(tl.float32) / scan_width.to(tl.float32)

    tl.store(FirstNz + row, first_nz.to(tl.int32))
    tl.store(LastNz + row, last_nz.to(tl.int32))

    # Global atomics: only one program per row writes, so atomic_add is
    # called nqb times total — cheap.
    tl.atomic_add(GlobalSums + 0, num_nonempty)
    tl.atomic_add(GlobalSums + 1, num_partial)
    tl.atomic_add(GlobalGapFloat, gap_for_row)


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


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def _fused_preproc(mask, block_m, block_n, want_packed=False):
    N = mask.shape[0]
    nqb = N // block_m
    nkb = N // block_n
    block_mask = torch.empty(nqb, nkb, dtype=torch.int8, device=mask.device)

    bn_chunks = 8
    while nkb % bn_chunks != 0 and bn_chunks > 1:
        bn_chunks //= 2
    _compute_block_mask[(nqb, nkb // bn_chunks)](
        mask, block_mask,
        mask.stride(0), mask.stride(1), nkb,
        BLOCK_M=block_m, BLOCK_N=block_n, BN_CHUNKS=bn_chunks,
    )

    # v160: fused reduce kernel replaces the PyTorch nz/argmax/metrics chain.
    first_nz = torch.empty(nqb, dtype=torch.int32, device=mask.device)
    last_nz = torch.empty(nqb, dtype=torch.int32, device=mask.device)
    global_sums = torch.zeros(2, dtype=torch.int32, device=mask.device)
    global_gap = torch.zeros(1, dtype=torch.float32, device=mask.device)
    nkb_pow2 = _next_pow2(nkb)
    _reduce_block_mask[(nqb,)](
        block_mask,
        first_nz, last_nz,
        global_sums, global_gap,
        block_mask.stride(0), block_mask.stride(1),
        nkb,
        NKB_POW2=nkb_pow2,
    )

    # Combine the 3 scalar tensors into one .tolist() transfer.
    metrics = torch.cat([
        global_sums.to(torch.float32),
        global_gap,
    ])
    num_nonempty_total, num_partial_total, global_gap_sum = metrics.tolist()

    density = num_nonempty_total / (nqb * nkb)
    partial_fraction = num_partial_total / max(num_nonempty_total, 1)
    avg_gap_ratio = global_gap_sum / nqb

    mask_packed = _pack_mask(mask) if want_packed else None
    return block_mask, first_nz, last_nz, avg_gap_ratio, density, partial_fraction, mask_packed


# ─── Main attention kernel (UNCHANGED from v149/v158/v159) ───

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
            bm128_bn64, fn, ln, gap_ratio, density, partial_fraction, _ = _fused_preproc(
                mask, 128, 64, want_packed=False,
            )
            if D == 64 and gap_ratio > _GAP_RATIO_THRESHOLD and N % 64 == 0:
                block_mask, first_nz, last_nz, _, _, _, mask_packed = _fused_preproc(
                    mask, 64, 64, want_packed=True,
                )
                chosen_bm, chosen_bn = 64, 64
                use_packed = 1
            elif D == 64 and density > _DENSITY_THRESHOLD_BN128 and N % 128 == 0:
                block_mask, first_nz, last_nz, _, _, _, mask_packed = _fused_preproc(
                    mask, 128, 128, want_packed=True,
                )
                chosen_bm, chosen_bn = 128, 128
                use_packed = 1
            else:
                block_mask, first_nz, last_nz = bm128_bn64, fn, ln
                use_packed = 1 if (
                    density > _DENSITY_THRESHOLD_PACKED
                    or gap_ratio > _GAP_RATIO_THRESHOLD
                    or partial_fraction > _PARTIAL_FRACTION_THRESHOLD
                ) else 0
                mask_packed = _pack_mask(mask) if use_packed else None
                chosen_bm, chosen_bn = 128, 64

            _bm_cache_key = key
            _bm_cache_val = (block_mask, first_nz, last_nz, mask_packed, chosen_bm, chosen_bn, use_packed)
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
