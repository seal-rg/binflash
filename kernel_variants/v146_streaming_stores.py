"""Binary Block Masked Flash Attention — v146: streaming stores (.cs cache modifier).

The kernel writes O (attention output) and LSE (log-sum-exp) exactly
once per program at the end. These values are never read back within
the kernel. Streaming stores (PTX cache modifier '.cs') tell the cache
controller to bypass L1/L2, so they don't pollute the cache with write
data. This could free up L2 space for subsequent K/V loads.

Dead end #30 tested `.cg` cache modifier on LOADS (neutral). Store-side
cache modifiers haven't been tested. This is the last cache-modifier
permutation remaining.

Applied to:
  - `tl.store(O_bp, ..., cache_modifier='.cs')`
  - `tl.store(LSE + ..., ..., cache_modifier='.cs')`

Kernel body otherwise identical to v132.
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


def _pack_mask(mask: torch.Tensor) -> torch.Tensor:
    N = mask.shape[0]
    assert N % 32 == 0, f"N={N} must be divisible by 32 for bit packing"
    weights = (1 << torch.arange(32, device=mask.device, dtype=torch.int32))
    mask_reshaped = mask.view(N, N // 32, 32).to(torch.int32)
    packed = (mask_reshaped * weights).sum(dim=-1, dtype=torch.int32).to(torch.int32)
    return packed


def _fused_preproc(mask, block_m, block_n, want_packed=False):
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
    mask_packed = _pack_mask(mask) if want_packed else None
    return block_mask, first_nz, last_nz, avg_gap_ratio, density, mask_packed


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
    # Streaming stores: bypass L1/L2 to avoid cache pollution
    tl.store(LSE + (off_hz * N_CTX) + offs_m, m_i + tl.math.log2(l_i), cache_modifier='.cs')
    tl.store(O_bp, acc.to(Out.type.element_ty), cache_modifier='.cs')


_bm_cache_key = None
_bm_cache_val = None

_GAP_RATIO_THRESHOLD = 0.3
_DENSITY_THRESHOLD_BN128 = 0.9
_DENSITY_THRESHOLD_PACKED = 0.8


def binflash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: torch.Tensor, sm_scale: float | None = None,
    block_m: int = 128, block_n: int | None = None,
    block_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """v132 + streaming stores (.cs cache modifier on O and LSE writes)."""
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
            bm128_bn64, fn, ln, gap_ratio, density, _ = _fused_preproc(mask, 128, 64, want_packed=False)
            if D == 64 and gap_ratio > _GAP_RATIO_THRESHOLD and N % 64 == 0:
                want = density > _DENSITY_THRESHOLD_PACKED
                block_mask, first_nz, last_nz, _, _, mask_packed = _fused_preproc(
                    mask, 64, 64, want_packed=want,
                )
                chosen_bm, chosen_bn = 64, 64
                use_packed = 1 if want else 0
            elif D == 64 and density > _DENSITY_THRESHOLD_BN128 and N % 128 == 0:
                block_mask, first_nz, last_nz, _, _, mask_packed = _fused_preproc(
                    mask, 128, 128, want_packed=True,
                )
                chosen_bm, chosen_bn = 128, 128
                use_packed = 1
            else:
                block_mask, first_nz, last_nz = bm128_bn64, fn, ln
                use_packed = 1 if density > _DENSITY_THRESHOLD_PACKED else 0
                mask_packed = _pack_mask(mask) if use_packed else None
                chosen_bm, chosen_bn = 128, 64
            _bm_cache_key = key
            _bm_cache_val = (block_mask, first_nz, last_nz, mask_packed, chosen_bm, chosen_bn, use_packed)
    else:
        raise NotImplementedError("External block_mask not supported in v146")

    if N % chosen_bm != 0 or N % chosen_bn != 0:
        raise ValueError(f"N must be divisible")

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
