"""Binary Block Masked Flash Attention — v143: INT8 K quantization.

Bandwidth-reduction attempt. Store K as int8 with per-row scale, cutting
K load bandwidth roughly in half (16KB → 8KB + 128B per block at D=128
BN=64). For sparse_10pct D=128 N=16384 this could save up to ~15% of
total runtime if the kernel is bandwidth-bound on K loads.

Correctness is the main risk. INT8 probe showed 1% relative error on a
simple inner product. Attention's softmax normalization might dampen
this in the output, but we won't know without measuring.

Applies only to pass 1 (full blocks) and USE_PACKED=0 branch of pass 2,
since those dominate for patterns where K bandwidth matters most.
Actually — for consistency, apply to both pass 1 and both pass 2
branches. The int8 load is always smaller than bf16 regardless of pattern.

Preprocessing: cached alongside block_mask. K_int8 and K_scale computed
once per unique mask key.
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


def _quantize_k(k: torch.Tensor):
    """Quantize K to INT8 with per-row scale.

    k: (B, H, N, D) bf16/fp16
    Returns (K_int8, K_scale) where K_scale matches k's dtype.
    Dequantization: K_bf16 = K_int8.to(k.dtype) * K_scale
    """
    k_fp32 = k.to(torch.float32)
    k_amax = k_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    k_scale = k_amax / 127.0  # (..., 1) fp32
    k_int8 = (k_fp32 / k_scale).round().clamp(-128, 127).to(torch.int8)
    # Preserve Q/K dtype for scale so broadcast multiply is a single dtype
    return k_int8, k_scale.to(k.dtype)


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
    K_int8_base, K_scale_base, V_base, Mask_base, MaskPacked_base,
    stride_kk, stride_kn,  # K_int8 strides (N, D) or (D, N)
    stride_ks,  # K_scale stride along N
    stride_vk, stride_vn,
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
            # Load INT8 K tile and per-row scales
            k_ptrs = K_int8_base + offs_d[:, None] * stride_kk + (start_n + offs_bn)[None, :] * stride_kn
            k_int8 = tl.load(k_ptrs)  # (D, BN) int8
            # Load scale (1, BN) - one per K row
            k_scale_ptrs = K_scale_base + (start_n + offs_bn) * stride_ks
            k_scale = tl.load(k_scale_ptrs)  # (BN,) q.dtype
            # Dequantize
            k = k_int8.to(q.dtype) * k_scale[None, :]  # (D, BN) matches q dtype
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
            k_ptrs = K_int8_base + offs_d[:, None] * stride_kk + (start_n + offs_bn)[None, :] * stride_kn
            k_int8 = tl.load(k_ptrs)
            k_scale_ptrs = K_scale_base + (start_n + offs_bn) * stride_ks
            k_scale = tl.load(k_scale_ptrs)
            k = k_int8.to(tl.bfloat16) * k_scale[None, :]
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
    Q, K_int8, K_scale, V, sm_scale, Out, Mask, MaskPacked, BlockMask, FirstNz, LastNz, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,  # K_int8 strides
    stride_ksz, stride_ksh, stride_ks,  # K_scale strides
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
    k_scale_offset = (off_hz // H).to(tl.int64) * stride_ksz + (off_hz % H).to(tl.int64) * stride_ksh
    Q_bp = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    O_bp = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    K_int8_base = K_int8 + qvk_offset
    K_scale_base = K_scale + k_scale_offset
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
        K_int8_base, K_scale_base, V_base, Mask, MaskPacked,
        stride_kk, stride_kn,
        stride_ks,
        stride_vk, stride_vn,
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
_k_quant_cache_key = None
_k_quant_cache_val = None

_GAP_RATIO_THRESHOLD = 0.3
_DENSITY_THRESHOLD_BN128 = 0.9
_DENSITY_THRESHOLD_PACKED = 0.8


def binflash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: torch.Tensor, sm_scale: float | None = None,
    block_m: int = 128, block_n: int | None = None,
    block_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """v132 + INT8 K quantization with per-row scale."""
    global _bm_cache_key, _bm_cache_val, _k_quant_cache_key, _k_quant_cache_val
    B, H, N, D = q.shape
    if block_n is None:
        block_n = min(D, 64)
    if sm_scale is None:
        sm_scale = D ** -0.5
    if D not in {16, 32, 64, 128}:
        raise ValueError(f"HEAD_DIM ({D}) must be one of {{16, 32, 64, 128}}")

    # Quantize K — cached by tensor ptr
    k_key = (k.data_ptr(), k.shape, k.stride())
    if k_key == _k_quant_cache_key:
        k_int8, k_scale = _k_quant_cache_val
    else:
        k_int8, k_scale = _quantize_k(k)
        _k_quant_cache_key = k_key
        _k_quant_cache_val = (k_int8, k_scale)

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
        raise NotImplementedError("External block_mask not supported in v143")

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

    # k_int8 has shape (B, H, N, D) int8 — same layout as K
    # k_scale has shape (B, H, N, 1) bf16 — per-row scale
    _binflash_fwd[grid](
        q, k_int8, k_scale, v, sm_scale, o, mask, mp_tensor, block_mask, first_nz, last_nz, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
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
