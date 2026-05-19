"""Binary Block Masked Flash Attention — v107: persistent kernel.

Fundamentally different dispatch: instead of grid (N/BM, B*H) with one
program per Q-row × head, launch a fixed small grid (~num_SMs) and have
each program atomically claim work items from a global counter. This
enables dynamic load balancing — programs that finish light rows pick
up heavy rows immediately, avoiding wave boundary stalls.

Raw pointers throughout (block_ptr can't be used in while loops).
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
    return block_mask, first_nz, last_nz


@triton.autotune(
    configs=[triton.Config({}, num_stages=s, num_warps=w)
             for s in [1, 2, 3, 4, 5] for w in [4, 8]],
    key=["N_CTX", "HEAD_DIM", "Z_TIMES_H"],
)
@triton.jit
def _binflash_persistent_fwd(
    Q, K, V, sm_scale, Out, Mask, BlockMask, FirstNz, LastNz, LSE,
    WorkCounter, total_work, num_q_blocks,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_mask_m, stride_mask_n,
    stride_bm_row, stride_bm_col,
    Z, H, N_CTX, Z_TIMES_H,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    offs_d = tl.arange(0, HEAD_DIM)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_bm_local = tl.arange(0, BLOCK_M)

    # Atomically claim work items until all are consumed
    work_id = tl.atomic_add(WorkCounter, 1)
    while work_id < total_work:
        # Decode work_id into (start_m, off_hz)
        start_m = work_id % num_q_blocks
        off_hz = work_id // num_q_blocks
        qvk_offset = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh

        # Raw pointer setup for Q (BLOCK_M, HEAD_DIM)
        offs_qm = start_m * BLOCK_M + offs_bm_local
        q_ptrs = Q + qvk_offset + offs_qm[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(q_ptrs)
        q = (q * (sm_scale * 1.44269504)).to(q.dtype)

        # K, V, Mask bases
        K_base = K + qvk_offset
        V_base = V + qvk_offset
        Mask_base = Mask  # (N_CTX, N_CTX) no head offset

        # Initialize accumulators
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Load bounds
        first_nz = tl.load(FirstNz + start_m)
        last_nz = tl.load(LastNz + start_m)

        # ── Pass 1: full blocks (bm_val==2) ──
        col_idx = first_nz
        while col_idx <= last_nz:
            bm_val = tl.load(BlockMask + start_m * stride_bm_row + col_idx * stride_bm_col)
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

        # ── Pass 2: partial blocks (bm_val==1) ──
        col_idx = first_nz
        while col_idx <= last_nz:
            bm_val = tl.load(BlockMask + start_m * stride_bm_row + col_idx * stride_bm_col)
            if bm_val == 1:
                start_n = col_idx * BLOCK_N
                k_ptrs = K_base + offs_d[:, None] * stride_kk + (start_n + offs_bn)[None, :] * stride_kn
                k = tl.load(k_ptrs)
                v_ptrs = V_base + (start_n + offs_bn)[:, None] * stride_vk + offs_d[None, :] * stride_vn
                v = tl.load(v_ptrs)
                mask_ptrs = Mask_base + offs_qm[:, None] * stride_mask_m + (start_n + offs_bn)[None, :] * stride_mask_n
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

        # Finalize and store
        l_i = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_i[:, None]
        o_ptrs = Out + qvk_offset + offs_qm[:, None] * stride_om + offs_d[None, :] * stride_on
        tl.store(o_ptrs, acc.to(Out.type.element_ty))
        # LSE
        tl.store(LSE + (off_hz * N_CTX) + offs_qm, m_i + tl.math.log2(l_i))

        # Claim next work item
        work_id = tl.atomic_add(WorkCounter, 1)


_bm_cache_key = None
_bm_cache_val = None
_num_sms = None


def _get_num_sms():
    global _num_sms
    if _num_sms is None:
        _num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    return _num_sms


def binflash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: torch.Tensor, sm_scale: float | None = None,
    block_m: int = 128, block_n: int | None = None,
    block_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary-block-masked flash attention with persistent kernel + work-stealing."""
    global _bm_cache_key, _bm_cache_val
    B, H, N, D = q.shape
    if block_n is None:
        block_n = min(D, 64)
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
    num_q_blocks = N // block_m
    total_work = num_q_blocks * B * H

    # Persistent grid: fixed small size (2x SM count for good occupancy)
    num_sms = _get_num_sms()
    grid_size = 2 * num_sms
    # Work counter: zeroed for each launch
    work_counter = torch.zeros(1, dtype=torch.int32, device=q.device)

    _binflash_persistent_fwd[(grid_size,)](
        q, k, v, sm_scale, o, mask, block_mask, first_nz, last_nz, lse,
        work_counter, total_work, num_q_blocks,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        mask.stride(0), mask.stride(1),
        block_mask.stride(0), block_mask.stride(1),
        B, H, N, B * H, HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
    )
    return o
