"""BinFlash v14: Split-KV parallelism.

Radically different architecture:
- Phase 1: One thread block per (Q-row, KV-block) pair. No inner loop.
  Each computes partial attention and stores (output_partial, m, l).
- Phase 2: Reduction kernel combines partial results per Q-row using
  online softmax correction.

Advantages: zero wasted iterations, no branching, perfect load balancing.
Disadvantage: extra global memory for partial results + reduction kernel overhead.
"""

import torch
import triton
import triton.language as tl

from masks import make_block_mask


# ─────────── Phase 1: Compute partial attention per (Q-row, KV-block) ───────


@triton.jit
def _splitkv_phase1(
    Q, K, V, Mask, KV_indices, KV_num_blocks,
    Partial_out, Partial_m, Partial_l,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_mask_m, stride_mask_n,
    stride_idx_row, stride_idx_col,
    stride_po_row, stride_po_kv, stride_po_m, stride_po_d,
    Z, H, N_CTX, MAX_KV_BLOCKS,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Each thread block handles one (Q-row, KV-block-index) pair."""
    q_row = tl.program_id(0)
    kv_idx = tl.program_id(1)  # which of the non-zero KV blocks for this Q-row
    off_hz = tl.program_id(2)
    off_z = off_hz // H
    off_h = off_hz % H

    # Check if this KV block index is valid for this Q-row
    num_blocks = tl.load(KV_num_blocks + q_row)
    if kv_idx >= num_blocks:
        # Padding block — write identity values
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        po_base = off_hz * (tl.cdiv(N_CTX, BLOCK_M)) * MAX_KV_BLOCKS
        m_ptr = Partial_m + (po_base + q_row * MAX_KV_BLOCKS + kv_idx) * BLOCK_M + offs_m
        l_ptr = Partial_l + (po_base + q_row * MAX_KV_BLOCKS + kv_idx) * BLOCK_M + offs_m
        tl.store(m_ptr, float("-inf") + tl.zeros([BLOCK_M], dtype=tl.float32))
        tl.store(l_ptr, tl.zeros([BLOCK_M], dtype=tl.float32))
        return

    # Load the actual KV column index
    col_idx = tl.load(KV_indices + q_row * stride_idx_row + kv_idx * stride_idx_col)
    kv_offset = col_idx * BLOCK_N

    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # Load Q
    Q_bp = tl.make_block_ptr(
        base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk), offsets=(q_row * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    q = tl.load(Q_bp)

    # Load K at the right column
    K_bp = tl.make_block_ptr(
        base=K + qvk_offset, shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn), offsets=(0, kv_offset),
        block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1),
    )
    k = tl.load(K_bp)

    # Compute QK^T
    qk_scale = sm_scale * 1.44269504
    qk = tl.dot(q, k) * qk_scale

    # Apply mask
    M_bp = tl.make_block_ptr(
        base=Mask, shape=(N_CTX, N_CTX),
        strides=(stride_mask_m, stride_mask_n),
        offsets=(q_row * BLOCK_M, kv_offset),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    mask = tl.load(M_bp) != 0
    qk += tl.where(mask, 0.0, -1.0e6)

    # Local softmax
    m_ij = tl.max(qk, 1)  # (BLOCK_M,)
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk)  # (BLOCK_M, BLOCK_N)
    l_ij = tl.sum(p, 1)   # (BLOCK_M,)

    # Load V and compute partial output
    V_bp = tl.make_block_ptr(
        base=V + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn), offsets=(kv_offset, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0),
    )
    v = tl.load(V_bp)
    p = p.to(v.dtype)
    partial_out = tl.dot(p, v)  # (BLOCK_M, HEAD_DIM) — unnormalized

    # Store partial results
    num_q_rows = tl.cdiv(N_CTX, BLOCK_M)
    po_base = off_hz * num_q_rows * MAX_KV_BLOCKS
    slot = po_base + q_row * MAX_KV_BLOCKS + kv_idx

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    # Store partial output: (BLOCK_M, HEAD_DIM)
    out_ptrs = Partial_out + slot * BLOCK_M * HEAD_DIM + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    tl.store(out_ptrs, partial_out.to(Partial_out.type.element_ty))
    # Store m and l: (BLOCK_M,)
    tl.store(Partial_m + slot * BLOCK_M + offs_m, m_ij)
    tl.store(Partial_l + slot * BLOCK_M + offs_m, l_ij)


# ─────────── Phase 2: Reduce partial results per Q-row ───────


@triton.jit
def _splitkv_phase2(
    Partial_out, Partial_m, Partial_l, KV_num_blocks,
    Out,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX, MAX_KV_BLOCKS,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """Combine partial results for each Q-row using online softmax correction."""
    q_row = tl.program_id(0)
    off_hz = tl.program_id(1)

    num_q_rows = tl.cdiv(N_CTX, BLOCK_M)
    num_blocks = tl.load(KV_num_blocks + q_row)
    po_base = off_hz * num_q_rows * MAX_KV_BLOCKS

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Initialize with first partial result
    slot0 = po_base + q_row * MAX_KV_BLOCKS
    m_acc = tl.load(Partial_m + slot0 * BLOCK_M + offs_m)  # (BLOCK_M,)
    l_acc = tl.load(Partial_l + slot0 * BLOCK_M + offs_m)  # (BLOCK_M,)
    out_ptrs0 = Partial_out + slot0 * BLOCK_M * HEAD_DIM + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    acc = tl.load(out_ptrs0).to(tl.float32)  # (BLOCK_M, HEAD_DIM)

    # Combine remaining partial results
    for i in range(1, MAX_KV_BLOCKS):
        if i < num_blocks:
            slot_i = po_base + q_row * MAX_KV_BLOCKS + i
            m_i = tl.load(Partial_m + slot_i * BLOCK_M + offs_m)
            l_i = tl.load(Partial_l + slot_i * BLOCK_M + offs_m)
            out_ptrs_i = Partial_out + slot_i * BLOCK_M * HEAD_DIM + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
            out_i = tl.load(out_ptrs_i).to(tl.float32)

            # Online softmax correction
            m_new = tl.maximum(m_acc, m_i)
            alpha = tl.math.exp2(m_acc - m_new)
            beta = tl.math.exp2(m_i - m_new)
            l_acc = l_acc * alpha + l_i * beta
            acc = acc * alpha[:, None] + out_i * beta[:, None]
            m_acc = m_new

    # Normalize
    l_acc = tl.where(l_acc == 0.0, 1.0, l_acc)
    acc = acc / l_acc[:, None]

    # Store final output
    off_z = off_hz // H
    off_h = off_hz % H
    out_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    out_ptrs = Out + out_offset + (q_row * BLOCK_M + offs_m[:, None]) * stride_om + offs_d[None, :] * stride_on
    tl.store(out_ptrs, acc.to(Out.type.element_ty))


# ─────────── Python wrapper ───────


def _build_kv_indices(block_mask):
    num_rows, num_cols = block_mask.shape
    kv_num_blocks = block_mask.sum(dim=1).to(torch.int32)
    col_indices = torch.arange(num_cols, device=block_mask.device).expand(num_rows, -1)
    sorted_order = block_mask.int().mul(-1).argsort(dim=1, stable=True)
    kv_indices = col_indices.gather(1, sorted_order).to(torch.int32)
    return kv_num_blocks, kv_indices


def binflash_splitkv_attention(
    q, k, v, mask, sm_scale=None, block_m=128, block_n=64, block_mask=None,
):
    """Split-KV block-sparse attention."""
    B, H, N, D = q.shape
    assert N % block_m == 0 and N % block_n == 0
    assert block_n <= D and D in {16, 32, 64, 128}
    if sm_scale is None:
        sm_scale = D ** -0.5
    if block_mask is None:
        block_mask = make_block_mask(mask, block_m, block_n)

    kv_num_blocks, kv_indices = _build_kv_indices(block_mask)
    max_kv_blocks = int(kv_num_blocks.max().item())
    num_q_rows = N // block_m

    mask_int = mask.to(torch.int8)

    # Allocate partial result buffers
    total_slots = B * H * num_q_rows * max_kv_blocks
    partial_out = torch.empty(total_slots * block_m * D, device=q.device, dtype=q.dtype)
    partial_m = torch.empty(total_slots * block_m, device=q.device, dtype=torch.float32)
    partial_l = torch.empty(total_slots * block_m, device=q.device, dtype=torch.float32)

    # Phase 1: compute partials
    grid1 = (num_q_rows, max_kv_blocks, B * H)
    _splitkv_phase1[grid1](
        q, k, v, mask_int, kv_indices, kv_num_blocks,
        partial_out, partial_m, partial_l,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        mask_int.stride(0), mask_int.stride(1),
        kv_indices.stride(0), kv_indices.stride(1),
        block_m * D, D, D, 1,  # partial_out strides (slot, BLOCK_M row, HEAD_DIM)
        B, H, N, max_kv_blocks,
        HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
    )

    # Phase 2: reduce
    o = torch.empty_like(q)
    grid2 = (num_q_rows, B * H)
    _splitkv_phase2[grid2](
        partial_out, partial_m, partial_l, kv_num_blocks,
        o,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, N, max_kv_blocks,
        HEAD_DIM=D, BLOCK_M=block_m,
    )

    return o
