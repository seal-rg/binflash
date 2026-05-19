"""Binary Block Masked Flash Attention (BinFlash).

Dual dispatch: scan-and-skip for low sparsity, index+tl.advance for high sparsity.
Selected at Python level based on block sparsity threshold.
"""

import torch
import triton
import triton.language as tl

from masks import make_block_mask


# ───────────────── Kernel 1: Scan-and-skip (low sparsity) ─────────────────


@triton.jit
def _scan_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr, Mask_block_ptr,
    block_mask_ptr, bm_stride_row, bm_stride_col, start_m,
    qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
    MASK_MODE: tl.constexpr,
):
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        col_idx = start_n // BLOCK_N
        bm_val = tl.load(block_mask_ptr + start_m * bm_stride_row + col_idx * bm_stride_col)
        # Speculative loads: unconditional so Triton can pipeline across iterations.
        # Wasted bandwidth for empty blocks is negligible (<1% of total).
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        if bm_val != 0:
            qk = tl.dot(q, k) * qk_scale
            mask = tl.load(Mask_block_ptr) != 0
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
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        Mask_block_ptr = tl.advance(Mask_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.autotune(
    configs=[triton.Config({}, num_stages=s, num_warps=w)
             for s in [2, 3, 4] for w in [4, 8]],
    key=["N_CTX", "HEAD_DIM"],
)
@triton.jit
def _scan_fwd(
    Q, K, V, sm_scale, Out, Mask, BlockMask, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_mask_m, stride_mask_n,
    stride_bm_row, stride_bm_col,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    MASK_MODE: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh
    Q_bp = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    K_bp = tl.make_block_ptr(base=K + qvk_offset, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    V_bp = tl.make_block_ptr(base=V + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vk, stride_vn), offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    O_bp = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    M_bp = tl.make_block_ptr(base=Mask, shape=(N_CTX, N_CTX), strides=(stride_mask_m, stride_mask_n), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_bp)
    acc, l_i, m_i = _scan_fwd_inner(acc, l_i, m_i, q, K_bp, V_bp, M_bp, BlockMask, stride_bm_row, stride_bm_col, start_m, sm_scale * 1.44269504, BLOCK_M, BLOCK_N, HEAD_DIM, N_CTX, MASK_MODE)
    acc = acc / l_i[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(LSE + (off_hz * N_CTX) + offs_m, m_i + tl.math.log2(l_i))
    tl.store(O_bp, acc.to(Out.type.element_ty))


# ───────────────── Kernel 2: Index dispatch (high sparsity) ─────────────────


@triton.jit
def _index_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr, Mask_block_ptr,
    kv_indices_ptr, num_blocks, start_m,
    qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    MASK_MODE: tl.constexpr,
):
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    prev_col = tl.zeros([], dtype=tl.int32)
    for idx in range(num_blocks):
        col_idx = tl.load(kv_indices_ptr + idx)
        delta = (col_idx - prev_col) * BLOCK_N
        K_block_ptr = tl.advance(K_block_ptr, (0, delta))
        V_block_ptr = tl.advance(V_block_ptr, (delta, 0))
        if MASK_MODE == 0:
            Mask_block_ptr = tl.advance(Mask_block_ptr, (0, delta))
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k) * qk_scale
        if MASK_MODE == 0:
            mask = tl.load(Mask_block_ptr) != 0
            qk += tl.where(mask, 0.0, -1.0e6)
        elif MASK_MODE == 1:
            kv_start = col_idx * BLOCK_N
            offs_n = kv_start + tl.arange(0, BLOCK_N)
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk += tl.where(causal_mask, 0.0, -1.0e6)
        # MASK_MODE == 2: no mask
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
        prev_col = col_idx
    return acc, l_i, m_i


@triton.autotune(
    configs=[triton.Config({}, num_stages=s, num_warps=w)
             for s in [2, 3, 4] for w in [4, 8]],
    key=["N_CTX", "HEAD_DIM"],
)
@triton.jit
def _index_fwd(
    Q, K, V, sm_scale, Out, Mask, KV_indices, KV_num_blocks, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_mask_m, stride_mask_n,
    stride_idx_row, stride_idx_col,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    MASK_MODE: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh
    Q_bp = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    K_bp = tl.make_block_ptr(base=K + qvk_offset, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    V_bp = tl.make_block_ptr(base=V + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vk, stride_vn), offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    O_bp = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    M_bp = tl.make_block_ptr(base=Mask, shape=(N_CTX, N_CTX), strides=(stride_mask_m, stride_mask_n), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    num_blocks = tl.load(KV_num_blocks + start_m)
    kv_idx_ptr = KV_indices + start_m * stride_idx_row
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_bp)
    acc, l_i, m_i = _index_fwd_inner(acc, l_i, m_i, q, K_bp, V_bp, M_bp, kv_idx_ptr, num_blocks, start_m, sm_scale * 1.44269504, BLOCK_M, BLOCK_N, HEAD_DIM, MASK_MODE)
    acc = acc / l_i[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(LSE + (off_hz * N_CTX) + offs_m, m_i + tl.math.log2(l_i))
    tl.store(O_bp, acc.to(Out.type.element_ty))


# ───────────────── Kernel 3: Causal scan (no mask load) ─────────────────


@triton.jit
def _causal_scan_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    start_m, qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
):
    """Causal attention: compute q_idx >= kv_idx on-the-fly. No mask load."""
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Only iterate up to (start_m + 1) * BLOCK_M blocks (causal: nothing beyond diagonal)
    hi = (start_m + 1) * BLOCK_M
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if start_n < hi:
            k = tl.load(K_block_ptr)
            qk = tl.dot(q, k) * qk_scale
            offs_n = start_n + tl.arange(0, BLOCK_N)
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk += tl.where(causal_mask, 0.0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            v = tl.load(V_block_ptr)
            p = p.to(v.dtype)
            acc = tl.dot(p, v, acc)
            m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.autotune(
    configs=[triton.Config({}, num_stages=s, num_warps=w)
             for s in [2, 3, 4] for w in [4, 8]],
    key=["N_CTX", "HEAD_DIM"],
)
@triton.jit
def _causal_fwd(
    Q, K, V, sm_scale, Out, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh
    Q_bp = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    K_bp = tl.make_block_ptr(base=K + qvk_offset, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    V_bp = tl.make_block_ptr(base=V + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vk, stride_vn), offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    O_bp = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_bp)
    acc, l_i, m_i = _causal_scan_inner(acc, l_i, m_i, q, K_bp, V_bp, start_m, sm_scale * 1.44269504, BLOCK_M, BLOCK_N, HEAD_DIM, N_CTX)
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(LSE + (off_hz * N_CTX) + offs_m, m_i + tl.math.log2(l_i))
    tl.store(O_bp, acc.to(Out.type.element_ty))


# ───────── Kernel 4: Maskless scan (all-full-blocks patterns) ─────────


@triton.jit
def _maskless_scan_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    block_mask_ptr, bm_stride_row, bm_stride_col, start_m,
    qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
):
    """Scan-and-skip with NO mask load — for patterns where all non-zero blocks are full."""
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        col_idx = start_n // BLOCK_N
        bm_val = tl.load(block_mask_ptr + start_m * bm_stride_row + col_idx * bm_stride_col)
        if bm_val != 0:
            k = tl.load(K_block_ptr)
            qk = tl.dot(q, k) * qk_scale
            # No mask load — all entries in non-zero blocks are True
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            v = tl.load(V_block_ptr)
            p = p.to(v.dtype)
            acc = tl.dot(p, v, acc)
            m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.autotune(
    configs=[triton.Config({}, num_stages=s, num_warps=w)
             for s in [2, 3, 4] for w in [4, 8]],
    key=["N_CTX", "HEAD_DIM"],
)
@triton.jit
def _maskless_scan_fwd(
    Q, K, V, sm_scale, Out, BlockMask, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_bm_row, stride_bm_col,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh
    Q_bp = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    K_bp = tl.make_block_ptr(base=K + qvk_offset, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    V_bp = tl.make_block_ptr(base=V + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vk, stride_vn), offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    O_bp = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_bp)
    acc, l_i, m_i = _maskless_scan_inner(acc, l_i, m_i, q, K_bp, V_bp, BlockMask, stride_bm_row, stride_bm_col, start_m, sm_scale * 1.44269504, BLOCK_M, BLOCK_N, HEAD_DIM, N_CTX)
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(LSE + (off_hz * N_CTX) + offs_m, m_i + tl.math.log2(l_i))
    tl.store(O_bp, acc.to(Out.type.element_ty))


# ───────────────── Kernel 4: Sliding window scan (computed mask) ─────────────────


@triton.jit
def _window_scan_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    start_m, qk_scale, WINDOW_SIZE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
):
    """Sliding window: |q_idx - kv_idx| <= W. Computed mask, bounded iteration."""
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Window bounds: Q tokens in [q_lo, q_hi), window extends to [q_lo - W, q_hi + W)
    q_lo = start_m * BLOCK_M
    q_hi = q_lo + BLOCK_M
    kv_lo = q_lo - WINDOW_SIZE  # can be negative
    kv_hi = q_hi + WINDOW_SIZE  # can exceed N_CTX
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_block_end = start_n + BLOCK_N
        # Skip blocks entirely outside the window
        if start_n < kv_hi and kv_block_end > kv_lo:
            k = tl.load(K_block_ptr)
            qk = tl.dot(q, k) * qk_scale
            # Compute window mask on-the-fly
            offs_n = start_n + tl.arange(0, BLOCK_N)
            diff = offs_m[:, None] - offs_n[None, :]
            window_mask = (diff >= -WINDOW_SIZE) & (diff <= WINDOW_SIZE)
            qk += tl.where(window_mask, 0.0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            v = tl.load(V_block_ptr)
            p = p.to(v.dtype)
            acc = tl.dot(p, v, acc)
            m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.autotune(
    configs=[triton.Config({}, num_stages=s, num_warps=w)
             for s in [2, 3, 4] for w in [4, 8]],
    key=["N_CTX", "HEAD_DIM"],
)
@triton.jit
def _window_fwd(
    Q, K, V, sm_scale, Out, LSE, WINDOW_SIZE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh
    Q_bp = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    K_bp = tl.make_block_ptr(base=K + qvk_offset, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    V_bp = tl.make_block_ptr(base=V + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vk, stride_vn), offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    O_bp = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_bp)
    acc, l_i, m_i = _window_scan_inner(acc, l_i, m_i, q, K_bp, V_bp, start_m, sm_scale * 1.44269504, WINDOW_SIZE, BLOCK_M, BLOCK_N, HEAD_DIM, N_CTX)
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(LSE + (off_hz * N_CTX) + offs_m, m_i + tl.math.log2(l_i))
    tl.store(O_bp, acc.to(Out.type.element_ty))


# ────────────────────────── Preprocessing ──────────────────────────


def _build_kv_indices(block_mask: torch.Tensor):
    num_rows, num_cols = block_mask.shape
    kv_num_blocks = block_mask.sum(dim=1).to(torch.int32)
    col_indices = torch.arange(num_cols, device=block_mask.device).expand(num_rows, -1)
    sorted_order = block_mask.int().mul(-1).argsort(dim=1, stable=True)
    kv_indices = col_indices.gather(1, sorted_order).to(torch.int32)
    return kv_num_blocks, kv_indices


# ────────────────────────── Python wrapper ──────────────────────────


def _detect_mask_mode(mask: torch.Tensor, block_mask: torch.Tensor, N: int) -> int:
    """Cheaply detect if mask matches a known pattern. Returns MASK_MODE constexpr.
    0 = general (load from memory), 1 = causal, 2 = full (no mask),
    3 = block-diagonal (all non-zero blocks are fully True).
    """
    # Check if mask is all-True (full attention)
    if mask.all():
        return 2

    # Check if mask is lower-triangular (causal)
    if N > 1 and not mask[0, N - 1].item() and mask[N - 1, 0].item():
        checks = min(N, 16)
        step = max(1, N // checks)
        is_causal = True
        for i in range(0, N, step):
            j_false = min(i + 1, N - 1)
            j_true = max(i - 1, 0)
            if j_false < N and i < j_false and mask[i, j_false].item():
                is_causal = False
                break
            if i > j_true and not mask[i, j_true].item():
                is_causal = False
                break
        if is_causal:
            return 1

    # Check if all non-zero blocks are fully True (block-diagonal-like)
    # This means we never need to load the fine-grained mask
    num_rows, num_cols = block_mask.shape
    block_m = N // num_rows
    block_n = N // num_cols
    nz_blocks = block_mask.sum().item()
    if nz_blocks > 0:
        block_sums = mask.float().view(num_rows, block_m, num_cols, block_n).sum(dim=(1, 3))
        full_blocks = (block_sums == block_m * block_n).sum().item()
        if full_blocks == nz_blocks:
            return 3  # all non-zero blocks are full → no mask needed

    return 0  # general


def binflash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: torch.Tensor, sm_scale: float | None = None,
    block_m: int = 128, block_n: int = 64,
    block_mask: torch.Tensor | None = None,
    mask_mode: int | None = None,
) -> torch.Tensor:
    """Binary-block-masked flash attention.

    Args:
        mask_mode: 0=general (load mask), 1=causal (compute), 2=full (skip mask).
                   If None, auto-detected from the mask.
    """
    B, H, N, D = q.shape
    assert N % block_m == 0 and N % block_n == 0
    assert block_n <= D and D in {16, 32, 64, 128}
    if sm_scale is None:
        sm_scale = D ** -0.5
    if block_mask is None:
        block_mask = make_block_mask(mask, block_m, block_n)

    if mask_mode is None:
        mask_mode = _detect_mask_mode(mask, block_mask, N)

    sparsity = 1.0 - block_mask.float().mean().item()
    total_blocks_per_row = N // block_n
    use_index = sparsity > 0.8 and total_blocks_per_row >= 128

    o = torch.empty_like(q)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    mask_int = mask.to(torch.int8)
    grid = lambda META: (triton.cdiv(N, block_m), B * H)

    if use_index:
        kv_num_blocks, kv_indices = _build_kv_indices(block_mask)
        _index_fwd[grid](
            q, k, v, sm_scale, o, mask_int, kv_indices, kv_num_blocks, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            mask_int.stride(0), mask_int.stride(1),
            kv_indices.stride(0), kv_indices.stride(1),
            B, H, N, HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
            MASK_MODE=mask_mode,
        )
    else:
        bm_int = block_mask.to(torch.int8)
        _scan_fwd[grid](
            q, k, v, sm_scale, o, mask_int, bm_int, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            mask_int.stride(0), mask_int.stride(1),
            bm_int.stride(0), bm_int.stride(1),
            B, H, N, HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
            MASK_MODE=mask_mode,
        )
    return o
