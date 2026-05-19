"""Binary Block Masked Flash Attention (BinFlash).

Two dispatch strategies (selected at Python level):
- Scan-and-skip: low sparsity (<60%). Static loop bounds.
- Index-based: high sparsity (>=60%). Two-pass like FlexAttention:
  1. Partial blocks: load and apply mask
  2. Full blocks: skip mask (all entries True)
  Both passes use tl.advance with incremental deltas. IS_FULL_BLOCKS is constexpr
  so Triton compiles two specialized inner functions (no runtime branch).
"""

import torch
import triton
import triton.language as tl

from masks import make_block_mask


# ───────────────── Shared inner function (constexpr specialization) ─────────


@triton.jit
def _fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr, Mask_block_ptr,
    kv_indices_ptr, num_blocks,
    qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    IS_FULL_BLOCKS: tl.constexpr,
):
    """Process a list of KV blocks. IS_FULL_BLOCKS=True skips mask load."""
    prev_col = tl.zeros([], dtype=tl.int32)
    for idx in range(num_blocks):
        col_idx = tl.load(kv_indices_ptr + idx)
        delta = (col_idx - prev_col) * BLOCK_N
        K_block_ptr = tl.advance(K_block_ptr, (0, delta))
        V_block_ptr = tl.advance(V_block_ptr, (delta, 0))
        if not IS_FULL_BLOCKS:
            Mask_block_ptr = tl.advance(Mask_block_ptr, (0, delta))

        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k) * qk_scale

        if not IS_FULL_BLOCKS:
            mask = tl.load(Mask_block_ptr) != 0
            qk += tl.where(mask, 0.0, -1.0e6)

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


# ───────────────── Kernel 1: Scan-and-skip (low sparsity) ─────────────────


@triton.jit
def _scan_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr, Mask_block_ptr,
    block_mask_ptr, bm_stride_row, bm_stride_col, start_m,
    qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
):
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        col_idx = start_n // BLOCK_N
        bm_val = tl.load(block_mask_ptr + start_m * bm_stride_row + col_idx * bm_stride_col)
        if bm_val != 0:
            k = tl.load(K_block_ptr)
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
            v = tl.load(V_block_ptr)
            p = p.to(v.dtype)
            acc = tl.dot(p, v, acc)
            m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        Mask_block_ptr = tl.advance(Mask_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.autotune(
    configs=[triton.Config({}, num_stages=s, num_warps=w)
             for s in [1, 2, 3, 4] for w in [4, 8]],
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
    acc, l_i, m_i = _scan_fwd_inner(acc, l_i, m_i, q, K_bp, V_bp, M_bp, BlockMask, stride_bm_row, stride_bm_col, start_m, sm_scale * 1.44269504, BLOCK_M, BLOCK_N, HEAD_DIM, N_CTX)
    acc = acc / l_i[:, None]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(LSE + (off_hz * N_CTX) + offs_m, m_i + tl.math.log2(l_i))
    tl.store(O_bp, acc.to(Out.type.element_ty))


# ──────── Kernel 2: Two-pass index dispatch (high sparsity) ──────


@triton.autotune(
    configs=[triton.Config({}, num_stages=s, num_warps=w)
             for s in [1, 2, 3, 4] for w in [4, 8]],
    key=["N_CTX", "HEAD_DIM"],
)
@triton.jit
def _index_fwd(
    Q, K, V, sm_scale, Out, Mask,
    Partial_indices, Partial_counts,
    Full_indices, Full_counts,
    LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_mask_m, stride_mask_n,
    stride_pidx_row, stride_pidx_col,
    stride_fidx_row, stride_fidx_col,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_FULL_BLOCKS: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh
    Q_bp = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    O_bp = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_bp)

    # Pass 1: partial blocks (need mask)
    num_partial = tl.load(Partial_counts + start_m)
    if num_partial > 0:
        K_bp = tl.make_block_ptr(base=K + qvk_offset, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
        V_bp = tl.make_block_ptr(base=V + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vk, stride_vn), offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
        M_bp = tl.make_block_ptr(base=Mask, shape=(N_CTX, N_CTX), strides=(stride_mask_m, stride_mask_n), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
        partial_ptr = Partial_indices + start_m * stride_pidx_row
        acc, l_i, m_i = _fwd_inner(
            acc, l_i, m_i, q, K_bp, V_bp, M_bp,
            partial_ptr, num_partial, qk_scale,
            BLOCK_M, BLOCK_N, HEAD_DIM, IS_FULL_BLOCKS=False,
        )

    # Pass 2: full blocks (skip mask — constexpr specialization)
    if HAS_FULL_BLOCKS:
        num_full = tl.load(Full_counts + start_m)
        if num_full > 0:
            K_bp2 = tl.make_block_ptr(base=K + qvk_offset, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
            V_bp2 = tl.make_block_ptr(base=V + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vk, stride_vn), offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
            M_bp2 = tl.make_block_ptr(base=Mask, shape=(N_CTX, N_CTX), strides=(stride_mask_m, stride_mask_n), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
            full_ptr = Full_indices + start_m * stride_fidx_row
            acc, l_i, m_i = _fwd_inner(
                acc, l_i, m_i, q, K_bp2, V_bp2, M_bp2,
                full_ptr, num_full, qk_scale,
                BLOCK_M, BLOCK_N, HEAD_DIM, IS_FULL_BLOCKS=True,
            )

    # Handle fully masked rows
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


def _build_full_partial_indices(mask, block_mask, block_m, block_n):
    """Separate full and partial block indices (like FlexAttention's BlockMask)."""
    num_rows, num_cols = block_mask.shape
    block_sums = mask.view(num_rows, block_m, num_cols, block_n).sum(dim=(1, 3))
    is_full = block_sums == (block_m * block_n)
    is_partial = block_mask & ~is_full

    partial_indices, partial_counts = _build_kv_indices(is_partial)
    full_indices, full_counts = _build_kv_indices(is_full)
    has_full = is_full.any().item()

    return partial_indices, partial_counts, full_indices, full_counts, has_full


# ────────────────────────── Python wrapper ──────────────────────────


_SPARSITY_THRESHOLD = 0.6


def binflash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: torch.Tensor, sm_scale: float | None = None,
    block_m: int = 128, block_n: int = 64,
    block_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary-block-masked flash attention."""
    B, H, N, D = q.shape
    assert N % block_m == 0 and N % block_n == 0
    assert block_n <= D and D in {16, 32, 64, 128}
    if sm_scale is None:
        sm_scale = D ** -0.5
    if block_mask is None:
        block_mask = make_block_mask(mask, block_m, block_n)

    sparsity = 1.0 - block_mask.float().mean().item()
    o = torch.empty_like(q)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    mask_int = mask.to(torch.int8)
    grid = lambda META: (triton.cdiv(N, block_m), B * H)

    if sparsity > _SPARSITY_THRESHOLD:
        pi, pc, fi, fc, has_full = _build_full_partial_indices(mask, block_mask, block_m, block_n)
        _index_fwd[grid](
            q, k, v, sm_scale, o, mask_int,
            pi, pc, fi, fc, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            mask_int.stride(0), mask_int.stride(1),
            pi.stride(0), pi.stride(1),
            fi.stride(0), fi.stride(1),
            B, H, N, HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
            HAS_FULL_BLOCKS=has_full,
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
        )
    return o
