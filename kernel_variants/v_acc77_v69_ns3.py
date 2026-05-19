"""BinFlash backward pass — v52b production (promoted 2026-04-24).

Four updates in this snapshot:
1. **v52b (NEW)**: dQ inner loops use `tl.range(num_stages=2)` for cross-iteration
   K, V load pipelining. dQ has fewer matmuls (3 vs 4 in dKdV) and 1 accumulator
   (vs 2), so its SMEM/register pressure is much lower → tl.range pipelining helps,
   even though it hurt dKdV (v38 finding still applies). dKdV's tl.range stays
   unrolled (v38). Also tested ns=3 on dQ: regression (+3.4% mean), too aggressive.

   Full 165-cell bench (v52b vs v51): **−0.54% time-weighted**. Helps biggest
   workloads most (qwen3-32b −0.78%); near-zero on small workloads (70B, 27B).

2. **v51**: AUTOTUNE `_bwd_preprocess_kernel` over BLOCK_M ∈ {64, 128, 256},
   num_stages ∈ {1, 2, 3}, num_warps ∈ {4, 8}. Triton picks BM=256 in most cases.

   Full 165-cell benchmark (v51 vs v38): **−3.63% time-weighted aggregate**.
   Per-workload: llama3-8b −4.79%, 7B-batch8 −3.52%, qwen3-32b −3.42%,
   70B-kv-heads −3.39%, 27B-kv-heads −3.56%.

Cumulative v51+v52b vs v38: ~−4.2% time-weighted.
2. **Triton-kernel _compute_bwd_block_mask** (replaces 4 PyTorch reductions over
   the N×N bool mask). 1.5-2.0x faster on the bool→int8 reduction. Cached in
   _autograd_preproc so doesn't show in benchmark median timing, but matters
   for real workloads where the mask changes per call.
3. **v38** (already promoted): REMOVE `num_stages=3` from `tl.range(...)` in bwd inner loops.

Both `@triton.autotune(num_stages=...)` (kernel-level) and `tl.range(num_stages=...)`
(loop-level) pipeline loads; keeping both creates a DOUBLE-PIPELINING conflict
that hurts SMEM/register allocation. Keeping only the autotune-controlled
pipelining (letting the autotuner pick the right depth for each shape) is
strictly better. The change removes the conflict and yields **uniform
−2 to −3.4% wins on ALL 11 patterns** vs v26.

v26 foundation kept unchanged:
- N-gated BM=128 BN=32 dQ dispatch (wins on dense patterns at large N)
- work_ratio gate to route gap-heavy patterns back to BM=64

Full-matrix 165-cell bench (v38 vs v26):
  - Aggregate −3.04% time-weighted across all 165 cells.
  - Per-pattern (weighted), ALL strictly negative:
      sparse_10 −2.98%, sparse_30 −2.91%, prefix_lm −3.31%, causal −3.38%,
      log_tree −3.16%, causal_doc_7 −2.90%, causal_window −2.62%,
      sliding_window −2.55%, longformer −2.46%, cdw_sinks −2.33%,
      block_diag −1.46%.
  - Cumulative v4 → v38: roughly −6.5% (v26 −3.54% compounded with v38 −3.04%).

v22 and v26 kept as past_attempts_bwd snapshots.
"""

from __future__ import annotations

import torch  # type: ignore
import triton  # type: ignore
import triton.language as tl  # type: ignore

from binflash_attention import (
    _build_gathered_indices,
    _dispatch_and_preprocess,
    _gathered_fwd,
)


# ─── Preprocessing: per-row delta = sum(dO * O) ───
# v51: autotune over BLOCK_M, num_warps, num_stages. Memory-bound kernel; bigger
# tiles reduce program-launch overhead. dKdV/dQ inner loop reads delta per row,
# so BM here is independent of dkdv's BM.

_preproc_autotune_configs = [
    triton.Config({"BLOCK_M": bm}, num_stages=ns, num_warps=nw)
    for bm in [64, 128, 256]
    for ns in [1, 2, 3]
    for nw in [4, 8]
]


@triton.autotune(configs=_preproc_autotune_configs, key=["N_CTX", "HEAD_DIM", "Z_TIMES_H"])
@triton.jit
def _bwd_preprocess_kernel(
    Out,
    DO,
    Delta,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_doz, stride_doh, stride_dom, stride_don,
    Z, H, N_CTX, Z_TIMES_H,
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


# ─── Autotune configs for bwd kernels (mirror fwd's _autotune_configs) ───

_bwd_autotune_configs = []
# v_acc69: extended autotune to address fp16-cast lottery — more nw, more maxnreg options
for _s in [1, 2, 3]:
    for _w in [2, 4, 8]:
        for _mnr in [None, 128, 168, 192, 224]:
            _kw = {}
            if _mnr is not None:
                _kw["maxnreg"] = _mnr
            _bwd_autotune_configs.append(triton.Config({}, num_stages=_s, num_warps=_w, **_kw))


# ─── v68: FUSED dKdV+dQ kernel via atomic_add (FA-Triton style) ───
# Single kernel computes dK, dV (registers) AND dq via atomic_add.
# Saves 2 redundant matmuls (qk, dp) per (Q,K) pair vs separate dQ kernel.
# FORCED num_warps=8 to relieve register pressure (dq_partial intermediate adds
# (BM, D) fp32 = 16KB transient regs).

_bwd_fused_configs = [
    triton.Config({}, num_stages=1, num_warps=4),
]


@triton.autotune(
    configs=_bwd_fused_configs,
    key=["N_CTX", "HEAD_DIM", "Z_TIMES_H", "BLOCK_M", "BLOCK_N"],
)
@triton.jit
def _bwd_fused_kernel(
    Q, K, V, sm_scale,
    DO, DK, DV, DQ,
    MaskPacked,
    LSE, Delta,
    RowIndicesKT, NFullKT, NTotalKT,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_doz, stride_doh, stride_dom, stride_don,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvk, stride_dvn,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_mp_m, stride_mp_n,
    stride_ri_row,
    Z, H, N_CTX,
    Z_TIMES_H,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset_q = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh
    qvk_offset_k = (off_hz // H).to(tl.int64) * stride_kz + (off_hz % H).to(tl.int64) * stride_kh
    qvk_offset_v = (off_hz // H).to(tl.int64) * stride_vz + (off_hz % H).to(tl.int64) * stride_vh
    qvk_offset_do = (off_hz // H).to(tl.int64) * stride_doz + (off_hz % H).to(tl.int64) * stride_doh
    qvk_offset_dk = (off_hz // H).to(tl.int64) * stride_dkz + (off_hz % H).to(tl.int64) * stride_dkh
    qvk_offset_dv = (off_hz // H).to(tl.int64) * stride_dvz + (off_hz % H).to(tl.int64) * stride_dvh
    qvk_offset_dq = (off_hz // H).to(tl.int64) * stride_dqz + (off_hz % H).to(tl.int64) * stride_dqh

    K_bp = tl.make_block_ptr(
        base=K + qvk_offset_k, shape=(N_CTX, HEAD_DIM),
        strides=(stride_kn, stride_kk), offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0),
    )
    V_bp = tl.make_block_ptr(
        base=V + qvk_offset_v, shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn), offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0),
    )
    k = tl.load(K_bp)
    v = tl.load(V_bp)
    k_scaled = (k * (sm_scale * 1.44269504)).to(tl.float16)  # v_acc68: fp16 k_scaled (dKdV only — q_scaled in dQ stays bf16)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    n_full = tl.load(NFullKT + start_n)
    n_total = tl.load(NTotalKT + start_n)
    tl.assume(n_full >= 0)
    tl.assume(n_total >= n_full)
    ri_base = RowIndicesKT + start_n * stride_ri_row

    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = tl.arange(0, BLOCK_M)
    BN_PACKED: tl.constexpr = BLOCK_N // 32
    offs_bn_packed = tl.arange(0, BN_PACKED)
    bit_offs = tl.arange(0, 32)
    start_n_packed = (start_n * BLOCK_N) // 32
    Q_base = Q + qvk_offset_q
    DO_base = DO + qvk_offset_do
    DQ_base = DQ + qvk_offset_dq
    MaskPacked_base = MaskPacked
    # Conversion factor: dq_partial = ds @ k_scaled * (1/log2(e)) = ds @ (k * sm_scale) * 1
    # (since k_scaled = k * sm_scale * log2(e), then ds @ k_scaled / log2(e) = ds @ k * sm_scale)
    SCALE_FOR_DQ = 0.6931471805599453  # ln(2) = 1 / log2(e)

    # Pass 1: full Q-blocks
    for idx in tl.range(0, n_full):
        row_idx = tl.load(ri_base + idx)
        m_cur = row_idx * BLOCK_M + offs_m
        q_ptrs = Q_base + m_cur[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = DO_base + m_cur[:, None] * stride_dom + offs_d[None, :] * stride_don
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        lse_i = tl.load(LSE + off_hz * N_CTX + m_cur)
        d_i = tl.load(Delta + off_hz * N_CTX + m_cur)

        qk = tl.dot(q.to(tl.float16), tl.trans(k_scaled))  # v_acc68: cast q to fp16 to match k_scaled fp16
        p = tl.math.exp2(qk - lse_i[:, None])
        dv = tl.dot(tl.trans(p.to(tl.float16)), do.to(tl.float16), dv)  # v_acc51: fp16 dV+dK
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - d_i[:, None])
        dk = tl.dot(tl.trans(ds.to(tl.float16)), q.to(tl.float16), dk)  # v_acc51: fp16 dV+dK
        # dq via atomic_add
        dq_partial = tl.dot(ds.to(k.dtype), k).to(tl.float32) * sm_scale
        dq_ptrs = DQ_base + m_cur[:, None] * stride_dqm + offs_d[None, :] * stride_dqk
        tl.atomic_add(dq_ptrs, dq_partial)

    # Pass 2: partial Q-blocks
    for idx in tl.range(n_full, n_total):
        row_idx = tl.load(ri_base + idx)
        m_cur = row_idx * BLOCK_M + offs_m
        q_ptrs = Q_base + m_cur[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = DO_base + m_cur[:, None] * stride_dom + offs_d[None, :] * stride_don
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        lse_i = tl.load(LSE + off_hz * N_CTX + m_cur)
        d_i = tl.load(Delta + off_hz * N_CTX + m_cur)

        qk = tl.dot(q.to(tl.float16), tl.trans(k_scaled))  # v_acc68: cast q to fp16 to match k_scaled fp16
        mp_ptrs = MaskPacked_base + m_cur[:, None] * stride_mp_m + (start_n_packed + offs_bn_packed)[None, :] * stride_mp_n
        packed = tl.load(mp_ptrs)
        bits_3d = (packed[:, :, None] >> bit_offs[None, None, :]) & 1
        mask_ = tl.reshape(bits_3d, (BLOCK_M, BLOCK_N)) != 0
        qk += tl.where(mask_, 0.0, -1.0e6)
        p = tl.math.exp2(qk - lse_i[:, None])
        dv = tl.dot(tl.trans(p.to(tl.float16)), do.to(tl.float16), dv)  # v_acc51: fp16 dV+dK
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - d_i[:, None])
        dk = tl.dot(tl.trans(ds.to(tl.float16)), q.to(tl.float16), dk)  # v_acc51: fp16 dV+dK
        # dq via atomic_add
        dq_partial = tl.dot(ds.to(k.dtype), k).to(tl.float32) * sm_scale
        dq_ptrs = DQ_base + m_cur[:, None] * stride_dqm + offs_d[None, :] * stride_dqk
        tl.atomic_add(dq_ptrs, dq_partial)

    dk = dk * sm_scale

    DK_bp = tl.make_block_ptr(
        base=DK + qvk_offset_dk, shape=(N_CTX, HEAD_DIM),
        strides=(stride_dkn, stride_dkk), offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0),
    )
    DV_bp = tl.make_block_ptr(
        base=DV + qvk_offset_dv, shape=(N_CTX, HEAD_DIM),
        strides=(stride_dvk, stride_dvn), offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0),
    )
    tl.store(DK_bp, dk.to(DK.type.element_ty))
    tl.store(DV_bp, dv.to(DV.type.element_ty))


# ─── dK, dV inner loop (K-outer, Q-inner — "transposed fwd") ───

@triton.jit
def _bwd_dkdv_inner(
    dk, dv,
    k_scaled, v,
    Q_base, DO_base,
    Mask_base, MaskPacked_base,
    LSE, Delta,
    stride_qm, stride_qk,
    stride_dom, stride_don,
    stride_mask_m, stride_mask_n,
    stride_mp_m, stride_mp_n,
    off_hz,
    ri_base,
    n_full, n_total,
    start_n,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
    USE_PACKED: tl.constexpr,
):
    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    BN_PACKED: tl.constexpr = BLOCK_N // 32
    offs_bn_packed = tl.arange(0, BN_PACKED)
    bit_offs = tl.arange(0, 32)
    start_n_packed = (start_n * BLOCK_N) // 32

    # Pass 1: full Q-blocks — no mask.
    for idx in tl.range(0, n_full):
        row_idx = tl.load(ri_base + idx)
        m_cur = row_idx * BLOCK_M + offs_m
        q_ptrs = Q_base + m_cur[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = DO_base + m_cur[:, None] * stride_dom + offs_d[None, :] * stride_don
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)

        lse_i = tl.load(LSE + off_hz * N_CTX + m_cur)
        d_i = tl.load(Delta + off_hz * N_CTX + m_cur)

        qk = tl.dot(q.to(tl.float16), tl.trans(k_scaled))  # v_acc68: cast q to fp16 to match k_scaled fp16   # already log2-basis because k is pre-scaled
        p = tl.math.exp2(qk - lse_i[:, None])

        dv = tl.dot(tl.trans(p.to(tl.float16)), do.to(tl.float16), dv)  # v_acc51: fp16 dV+dK

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - d_i[:, None])
        dk = tl.dot(tl.trans(ds.to(tl.float16)), q.to(tl.float16), dk)  # v_acc51: fp16 dV+dK

    # Pass 2: partial Q-blocks — packed or bool mask.
    for idx in tl.range(n_full, n_total):
        row_idx = tl.load(ri_base + idx)
        m_cur = row_idx * BLOCK_M + offs_m
        q_ptrs = Q_base + m_cur[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = DO_base + m_cur[:, None] * stride_dom + offs_d[None, :] * stride_don
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)

        lse_i = tl.load(LSE + off_hz * N_CTX + m_cur)
        d_i = tl.load(Delta + off_hz * N_CTX + m_cur)

        qk = tl.dot(q.to(tl.float16), tl.trans(k_scaled))  # v_acc68: cast q to fp16 to match k_scaled fp16
        if USE_PACKED:
            mp_ptrs = MaskPacked_base + m_cur[:, None] * stride_mp_m + (start_n_packed + offs_bn_packed)[None, :] * stride_mp_n
            packed = tl.load(mp_ptrs)
            bits_3d = (packed[:, :, None] >> bit_offs[None, None, :]) & 1
            mask_ = tl.reshape(bits_3d, (BLOCK_M, BLOCK_N)) != 0
        else:
            mask_ptrs = Mask_base + m_cur[:, None] * stride_mask_m + offs_n[None, :] * stride_mask_n
            mask_ = tl.load(mask_ptrs) != 0
        qk += tl.where(mask_, 0.0, -1.0e6)
        p = tl.math.exp2(qk - lse_i[:, None])

        dv = tl.dot(tl.trans(p.to(tl.float16)), do.to(tl.float16), dv)  # v_acc51: fp16 dV+dK

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - d_i[:, None])
        dk = tl.dot(tl.trans(ds.to(tl.float16)), q.to(tl.float16), dk)  # v_acc51: fp16 dV+dK

    return dk, dv


@triton.autotune(
    configs=_bwd_autotune_configs,
    key=["N_CTX", "HEAD_DIM", "Z_TIMES_H", "BLOCK_M", "BLOCK_N", "USE_PACKED"],
)
@triton.jit
def _bwd_dkdv_kernel(
    Q, K, V, sm_scale,
    DO, DK, DV,
    Mask,
    MaskPacked,
    LSE, Delta,
    RowIndicesKT, NFullKT, NTotalKT,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_doz, stride_doh, stride_dom, stride_don,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvk, stride_dvn,
    stride_mask_m, stride_mask_n,
    stride_mp_m, stride_mp_n,
    stride_ri_row,
    Z, H, N_CTX,
    Z_TIMES_H,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    USE_PACKED: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset_q = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh
    qvk_offset_k = (off_hz // H).to(tl.int64) * stride_kz + (off_hz % H).to(tl.int64) * stride_kh
    qvk_offset_v = (off_hz // H).to(tl.int64) * stride_vz + (off_hz % H).to(tl.int64) * stride_vh
    qvk_offset_do = (off_hz // H).to(tl.int64) * stride_doz + (off_hz % H).to(tl.int64) * stride_doh
    qvk_offset_dk = (off_hz // H).to(tl.int64) * stride_dkz + (off_hz % H).to(tl.int64) * stride_dkh
    qvk_offset_dv = (off_hz // H).to(tl.int64) * stride_dvz + (off_hz % H).to(tl.int64) * stride_dvh

    K_bp = tl.make_block_ptr(
        base=K + qvk_offset_k,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_kn, stride_kk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    V_bp = tl.make_block_ptr(
        base=V + qvk_offset_v,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    k = tl.load(K_bp)
    v = tl.load(V_bp)
    k_scaled = (k * (sm_scale * 1.44269504)).to(tl.float16)  # v_acc68: fp16 k_scaled (dKdV only — q_scaled in dQ stays bf16)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    n_full = tl.load(NFullKT + start_n)
    n_total = tl.load(NTotalKT + start_n)
    tl.assume(n_full >= 0)
    tl.assume(n_total >= n_full)
    ri_base = RowIndicesKT + start_n * stride_ri_row

    dk, dv = _bwd_dkdv_inner(
        dk, dv, k_scaled, v,
        Q + qvk_offset_q, DO + qvk_offset_do,
        Mask, MaskPacked,
        LSE, Delta,
        stride_qm, stride_qk,
        stride_dom, stride_don,
        stride_mask_m, stride_mask_n,
        stride_mp_m, stride_mp_n,
        off_hz, ri_base,
        n_full, n_total, start_n, sm_scale,
        BLOCK_M, BLOCK_N, HEAD_DIM, N_CTX, USE_PACKED,
    )

    dk = dk * sm_scale

    DK_bp = tl.make_block_ptr(
        base=DK + qvk_offset_dk,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dkn, stride_dkk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    DV_bp = tl.make_block_ptr(
        base=DV + qvk_offset_dv,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dvk, stride_dvn),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(DK_bp, dk.to(DK.type.element_ty))
    tl.store(DV_bp, dv.to(DV.type.element_ty))


# ─── dQ inner loop (Q-outer, K-inner — direct port of fwd) ───

@triton.jit
def _bwd_dq_inner(
    dq,
    q_scaled, do,
    K_base, V_base,
    Mask_base, MaskPacked_base,
    lse_i, d_i,
    stride_kk, stride_kn,
    stride_vk, stride_vn,
    stride_mask_m, stride_mask_n,
    stride_mp_m, stride_mp_n,
    ci_base,
    n_full, n_total,
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

    # v52b: dQ inner has fewer matmuls + 1 accumulator vs dKdV (3 matmuls + 1 acc),
    # so its SMEM/register pressure is much less than dKdV. Add tl.range(num_stages=2)
    # for cross-iteration K, V load pipelining. dKdV's tl.range stays unrolled (v38).
    for idx in tl.range(0, n_full, num_stages=3):  # v_acc77: ns=3 deeper pipelining
        col_idx = tl.load(ci_base + idx)
        start_n = col_idx * BLOCK_N
        k_ptrs = K_base + offs_d[:, None] * stride_kk + (start_n + offs_bn)[None, :] * stride_kn
        k = tl.load(k_ptrs)
        v_ptrs = V_base + (start_n + offs_bn)[:, None] * stride_vk + offs_d[None, :] * stride_vn
        v = tl.load(v_ptrs)

        qk = tl.dot(q_scaled, k)
        p = tl.math.exp2(qk - lse_i[:, None])

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - d_i[:, None])
        dq = tl.dot(ds.to(k.dtype), tl.trans(k), dq)

    for idx in tl.range(n_full, n_total, num_stages=3):  # v_acc77: ns=3
        col_idx = tl.load(ci_base + idx)
        start_n = col_idx * BLOCK_N
        k_ptrs = K_base + offs_d[:, None] * stride_kk + (start_n + offs_bn)[None, :] * stride_kn
        k = tl.load(k_ptrs)
        v_ptrs = V_base + (start_n + offs_bn)[:, None] * stride_vk + offs_d[None, :] * stride_vn
        v = tl.load(v_ptrs)

        qk = tl.dot(q_scaled, k)
        if USE_PACKED:
            start_n_packed = (col_idx * BLOCK_N) // 32
            mp_ptrs = MaskPacked_base + offs_bm[:, None] * stride_mp_m + (start_n_packed + offs_bn_packed)[None, :] * stride_mp_n
            packed = tl.load(mp_ptrs)
            bits_3d = (packed[:, :, None] >> bit_offs[None, None, :]) & 1
            mask_ = tl.reshape(bits_3d, (BLOCK_M, BLOCK_N)) != 0
        else:
            mask_ptrs = Mask_base + offs_bm[:, None] * stride_mask_m + (start_n + offs_bn)[None, :] * stride_mask_n
            mask_ = tl.load(mask_ptrs) != 0
        qk += tl.where(mask_, 0.0, -1.0e6)
        p = tl.math.exp2(qk - lse_i[:, None])

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - d_i[:, None])
        dq = tl.dot(ds.to(k.dtype), tl.trans(k), dq)

    return dq


@triton.autotune(
    configs=_bwd_autotune_configs,
    key=["N_CTX", "HEAD_DIM", "Z_TIMES_H", "BLOCK_M", "BLOCK_N", "USE_PACKED"],
)
@triton.jit
def _bwd_dq_kernel(
    Q, K, V, sm_scale,
    DO, DQ,
    Mask,
    MaskPacked,
    LSE, Delta,
    ColIndices, NFull, NTotal,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_doz, stride_doh, stride_dom, stride_don,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_mask_m, stride_mask_n,
    stride_mp_m, stride_mp_n,
    stride_ci_row,
    Z, H, N_CTX,
    Z_TIMES_H,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    USE_PACKED: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset_q = (off_hz // H).to(tl.int64) * stride_qz + (off_hz % H).to(tl.int64) * stride_qh
    qvk_offset_k = (off_hz // H).to(tl.int64) * stride_kz + (off_hz % H).to(tl.int64) * stride_kh
    qvk_offset_v = (off_hz // H).to(tl.int64) * stride_vz + (off_hz % H).to(tl.int64) * stride_vh
    qvk_offset_do = (off_hz // H).to(tl.int64) * stride_doz + (off_hz % H).to(tl.int64) * stride_doh
    qvk_offset_dq = (off_hz // H).to(tl.int64) * stride_dqz + (off_hz % H).to(tl.int64) * stride_dqh

    Q_bp = tl.make_block_ptr(
        base=Q + qvk_offset_q,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    DO_bp = tl.make_block_ptr(
        base=DO + qvk_offset_do,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dom, stride_don),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    q = tl.load(Q_bp)
    do = tl.load(DO_bp)
    q_scaled = (q * (sm_scale * 1.44269504)).to(q.dtype)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_i = tl.load(LSE + off_hz * N_CTX + offs_m)
    d_i = tl.load(Delta + off_hz * N_CTX + offs_m)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    n_full = tl.load(NFull + start_m)
    n_total = tl.load(NTotal + start_m)
    tl.assume(n_full >= 0)
    tl.assume(n_total >= n_full)
    ci_base = ColIndices + start_m * stride_ci_row

    dq = _bwd_dq_inner(
        dq, q_scaled, do,
        K + qvk_offset_k, V + qvk_offset_v,
        Mask, MaskPacked,
        lse_i, d_i,
        stride_kk, stride_kn,
        stride_vk, stride_vn,
        stride_mask_m, stride_mask_n,
        stride_mp_m, stride_mp_n,
        ci_base,
        n_full, n_total, start_m,
        BLOCK_M, BLOCK_N, HEAD_DIM, N_CTX, USE_PACKED,
    )

    dq = dq * sm_scale

    DQ_bp = tl.make_block_ptr(
        base=DQ + qvk_offset_dq,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dqm, stride_dqk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(DQ_bp, dq.to(DQ.type.element_ty))


# ─── Autograd-path preproc cache ───

_BWD_BLOCK_M = 64
_BWD_BLOCK_N = 64
# v22: at N >= 8192, dispatch dQ to BM=128 BN=32 — halves programs while keeping
# per-tile flops identical (128*32 = 64*64). Full-matrix benchmark: -3.52% vs v4.
_BWD_DQ_BLOCK_M = 128
_BWD_DQ_BLOCK_N = 32

_ag_cache_key = None
_ag_cache_val = None


@triton.jit
def _compute_bwd_block_mask_kernel(
    Mask, BlockMask,
    stride_mask_m, stride_mask_n,
    stride_bm_m,
    nkb,
    BM_COARSE: tl.constexpr,
    BN_CHUNKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce a (BM_COARSE, BN_CHUNKS*BLOCK_N) tile of bool mask into BN_CHUNKS
    int8 entries (0=all-masked, 1=mixed, 2=all-attended). One program per
    (Q-block-row, K-chunk)."""
    row_coarse = tl.program_id(0)
    col_chunk = tl.program_id(1)
    offs_m = row_coarse * BM_COARSE + tl.arange(0, BM_COARSE)
    offs_n_base = col_chunk * (BN_CHUNKS * BLOCK_N)
    offs_n = offs_n_base + tl.arange(0, BN_CHUNKS * BLOCK_N)
    tile_ptrs = Mask + offs_m[:, None] * stride_mask_m + offs_n[None, :] * stride_mask_n
    tile = tl.load(tile_ptrs)
    tile_3d = tl.reshape(tile, (BM_COARSE, BN_CHUNKS, BLOCK_N))
    tile_i32 = tile_3d.to(tl.int32)
    sum_col = tl.sum(tile_i32, axis=0)        # (BN_CHUNKS, BLOCK_N)
    coarse_sums = tl.sum(sum_col, axis=1)     # (BN_CHUNKS,)
    has_any = coarse_sums > 0
    has_all = coarse_sums == (BM_COARSE * BLOCK_N)
    bm_val = has_any.to(tl.int8) + has_all.to(tl.int8)
    col_start = col_chunk * BN_CHUNKS + tl.arange(0, BN_CHUNKS)
    tl.store(BlockMask + row_coarse * stride_bm_m + col_start, bm_val)


def _compute_bwd_block_mask(mask, block_m, block_n):
    """Reduce (N, N) bool mask to (N/block_m, N/block_n) int8:
    0 = all masked, 1 = mixed, 2 = all attended.

    Triton implementation — single kernel that does the reduction in-tile,
    replacing 4 PyTorch reductions over the full N×N bool tensor.
    """
    N = mask.shape[0]
    nqb = N // block_m
    nkb = N // block_n
    block_mask = torch.empty(nqb, nkb, dtype=torch.int8, device=mask.device)
    # Pick BN_CHUNKS to keep tile size ≤ 64KB and divide nkb evenly.
    bn_chunks = 8
    while bn_chunks > 1 and (nkb % bn_chunks != 0 or block_m * bn_chunks * block_n > 65536):
        bn_chunks //= 2
    grid = (nqb, nkb // bn_chunks)
    _compute_bwd_block_mask_kernel[grid](
        mask, block_mask,
        mask.stride(0), mask.stride(1),
        block_mask.stride(0),
        nkb,
        BM_COARSE=block_m,
        BN_CHUNKS=bn_chunks,
        BLOCK_N=block_n,
    )
    return block_mask


def _autograd_preproc(mask, D):
    """Compute (and cache) all fwd + bwd preproc outputs for the autograd path."""
    global _ag_cache_key, _ag_cache_val
    key = (mask.data_ptr(), mask.shape[0], mask.shape[1], D)
    if key == _ag_cache_key:
        return _ag_cache_val

    col_indices_fwd, n_full_fwd, n_total_fwd, mask_packed_fwd, chosen_bm, chosen_bn, use_packed = (
        _dispatch_and_preprocess(mask, D)
    )

    N = mask.shape[0]
    bwd_block_mask = _compute_bwd_block_mask(mask, _BWD_BLOCK_M, _BWD_BLOCK_N)
    col_indices_bwd, n_full_bwd, n_total_bwd = _build_gathered_indices(bwd_block_mask)
    bwd_block_mask_kt = bwd_block_mask.T.contiguous()
    row_indices_kt, n_full_kt, n_total_kt = _build_gathered_indices(bwd_block_mask_kt)

    # v22: separate (N/128, N/32) block mask for the BM=128 BN=32 dQ path
    # (used when N >= 8192 per _binflash_bwd's dispatch).
    bwd_dq_block_mask = _compute_bwd_block_mask(mask, _BWD_DQ_BLOCK_M, _BWD_DQ_BLOCK_N)
    col_indices_dq, n_full_dq, n_total_dq = _build_gathered_indices(bwd_dq_block_mask)

    if mask_packed_fwd is not None and mask_packed_fwd.shape == (N, N // 32):
        mask_packed_bwd = mask_packed_fwd
    else:
        mask_packed_bwd = torch.empty((N // 32, N), dtype=torch.int32, device=mask.device).T
        _mask_i32 = mask.view(N, N // 32, 32).to(torch.int32)
        _bit_weights = (1 << torch.arange(32, device=mask.device, dtype=torch.int32))
        torch.sum(_mask_i32 * _bit_weights, dim=-1, out=mask_packed_bwd)

    # v26: work-ratio metric. BM=128 dQ halves program count at zero flop cost for
    # dense patterns (ratio ≈ 1) but adds flops for gap-heavy patterns where the
    # 128-row Q union covers more K-block columns (ratio > 1). Gate BM=128 on this
    # ratio so we only opt in when BM=128 is a clean program-count halving.
    visits_128 = int(n_total_dq.sum().item())
    visits_64 = int(n_total_bwd.sum().item())
    work_ratio = visits_128 / max(visits_64, 1)

    val = {
        "col_indices_fwd": col_indices_fwd, "n_full_fwd": n_full_fwd, "n_total_fwd": n_total_fwd,
        "mask_packed_fwd": mask_packed_fwd, "chosen_bm": chosen_bm, "chosen_bn": chosen_bn,
        "use_packed": use_packed,
        "col_indices_bwd": col_indices_bwd, "n_full_bwd": n_full_bwd, "n_total_bwd": n_total_bwd,
        "row_indices_kt": row_indices_kt, "n_full_kt": n_full_kt, "n_total_kt": n_total_kt,
        "mask_packed_bwd": mask_packed_bwd,
        # v22: BM=128 BN=32 dQ dispatch path
        "col_indices_dq": col_indices_dq, "n_full_dq": n_full_dq, "n_total_dq": n_total_dq,
        # v26: work-ratio gate
        "work_ratio_dq": work_ratio,
    }
    _ag_cache_key = key
    _ag_cache_val = val
    return val


# ─── Fwd that returns LSE (for the autograd wrapper) ───

def _binflash_fwd_with_lse(q, k, v, mask, sm_scale, preproc=None):
    B, H, N, D = q.shape
    if sm_scale is None:
        sm_scale = D ** -0.5

    if preproc is None:
        preproc = _autograd_preproc(mask, D)
    col_indices = preproc["col_indices_fwd"]
    n_full = preproc["n_full_fwd"]
    n_total = preproc["n_total_fwd"]
    mask_packed = preproc["mask_packed_fwd"]
    chosen_bm = preproc["chosen_bm"]
    chosen_bn = preproc["chosen_bn"]
    use_packed = preproc["use_packed"]

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

def _binflash_bwd(do, q, k, v, o, lse, mask, sm_scale, preproc=None):
    B, H, N, D = q.shape
    assert N % _BWD_BLOCK_M == 0
    assert N % _BWD_BLOCK_N == 0
    do = do.contiguous()

    delta = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    if preproc is None:
        preproc = _autograd_preproc(mask, D)
    col_indices = preproc["col_indices_bwd"]
    n_full = preproc["n_full_bwd"]
    n_total = preproc["n_total_bwd"]
    row_indices_kt = preproc["row_indices_kt"]
    n_full_kt = preproc["n_full_kt"]
    n_total_kt = preproc["n_total_kt"]
    mask_packed = preproc["mask_packed_bwd"]
    USE_PACKED = 1

    # v51: autotuned preproc; grid uses BLOCK_M from autotune
    grid_pre = lambda META: (triton.cdiv(N, META["BLOCK_M"]), B * H)  # noqa: E731
    _bwd_preprocess_kernel[grid_pre](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        B, H, N, B * H,
        HEAD_DIM=D,
    )

    # v26: BM=128 dQ gated on N AND work_ratio. For dense patterns (sparse_*/
    # causal/prefix_lm/log_tree/causal_doc_7) work_ratio ≈ 1.0 → BM=128 wins.
    # For gap-heavy patterns (longformer/sliding_window/causal_window/cdw_sinks),
    # BM=128's Q-block union adds K-block visits → work_ratio > 1.1 → BM=128
    # would add flops. Fall back to BM=64 (v4 path) for those.
    work_ratio = preproc.get("work_ratio_dq", 99.0)
    if N >= 8192 and work_ratio <= 1.1:
        dq_bm, dq_bn = _BWD_DQ_BLOCK_M, _BWD_DQ_BLOCK_N  # 128, 32
        col_indices_dq = preproc["col_indices_dq"]
        n_full_dq = preproc["n_full_dq"]
        n_total_dq = preproc["n_total_dq"]
    else:
        dq_bm, dq_bn = _BWD_BLOCK_M, _BWD_BLOCK_N  # 64, 64 (v4 path)
        col_indices_dq = col_indices
        n_full_dq = n_full
        n_total_dq = n_total

    grid_dkdv = (triton.cdiv(N, _BWD_BLOCK_N), B * H)
    _bwd_dkdv_kernel[grid_dkdv](
        q, k, v, sm_scale, do, dk, dv, mask, mask_packed, lse, delta,
        row_indices_kt, n_full_kt, n_total_kt,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        mask.stride(0), mask.stride(1),
        mask_packed.stride(0), mask_packed.stride(1),
        row_indices_kt.stride(0),
        B, H, N, B * H,
        BLOCK_M=_BWD_BLOCK_M, BLOCK_N=_BWD_BLOCK_N, HEAD_DIM=D,
        USE_PACKED=USE_PACKED,
    )

    grid_dq = (triton.cdiv(N, dq_bm), B * H)
    _bwd_dq_kernel[grid_dq](
        q, k, v, sm_scale, do, dq, mask, mask_packed, lse, delta,
        col_indices_dq, n_full_dq, n_total_dq,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        mask.stride(0), mask.stride(1),
        mask_packed.stride(0), mask_packed.stride(1),
        col_indices_dq.stride(0),
        B, H, N, B * H,
        BLOCK_M=dq_bm, BLOCK_N=dq_bn, HEAD_DIM=D,
        USE_PACKED=USE_PACKED,
    )

    return dq, dk, dv


class _BinFlashFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, sm_scale):
        if sm_scale is None:
            sm_scale = q.shape[-1] ** -0.5
        D = q.shape[-1]
        preproc = _autograd_preproc(mask, D)
        o, lse = _binflash_fwd_with_lse(q, k, v, mask, sm_scale, preproc=preproc)
        ctx.save_for_backward(q, k, v, o, lse, mask)
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, mask = ctx.saved_tensors
        D = q.shape[-1]
        preproc = _autograd_preproc(mask, D)
        dq, dk, dv = _binflash_bwd(do, q, k, v, o, lse, mask, ctx.sm_scale, preproc=preproc)
        return dq, dk, dv, None, None


def binflash_attention_autograd(q, k, v, mask, sm_scale=None):
    """Drop-in replacement for binflash_attention with autograd support."""
    return _BinFlashFunction.apply(q, k, v, mask, sm_scale)
