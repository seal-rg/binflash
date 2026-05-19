"""FlexAttention wrapper using PyTorch's built-in torch.nn.attention.flex_attention."""

import torch  # type: ignore
from torch.nn.attention.flex_attention import (  # type: ignore
    create_block_mask,
)
from torch.nn.attention.flex_attention import (
    flex_attention as _flex_attention,
)

# Pre-compile flex_attention for best performance.
# `mode="max-autotune-no-cudagraphs"` is recommended by PyTorch FlexAttention
# docs for production training. Default mode picks block sizes that overflow
# Ada SMEM at D=128 (OOM); max-autotune chooses a feasible config.
# We use `-no-cudagraphs` to avoid the autograd-incompatible CUDAGraph capture.
_compiled_flex = torch.compile(_flex_attention, mode="max-autotune-no-cudagraphs")


_flex_attention_from_mask_cache: dict = {}


def flex_attention_from_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    sm_scale: float | None = None,
    block_size: int = 128,
) -> torch.Tensor:
    """Run FlexAttention given a dense boolean mask.

    Caches BlockMask per mask data_ptr — so the bwd bench harness, which calls
    this every step, doesn't pay create_block_mask cost on every call. Matches
    binflash's preproc-cache behavior for fair comparison.

    Args:
        q, k, v: (B, H, N, D)
        mask: (N, N) bool tensor on CUDA
        sm_scale: softmax scale, defaults to 1/sqrt(D)
        block_size: block size for BlockMask construction
    """
    B, H, N, D = q.shape
    if sm_scale is None:
        sm_scale = D**-0.5

    cache_key = (mask.data_ptr(), N, block_size, q.device.index if q.device.type == "cuda" else -1)
    block_mask = _flex_attention_from_mask_cache.get(cache_key)
    if block_mask is None:
        # mask_mod closure over the dense mask tensor.
        m = mask

        def mask_mod(b, h, q_idx, kv_idx):
            return m[q_idx, kv_idx]

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=N,
            KV_LEN=N,
            device=q.device,
            BLOCK_SIZE=block_size,
        )
        _flex_attention_from_mask_cache[cache_key] = block_mask

    # Use full _FLEX_KERNEL_OPTIONS (with proper bwd_ prefixes) for D=128
    # where unconstrained sizes overflow Ada SMEM.
    kernel_options = _FLEX_KERNEL_OPTIONS if D > 64 else None

    return _compiled_flex(
        q,
        k,
        v,
        block_mask=block_mask,
        scale=sm_scale,
        kernel_options=kernel_options,
    )


def flex_attention_causal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
    block_size: int = 128,
) -> torch.Tensor:
    """Causal attention via FlexAttention"""
    B, H, N, D = q.shape
    if sm_scale is None:
        sm_scale = D**-0.5

    def causal_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(
        causal_mod,
        B=None,
        H=None,
        Q_LEN=N,
        KV_LEN=N,
        device=q.device,
        BLOCK_SIZE=block_size,
    )

    return _compiled_flex(q, k, v, block_mask=block_mask, scale=sm_scale)


def flex_attention_sliding_window(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    sm_scale: float | None = None,
    block_size: int = 128,
) -> torch.Tensor:
    """Sliding window attention via FlexAttention."""
    B, H, N, D = q.shape
    if sm_scale is None:
        sm_scale = D**-0.5

    def window_mod(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx).abs() <= window_size

    block_mask = create_block_mask(
        window_mod,
        B=None,
        H=None,
        Q_LEN=N,
        KV_LEN=N,
        device=q.device,
        BLOCK_SIZE=block_size,
    )

    return _compiled_flex(q, k, v, block_mask=block_mask, scale=sm_scale)


# ────────────────── Benchmark wrappers ──────────────────
#
# Factories used by benchmark.py. Each builds a flex_attention caller with
# its own per-mask block-mask cache so the timed loop measures kernel cost
# only (no re-running create_block_mask on every call).
#
# Two flavors:
#   make_flex_wrapper          — dense-tensor-lookup mask_mod. Apples-to-apples
#                                with the other tensor-interface methods.
#   make_symbolic_flex_wrapper — closed-form predicates for patterns where one
#                                exists (causal, sliding_window, etc.). Flex
#                                compiles these into register arithmetic for
#                                its partial-block path, avoiding HBM mask
#                                loads. A roofline / upper-bound reference.
#                                Falls back to dense lookup for patterns
#                                without a closed form (random, log_tree,
#                                document masks).


# Default kernel_options constrained so the compiled kernel fits in the SRAM
# of consumer Ada GPUs (~100 KB/SM). Flex's unconstrained defaults overflow
# at D=128.
#
# IMPORTANT: per PyTorch 2.11 FlexAttention docs (and source), the keys
# BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 are bwd-specific and MUST be prefixed
# with `bwd_` to apply. Without prefix they are silently ignored. The prior
# version of this file had `BLOCK_M1` etc. without prefix — those keys had
# zero effect (bwd used default autotune). Verified empirically (2026-04-27)
# that adding bwd_ prefix gives the same result as autotune-without-options,
# so the previous unprefixed keys were no-ops.
_FLEX_KERNEL_OPTIONS = {
    "fwd_BLOCK_M": 64,
    "fwd_BLOCK_N": 64,
    "bwd_BLOCK_M1": 32,
    "bwd_BLOCK_N1": 64,
    "bwd_BLOCK_M2": 64,
    "bwd_BLOCK_N2": 32,
    "num_stages": 2,
}


def make_flex_wrapper():
    """Build a flex_attention caller that caches block_mask per mask tensor.

    mask_mod is a dense tensor lookup, matching how the other benchmark methods
    consume the (N, N) bool mask.
    """
    cache = {}

    def wrapper(q, k, v, mask, sm_scale):
        key = mask.data_ptr()
        if key not in cache:
            N = mask.shape[0]

            def mask_mod(b, h, q_idx, kv_idx):
                return mask[q_idx, kv_idx]

            cache[key] = create_block_mask(mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N, device=q.device)
        return _compiled_flex(q, k, v, block_mask=cache[key], scale=sm_scale, kernel_options=_FLEX_KERNEL_OPTIONS)

    return wrapper


def flex_preproc(mask, q, k, v, sm_scale):
    """Preproc cost for plain flex: just create_block_mask with a dense lookup."""
    N = mask.shape[0]

    def mask_mod(b, h, q_idx, kv_idx):
        return mask[q_idx, kv_idx]

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N, device=q.device)


def symbolic_mask_mod_for(pattern: str, N: int):
    """Return a closed-form mask_mod for `pattern`, or None if no such form exists.

    Closed forms compile into register-level arithmetic inside flex_attention's
    partial-block path — no HBM mask loads. These must match the exact mask
    produced by the corresponding MASK_FACTORIES entry in masks.py.
    """
    if pattern == "causal":

        def m(b, h, q, kv):
            return q >= kv

        return m
    if pattern == "sliding_window_128":
        W = 128

        def m(b, h, q, kv):
            return ((q - kv) <= W) & ((kv - q) <= W)

        return m
    if pattern == "causal_window_256":
        W = 256

        def m(b, h, q, kv):
            return (q >= kv) & ((q - kv) <= W)

        return m
    if pattern == "block_diagonal_128":
        BS = 128

        def m(b, h, q, kv):
            return (q // BS) == (kv // BS)

        return m
    if pattern == "prefix_lm_quarter":
        P = N // 4

        def m(b, h, q, kv):
            return (kv < P) | (q >= kv)

        return m
    if pattern == "longformer_128_16":
        W = 128
        G = 16

        def m(b, h, q, kv):
            return (((q - kv) <= W) & ((kv - q) <= W)) | (q < G) | (kv < G)

        return m
    return None


def make_symbolic_flex_wrapper():
    """Flex with pattern-specific symbolic mask_mod — roofline upper bound.

    Contract: call `wrapper.register(mask, pattern_name)` before the first
    call on a fresh mask so the wrapper knows which closed form to apply.
    Patterns without a closed form fall back to a dense tensor lookup, so
    those columns read identically to `make_flex_wrapper`'s output.

    Exposes `.preproc` for measuring create_block_mask cost and `.register`
    for pattern registration.
    """
    cache = {}
    pattern_registry = {}

    def _resolve_mask_mod(mask):
        N = mask.shape[0]
        pattern = pattern_registry.get(mask.data_ptr())
        if pattern is not None:
            mm = symbolic_mask_mod_for(pattern, N)
            if mm is not None:
                return mm

        def dense(b, h, q_idx, kv_idx):
            return mask[q_idx, kv_idx]

        return dense

    def wrapper(q, k, v, mask, sm_scale):
        key = mask.data_ptr()
        if key not in cache:
            N = mask.shape[0]
            mask_mod = _resolve_mask_mod(mask)
            cache[key] = create_block_mask(mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N, device=q.device)
        return _compiled_flex(q, k, v, block_mask=cache[key], scale=sm_scale, kernel_options=_FLEX_KERNEL_OPTIONS)

    def preproc(mask, q, k, v, sm_scale):
        N = mask.shape[0]
        mask_mod = _resolve_mask_mod(mask)
        return create_block_mask(mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N, device=q.device)

    def register(mask, pattern_name):
        pattern_registry[mask.data_ptr()] = pattern_name

    wrapper.register = register
    wrapper.preproc = preproc
    return wrapper
