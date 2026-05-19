"""Bwd correctness tests: binflash backward vs PyTorch autograd on the reference."""

import sys

import torch

from binflash import binflash_attention
from binflash.masks import (
    block_diagonal_mask,
    causal_mask,
    log_tree_mask,
    longformer_mask,
    prefix_lm_mask,
    random_sparse_mask,
    sliding_window_mask,
)
from binflash.reference import reference_attention


def check_close(name, out, ref, atol=5e-2):
    diff = (out.float() - ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ok = max_err < atol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max_err={max_err:.4f}, mean_err={mean_err:.6f}")
    return ok


def run_case(mask_name, mask, B, H, N, D):
    print(f"\n--- {mask_name} (N={N}, B={B}, H={H}, D={D}) ---")
    torch.manual_seed(42)
    device = "cuda"

    q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    do = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    sm_scale = D**-0.5

    # fp32 reference
    qf = q.detach().float().requires_grad_()
    kf = k.detach().float().requires_grad_()
    vf = v.detach().float().requires_grad_()
    out_ref = reference_attention(qf, kf, vf, mask, sm_scale)
    out_ref.backward(do.float())
    dq_ref, dk_ref, dv_ref = qf.grad, kf.grad, vf.grad

    # Modes: (precise, approximate_softmax). approximate_softmax=True uses a
    # conservative threshold so the skip drops only truly-negligible contributions.
    modes = [
        (False, False),
        (True, False),
        (False, True),
    ]
    ok = True
    for precise, approx in modes:
        for t in (q, k, v):
            t.grad = None
        out = binflash_attention(
            q, k, v, mask, sm_scale,
            precise=precise,
            approximate_softmax=approx,
            softmax_threshold=1e-6,
        )
        out.backward(do)
        tag = f"{mask_name}[precise={precise},approx={approx}]"
        ok &= check_close(f"{tag}/out", out, out_ref, atol=1e-2)
        ok &= check_close(f"{tag}/dq", q.grad, dq_ref, atol=5e-2)
        ok &= check_close(f"{tag}/dk", k.grad, dk_ref, atol=5e-2)
        ok &= check_close(f"{tag}/dv", v.grad, dv_ref, atol=5e-2)
    return ok


def main():
    device = "cuda"
    configs = [
        # (N, D, B, H)
        (1024, 64, 2, 4),
        (2048, 64, 1, 4),
        (1024, 128, 1, 4),
        # N>=8192 trips the BM=128 BN=32 dQ dispatch + work_ratio gate.
        # Without these configs the most-used bwd code path is untested.
        (8192, 64, 1, 4),
        (8192, 128, 1, 4),
        (16384, 128, 1, 1),
    ]
    all_pass = True
    for N, D, B, H in configs:
        print(f"\n{'=' * 60}\nConfig: N={N}, D={D}, B={B}, H={H}\n{'=' * 60}")
        masks = {
            "causal": causal_mask(N, device),
            "sliding_window_128": sliding_window_mask(N, 128, device),
            "block_diagonal_128": block_diagonal_mask(N, 128, device),
            "sparse_10pct": random_sparse_mask(N, 0.1, device),
            # work_ratio gate falls back to BM=64 dQ on gap-heavy patterns
            "longformer_128_16": longformer_mask(N, 128, 16, device),
            # gathered dispatch heavily exercised on log_tree
            "log_tree": log_tree_mask(N, device),
            # mixed dense + causal regions
            "prefix_lm_quarter": prefix_lm_mask(N, N // 4, device),
        }
        for name, mask in masks.items():
            all_pass &= run_case(name, mask, B=B, H=H, N=N, D=D)

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL BWD TESTS PASSED")
    else:
        print("SOME BWD TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
