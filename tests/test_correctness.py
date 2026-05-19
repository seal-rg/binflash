"""Correctness tests: verify binflash forward matches the reference."""

import sys

import torch

from binflash import binflash_attention
from binflash.masks import (
    block_diagonal_mask,
    causal_doc_window_sinks,
    causal_document_mask,
    causal_mask,
    log_tree_mask,
    mask_sparsity,
    prefix_lm_mask,
    random_sparse_mask,
    sliding_window_mask,
)
from binflash.reference import reference_attention


def check_close(name: str, out: torch.Tensor, ref: torch.Tensor, atol: float = 1e-2):
    diff = (out.float() - ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ok = max_err < atol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    return ok


def run_test(mask_name: str, mask: torch.Tensor, B=2, H=4, N=None, D=64):
    if N is None:
        N = mask.shape[0]

    print(f"\n--- {mask_name} (N={N}, sparsity={mask_sparsity(mask):.1%}) ---")

    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    sm_scale = D**-0.5

    ref = reference_attention(q, k, v, mask, sm_scale)

    # Modes: precise=False/True, plus approximate_softmax=True at a tight
    # threshold (where the skip should hit very few — or zero — blocks and the
    # result should still match the reference within tolerance).
    modes = [
        (False, False),
        (True, False),
        (False, True),
    ]
    ok = True
    for precise, approx in modes:
        with torch.no_grad():
            out = binflash_attention(
                q,
                k,
                v,
                mask,
                sm_scale,
                precise=precise,
                approximate_softmax=approx,
                softmax_threshold=1e-6,
            )
        ok &= check_close(f"binflash(precise={precise},approx={approx})", out, ref)
    return ok


def main():
    torch.manual_seed(42)
    device = "cuda"

    test_configs = [
        # (N, D, B, H, description)
        (1024, 64, 2, 4, "small"),
        (4096, 64, 2, 8, "medium D=64"),
        (8192, 64, 2, 16, "large D=64"),
        (4096, 128, 1, 8, "medium D=128"),
        (8192, 128, 1, 16, "large D=128"),
        (16384, 128, 1, 8, "xlarge D=128 (matches benchmark high end)"),
    ]

    def make_masks(N, device):
        return {
            "causal": causal_mask(N, device),
            "sliding_window_64": sliding_window_mask(N, 64, device),
            "sliding_window_128": sliding_window_mask(N, 128, device),
            "block_diagonal_128": block_diagonal_mask(N, 128, device),
            "prefix_lm_quarter": prefix_lm_mask(N, N // 4, device),
            "random_sparse_30pct": random_sparse_mask(N, 0.3, device),
            "random_sparse_10pct": random_sparse_mask(N, 0.1, device),
            "log_tree": log_tree_mask(N, device),
            "causal_doc_4": causal_document_mask(N, [N // 4] * 4, device),
            "causal_doc_win_sinks": causal_doc_window_sinks(N, [N // 2, N // 2], 128, 4, device),
        }

    print("=" * 60)
    print("Correctness Tests: binflash forward vs reference")
    print("=" * 60)

    all_pass = True
    for N, D, B, H, desc in test_configs:
        print(f"\n{'-' * 60}")
        print(f"Config: N={N}, D={D}, B={B}, H={H} ({desc})")
        print(f"{'-' * 60}")
        masks = make_masks(N, device)
        for name, mask in masks.items():
            all_pass &= run_test(name, mask, B=B, H=H, N=N, D=D)

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
