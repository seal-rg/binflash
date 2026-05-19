"""Forward-pass benchmark: binflash vs flex_attention vs torch SDPA.

Sweeps a fixed (pattern x seq_len x workload) matrix and writes a row per
configuration to CSV. The workload list and patterns are intentionally fixed
so progress comparisons across kernel versions stay honest.

Usage:
    uv run python benchmarks/benchmark.py
    uv run python benchmarks/benchmark.py --csv fwd_results.csv
    uv run python benchmarks/benchmark.py --methods binflash flex
"""

import argparse
import csv
import os
import sys


# Pre-parse `--gpu N` before torch imports — PyTorch snapshots CUDA_VISIBLE_DEVICES
# on first CUDA init (which can happen during import).
def _preparse_gpu():
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            return
        if arg.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.split("=", 1)[1]
            return


_preparse_gpu()

import torch  # noqa: E402
import torch._dynamo  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import triton  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Larger compile cache so we don't hit recompile limits when sweeping shapes.
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_recompile_limit = 512

from binflash import binflash_attention  # noqa: E402
from binflash.flex_attention_impl import flex_attention_from_mask  # noqa: E402
from binflash.masks import (  # noqa: E402
    block_diagonal_mask,
    causal_doc_window_sinks,
    causal_document_mask,
    causal_mask,
    causal_sliding_window_mask,
    log_tree_mask,
    longformer_mask,
    mask_sparsity,
    prefix_lm_mask,
    random_sparse_mask,
    sliding_window_mask,
)
from binflash.reference import reference_attention  # noqa: E402


# ─────────────────────── Mask factory ───────────────────────


MASK_FACTORIES = {
    "causal": lambda N, device: causal_mask(N, device),
    "sliding_window_128": lambda N, device: sliding_window_mask(N, 128, device),
    "causal_window_256": lambda N, device: causal_sliding_window_mask(N, 256, device),
    "block_diagonal_128": lambda N, device: block_diagonal_mask(N, 128, device),
    "prefix_lm_quarter": lambda N, device: prefix_lm_mask(N, N // 4, device),
    "sparse_10pct": lambda N, device: random_sparse_mask(N, 0.1, device),
    "sparse_30pct": lambda N, device: random_sparse_mask(N, 0.3, device),
    "longformer_128_16": lambda N, device: longformer_mask(N, 128, 16, device),
    "log_tree": lambda N, device: log_tree_mask(N, device),
    "causal_doc_7": lambda N, device: causal_document_mask(N, None, device),
    "cdw_sinks": lambda N, device: causal_doc_window_sinks(N, None, 128, 4, device),
}


# ─────────────────────── Methods ───────────────────────


def _torch_sdpa(q, k, v, mask, sm_scale):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=sm_scale)


def _binflash_no_grad(q, k, v, mask, sm_scale):
    with torch.no_grad():
        return binflash_attention(q, k, v, mask, sm_scale)


def get_default_methods():
    return [
        {"name": "torch_sdpa", "fn": _torch_sdpa},
        {"name": "flex", "fn": flex_attention_from_mask},
        {"name": "binflash", "fn": _binflash_no_grad},
    ]


# ─────────────────────── Bench helpers ───────────────────────


def bench_fn(fn, warmup=50, rep=200):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def _save_csv(results, path):
    if not results:
        return
    all_keys = []
    seen = set()
    for row in results:
        for key in row:
            if key not in seen:
                all_keys.append(key)
                seen.add(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, restval="")
        writer.writeheader()
        writer.writerows(results)


# ─────────────────────── Runner ───────────────────────


def run_benchmarks(seq_lens, patterns, methods, B, H, D, csv_path, warmup, rep, results, workload_name):
    device = "cuda"
    method_names = {m["name"] for m in methods}
    baseline_name = "flex" if "flex" in method_names else (methods[0]["name"] if methods else None)
    torch.manual_seed(42)

    completed = set()
    if csv_path and os.path.exists(csv_path):
        try:
            with open(csv_path) as f:
                for r in csv.DictReader(f):
                    if r.get("workload") == workload_name:
                        try:
                            completed.add((r["pattern"], int(r["seq_len"])))
                        except (KeyError, ValueError):
                            pass
                        results.append(r)
        except Exception as e:
            tqdm.write(f"  (resume: failed to read {csv_path}: {e})")
    if completed:
        tqdm.write(f"  (resume: skipping {len(completed)} already-completed configs in [{workload_name}])")

    configs = [(p, n) for p in patterns for n in seq_lens]
    pbar = tqdm(configs, desc=f"[{workload_name}]", unit="cfg", dynamic_ncols=True, leave=True)
    for pattern_name, N in pbar:
        if (pattern_name, N) in completed:
            continue
        pbar.set_postfix_str(f"{pattern_name} N={N}")

        mask = MASK_FACTORIES[pattern_name](N, device)
        sparsity = mask_sparsity(mask)
        density = 1.0 - sparsity

        q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
        sm_scale = D**-0.5

        ref = reference_attention(q.float(), k.float(), v.float(), mask, sm_scale)
        dense_flops = 4.0 * B * H * N * N * D
        real_flops = dense_flops * density

        row = {
            "workload": workload_name,
            "pattern": pattern_name,
            "seq_len": N,
            "B": B,
            "H": H,
            "D": D,
            "sparsity": f"{sparsity:.1%}",
            "dense_gflops": f"{dense_flops / 1e9:.1f}",
            "real_gflops": f"{real_flops / 1e9:.1f}",
        }

        ms_by_name = {}
        for method in methods:
            name = method["name"]
            fn = method["fn"]
            pbar.set_postfix_str(f"{pattern_name} N={N} - {name}")

            ms_med = bench_fn(lambda fn=fn: fn(q, k, v, mask, sm_scale), warmup, rep)  # noqa: B023
            ms_by_name[name] = ms_med
            row[f"{name}_ms"] = f"{ms_med:.3f}"
            row[f"{name}_tflops"] = f"{dense_flops / (ms_med * 1e-3) / 1e12:.1f}"
            row[f"{name}_real_tflops"] = f"{real_flops / (ms_med * 1e-3) / 1e12:.1f}"

            out = fn(q, k, v, mask, sm_scale)
            err = (out.float() - ref).abs().max().item()
            row[f"{name}_err"] = f"{err:.4e}"

        baseline_ms = ms_by_name.get(baseline_name)
        if baseline_ms is not None:
            for method in methods:
                name = method["name"]
                if name == baseline_name or name not in ms_by_name:
                    continue
                row[f"{name}_speedup"] = f"{baseline_ms / ms_by_name[name]:.2f}x"

        results.append(row)
        parts = [f"  [{workload_name}] {pattern_name:20s} N={N:5d} sparsity={sparsity:5.1%}"]
        for method in methods:
            name = method["name"]
            ms_str = row.get(f"{name}_ms", "ERR")
            speedup = row.get(f"{name}_speedup", "")
            sp_str = f" ({speedup})" if speedup else ""
            parts.append(f"{name} {ms_str:>7s}ms{sp_str}")
        tqdm.write(" | ".join(parts))

        if csv_path:
            _save_csv(results, csv_path)

    pbar.close()
    return results


# ─────────────────────── CLI ───────────────────────


# Fixed benchmark workload. Patterns and seq lens are intentionally not exposed
# as CLI flags so version-to-version comparisons remain comparable.
_BENCH_SEQ_LENS = [4096, 8192, 16384]
_WORKLOADS = [  # (name, B, H, D)
    ("llama3-8b", 4, 32, 128),
    ("7B-batch8", 8, 32, 128),
    ("qwen3-32b", 8, 64, 128),
    ("70B-kv-heads", 2, 8, 128),
    ("27B-kv-heads", 4, 16, 128),
]


def main():
    parser = argparse.ArgumentParser(description="BinFlash forward benchmark")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--methods", type=str, nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    all_defaults = get_default_methods()
    if args.methods is not None:
        default_names = {m["name"] for m in all_defaults}
        methods = [m for m in all_defaults if m["name"] in args.methods]
        for name in args.methods:
            if name not in default_names:
                print(f"Warning: unknown method '{name}', available: {sorted(default_names)}")
    else:
        methods = all_defaults

    gpu_name = torch.cuda.get_device_name(0)
    patterns = list(MASK_FACTORIES.keys())
    print("=" * 120)
    print("BinFlash forward benchmark")
    print(f"  GPU: {gpu_name}")
    print(f"  warmup={args.warmup}, rep={args.rep}, dtype=bfloat16")
    print(f"  Seq lens: {_BENCH_SEQ_LENS}")
    print(f"  Methods: {[m['name'] for m in methods]}")
    print("=" * 120)

    results = []
    for name, B, H, D in _WORKLOADS:
        print(f"  > workload {name} (B={B}, H={H}, D={D})")
        run_benchmarks(
            _BENCH_SEQ_LENS,
            patterns,
            methods,
            B=B,
            H=H,
            D=D,
            csv_path=args.csv,
            warmup=args.warmup,
            rep=args.rep,
            results=results,
            workload_name=name,
        )

    if args.csv:
        print(f"Results saved to {args.csv}")


if __name__ == "__main__":
    main()
