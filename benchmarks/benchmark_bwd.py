"""Backward-pass benchmark: binflash autograd vs flex_attention vs torch SDPA.

Mirrors benchmark.py but measures fwd+bwd time and grad accuracy. TFLOPs count
uses 2.5x the fwd FLOPs (standard bwd-cost convention).

Usage:
    uv run python benchmarks/benchmark_bwd.py --csv bwd_results.csv
    uv run python benchmarks/benchmark_bwd.py --methods torch_sdpa binflash
"""

import argparse
import csv
import os
import sys


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
import torch.nn.functional as F  # noqa: E402
import triton  # noqa: E402
from benchmark import MASK_FACTORIES, _save_csv  # noqa: E402
from tqdm import tqdm  # noqa: E402

from binflash import binflash_attention  # noqa: E402
from binflash.flex_attention_impl import flex_attention_from_mask  # noqa: E402
from binflash.masks import mask_sparsity  # noqa: E402
from binflash.reference import reference_attention  # noqa: E402


def _torch_sdpa(q, k, v, mask, sm_scale):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=sm_scale)


def get_default_methods():
    return [
        {"name": "torch_sdpa", "fn": _torch_sdpa},
        {"name": "flex", "fn": flex_attention_from_mask},
        {"name": "binflash", "fn": binflash_attention},
    ]


def bench_fn(fn, warmup=50, rep=200):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def _make_bwd_call(method_fn, q, k, v, mask, sm_scale, do):
    def call():
        if q.grad is not None:
            q.grad = None
        if k.grad is not None:
            k.grad = None
        if v.grad is not None:
            v.grad = None
        out = method_fn(q, k, v, mask, sm_scale)
        out.backward(do)

    return call


def _extract_grads(method_fn, q, k, v, mask, sm_scale, do):
    for t in (q, k, v):
        if t.grad is not None:
            t.grad = None
    out = method_fn(q, k, v, mask, sm_scale)
    out.backward(do)
    return q.grad.clone(), k.grad.clone(), v.grad.clone()


def _reference_grads(q, k, v, mask, sm_scale, do):
    qf = q.detach().float().requires_grad_()
    kf = k.detach().float().requires_grad_()
    vf = v.detach().float().requires_grad_()
    out = reference_attention(qf, kf, vf, mask, sm_scale)
    out.backward(do.float())
    return qf.grad.detach(), kf.grad.detach(), vf.grad.detach()


def _grad_err(g, g_ref):
    return (g.float() - g_ref.float()).abs().max().item()


def run_benchmarks_bwd(seq_lens, patterns, methods, B, H, D, csv_path, warmup, rep, results, workload_name):
    device = "cuda"
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

        q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        do = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
        sm_scale = D**-0.5

        try:
            dq_ref, dk_ref, dv_ref = _reference_grads(q, k, v, mask, sm_scale, do)
            ref_ok = True
        except torch.cuda.OutOfMemoryError:
            ref_ok = False
            dq_ref = dk_ref = dv_ref = None

        fwd_flops = 4.0 * B * H * N * N * D
        bwd_flops = 2.5 * fwd_flops
        real_flops = bwd_flops * density

        row = {
            "workload": workload_name,
            "pattern": pattern_name,
            "seq_len": N,
            "B": B,
            "H": H,
            "D": D,
            "sparsity": f"{sparsity:.1%}",
            "dense_gflops": f"{bwd_flops / 1e9:.1f}",
            "real_gflops": f"{real_flops / 1e9:.1f}",
        }

        for method in methods:
            name = method["name"]
            fn = method["fn"]
            pbar.set_postfix_str(f"{pattern_name} N={N} - {name}")

            try:
                call = _make_bwd_call(fn, q, k, v, mask, sm_scale, do)
                ms_med = bench_fn(call, warmup, rep)
            except Exception as e:
                tqdm.write(f"  [{name}] bench failed: {e}")
                row[f"{name}_ms"] = ""
                row[f"{name}_tflops"] = ""
                row[f"{name}_real_tflops"] = ""
                row[f"{name}_err"] = ""
                continue

            row[f"{name}_ms"] = f"{ms_med:.3f}"
            row[f"{name}_tflops"] = f"{bwd_flops / (ms_med * 1e-3) / 1e12:.1f}"
            row[f"{name}_real_tflops"] = f"{real_flops / (ms_med * 1e-3) / 1e12:.1f}"

            if ref_ok:
                try:
                    dq, dk, dv = _extract_grads(fn, q, k, v, mask, sm_scale, do)
                    err = max(_grad_err(dq, dq_ref), _grad_err(dk, dk_ref), _grad_err(dv, dv_ref))
                    row[f"{name}_err"] = f"{err:.4e}"
                except Exception as e:
                    tqdm.write(f"  [{name}] err extraction failed: {e}")
                    row[f"{name}_err"] = ""
            else:
                row[f"{name}_err"] = ""

        results.append(row)
        parts = [f"  [{workload_name}] {pattern_name:20s} N={N:5d} sparsity={sparsity:5.1%}"]
        for method in methods:
            name = method["name"]
            parts.append(f"{name} {row.get(f'{name}_ms', 'ERR'):>7}ms")
        tqdm.write(" | ".join(parts))

        if csv_path:
            _save_csv(results, csv_path)

    pbar.close()
    return results


# Fixed workload — see benchmark.py for the same matrix on forward.
_BENCH_SEQ_LENS = [4096, 8192, 16384]
_WORKLOADS = [
    ("llama3-8b", 4, 32, 128),
    ("7B-batch8", 8, 32, 128),
    ("qwen3-32b", 8, 64, 128),
    ("70B-kv-heads", 2, 8, 128),
    ("27B-kv-heads", 4, 16, 128),
]


def main():
    parser = argparse.ArgumentParser(description="BinFlash backward benchmark")
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
    print("BinFlash backward benchmark")
    print(f"  GPU: {gpu_name}")
    print(f"  warmup={args.warmup}, rep={args.rep}, dtype=bfloat16")
    print(f"  Seq lens: {_BENCH_SEQ_LENS}")
    print(f"  Methods: {[m['name'] for m in methods]}")
    print("=" * 120)

    results = []
    for name, B, H, D in _WORKLOADS:
        print(f"  > workload {name} (B={B}, H={H}, D={D})")
        run_benchmarks_bwd(
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
