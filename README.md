# BinFlash

Flash attention with **binary block-mask skipping** for sparse attention
patterns that can be described by binary masks. Triton kernels with forward and backward
support; drop-in replacement for `F.scaled_dot_product_attention` in all situations where an `(N, N)` bool mask is given.

Notably, the interface is strictly tensor-> tensor, there is no pre-compilation per mask required, it's just 
`out = binflash_attention(q, k, v, mask)`. 

This is a (model-assisted!) cleanup and a small usability update of our code for ["Efficiently Dispatching Flash
Attention For Partially Filled Attention Masks"](https://arxiv.org/abs/2409.15097)
(Sharma & Geiping, NeurIPS ENLSP Workshop 2024).

## Efficiently Dispatching FlashAttention For Partially Filled Attention Masks

BinFlash adds a preprocessing that
reduces the full `(N, N)` bool mask to a coarse `(N/BM × N/BN)` block mask
plus a 32-bit-packed fine mask. The inner kernel iterates over a pre-sorted list of non-empty K-blocks per Q-row, runs the "all-True"
blocks with zero mask loading, and applies the packed mask only for partial
blocks. All-False blocks are skipped entirely.

The dispatcher tries to autotune tiling based on density, gap ratio, and partial
fraction measured on the coarse block mask. Backward uses two passes
(dKdV over K-blocks, dQ over Q-blocks) with the same gathered-dispatch
machinery.

## Install

```bash
pip install -e .
# or, with the benchmark dependencies:
pip install -e ".[benchmark]"
```

Requires Python ≥ 3.10, a CUDA GPU, PyTorch ≥ 2.4, Triton ≥ 3.0. Kernels
target Ampere/Ada/Hopper. Tested primarily on Ada (RTX A600-ada).

## Use

```python
import torch
from binflash import binflash_attention
from binflash.masks import causal_mask

B, H, N, D = 4, 32, 8192, 128
q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
mask = causal_mask(N, device="cuda")           # (N, N) bool, True = attend

out = binflash_attention(q, k, v, mask)        # (B, H, N, D)
out.sum().backward()                           # q.grad, k.grad, v.grad populated
```

Inputs:

- `q`, `k`, `v`: `(B, H, N, D)`, `D in {16, 32, 64, 128}`, dtype fp16/bf16.
- `mask`: `(N, N)` bool tensor on the same CUDA device, `True` = attend.
- `sm_scale`: optional softmax scale; defaults to `1 / sqrt(D)`.
- `precise`: optional bool, default `False`. When `True`, applies log2e post-matmul in fp32 and uses fp16 P@V / dV / dK matmuls. Lowers forward max-error by ~25% and dK/dV by ~10% at some cost to latency.
- `approximate_softmax` + `softmax_threshold`: optional. When `approximate_softmax=True`, the kernel applies BLASST-style content-based block skipping ([arXiv:2512.12087](https://arxiv.org/abs/2512.12087)).

For inference, wrap in `torch.no_grad()` to skip ctx saves.


## Run tests and benchmarks

```bash
python tests/test_correctness.py
python tests/test_correctness_bwd.py

python benchmarks/benchmark.py --csv fwd.csv
python benchmarks/benchmark_bwd.py --csv bwd.csv

# Subset a few methods:
python benchmarks/benchmark.py --methods binflash flex --csv fwd_only.csv
```

## Citation

```bibtex
@inproceedings{sharma2024binflash,
  title={Efficiently Dispatching Flash Attention For Partially Filled Attention Masks},
  author={Sharma, Agniv and Geiping, Jonas},
  booktitle={NeurIPS Workshop on Efficient Natural Language and Speech Processing},
  year={2024},
  url={https://arxiv.org/abs/2409.15097},
}
```

## License

MIT — see [LICENSE](LICENSE).
