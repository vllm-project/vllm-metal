# Tools

## Greedy parity

Compare serving with a GPU reference: `mlx-lm` for MLX, Transformers/SDPA on MPS for MPS.

```bash
python tools/check_parity.py --model Qwen/Qwen3-0.6B --top-k 5
VLLM_METAL_BACKEND=mps python tools/check_parity.py --model Qwen/Qwen3-0.6B --top-k 5
```

Defaults: 40 prompts, 10 output tokens, request batches of 1 and 2, prefix caching off.
Both sides use identical input IDs; MPS also matches the checkpoint's FP16/BF16 precision
and disables CPU fallback. Use `--help` for options and `--output-dir` for logs and results.

- `EXACT`: every output token matches.
- `TOP_K_MATCH`: each side's chosen token is in the other's top-K at the first divergence; later tokens are not compared.
- `FAIL`: a mismatch fails the comparison, or generation is incomplete.

Exit status is 0 when all prompts pass, otherwise nonzero.

## MLX versus PyTorch MPS performance

`tools/benchmark_mps.py` runs `vllm bench serve` in MLX/MPS/MPS/MLX order with
matched checkpoint precision, workload and cache capacity. It saves raw results
and `summary.md` locally. Use `--help` for options.

## Scheduled and requested CI

Parity runs daily at 07:17 UTC on `main`. Users with repository write access can also comment `/ci parity` on an open PR once the workflow is on the default branch.

Both triggers invoke this same tool for Qwen3-0.6B and Qwen3.5-0.8B on macOS 15 and 26 with Xcode 26.3: 40 shared prompts, 20 output tokens, top-K 5, and HTTP request batch sizes 1/2. CI supplies the checkpoint and environment and collects the tool's results.

The `Parity` check links to job summaries and artifacts containing the native reference, server log, and per-batch results, retained for 14 days. Regular PR CI runs fast tests without starting model servers.
