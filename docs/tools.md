# Tools

## Greedy parity

Compare serving against a fresh GPU reference: `mlx-lm` for MLX, or Transformers on MPS for the MPS backend:

```bash
python tools/check_parity.py --model Qwen/Qwen3-0.6B
VLLM_METAL_BACKEND=mps python tools/check_parity.py --model Qwen/Qwen3-0.6B --top-k 5
python tools/check_parity.py --model /path/to/checkpoint --top-k 5 --batch-size 1 2 --output-dir parity-results
```

The tool generates the reference once and exits that process before starting one `vllm serve` instance. It checks `/health`, compares individual prompts and pairs of prompts through `/v1/completions`, then stops the server. The server's sequence limit is the largest requested batch size, and prefix caching is disabled. Each pair is submitted in one HTTP request. These are request batch sizes, not assertions about the actual GPU batch size. Both backends use the same checkpoint and input token IDs. For MPS, both use the checkpoint’s actual FP16/BF16 precision; uniform unquantized safetensors are required. The Transformers reference uses SDPA, neutral greedy settings, and no CPU fallback.

The default is the 40 prompts in `tools/parity_prompts.py`, greedy decoding, and 10 output tokens, ignoring EOS. Use `--prompt` for custom text, `--max-tokens` for output length, and `--help` for all options. Inputs are plain text without a chat template. Logs, the native reference, and a summary are saved to `--output-dir`, or a new temporary directory printed at startup. The tool selects an available localhost port automatically.

- `EXACT`: every output token matches.
- `TOP_K_MATCH`: the first divergence passes mutual top-K membership; later tokens are not compared. Enabled with `--top-k K`.
- `FAIL`: a mismatch fails the comparison, or generation is incomplete.

Exit status is 0 when all prompts pass, otherwise nonzero. No saved golden token IDs or regeneration step is needed.

## MLX versus PyTorch MPS performance

Run the same serving workload on both backends:

```bash
curl -L https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/sonnet.txt -o sonnet.txt
python tools/benchmark_mps.py --model Qwen/Qwen3-0.6B --output-dir .cache/qwen-mps
```

The tool reuses `vllm bench serve` and starts four fresh servers in
MLX/MPS/MPS/MLX order. Defaults match the MPS stack's serving comparison:
Sonnet 1024/128, 100 requests, 10 requests/s, concurrency 32, four warmup requests,
2,340 cache blocks of 16 tokens, prefix caching off, and async scheduling on both.
Use `--help` to change the model or workload. Avoid other GPU work during the run.

Both servers use one resolved checkpoint and its actual FP16/BF16 weight dtype;
uniform unquantized safetensors are required because MLX preserves stored weight
precision. Generation is greedy and ignores EOS. Incomplete requests or unequal
input/output token counts fail the comparison.

`summary.md` reports two-run means for output tok/s, TTFT, TPOT, and preemptions.
Raw benchmark JSON, server/client logs, metrics, and `config.json` record the
checkpoint, commands/settings, dataset hash, hardware, versions, and source commit.
Results are local artifacts; there is no CI timing threshold. Use the greedy
parity command above for correctness.

## Scheduled and requested CI

Parity runs daily at 07:17 UTC on `main`. Users with repository write access can also comment `/ci parity` on an open PR once the workflow is on the default branch.

Both triggers invoke this same tool for Qwen3-0.6B and Qwen3.5-0.8B on macOS 15 and 26 with Xcode 26.3: 40 shared prompts, 20 output tokens, top-K 5, and HTTP request batch sizes 1/2. CI supplies the checkpoint and environment and collects the tool's results.

The `Parity` check links to job summaries and artifacts containing the native reference, server log, and per-batch results, retained for 14 days. Regular PR CI runs fast tests without starting model servers.
