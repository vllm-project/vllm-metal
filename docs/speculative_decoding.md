# Speculative Decoding

vllm-metal implements four speculative decoding methods on the paged-attention
path. Use vLLM's [speculative decoding guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
for method behavior and configuration details.

| | MTP | EAGLE3 (experimental) | Draft model | N-gram |
|---|---|---|---|---|
| `--speculative-config` method | `mtp` | `eagle3` | `draft_model` | `ngram` |
| Target models | Gemma4 | Dense Qwen3, Llama, and Gemma4 text | Non-hybrid paged-attention models | Non-hybrid paged-attention models |
| Draft source | Matching Gemma4 assistant checkpoint | Matching EAGLE3 speculator checkpoint | Separate smaller model | Prompt and output token history |
| `num_speculative_tokens` | Configurable (2–3 typical) | Linear chain; 1–3 tested | Configurable (3–5 typical) | Configurable (3–5 typical) |
| Additional model weights | Assistant checkpoint | EAGLE3 head | Draft model | None |
| Additional KV cache | None; reads target KV | Scheduler-managed draft cache | Second scheduler-managed cache | None |

All methods currently have these Metal-specific constraints:

- Only plain greedy requests (`temperature=0`, without penalties, token
  constraints, or sample logprobs) are drafted. Other requests run without
  speculation.
- Scheduling must be synchronous. The Metal platform disables async scheduling
  when speculative decoding is configured.
- Pipeline parallelism is not supported with speculative decoding.
- Hybrid GDN targets and heterogeneous draft vocabularies are not supported.
- `long_prefill_token_threshold`, when set, must be at least
  `1 + num_speculative_tokens`.

EAGLE3's reduced output vocabulary is mapped to target token IDs before
verification. It does not require `use_heterogeneous_vocab`.

EAGLE3 and draft-model decoding stop proposing tokens for a request when its
current target context plus K exceeds the effective draft-model context limit.

## EAGLE3

The implementation drafts one linear chain and uses the existing greedy target
verifier. It does not construct or verify trees. Initial checkpoint pairs are:

| Target | EAGLE3 head |
|---|---|
| `Qwen/Qwen3-8B` | `RedHatAI/Qwen3-8B-speculator.eagle3` |
| `meta-llama/Llama-3.1-8B-Instruct` | `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` |
| `meta-llama/Llama-3.1-8B-Instruct` | `RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3` |
| `mlx-community/gemma-4-31b-it-4bit` | `RedHatAI/gemma-4-31B-it-speculator.eagle3` |

These heads use one Llama-style draft layer. The Qwen3 checkpoint was trained
with thinking disabled; use its target chat template with
`enable_thinking=false`. Tensor parallelism and LoRA are not supported by the
Metal EAGLE3 path.

Gemma4 validation uses the 4-bit target above with the original Red Hat head;
the full BF16 31B target has not been validated.

```bash
vllm serve Qwen/Qwen3-8B \
  --max-model-len 2048 \
  --enable-prefix-caching \
  --no-async-scheduling \
  --speculative-config '{"method":"eagle3","model":"RedHatAI/Qwen3-8B-speculator.eagle3","num_speculative_tokens":3}'
```

Original EAGLE checkpoints and Red Hat's speculators format are supported.
Original checkpoints may omit token embeddings; the loader then uses the
matching target embedding weights.

Batch-dependent rounding can change greedy winners. Exact SD-on/off token
identity is not guaranteed.

## Gemma4 MTP

Follow the upstream [MTP guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/)
for Gemma4 assistant behavior. Use matching target and assistant families:

| Target | Assistant |
|---|---|
| Gemma4 E2B-it | Gemma4 E2B-it assistant bf16 |
| Gemma4 E4B-it | Gemma4 E4B-it assistant bf16 |
| Gemma4 31B-it bf16 | Gemma4 31B-it assistant bf16 |

Start with `num_speculative_tokens=3`. On the measured E4B workload, higher
values improved single-stream throughput but reduced saturated throughput.
Benchmark the intended batch shape before changing it.

### Example

```bash
export TARGET=/path/to/gemma-4-E2B-it
export ASSISTANT=/path/to/gemma-4-E2B-it-assistant-bf16

VLLM_METAL_MEMORY_FRACTION=0.5 \
  vllm serve "$TARGET" \
    --max-model-len 1024 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 4 \
    --no-async-scheduling \
    --speculative-config "{\"method\":\"mtp\",\"model\":\"$ASSISTANT\",\"num_speculative_tokens\":3}"
```

Remote Hugging Face checkpoints are supported. Pin `revision` in
`speculative_config` when publishing benchmark results.

## Draft model

Follow the upstream [draft-model guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/draft_model/)
for configuration details. The draft must use the target vocabulary and full
attention. Sliding-window and hybrid draft models are rejected at startup.
Its committed KV cache shares the Metal KV memory budget with the target.

### Example

```bash
VLLM_METAL_MEMORY_FRACTION=0.55 \
  vllm serve Qwen/Qwen3-8B \
    --max-model-len 2048 \
    --no-async-scheduling \
    --speculative-config '{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":3}'
```

### Dynamic speculative decoding

See the upstream [dynamic speculative decoding guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/)
for configuration details. This example sets `K=3` for one scheduled request
and `K=0` for two:

```bash
VLLM_METAL_MEMORY_FRACTION=0.55 \
  vllm serve Qwen/Qwen3-8B \
    --max-model-len 2048 \
    --max-num-seqs 2 \
    --no-async-scheduling \
    --speculative-config '{
      "method": "draft_model",
      "model": "Qwen/Qwen3-0.6B",
      "num_speculative_tokens": 3,
      "num_speculative_tokens_per_batch_size": [[1, 1, 3], [2, 2, 0]]
    }'
```

## N-gram

Follow the upstream [N-gram guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/)
for configuration details. N-gram speculation needs no additional model or KV
cache. Its benefit depends on repeated token spans in the request history.

### Example

```bash
  vllm serve Qwen/Qwen3-8B \
    --max-model-len 2048 \
    --no-async-scheduling \
    --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_min":2,"prompt_lookup_max":3}'
```

## Benchmarking

Use vLLM's benchmark CLI for serving workloads. For a reproducible Gemma4
target-only versus MTP comparison, use the in-tree benchmark:

```bash
python -m tools.benchmark.gemma4_mtp_benchmark --help
```

`tools/README.md` documents the before-and-after commands and the natural-prompt
dataset used for speculative-decoding measurements.
