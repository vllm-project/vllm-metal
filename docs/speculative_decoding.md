# Speculative Decoding

vllm-metal supports the following speculative decoding methods on the paged-attention
path. Use vLLM's [speculative decoding guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
for method behavior and configuration details.

| | MTP | Draft model | N-gram |
|---|---|---|---|
| `--speculative-config` method | `mtp` | `draft_model` | `ngram` |
| Target models | Gemma4 | Non-hybrid paged-attention models | Non-hybrid paged-attention models and hybrid GDN targets |
| Draft source | Matching Gemma4 assistant checkpoint | Separate smaller model | Prompt and output token history |
| `num_speculative_tokens` | Configurable (2–3 typical) | Configurable (3–5 typical) | Configurable (3–5 typical) |
| Additional model weights | Assistant checkpoint | Draft model | None |
| Additional KV cache | None; reads target KV | Second scheduler-managed cache | None |

Experimental [DFlash serving](dflash.md) is also available for a qualified Qwen3
target/draft pair, with scheduler-owned prefix reuse. It supports fixed widths and
vLLM's batch-size-based draft schedule, including K=0 pause/resume.

Experimental [DSpark serving](dspark.md) uses the same scheduler-owned cache
lifecycle with DSpark's own embeddings and Markov prediction head. It supports
scheduler-owned prefix reuse and verifies greedy proposals on the target.

These methods currently have these Metal-specific constraints:

- Requests that ask for sample logprobs run without speculation.
- Scheduling must be synchronous. The Metal platform disables async scheduling
  when speculative decoding is configured.
- Pipeline parallelism is not supported with speculative decoding.
- Multi-token verification is not supported on the multimodal paged path,
  including text speculative decode sharing a batch with multimodal prefill.
- Heterogeneous draft vocabularies are not supported.
- `long_prefill_token_threshold`, when set, must be at least
  `1 + num_speculative_tokens`.

## Hybrid GDN targets

N-gram speculation supports the text-only execution path on hybrid GDN targets
(Qwen3.5/Qwen3.8-style linear-attention hybrids):

- The verification forward runs the draft span through the same paged kernels
  as plain decode — full-attention layers take the extra query rows, and the
  GDN layers run their state scan over the span. No kernels changed.
- GDN state is *staged*: each verify span defers its span-final state, and a
  partial acceptance re-scans the accepted prefix from the untouched
  pre-span state (see `vllm_metal/attention/impls/gdn_spec.py`). When every
  request in the batch fully accepts its span, no acceptance re-scan runs.
  A partial acceptance currently re-scans all requests in that batch,
  including any prefill spans.
- Hybrid verification requires the lazy GDN kernels
  (`VLLM_METAL_GDN_LAZY_KERNELS`, on by default); disabling them is rejected
  at startup.
- The resolved recurrent-state dtype must be `float32`. Set
  `--mamba-ssm-cache-dtype float32` if `auto` resolves to a lower precision.
  Low precision recurrent state is rejected with hybrid speculation because
  the multi-token scan and ordinary decode round state at different depths;
  a supported BF16 cache configuration failed exact greedy parity. Model
  weights and convolution state do not need to be FP32.
- Hybrid verification requires one window segment per request and attention
  head sizes of at most 256. An incompatible layout is rejected during
  model warmup rather than falling back to expanded per-token segments.
- With align-mode prefix caching, per-request drafts are capped at the
  state-block boundary so prefix checkpoints stay exact.
- **When to enable**: n-gram drafting needs repeated token spans in the
  prompt or generated history, and verification must accept enough drafts
  to offset its cost. Long-context rejections and mixed-batch state re-scans
  can make speculation slower than ordinary decode. Measure acceptance and
  end-to-end serving throughput on representative workloads and concurrency
  before enabling it.

Draft-model and MTP methods on hybrid targets remain unsupported until their
proposers are validated with the state staging and boundary caps. Unsupported
hybrid speculative pairings — a non-ngram method, or a non-GDN hybrid state
family (mamba2, KDA, shortconv) — are rejected at startup, not at the first
verification.

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

vllm serve "$TARGET" \
    --gpu-memory-utilization 0.5 \
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
Its committed KV and speculative lookahead share the Metal KV memory budget
with the target. The scheduler allocates both; no separate draft scratch pool
is reserved. Drafting stops when its write span would exceed the smaller of
the draft model's context limit and the engine's final target context limit.
When no configured positive draft width can fit, the engine logs the request
and effective context limit once and continues with target-only decoding.
Temporary dynamic-width skips and K=0 steps do not trigger this message.

For benchmark diagnostics, read one statistics snapshot per worker. The RPC
returns a list with one entry per worker (`None` for non-draft-model workers):

```python
stats_per_worker = llm.collective_rpc("get_draft_model_stats")
stats = stats_per_worker[0]  # single-worker setups
```

Each ordinary draft-model worker returns `num_context_limit_fallback_requests`,
`min_draft_tokens`, and `max_model_len`. The counter counts affected request
lifetimes once, survives preemption and completion, and resets when the drafter
is recreated. Reading it does not reset it. `min_draft_tokens` is the smallest
positive width reachable in the configured batch-size schedule (zero if all
widths are zero); `max_model_len` is the effective target/draft limit.
Other decoding methods return `None`. These worker diagnostics are not exported
as Prometheus metrics or added to vLLM's scheduler statistics.

### Example

```bash
vllm serve Qwen/Qwen3-8B \
    --gpu-memory-utilization 0.55 \
    --max-model-len 2048 \
    --no-async-scheduling \
    --speculative-config '{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":3}'
```

### Dynamic speculative decoding

See the upstream [dynamic speculative decoding guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/)
for configuration details. This example sets `K=3` for one scheduled request
and `K=0` for two:

```bash
vllm serve Qwen/Qwen3-8B \
    --gpu-memory-utilization 0.55 \
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

The [DFlash guide](dflash.md) covers its experimental serving restrictions and
standalone checkpoint qualification.

Use vLLM's benchmark CLI for serving workloads. For a reproducible Gemma4
target-only versus MTP comparison, use the in-tree benchmark:

```bash
python -m tools.benchmark.gemma4_mtp_benchmark --help
```

`tools/README.md` documents the before-and-after commands and the natural-prompt
dataset used for speculative-decoding measurements.
