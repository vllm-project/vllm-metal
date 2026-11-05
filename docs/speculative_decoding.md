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

- Only plain greedy requests (`temperature=0`, without penalties, token
  constraints, or sample logprobs) are drafted. Other requests run without
  speculation.
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

## Suffix

Suffix decoding ([Snowflake, arXiv 2411.04975](https://arxiv.org/pdf/2411.04975))
drafts by walking a suffix tree built over the prompts and previously
generated responses of the serving process. Like n-gram it needs no draft
model and no KV cache; unlike n-gram it learns *across* requests, so its
wins concentrate in agentic and templated workloads where responses re-emit
near-identical scaffolding (tool-call envelopes, structured lists, repeated
phrasing). Draft lengths are dynamic per request and per step, bounded by
`num_speculative_tokens`. Support is limited to MLX, text-only,
full-attention paged-attention models
(hybrid GDN targets keep the ngram-only gate for now). Verification is
greedy-only and requires `--no-async-scheduling`, like every Metal
speculator.

Follow the upstream
[Suffix Decoding guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix_decoding/)
for the tuning knobs (`suffix_decoding_max_tree_depth`,
`suffix_decoding_max_spec_factor`, `suffix_decoding_min_token_prob`,
`suffix_decoding_max_cached_requests`).

### Installing arctic_inference on macOS

The suffix tree lives in the [`arctic_inference`](https://pypi.org/project/arctic-inference/)
package. The tested version is **0.3.0**. Its suffix-tree core uses
C++/nanobind and builds from source on macOS without CUDA.

Activate the same native arm64 environment used to run vLLM Metal, then
install the build tools and the pinned backend. Build without isolation
to use the environment's existing PyTorch and nanobind versions, and pin
the interpreter so CMake does not select a different Homebrew Python:

```bash
  python -m pip install cmake ninja grpcio-tools wheel
  Python_ROOT_DIR="$VIRTUAL_ENV" Python_FIND_STRATEGY=LOCATION \
    python -m pip install --no-build-isolation --no-deps \
    --no-binary arctic-inference arctic-inference==0.3.0
  python -c 'from arctic_inference.suffix_decoding import SuffixDecodingCache; SuffixDecodingCache(max_tree_depth=24)'
```

Serving without the package installed fails at config time with vLLM's own
error, which may suggest `arctic-inference==0.1.1`. Use the tested 0.3.0
version above for this backend. It is not required for other speculative
methods. The package is Apache-2.0 licensed.

`scripts/test.sh` and CI include the real Arctic tests in the main pytest
run. CI caches the compiled wheel for the matching toolchain; cache hits
skip the build tools and source compilation. Local runs keep the wheel in
the test environment's `arctic-wheels` directory. To omit the optional
backend for a shorter local check, use `VLLM_METAL_TEST_SUFFIX=0 scripts/test.sh`;
lightweight suffix unit tests still run. For an already prepared environment,
`pytest -m suffix tests/test_suffix_proposer.py` runs just the real-backend tests.

Suffix uses the existing greedy verifier: each accepted draft must match the
target's argmax at that position, and the first rejection emits the target's
correction. It does not use approximate acceptance rules. As described in
vLLM's [numerical limitations](https://docs.vllm.ai/en/latest/features/speculative_decoding/#lossless-guarantees-of-speculative-decoding),
floating-point and batch-shape effects can still change output tokens between
ordinary and speculative execution. This integration keeps the default Metal
model kernels; it does not enable batch-invariant execution or guarantee
identical complete output sequences. A separate opt-in reproducibility mode
is outside this feature's scope.

For diagnostics, compare fresh spec-off/spec-on runs with identical input
token IDs and settings using `tools/check_parity.py --method suffix --top-k 5`.
The tool retains its stricter check: any suffix-versus-target token divergence
returns an unsuccessful exit status and is recorded in the summary.
`TOP_K_MATCH` and first-divergence logprobs do not prove equivalence or identify
the cause. Investigate divergences to distinguish numerical variation from
incorrect verification or request state; the diagnostic tool does not change
the engine's execution mode.

With `--enable-sleep-mode`, suffix decoding supports the worker's
`sleep(level=1)` → `wake_up()` cycle. Suffix trees remain in host memory
while the KV cache is released, so cached responses can be reused after
wake-up. Level 2 sleep is not supported on Metal.

### Example

```bash
  vllm serve Qwen/Qwen3-8B \
    --max-model-len 2048 \
    --no-async-scheduling \
    --speculative-config '{"method":"suffix","num_speculative_tokens":3}'
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
