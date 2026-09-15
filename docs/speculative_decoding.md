# Speculative Decoding

vllm-metal supports four speculative decoding methods on the paged-attention
path. Use vLLM's [speculative decoding guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
for method behavior and configuration details.

| | MTP | DSpark | Draft model | N-gram |
|---|---|---|---|---|
| `--speculative-config` method | `mtp` | `dspark` | `draft_model` | `ngram` |
| Target models | Gemma4 | Qwen3 4B/8B/14B (needs a matched drafter) | Non-hybrid paged-attention models | Non-hybrid paged-attention models |
| Draft source | Matching Gemma4 assistant checkpoint | Parallel backbone and sequential Markov head (consumes target hidden states) | Separate smaller model | Prompt and output token history |
| `num_speculative_tokens` | Configurable (2–3 typical) | 1 to the checkpoint `block_size` (7); 4 measured best | Configurable (3–5 typical) | Configurable (3–5 typical) |
| Additional model weights | Assistant checkpoint | Drafter checkpoint | Draft model | None |
| Additional KV cache | None; reads target KV | Proposer-owned drafter context, reserved before the target KV cache is sized | Second scheduler-managed cache | None |

All four methods currently have these Metal-specific constraints:

- Only plain greedy requests (`temperature=0`, without penalties, token
  constraints, or sample logprobs) are drafted. Other requests run without
  speculation.
- Scheduling must be synchronous. The Metal platform disables async scheduling
  when speculative decoding is configured.
- Pipeline parallelism is not supported with speculative decoding.
- Hybrid GDN targets and heterogeneous draft vocabularies are not supported.
- `long_prefill_token_threshold`, when set, must be at least
  `1 + num_speculative_tokens`.

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

## DSpark

[DSpark](https://github.com/deepseek-ai/DeepSpec) drafts a block of tokens from
the *target's own hidden states*: a small parallel backbone cross-attends over
the fused residuals of a few target layers (the drafter's `target_layer_ids`),
proposes a `block_size`-token block (7 in the published checkpoints) in one
pass, and a rank-256 Markov head corrects each position on the token before
it. The target then verifies the block exactly as for the other methods, so
greedy output is unchanged. The Metal implementation covers greedy drafting
and verification with a per-request drafter context; DeepSpec's adaptive and
sampled drafting modes are not implemented.

A DSpark drafter is **trained per target** — it consumes that target's
hidden states and predicts that target's continuations, so it only works for
models with a published matched drafter:

| Trained target | DSpark drafter |
| --- | --- |
| `Qwen/Qwen3-4B` | `deepseek-ai/dspark_qwen3_4b_block7` |
| `Qwen/Qwen3-8B` | `deepseek-ai/dspark_qwen3_8b_block7` |
| `Qwen/Qwen3-14B` | `deepseek-ai/dspark_qwen3_14b_block7` |

These pairings are from the [official DeepSpec release](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/README.md).
A quantized conversion of a trained target (for example the `mlx-community`
4-bit Qwen3-4B used below) works with that target's drafter; the drafter
itself is converted to the MLX 4-bit recipe at load. Only the 4B pair has been
measured here. Gemma4 drafters and the integrated DeepSeek-V4 form are not
supported.

For targets without a matched DSpark drafter, use [N-gram](#n-gram)
(model-agnostic) instead.

### Example

```bash
vllm serve mlx-community/Qwen3-4B-4bit \
  --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.3 --max-model-len 4096 --max-num-seqs 4 \
  --generation-config vllm \
  --speculative-config '{"method":"dspark","model":"deepseek-ai/dspark_qwen3_4b_block7","revision":"3457dff1417cb84927f6098a5fcb7cee85c934b7","num_speculative_tokens":4}'
```

The drafter can be an HF repo id or a local path; pin `revision` for
reproducible runs. `num_speculative_tokens` must be between 1 and the
checkpoint's `block_size`. vLLM itself implements DSpark only in its GPU
V2 model runner and rejects the `dspark` method on the V1 runner that Metal
uses, so after validating the pair the platform presents it to vLLM as the
`draft_model` method; the Metal runner recognises the drafter under either
name. Async scheduling is downgraded to synchronous by the platform, as for
every Metal speculative method.

Startup validates the pair — a `Qwen3ForCausalLM` target with a
`Qwen3DSparkModel` drafter of matching hidden size, vocabulary and layer
count — and rejects the vLLM speculative options this proposer does not use:
adaptive verification, non-default draft or rejection sampling methods,
`dspark_draft_topk`, draft `quantization`, `kv_cache_dtype`, `max_model_len`,
`attention_backend` and `draft_load_config` overrides,
`disable_padded_drafter_batch`, `use_local_argmax_reduction`, draft or target
tensor parallelism, and LoRA. The loader checks the checkpoint's shards and
headers against the config, rejects non-finite weights, converts one tensor
at a time within the memory budget, and materializes the drafter before the
target KV cache is sized.

Confirm speculative decoding is active: the server log shows
`DSpark drafter loaded for speculative decoding: <model> (block_size=7,
target_layer_ids=[...])` with the reserved context, capture and workspace
sizes, and the periodic `SpecDecoding metrics ... Avg Draft acceptance rate`
line reflects the live acceptance. `tools/check_sd_lossless.py --method dspark`
compares a pair's greedy output against a target-only engine
(see [Tools](tools.md#speculative-decoding-losslessness)).

### Characteristics

- **Greedy only**, like every Metal speculative method: sampled requests run
  target-only.
- **Contiguous per-request context.** The proposer keeps, for each request,
  the drafter's context KV over every committed target position. Every
  prefill chunk contributes its captured features (including chunks that
  sample nothing), and each verification step ingests the accepted rows at
  their absolute positions. A request whose features are unavailable — a
  target prefix-cache hit, a context slot or budget exhausted — runs
  target-only for its lifetime; the proposer never replays a prompt. With
  prefix caching on (the default), requests that share a cached prefix
  of at least one KV block are therefore not drafted; pass
  `--no-enable-prefix-caching` to draft such workloads.
- **Batched drafting.** Every eligible request drafts in one backbone pass
  over padded contexts. Memory is reserved for `min(max_num_seqs, 32)`
  complete contexts; requests beyond that use target-only generation.
- **Bounded memory.** The drafter's context, capture staging and per-step
  workspace are reserved before the target KV cache is sized and appear as
  `dspark_context_and_workspace` in the paged-attention plan log line. A
  configuration that cannot fit fails at startup naming the limits to lower
  (`--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`); a
  recoverable allocation failure while drafting releases the draft context
  and keeps the target's output.

### Performance

Measured on an Apple M5 Max (48 GB) with `mlx-community/Qwen3-4B-4bit` and
`deepseek-ai/dspark_qwen3_4b_block7`, against a target-only server started
with identical flags. `vllm bench serve --temperature 0`: only greedy
requests are drafted, so a sampled benchmark would measure an idle drafter.
The `/metrics` spec-decode counters confirm each speculative run drafted.

Natural short prompts (10 each from the DeepSpec gsm8k, humaneval, mbpp and
alpaca sets, chat template, 256 output tokens), `--max-num-seqs 4`:

| Server | Clients | Output tok/s | Median TPOT | Accepted per round |
| --- | --- | --- | --- | --- |
| target-only | 1 | 153.1 | 6.4 ms | – |
| DSpark K=4 | 1 | 178.7 (+17%) | 5.1 ms | 1.79 of 4 |
| DSpark K=7 | 1 | 149.9 (−2%) | 6.5 ms | 2.05 of 7 |
| target-only | 4 | 463.3 | 8.5 ms | – |
| DSpark K=4 | 4 | 374.3 (−19%) | 9.7 ms | 1.39 of 4 |
| DSpark K=7 | 4 | 358.5 (−23%) | 9.6 ms | 1.53 of 7 |

The `sonnet` benchmark (≈550-token prompts sharing a 200-token prefix, 150
output tokens) with prefix caching disabled on both servers, so the shared
prefix does not turn the requests target-only:

| Server | Clients | Output tok/s | Median TPOT | Accepted per round |
| --- | --- | --- | --- | --- |
| target-only | 1 | 138.1 | 6.5 ms | – |
| DSpark K=4 | 1 | 159.4 (+15%) | 5.7 ms | 1.66 of 4 |
| target-only | 4 | 327.0 | 9.5 ms | – |
| DSpark K=4 | 4 | 248.5 (−24%) | 14.3 ms | 1.62 of 4 |

A fixed draft width is a single-stream win at `K=4` and a loss once several
requests share a step: verification adds `K` rows per request per step, and a
wider block adds rows faster than it adds accepted tokens. Choose `K` for the
workload, and leave the method off for a server that is usually busy.

Requests that hit the target prefix cache are never drafted — their hidden
states were never captured — so with the default prefix caching a workload
with a long shared prefix pays the idle drafter's cost (11–13% on the sonnet
workload above). Pass `--no-enable-prefix-caching` for such workloads.

### Limitations

- **Matched models required.** Do not infer support for another target from
  a similar model name or tensor shape; only the Qwen3 family has a qualified
  hidden-state capture.
- **Low-concurrency method.** The single-stream gain turns into a loss once
  several requests share a step (see Performance).
- **Output parity is exact up to the target's own numerical stability.**
  `tools/check_sd_lossless.py` on the 4B pair (12 prompts, 64 greedy tokens):
  served one prompt at a time, K=4 and K=7 each give 5 exact outputs and 7
  first divergences between tokens within one bfloat16 ULP of each other in
  the target's own logits (gap ≤ 0.125 nats), and the target-only engine
  served one prompt at a time differs from its own 12-prompt batch on those
  same 7 prompts. In a 12-prompt batch, K=2 gives 10 exact outputs and 2 such
  ties; K=7 gives 6 exact, 5 ties and one prompt whose speculative token is the
  target's runner-up by one ULP single-row and the target's own argmax under
  16-token prefill chunks. The n-gram method on the same target and batch
  shows the same class (9 exact, 3 ties). Degenerate repetitive prompts can
  flip between near-tied tokens on any execution path.
- **Not implemented:** sampled-request verification, DeepSpec's adaptive and
  confidence-based draft lengths, asynchronous scheduling, LoRA and tensor or
  pipeline parallelism.

---

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
