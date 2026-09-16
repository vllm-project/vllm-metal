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

- Only plain requests (greedy, or plain temperature/top-k/top-p without
  penalties, token constraints, or sample logprobs) are drafted. MTP,
  draft-model and n-gram drafting requires greedy requests; DSpark also
  drafts plain sampled requests. Other requests run without speculation.
- MTP, draft-model and n-gram scheduling must be synchronous: the Metal
  platform disables async scheduling for them. A DSpark server keeps vLLM's
  asynchronous scheduler (see [DSpark](#dspark)).
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
greedy output is unchanged. The Metal implementation covers greedy and
sampled drafting and verification with a bounded per-request drafter context,
adds calibrated adaptive planning that skips drafting when it does not pay,
and runs under vLLM's asynchronous scheduler.

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
name. A DSpark server runs vLLM's asynchronous scheduler, vLLM's own
production default; the other Metal speculative methods downgrade async
scheduling to synchronous.

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

The serving mode is selected with `VLLM_METAL_DSPARK_MODE`: `fixed`, the
default, verifies the configured width for every eligible request;
`adaptive` plans each request's draft prefix from its calibrated confidence
and the measured step costs, and skips drafting altogether when the bypass
step is predicted to be faster; `bypass` keeps the drafter loaded but never
drafts. The adaptive mode needs the pair's calibration artifact and this
machine's cost model (`VLLM_METAL_DSPARK_CALIBRATION`,
`VLLM_METAL_DSPARK_COST_MODEL`), produced by the tools described in
[Tools](tools.md#dspark-adaptive-mode-artifacts); startup fails with the
reason when either is missing or belongs to another pair.
`VLLM_METAL_DSPARK_DRAFT_PRECISION=source` keeps the drafter in the
checkpoint's own precision instead of the qualified affine 4-bit conversion
(more memory, a different cost profile; re-profile the cost model for it).

Confirm speculative decoding is active: the server log shows
`DSpark drafter loaded for speculative decoding: <model> (block_size=7,
target_layer_ids=[...]) ... mode=<fixed|adaptive|bypass>` with the reserved
context, capture and workspace sizes, the periodic
`SpecDecoding metrics ... Avg Draft acceptance rate` line reflects the live
acceptance, and every 2,000 drafting steps the proposer logs a
`DSpark counters` snapshot (bypass reasons, proposed, scheduled and accepted
tokens, per-position acceptance, planner lengths). `tools/check_sd_lossless.py --method dspark`
compares a pair's greedy output against a target-only engine
(see [Tools](tools.md#speculative-decoding-losslessness)).

### Characteristics

- **Greedy and sampled requests.** Greedy requests are verified exactly.
  A plain temperature/top-k/top-p request samples its drafts from exact
  float32 proposal distributions that stay attached to the scheduled proposal,
  and verification accepts each draft with probability `min(1, p/q)`, samples
  the first rejected position from the normalized residual and the bonus
  token from the target distribution, all from the request's own random
  streams (at a fixed draft width, a seeded request reproduces whatever else is
  in the batch; under the adaptive planner the width itself depends on the batch).
  The emitted distribution equals the
  target's; the tokens at a given seed differ from target-only serving.
- **Load regime.** When the calibrated planner declines to draft for 32
  consecutive steps (the batch is too large for verification to pay on this
  machine), the proposer lapses: no target feature is captured, no context is
  kept, and the server runs at target-only cost; it primes new requests again
  once the load has dropped below the count it lapsed at and the planner
  would draft, on 8 consecutive steps. The bypass mode lapses from the start
  (`VLLM_METAL_DSPARK_LAPSE`).
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
- **Batched drafting.** The backbone runs across selected requests, each row
  attending to its own slot of a per-layer context arena (no padding, gather
  or mask), and a step's accepted rows are ingested in one batched pass per
  layer. Memory is reserved for up to
  `min(max_num_seqs, VLLM_METAL_DSPARK_MAX_CONTEXTS)` complete contexts
  (default 32); a request scheduled while every slot is held uses target-only
  generation, and slots are reused as requests finish.
  `VLLM_METAL_DSPARK_MAX_DRAFTS_PER_STEP` bounds the requests drafted per
  step with least-recently-drafted rotation; see
  [configuration](configuration.md#environment-variables).
- **Decode pipeline on non-drafting steps.** The proposer can consume a
  pure-decode step whose sampling sync the runner's one-step-ahead pipeline
  defers (the `bypass` mode, or the `adaptive` planner's bypass decision for
  the batch), ingesting that step's target features without the sampled
  token values; drafting and verification steps stay synchronous.
- **Asynchronous scheduling.** A DSpark server runs vLLM's asynchronous
  scheduler (the production default for target-only serving): the scheduler
  books `num_speculative_tokens` placeholder slots per running request and
  the runner fills them with the drafts it produced at the end of the
  request's previous step, reporting unused slots so the speculative-decode
  metrics count real drafts. `--no-async-scheduling` keeps every step
  synchronous; the other Metal speculative methods always are.
- **Bounded memory.** The drafter's context arena is allocated when the
  drafter loads; its capture staging and per-step workspace are reserved
  before the target KV cache is sized and appear as
  `dspark_capture_and_workspace` in the paged-attention plan log line. A
  configuration that cannot fit fails at startup naming the limits to lower
  (`--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`); a
  recoverable allocation failure while drafting releases the draft context
  and keeps the target's output.

### Performance

Measured on an Apple M5 Max (48 GB) with `mlx-community/Qwen3-4B-4bit` and
`deepseek-ai/dspark_qwen3_4b_block7`. Serving figures are paired A/B runs
against a target-only server started with identical flags: the two servers
alternate, five repeats per cell, 95% confidence interval by bootstrap, and a
pair is discarded and retried if another process loads the machine while it
runs.

### Serving

`adaptive` mode against target-only, prompt length × output length, one to
eight clients:

| Prompt × output | Clients | Fixed `K=7` | `adaptive` |
| --- | --- | --- | --- |
| 128 × 128 | 1 | +5.7% [4.6, 12.7] | **+50.1%** [41.5, 57.0] |
| 128 × 128 | 8 | −11.8% [−21.2, 6.6] | **−5.9%** [−28.0, 38.0] |
| 1024 × 128 | 1 | −18.1% [−32.2, −6.3] | **+6.2%** [−27.3, 33.4] |
| 1024 × 128 | 8 | −30.4% [−37.3, −11.6] | **−6.3%** [−20.6, 3.3] |

The planner reaches this by not drafting when verification will not repay it.
Across the 128 × 128 cells at four, eight and sixteen clients it issues 12, 26
and 28 draft tokens in total; fixed `K=7` issues 24,063 at eight clients. The
residual few percent is the capture and planning the server still does while
bypassing, and the load regime bounds it further by stopping capture entirely
while the planner keeps declining.

Long outputs are where drafting pays most, and the planner keeps drafting
there: 128 × 512 at one client is +28.3% [24.9, 47.6], and at eight clients it
resumes drafting mid-run (5,525 accepted of 5,740 drafted) for −1.3%
[−14.1, 7.3].

### Single stream against the standalone runtime

The same three prompts through `mlx-dspark`, the reference standalone MLX
implementation of this drafter, and through this port (200 new tokens, three
trials, `MLX_ENABLE_TF32=0` on both):

| Prompt | `mlx-dspark` | This port |
| --- | --- | --- |
| chat | 181.3 → 212.4 tok/s (1.17×) | 142.8 → 217.6 tok/s (**1.52×**) |
| code | 182.8 → 205.9 tok/s (1.13×) | 142.3 → 311.2 tok/s (**2.19×**) |
| math | 182.1 → 209.4 tok/s (1.15×) | 138.5 → 317.0 tok/s (**2.29×**) |

Each row is that runtime's own target-only rate → its DSpark rate. The paged
target here starts slower than the standalone MLX one, and ends faster.

### Accepted length per drafting round

Full block width, `T=1.0`, 64 prompts per set from the DeepSpec evaluation
sets, against the DSpark paper's Table 1 (bf16 target, full sets, 2,048
tokens):

| Evaluation set | This port | Paper | Acceptance rate |
| --- | --- | --- | --- |
| gsm8k | 6.12 | 6.11 | 73.2% |
| math500 | 5.69 | 5.70 | 67.2% |
| humaneval | 5.11 | 5.38 | 58.8% |
| mbpp | 4.81 | 5.13 | 54.4% |
| mt-bench | 3.54 | 3.64 | 36.5% |
| alpaca | 3.40 | 3.54 | 34.3% |

The drafter reproduces the published acceptance lengths on a 4-bit target,
which is the check that the port is faithful rather than merely fast.

### What still costs

Requests that hit the target prefix cache are never drafted — their hidden
states were never captured — so with the default prefix caching a workload
with a long shared prefix pays the idle drafter's cost. Pass
`--no-enable-prefix-caching` for such workloads. Both the calibration and the
cost model are per machine and per model pair; without them the proposer
serves at its fixed width.

### Limitations

- **Matched models required.** Do not infer support for another target from
  a similar model name or tensor shape; only the Qwen3 family has a qualified
  hidden-state capture.
- **Concurrency.** Verifying `K+1` rows per request costs more than a decode
  step on this path, so a fixed width loses once several requests share a
  step. The `adaptive` mode weighs that trade-off per step from the pair's
  calibration and this machine's cost model, and the load regime stops
  capturing and drafting while the planner declines (see Performance).
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
- **Not implemented:** LoRA and tensor or pipeline parallelism. The `adaptive`
  mode needs a per-pair calibration artifact and a cost model measured on the
  serving machine; without them, serve the `fixed` mode.

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
