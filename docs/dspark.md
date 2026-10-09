# DSpark experimental serving

DSpark greedy serving is an experimental stage of [RFC #825](https://github.com/vllm-project/vllm-metal/issues/825).
It uses the shared DFlash target-capture and committed-feature lifecycle with
scheduler-owned draft KV. DSpark's own embeddings and Markov head propose tokens;
the target verifies every proposal. Prefix reuse is supported in synchronous
serving. Confidence-based planning, sampled verification, and asynchronous
scheduling remain subsequent work.

`vllm_metal/v1/dspark.py` adapts the [MIT-licensed DeepSpec implementation](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/LICENSE).
It retains DeepSpec's copyright and full MIT permission notice, following the
existing DFlash module's approach to third-party attribution.

## Serve the trained pair

```bash
vllm serve mlx-community/Qwen3-4B-4bit \
    --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
    --max-model-len 2048 \
    --enable-prefix-caching \
    --no-async-scheduling \
    --speculative-config '{
      "method": "dspark",
      "model": "deepseek-ai/dspark_qwen3_4b_block7",
      "revision": "3457dff1417cb84927f6098a5fcb7cee85c934b7",
      "num_speculative_tokens": 7
    }'
```

Send plain greedy requests (`temperature=0`). Requests with sampling, penalties,
grammar constraints, or sample logprobs use the target without drafting.
The first stage requires a single-device Qwen3 text target without LoRA or
TurboQuant, native cache blocks of 8/16/32, and matching target/draft activation
precision. Draft weights load before memory profiling and share the device
budget with target weights, activations, and scheduler-owned KV.
Pre-quantized draft checkpoints and an explicit draft-cache precision that differs
from the target activation precision are rejected. An optional runtime Q4
conversion is described below.

Widths 1 through the checkpoint's trained width are supported. The optional
`num_speculative_tokens_per_batch_size` schedule can use zero to pause drafting;
verified target features still commit during a pause. A K-token proposal writes
exactly K draft slots (anchor plus K-1 masks), including at the context boundary.
Drafting stops when the selected span would exceed the effective target/draft
context limit. Cancellation and preemption discard logical feature coverage;
resumed prefills adopt the scheduler's new common prefix and commit the suffix
before drafting resumes.

Prefix reuse follows the shared [DFlash cache lifecycle](dflash.md#cache-lifecycle-and-validation).
Both target and draft groups must have the prefix; an independently missing
group forces suffix recomputation. Fallback and zero-width requests still commit
draft features, so they can populate reusable prefixes. The scheduler owns
hashing, shared pages, eviction, and cache reset. Its existing speculative
block-drop policy stays in effect, and temporary draft slots are never committed
prefix data. Use `--no-enable-prefix-caching` for a cold-cache comparison.

KV offloading is rejected during configuration: the Metal offloader does not
support restoring the target and draft cache groups. Remove `--kv-offloading-size`
and any offloading connector from `--kv-transfer-config` when using DSpark.

`enable_adaptive_verification`, non-greedy `draft_sample_method`, and nonstandard
`rejection_sample_method` are rejected rather than ignored.
The Metal compatibility bridge exempts only `MetalWorker` from the GPU V1
runner's DSpark prohibition; all other upstream runner checks remain active.

### Limit Markov candidates

Optionally add `"dspark_draft_topk": 64` to the speculative configuration.
For each draft position, DSpark selects that many candidates from its base
logits and computes the sequential Markov correction only for those tokens.
This reduces Markov projection work but can exclude the full-vocabulary winner,
changing proposals and acceptance. The target still verifies every proposal
against its full vocabulary. Confidence uses the actual preceding draft token.

The limit must be an integer from 1 through the draft vocabulary size. An
explicit setting overrides the checkpoint's `dspark_draft_topk`; with neither
set, the existing full-vocabulary path is unchanged. A limit equal to the
vocabulary size also uses that path. Smaller limits trade candidate coverage for
less projection work; measure acceptance and end-to-end latency together.
This option does not enable adaptive verification or change the proposal width.

### Quantize draft linear layers

Add `--additional-config '{"dspark_draft_quantization":"q4"}'` to the serve
command to convert the draft backbone's linear layers and its vocabulary
projection to MLX affine 4-bit weights with group size 64. Conversion runs once,
after checkpoint validation and before memory profiling or compilation. Omit the
option to retain the checkpoint weights. Only `q4` is supported, and the option
requires `method="dspark"`.

Embeddings, feature fusion, normalization, and Markov/confidence heads retain
checkpoint precision. Draft activations and KV also retain FP16/BF16 precision;
the target is unchanged. This reduces resident draft weight memory, but startup
still loads the original floating checkpoint before conversion. Linear input
dimensions must be divisible by 64; incompatible checkpoint dimensions are
rejected before allocating the draft model or loading its weights.

Q4 can change draft proposals and acceptance. The target still verifies each
proposal against its full vocabulary. Measure acceptance and serving latency
together; reduced draft memory does not establish a speedup or native greedy
equivalence. The existing reduced-precision qualification limits still apply.

Both `tools.dflash_serving_parity` and
`tools.benchmark.dspark_serving_benchmark` accept
`--dspark-draft-quantization q4`, including with `--dspark-draft-topk 64`.
The option applies only to the DSpark arm; native, target-only and ordinary
draft-model controls retain their existing settings.

## Serving validation

The shared lifecycle tests exercise both DFlash and DSpark, in both target
verification layouts. They require exact output IDs against target-only serving,
actual drafting and rejection, chunked prefill, context/page boundaries, mixed
greedy/fallback batches, preemption/recomputation, cancellation and request-ID
reuse, stop/EOS handling, and scheduler-driven width changes through zero.

```bash
pytest -m slow tests/test_block_draft_serving_e2e.py tests/test_block_draft_schedule_e2e.py
pytest -m slow tests/test_block_draft_prefix_caching_e2e.py -k dspark
python -m tools.dflash_serving_parity \
    --method dspark --num-draft-tokens 7 \
    --target /path/to/target/snapshot --draft /path/to/draft/snapshot \
    --batch-size 1 2 --max-tokens 32 --output-dir /path/to/new-serving-results
```

The prefix-reuse tests pin the documented 4B target/DSpark revisions and
use K=7 with top-K=64 in both verification layouts. They require exact output IDs
against cache-disabled serving, observed cache hits and verified drafts after
reuse, reduced prefill work, fallback-produced prefixes, shared prompts,
cancellation, and cache-pressure resume. Per-case IDs, counters, and elapsed times
are retained in pytest's temporary directory; timings are observations rather
than performance gates. A retained-page preemption case requires a nonzero hit
on resume. A deliberately incorrect proposal exercises rejection without relying
on the model making a natural mistake; its output must still match the original
greedy continuation.
The first verification window also checks the sampled anchor, absolute position,
and first-proposal IDs against cache-disabled serving, so target corrections cannot
hide a draft shift after prefill or resume. Separately trained drafters are not
required to match an identical target/draft-model control's acceptance rate.

The shared parity tool compares native mlx-lm, target-only serving, and DSpark
serving. It records actual verification counts and reports `EXACT`, `TOP_K_MATCH`,
and failures separately; top-k agreement is not exact sequence equivalence.
These are correctness checks, not throughput or latency measurements.
Pass `--dspark-draft-topk 64` to the parity tool to exercise candidate limiting.

For the pinned pair above on Apple M5 Max, with vLLM 0.30.0, MLX 0.32.1,
and mlx-lm 0.32.0, the shared 40-prompt corpus (32 output tokens, K=7) reports:

| Mode | Batch size | EXACT vs native | TOP_K_MATCH | FAIL |
| --- | --- | --- | --- | --- |
| Target only | 1 | 27 | 13 | 0 |
| Target only | 2 | 27 | 13 | 0 |
| DSpark | 1 | 24 | 16 | 0 |
| DSpark | 2 | 28 | 12 | 0 |

`TOP_K_MATCH` checks the first divergent choice; it does not validate the rest of
the divergent continuation. DSpark matches same-batch target-only sequences
exactly on 29/40 prompts at batch 1 and 32/40 at batch 2. It verifies 3,395/3,297
draft tokens and accepts 830/842, respectively. Thus drafting is exercised, but
**bitwise serving losslessness is not established**. The exact lifecycle cases
above and the wider corpus's near-tie behavior are both part of qualification.

The reduced-precision **draft-forward** qualification failures below remain
unresolved. Different draft candidates can change acceptance and performance;
they do not bypass target verification. Serving output checks and checkpoint
candidate equivalence are separate requirements, and the experimental integration
does not establish complete DSpark qualification or a speedup.

### Full-continuation audit

Use `--audit-continuations` for a strict check that also covers every token after
the first divergence:

```bash
python -m tools.dflash_serving_parity \
    --method dspark --num-draft-tokens 7 --dspark-draft-topk 64 \
    --target /path/to/target/snapshot --draft /path/to/draft/snapshot \
    --batch-size 1 4 --max-tokens 32 --audit-continuations \
    --output-dir /path/to/new-continuation-audit
```

The tool observes the actual target verification rows without requesting sample
logprobs, which would disable drafting. It then replays each emitted continuation
through native mlx-lm with fresh KV, forcing the observed tokens so that every
comparison uses the prefix serving actually produced. A native self-replay
control checks this replay against the free-running reference.

`continuation-audit.json` retains every position's emitted token, native argmax,
candidate rank and score gap, and the serving verifier's own argmax. It labels
prefill, ordinary decode, accepted draft, correction, and bonus decisions. The
summary separates mismatches against native MLX from mismatches against the
serving verifier; it also records exact free-running sequence counts. Checkpoint
paths are resolved once for all workers and recorded with source hashes and
package versions in `metadata.json`.

Any greedy mismatch, including a tie with a different argmax, fails the strict
audit and makes the command exit nonzero. Ranks and gaps are diagnostics, not
tolerances. Missing or invalid evidence also fails. The legacy `TOP_K_MATCH`
report cannot override these failures. This qualifies greedy decisions for the
tested workload; it does not establish sampled verification or measure speed.

## Measure HTTP serving and memory

Run the matched three-way comparison from a source checkout on macOS:

```bash
python -m tools.benchmark.dspark_serving_benchmark \
    --target /path/to/Qwen3-4B-4bit/snapshot \
    --dspark /path/to/dspark_qwen3_4b_block7/snapshot \
    --draft /path/to/Qwen3-0.6B/snapshot \
    --concurrency 1 4 --repeats 2 --output-len 64 \
    --output-dir /path/to/new-benchmark-results
```

Use the pinned target/DSpark pair above and a compatible ordinary draft model.
All three arms use the same target, tokenizer, plain greedy prompts, output
length, context/prefill limits, memory fraction, and disabled prefix caching.
The default workload is the shared 40-prompt parity corpus. For representative
longer contexts, supply `--prompt-file` with JSONL `{"prompt": "..."}` rows and
`--num-prompts`; prompts must fit the context limit without truncation. Draft
widths are recorded separately (`--dspark-width 7`, `--draft-width 3`).
Use `--dspark-draft-topk 64` to measure candidate limiting in the DSpark arm;
the target-only and ordinary-draft arms retain their original configuration.

The tool starts fresh loopback-only `vllm serve` processes with multiprocessing
enabled, reverses arm and concurrency order between repeats, and shuts down each
server's process group before starting the next. Per concurrency, it performs
an untimed token-ID comparison pass, a discarded streaming warmup pass, then
measures with `vllm bench serve`. Full output IDs from the untimed pass are
retained and compared to that repeat's target-only arm. That pass submits
batches of prompts in one completion request; the timed pass sends independent
concurrent streaming requests. Exact-match counts describe only the untimed
pass. Sample logprobs are never requested, since they disable drafting. Timings
use the benchmark's streaming usage counts and latency definitions, not SSE
chunk counts.

`summary.json` retains every repeat's throughput and paired ratio, latency,
exact sequence counts, and `greedy_parity_passed` for each arm/concurrency.
`first_divergences_vs_target` lists the first differing token of every mismatched
prompt, grouped in `repeat_ids` order. Prompt and output-token indices are
zero-based; token IDs can be looked up in the saved `tokens.json` files.
Each run also saves commands, detailed benchmark results, token IDs, server logs,
source hashes, and before/after counters.
Counter snapshots wait for completed requests and exclude both warmup passes;
a missing required counter or a speculative arm with no actual verified drafts
fails. Failed, incomplete, or mismatched workloads cannot produce a successful
summary. Any token mismatch in either speculative arm fails greedy qualification
and makes the command exit nonzero after all measurements finish. The summary
and raw timings are preserved for diagnosis; their throughput ratios do not
establish a lossless speedup. Investigate divergences with the separate serving
parity tool.

Worker snapshots distinguish MLX active/peak allocation, allocator cache, worker
RSS and lifetime peak RSS, and physical KV backing bytes. The startup snapshot
includes model loading and profiling; the measured phase resets only the MLX
peak counter. These are different accounting domains on unified memory and
must not be added together. Worker RSS excludes the API/client processes;
reserved KV bytes do not mean every page is resident. The scheduler's own
reported token capacity accounts for cache groups and is saved separately.
Equal memory fractions can yield different usable capacities across arms.

Follow the [macOS benchmark guide](benchmarking-macos.md) for warmup, power,
thermal state and desktop contention. Before/after machine observations are
saved; they cannot rule out interference during a run. Natural continuations
can differ even under greedy sampling, so these measurements do not establish
bitwise losslessness or a general speedup.

To reproduce the RFC's Qwen3-8B/0.6B workload, use these pinned local snapshots
in the same command:

- Target: [mlx-community/Qwen3-8B-4bit](https://huggingface.co/mlx-community/Qwen3-8B-4bit/tree/545dc4251c05440727734bcd94334791f6ab0192).
- DSpark: [deepseek-ai/dspark_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7/tree/03326e5043815da1f81b109078b2889737c26017).
- Ordinary draft: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca).

Compare the full-vocabulary and candidate-limited configurations in separate
output directories, keeping all other settings fixed. This pair uses more
memory than the 4B pair; record the memory fraction and cache capacity for
every arm.

## Forward contract

The initial checkpoint is
[`deepseek-ai/dspark_qwen3_4b_block7`](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7/tree/3457dff1417cb84927f6098a5fcb7cee85c934b7),
paired with Qwen3-4B. It has five draft layers and a trained width of seven.
The loader checks target geometry before loading weights; use the trained pair,
since matching dimensions alone do not establish training/tokenizer compatibility.

- DSpark owns its trained embedding and output projection. Both load from the
  draft checkpoint, including when the target is quantized.
- To predict K tokens, the input block contains one anchor and K−1 masks.
  **Slot 0 produces the first proposal**, so K=1 still runs the anchor through
  the backbone. The supported range is 1 through the trained block width.
- Features cover the prefix immediately before the anchor. Use
  `DFlashTargetCapture(target, draft.config.backbone)` for the shared HF
  `hidden_states[layer_id + 1]` convention, including final normalization when
  a final-layer tap is configured. Embedding-output taps are not supported.
- Every block query attends the full committed prefix and the complete block.
  `DSparkModel.block_hidden` returns normalized states for all K positions.
- `greedy_proposal` adds the vanilla low-rank Markov correction sequentially:
  the first position uses the anchor; later positions use the actual preceding
  proposal. It returns IDs, corrected logits, and optional raw confidence logits.
  Confidence uses that same predecessor and does not truncate the proposal.
- The loader accepts local, unsharded, uniform FP32/FP16/BF16 safetensors and
  preserves their precision. Tensor names/shapes and finite weights are checked.
  DFlash and DSpark share these checks in `draft_checkpoint.py`.
  Pre-quantized, gated/RNN-head, GIDD, scaled/partial-RoPE, and non-Qwen3 checkpoints
  are rejected. Confidence heads may be absent or may use hidden states with
  or without Markov embeddings.
- Validate external anchor IDs with `draft.validate_anchors(anchors)` outside
  compilation. The repeated forward checks metadata without a CPU token readback.
  Input features and block states must match the draft's compute precision.

## Reproduce forward parity

Use a local checkout of the [official DeepSpec reference](https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2)
at `005e03b81cec38b7da6399833d609ee89a2587f2`. The tool imports that local code;
it does not fetch or execute remote code automatically. Download these snapshots:

```bash
hf download mlx-community/Qwen3-4B-4bit --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25
hf download deepseek-ai/dspark_qwen3_4b_block7 --revision 3457dff1417cb84927f6098a5fcb7cee85c934b7
python -m tools.dspark_parity \
    --target /path/to/target/snapshot \
    --draft /path/to/draft/snapshot \
    --reference /path/to/DeepSpec \
    --context-lengths 1 15 16 17 65 257 1025 \
    --output /path/to/new-results.json
```

The default compares both drafters in FP32 using the checkpoint's stored weights
and native mlx-lm target features. It checks eager and compiled greedy proposals,
normalized block states, corrected logits, and confidence against official
PyTorch eager attention at batches 1/2 and widths 1/3/7. It requires exact proposal
IDs and `atol=rtol=1e-3` for floating outputs, rejects incomplete/non-finite
comparisons, and records versions, source hashes, paths, and case results.
Compilation is reused across inputs. A failed run cannot leave a stale passing
report at the requested output path.

The qualified FP32 matrix passes 42 cases, with 231 proposal IDs identical in
each execution mode. This is forward equivalence, not generated-sequence parity
or a performance claim. Loading the target and two draft implementations requires
substantial host/unified memory; FP32 uses more memory than the stored BF16 weights.

`--dtype bfloat16` is an additional diagnostic with looser tensor tolerances
(`atol=0.25`, `rtol=0.02`) and the same exact-token requirement. It currently
**fails** the qualified pair at batch 1, context 17, K=3: the reference's first
divergent choice ties at 18.25, while MLX produces 18.5 versus 18.25. That choice
changes subsequent Markov corrections. BF16 token equivalence and FP16 checkpoint
qualification therefore remain unestablished. Serving output is checked separately
through target verification.

Small independent CPU-math tests cover block attention, Markov recurrence,
confidence predecessor alignment, optional heads, malformed checkpoints, and
compiled replay. Run them with `pytest tests/test_dspark.py`.

## Scheduler-owned draft KV

`DSparkPagedCache` binds the draft layers' views of a `KVCacheStorage` allocation.
It shares context projection, scatter, and block attention with `DFlashPagedCache`;
each adapter keeps its checkpoint's embeddings, prediction alignment, and heads.
The binding consumes scheduler block tables and never allocates request pages.
Bound layers must be distinct, belong to one scheduler group, and share the
cache block size and precision. Incompatible bindings are rejected before writes.

- `write_context(features, spans)` takes packed target features and
  `(block_ids, first_position, row_count)` spans. Commit only verified target rows.
  After verification, overwrite temporary block KV with those target-feature
  projections, including for accepted draft tokens.
- `compile_draft(num_draft_tokens=K)` returns a callable taking anchor IDs and
  `(block_ids, committed_length)` rows. It returns IDs, corrected logits, and
  optional raw confidence. It reserves **K slots**, including the anchor, and
  predicts from every slot. DFlash retains its separate K+1-slot contract.
- Each block query sees its entire committed prefix and the full draft block.
  K=1 uses ordinary paged decode because its prefix plus anchor is already the
  complete block. Larger blocks use explicit bidirectional ranges.
- Fixed-size block tables and device offsets let a compiled block replay as
  context grows. Cache writes retain dependencies through shared storage.
  The caller owns committed lengths, page ownership, and invalidation after
  preemption or request reuse; the cache does not track request identities.

`pytest tests/test_dspark_paged.py` exercises actual Metal kernels with FP16/BF16,
cache blocks 8/16/32, widths 1/3/7, optional confidence heads, ragged requests,
incremental verified commits, rejected tails, and reused pages. Independent CPU
attention checks logits and confidence. Exact-ID assertions use well-separated
Markov choices so numerical near ties do not obscure cache errors. Tests also
check page bounds, untouched target storage, and replay without retracing.

The full-checkpoint diagnostic compares paged proposals against native dense
DSpark at the same precision, using the snapshots above:

```bash
python -m tools.dspark_paged_parity \
    --target /path/to/target/snapshot \
    --draft /path/to/draft/snapshot \
    --dtype float16 \
    --output /path/to/new-paged-results.json
```

It records all 42 cases, source hashes, tensor errors, and exact-token checks,
and exits unsuccessfully if any gate fails. On the qualified pair, FP16 matches
all 231 proposal IDs, but six one-token-prefix cases exceed the strict final-logit
bound (`atol=0.015`, `rtol=0.02`; maximum absolute error 0.04004).
The bound is retained and these cases remain failures. This cache diagnostic does
not establish reduced-precision checkpoint equivalence or serving losslessness;
use the separate serving checks above to assess target-verified output.

`--dtype bfloat16` passes 41 of the 42 native comparisons, but fails at batch 2,
context 1, K=7: the dense path ties at 16.75, while paged attention produces
16.75 versus 16.625. The changed choice affects subsequent Markov corrections.
