# DSpark experimental serving

DSpark greedy serving is an experimental stage of [RFC #825](https://github.com/vllm-project/vllm-metal/issues/825).
It uses the shared DFlash target-capture and committed-feature lifecycle with
scheduler-owned draft KV. DSpark's own embeddings and Markov head propose tokens;
the target verifies every proposal. Confidence-based planning, sampled
verification, prefix reuse, and asynchronous scheduling remain subsequent work.

`vllm_metal/v1/dspark.py` adapts the [MIT-licensed DeepSpec implementation](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/LICENSE).
It retains DeepSpec's copyright and full MIT permission notice, following the
existing DFlash module's approach to third-party attribution.

## Serve the trained pair

```bash
vllm serve mlx-community/Qwen3-4B-4bit \
    --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
    --max-model-len 2048 \
    --no-enable-prefix-caching \
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
Draft quantization and an explicit draft-cache precision that differs from the
target activation precision are rejected.

Widths 1 through the checkpoint's trained width are supported. The optional
`num_speculative_tokens_per_batch_size` schedule can use zero to pause drafting;
verified target features still commit during a pause. A K-token proposal writes
exactly K draft slots (anchor plus K-1 masks), including at the context boundary.
Drafting stops when the selected span would exceed the effective target/draft
context limit. Cancellation and preemption discard logical feature coverage;
recomputation overwrites reused pages before drafting resumes.

`enable_adaptive_verification`, non-greedy `draft_sample_method`, nonstandard
`rejection_sample_method`, and `dspark_draft_topk` are rejected rather than ignored.
The vLLM 0.30 compatibility bridge exempts only `MetalWorker` from the GPU V1
runner's DSpark prohibition; all other upstream runner checks remain active.

## Serving validation

The shared lifecycle tests exercise both DFlash and DSpark, in both target
verification layouts. They require exact output IDs against target-only serving,
actual drafting and rejection, chunked prefill, context/page boundaries, mixed
greedy/fallback batches, preemption/recomputation, cancellation and request-ID
reuse, stop/EOS handling, and scheduler-driven width changes through zero.

```bash
pytest -m slow tests/test_block_draft_serving_e2e.py tests/test_block_draft_schedule_e2e.py
python -m tools.dflash_serving_parity \
    --method dspark --num-draft-tokens 7 \
    --target /path/to/target/snapshot --draft /path/to/draft/snapshot \
    --batch-size 1 2 --max-tokens 32 --output-dir /path/to/new-serving-results
```

The shared parity tool compares native mlx-lm, target-only serving, and DSpark
serving. It records actual verification counts and reports `EXACT`, `TOP_K_MATCH`,
and failures separately; top-k agreement is not exact sequence equivalence.
These are correctness checks, not throughput or latency measurements.

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
  Quantized, gated/RNN-head, GIDD, scaled/partial-RoPE, and non-Qwen3 checkpoints
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
