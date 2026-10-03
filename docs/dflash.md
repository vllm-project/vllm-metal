# Experimental DFlash serving

This implements the first experimental serving stage of
[RFC #825](https://github.com/vllm-project/vllm-metal/issues/825): synchronous
DFlash drafting with native Metal block attention and
scheduler-owned KV. The standalone model qualification tools remain available.

The first reference checkpoint is
[z-lab/Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16/tree/b74e3a329c4d963783143b1e970d95b002be72bd),
paired with Qwen3-4B. It has **five draft layers** and a 16-position block:
one anchor and 15 predictions. It does not fulfill the RFC's preferred
three-layer milestone. Quantized target conversions borrow their own actual
embedding and output projection; comparisons use that same target precision.

## Serve the qualified pair

```bash
vllm serve mlx-community/Qwen3-4B-4bit \
    --no-enable-prefix-caching --no-async-scheduling \
    --max-model-len 4096 --max-num-seqs 4 \
    --speculative-config '{"method":"dflash","model":"z-lab/Qwen3-4B-DFlash-b16","num_speculative_tokens":3}'
```

Use `temperature=0` without penalties, token constraints, or sample logprobs to
exercise drafting. Other requests use ordinary target sampling. The current
serving path requires a single-device Qwen3 text target, matching FP16 or BF16
target/draft activation precision, and a native cache block size (8, 16, or 32).
LoRA, TurboQuant, and prefix caching fail explicitly.
For this checkpoint, the configured maximum draft width may be 1–15. Near the context limit,
requests continue with target-only decoding when a complete block cannot fit.

## Choose draft widths by batch size

DFlash also accepts vLLM's `num_speculative_tokens_per_batch_size` schedule.
For example, this drafts three tokens for one scheduled request and pauses
drafting for larger batches:

```bash
vllm serve mlx-community/Qwen3-4B-4bit \
    --no-enable-prefix-caching --no-async-scheduling \
    --max-model-len 4096 --max-num-seqs 4 \
    --speculative-config '{
      "method": "dflash",
      "model": "z-lab/Qwen3-4B-DFlash-b16",
      "num_speculative_tokens": 3,
      "num_speculative_tokens_per_batch_size": [[1, 1, 3], [2, 4, 0]]
    }'
```

Ranges are inclusive and count requests scheduled in the current step, including
prefill. The upstream scheduler validates the schedule, carries widths through
gaps and the tail, and caps them at `num_speculative_tokens`. Intermediate widths
such as `[[1,1,3],[2,2,1],[3,4,0]]` are supported too. Each selected width controls
the next proposals; verification still consumes the previously issued width.
The context-limit check uses the selected width, so a shorter block can still fit
when the maximum width cannot.

At K=0, target sampling continues and committed features still update draft KV.
This lets drafting resume on the next step without replaying the prefix. Draft
weights, capture/projection work, KV capacity, and maximum-width lookahead remain
allocated or active, so a paused drafter does not have target-only memory or cost.
Compiled callables are reused per encountered nonzero width. The first use of a
new width/batch shape can incur compilation; warm the tiers before measuring.

This is an explicit batch-size policy, not a learned confidence or cost planner.
Measure it on your workload: fixed-width drafting can be slower than the target
alone at higher concurrency. The default remains fixed-width when no schedule
is supplied.

## Cache lifecycle and validation

Draft weights load before memory profiling. Draft layer specs are included in
the scheduler's cache budget, and target/draft views share its `KVCacheStorage`.
Captured target features are projected once per committed position. The
compiled block reads that paged context and writes the anchor/masks into the
scheduler's lookahead slots; each query sees the whole block. After verification,
only committed target-feature rows enter draft context. These overwrite temporary
KV even for accepted draft tokens. Preemption and cancellation clear logical
coverage before pages or request IDs are reused.

The full-prefix callable below is for qualification. Serving uses
`DFlashPagedCache.compile_draft`, passing cache views and fixed-width block
tables as graph inputs, so it does not gather or reproject the full prefix.
Different batch sizes can compile different graphs; context growth within the
configured limit does not change the block-table shape.

Run the real-checkpoint lifecycle test with:

```bash
pytest -m slow tests/test_block_draft_serving_e2e.py
pytest -m slow tests/test_block_draft_schedule_e2e.py
python -m tools.dflash_serving_parity --output-dir /path/to/new-parity-results
python -m tools.dflash_serving_parity --batch-size 1 2 4 \
    --draft-schedule '[[1,1,3],[2,2,1],[3,4,0]]' \
    --output-dir /path/to/new-scheduled-parity-results
```

It checks target-only parity, page boundaries, chunked prefill, acceptance and
rejection, constrained-cache preemption, cancellation, and context-limit fallback.
It also covers EOS, stop tokens, short output budgets, and mixed batches with
sampling/logprob fallback, including cache-page release after completion.
The scheduled-width test covers shrinking/growing blocks, consecutive K=0 steps,
cancellation, and preemption/recomputation. The parity tool records selected
widths and verified drafts; a configured K=0 batch need not verify any drafts.
Floating-point reduction differences between single-token and multi-token target
forwards can change greedy choices near ties. Report exact matches separately
from mutual top-k agreement; the latter checks only the first differing token.
Measure throughput for your target, draft width, and workload before enabling
this experimental mode in a deployment.

## Model contract

- Checkpoint target layer ID `i` selects Hugging Face `hidden_states[i + 1]`.
  Intermediate entries are decoder outputs before final normalization; the
  final entry is **after** the target's final norm. Use `DFlashTargetCapture`
  to adapt the shared bridge's pre-norm outputs to this contract. It preserves
  order and duplicates, and applies the target norm only for a final-layer tap.
  The reference checkpoint's `[1, 9, 17, 25, 33]` maps to bridge indices
  `[2, 10, 18, 26, 34]` and needs no final normalization.
- Each block attends to the complete committed context and every position
  within its own block. Proposal logits come from slots 1 onward.
- The caller supplies the target projections and full-prefix features.
  The model stores no borrowed target weights or persistent KV state.
- The initial loader accepts a local, unsharded z-lab Qwen3 checkpoint with
  uniform FP32, FP16, or BF16 weights, full attention, and default RoPE.
  Incompatible tensor names, shapes, precision, non-finite weights, and
  unsupported checkpoint semantics fail explicitly.
- Target geometry checks establish structural compatibility. Use the target
  named by the checkpoint's model card; equal geometry alone does not establish
  tokenizer identity or training compatibility. `load_dflash` requires the
  target's configuration as `target_config` and checks it before loading weights.
- `draft_logits` accepts valid target token IDs, normally supplied by the target
  sampler, and does not read token values back to the CPU. Validate external
  token IDs with `draft.validate_anchors(anchors)` at the input boundary, outside
  the compiled or repeated draft forward. Shape and dtype checks remain in the
  forward; mask IDs and anchors are represented as int64 without narrowing.

## Reuse compiled drafting across context lengths

Create a callable once after loading the draft and target weights, then pass it
unpadded full-prefix features on each step:

```python
compiled_draft = draft.compile_draft(
    num_draft_tokens=15,
    embed=target.model.embed_tokens,
    project=project,  # The same tied or untied target head used for qualification.
)
logits = compiled_draft(anchors, features)
```

The callable projects the real prefix, then pads its K/V to multiples of
`context_bucket_size` (default 256), capped at the checkpoint's position limit minus the block width.
Buckets also stop at MLX attention dispatch boundaries so padding does not
select a different reduction algorithm before the real sequence reaches it.
The callable validates the real prefix length before padding. The compiled
forward receives that length as a device scalar for RoPE and attention masking, so growing within
a bucket reuses its graph. Valid context and block keys remain contiguous, with
masked padding at the end, preserving their attention reduction order.

Call the returned wrapper directly. Prefix projections and padding run outside
its compiled block forward: padding raw feature rows can change the GEMM
reduction and BF16 rounding. The block graph receives the padded K/V and is
reused as the real prefix grows. All rows must still have the same real context
length. Draft width and weights stay fixed for the callable's lifetime; changing batch size, dtype, or
bucket can create another graph. External anchors need the same boundary
validation as `draft_logits`. Bucketing still recomputes full-context K/V and
adds padded work; it is not scheduler cache integration or a serving speedup claim.

## Reproduce the numerical comparison

Download these snapshots with the Hugging Face CLI; it prints each local path:

```bash
hf download mlx-community/Qwen3-4B-4bit \
    --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25
hf download z-lab/Qwen3-4B-DFlash-b16 \
    --revision b74e3a329c4d963783143b1e970d95b002be72bd
```

Save the official MIT-licensed
[model_mlx.py at 07ebd93](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model_mlx.py)
locally, then run from the repository's development environment:

```bash
python -m tools.dflash_parity \
    --target /path/to/target/snapshot \
    --draft /path/to/draft/snapshot \
    --reference /path/to/model_mlx.py \
    --output /path/to/new-results.json
```

The tool uses both capture implementations and compares eager, compiled, and
bucketed compiled draft logits and greedy proposal IDs at batch sizes 1 and 2,
context lengths 17, 33, 65, 255, 256, 257, 769, 1022, 1023, 1024, and 1025,
and block sizes 2, 5, 8, 9, and 16.
It reuses the compiled callables across lengths, including bucket
and attention dispatch boundaries. Use `--context-bucket-size N` to qualify
another bucket size; the report records that setting.
It records exact equality separately from the numerical tolerance
(`atol=rtol=1e-3`), rejects non-finite or incomplete comparisons,
and fails on any proposal mismatch. The report includes snapshot paths, native and
reference source hashes, and library versions. Use a new output file for each run.
If a checkpoint selects the final target layer, the tool normalizes the reference
hook's output to match the PyTorch/Hugging Face contract and records that adjustment
as `reference_final_norm_applied`. Independent tests compare captured features with
an actual Hugging Face Qwen3 forward, including the final layer and compiled replay.

This is forward parity, not generated-sequence parity or a performance benchmark.
The independent small-model tests also compare against explicit CPU attention
math and check that a causal block mask produces a different result.

## Remaining milestones

This is DFlash serving, not full DSpark support. A smaller qualified checkpoint,
DSpark serving integration (following [model/head qualification](dspark.md)),
confidence/cost-based planning, sampled verification, prefix
reuse, and asynchronous execution remain separate roadmap items. Keep each
change independently reviewable and qualify its lifecycle and serving behavior.
