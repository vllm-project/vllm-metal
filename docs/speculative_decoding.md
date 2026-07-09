# Speculative Decoding

vllm-metal supports the following speculative decoding methods on the paged
attention path. All require synchronous scheduling and greedy sampling.

| | Gemma4 MTP | Native MTP (GLM) | Draft Model | N-gram |
|---|---|---|---|---|
| `--speculative-config` method | `mtp` | `mtp` | `draft_model` | `ngram` |
| Target models | Gemma4 (paged) | GLM-4.7-Flash (paged) | Any paged-attention model | Any paged-attention model |
| Draft source | MTP assistant checkpoint (reads target KV cache) | Extracted nextn head (own 1-layer KV slab) | Separate smaller model (own KV cache) | Prompt/output token history (no model) |
| `num_speculative_tokens` | 1 | 1 | Configurable (3–5 typical) | Configurable (3–5 typical) |
| Extra memory | None (reads target KV cache) | ~1.2 KB/token/request (compressed MLA latent) | Second un-budgeted KV cache | None |
| Prefix caching | Tolerated | Must be off (`--no-enable-prefix-caching`) | Tolerated | Tolerated |

All methods:

- Require `--no-async-scheduling` (vLLM auto-disables async for most spec-decode
  methods; pass it explicitly to be safe).
- Only accelerate greedy requests (`temperature=0`). Non-greedy requests skip
  drafting silently.
- Are lossless under greedy decoding, up to the target's own kernel numerics —
  see the tie-flip note in the GLM section for MLA targets.

---

## Gemma4 MTP

Gemma4 MTP speculative decoding is experimental on Metal. It uses vLLM's
scheduler contract and verifies scheduled draft tokens on the target
paged-decode path.

Use matching target and assistant checkpoint families (assistant checkpoints
use `gemma4_assistant` or `gemma4_mtp` model config):

| Target | Assistant |
| --- | --- |
| Gemma4 E2B-it | Gemma4 E2B-it assistant bf16 |
| Gemma4 E4B-it | Gemma4 E4B-it assistant bf16 |
| Gemma4 31B-it bf16 | Gemma4 31B-it assistant bf16 |

The path is intentionally narrow. Unsupported speculative decode behavior fails
early instead of falling back silently.

### Serve

```bash
export TARGET=/path/to/gemma-4-E2B-it
export ASSISTANT=/path/to/gemma-4-E2B-it-assistant-bf16

VLLM_METAL_MEMORY_FRACTION=0.5 \
  vllm serve "$TARGET" \
    --max-model-len 1024 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 4 \
    --no-async-scheduling \
    --speculative-config "{\"method\":\"mtp\",\"model\":\"$ASSISTANT\",\"num_speculative_tokens\":1}"
```

For remote Hugging Face checkpoints, use the same shape and set `model` to the
assistant repository name. Pin `revision` in the JSON when publishing benchmark
numbers.

### Benchmark

Use the in-tree offline benchmark for before/after runs. It runs one mode per
process: omit `--assistant-model` for the baseline, then add it for MTP.

```bash
source .venv-vllm-metal/bin/activate
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_METAL_MEMORY_FRACTION=0.5
export TARGET=/path/to/gemma-4-E2B-it
export ASSISTANT=/path/to/gemma-4-E2B-it-assistant-bf16

python -m tools.benchmark.gemma4_mtp_benchmark \
  --model "$TARGET" \
  --batch-size 4 \
  --max-tokens 64 \
  --repeats 1 \
  --warmup 0 \
  --ignore-eos \
  --max-model-len 1024 \
  --max-num-batched-tokens 512 \
  --label e2b-baseline-bs4-64 \
  --output-json /tmp/gemma4-e2b-baseline-bs4-64.json

python -m tools.benchmark.gemma4_mtp_benchmark \
  --model "$TARGET" \
  --assistant-model "$ASSISTANT" \
  --batch-size 4 \
  --max-tokens 64 \
  --repeats 1 \
  --warmup 0 \
  --ignore-eos \
  --max-model-len 1024 \
  --max-num-batched-tokens 512 \
  --label e2b-mtp-bs4-64 \
  --output-json /tmp/gemma4-e2b-mtp-bs4-64.json
```

The JSON records the model paths, runtime package versions, relevant
environment variables, prompts, generated token IDs, elapsed time, and output
tokens per second. Attach those JSON files to performance PRs instead of copying
machine-specific results into the docs.

---

## GLM-4.7-Flash MTP (native head)

GLM-4.7-Flash ships one trained nextn (MTP) layer inside the target checkpoint
(`num_nextn_predict_layers: 1`). vllm-metal runs it as a native drafter: each
step, the head projects the target's post-final-norm hidden states plus the
committed tokens through one decoder layer and proposes exactly one draft token.
The head keeps its own per-request KV history (the layer's compressed MLA
latent, ~1.2 KB per token per request) — it never touches the target's paged
KV, needs no second block allocator, and never re-ingests the prompt.

### Extract the head

The `mlx-community` 4-bit target mirror strips the nextn layer, so extract it
once from the bf16 master (downloads only the ~4 GB of shards that hold the
layer, not the full repo):

```bash
python tools/extract_glm47_mtp_head.py --q-bits 4 --q-group-size 64 \
  --out ~/.cache/vllm-metal-dev/glm47-mtp-head/4bit
```

Omit `--q-bits` for a bf16 head. The 4-bit head is strongly recommended: the
head's cost is dominated by its untied lm_head, and a bf16 head spends a large
fraction of a target decode step on it.

### Serve

```bash
export HEAD=~/.cache/vllm-metal-dev/glm47-mtp-head/4bit

vllm serve mlx-community/GLM-4.7-Flash-4bit \
  --max-model-len 4096 \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --speculative-config "{\"method\":\"mtp\",\"model\":\"$HEAD\",\"num_speculative_tokens\":1}"
```

Confirm it is active: the server log shows
`Native MTP proposer ready: head=glm4_moe_lite_mtp (...)` at startup and
periodic `SpecDecoding metrics ... Avg Draft acceptance rate`.

### Benchmark

Use the offline benchmark (thin sibling of the Gemma4 one; it forces prefix
caching off and records acceptance counters in the JSON). One mode per process:
omit `--assistant-model` for the baseline, add it for MTP. `--chat` renders
prompts through the chat template, which is strongly recommended for
representative acceptance rates.

```bash
python -m tools.benchmark.native_mtp_benchmark \
  --model mlx-community/GLM-4.7-Flash-4bit \
  --batch-size 1 --max-tokens 128 --repeats 3 --warmup 1 \
  --ignore-eos --chat --max-model-len 1024 \
  --label baseline-bs1 --output-json /tmp/glm-mtp-baseline-bs1.json

python -m tools.benchmark.native_mtp_benchmark \
  --model mlx-community/GLM-4.7-Flash-4bit \
  --assistant-model "$HEAD" \
  --batch-size 1 --max-tokens 128 --repeats 3 --warmup 1 \
  --ignore-eos --chat --max-model-len 1024 \
  --label mtp-bs1 --output-json /tmp/glm-mtp-bs1.json
```

Attach the JSON files to performance PRs instead of copying machine-specific
results into the docs.

### Constraints

- **Greedy only, K=1.** The head is trained for one extra token
  (`num_speculative_tokens` above 1 is rejected at startup), and only
  `temperature=0` requests draft.
- **Prefix caching must be off** (`--no-enable-prefix-caching`). Cached prompt
  tokens skip the target forward that produces the hidden states the head
  consumes, so their draft-side state could never be built. The server rejects
  the config at startup otherwise. (Gemma4 MTP tolerates prefix caching because
  its assistant reads the target's paged KV instead of hidden states.)
- **Extracted head required.** Pointing `model` at the raw `zai-org` target
  repo fails at startup (keyed on `num_hidden_layers != 0`) with a message
  naming the extraction tool.
- **MLA tie-flip note.** On this target, greedy token-identity between SD and
  non-SD runs (and even between two non-SD runs with different batch sizes) is
  not bit-exact: the 4-bit MoE + MLA kernels are not batch-shape-invariant, so
  positions where the target's top-2 logits are close can flip either way.
  Parity is therefore judged tie-aware: at every divergence, the base model's
  fp32 top-2 logit gap must be within the same range that no-SD batching
  perturbations flip — divergences at gaps beyond that range indicate a real
  verify bug. The PR for this feature carries the calibration data.

---

## Draft Model

Draft-model speculative decoding uses a separate, smaller model to propose
`num_speculative_tokens` per step, verified greedily against the target. Any
paged-attention model can serve as the target; the draft must share the
target's tokenizer and vocabulary.

### Serve

```bash
VLLM_METAL_MEMORY_FRACTION=0.55 \
vllm serve Qwen/Qwen3-8B \
  --max-model-len 2048 \
  --no-async-scheduling \
  --speculative-config '{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":3}'
```

Confirm speculative decoding is active: the server log shows
`Draft model loaded for speculative decoding: <model>` and periodic
`SpecDecoding metrics ... Avg Draft acceptance rate`.

### Limitations

- **Draft KV cache is un-budgeted.** The draft gets its own KV cache sized to
  the target's `num_blocks`, allocated after the target KV budget and not
  subtracted from it. Keep `VLLM_METAL_MEMORY_FRACTION` well below 1.0.
- **SWA draft models untested.** The draft block allocator assumes full
  attention.
- **No pipeline parallelism.** `pipeline_parallel_size>1` with speculative
  config raises at startup.

### Performance

On an M5 Pro (64 GB) with Qwen3-8B target + Qwen3-0.6B draft, natural prompts
from RedHatAI/speculator_benchmarks:

- **Single-stream:** 1.36–1.48x TPOT improvement (K=3–5).
- **Batched (concurrency 32):** K=3 gives +11% throughput; K=5 turns
  net-negative as draft compute exceeds the savings.

---

## N-gram

N-gram (prompt-lookup) speculative decoding drafts with **no model at all**: each
step it matches the longest suffix n-gram of the request's token history against
an earlier occurrence and proposes the tokens that followed it. The match runs on
CPU (vLLM's Numba kernel), so there is no extra GPU memory and no second cache.

It shines on workloads with repeated spans — code completion, JSON/structured
output, summarization or RAG that echoes the prompt, and agentic loops. On free-form
prose with little repetition most drafts are rejected and it adds little.

### Serve

```bash
VLLM_METAL_USE_PAGED_ATTENTION=1 \
vllm serve Qwen/Qwen3-8B \
  --max-model-len 2048 \
  --no-async-scheduling \
  --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_min":2,"prompt_lookup_max":3}'
```

Confirm speculative decoding is active: the server log shows
`N-gram speculative decoding enabled (...)` at startup and periodic
`SpecDecoding metrics ... Avg Draft acceptance rate`.

### Tuning

- `num_speculative_tokens` — how many tokens to copy after a match (K).
- `prompt_lookup_min` / `prompt_lookup_max` — the n-gram length window to match.
  Lower `prompt_lookup_min` matches more aggressively (more drafts, more
  rejections); higher values match only longer, higher-confidence repeats.
  Defaults to 5/5 when unset.

### Limitations

- Paged path only (`VLLM_METAL_USE_PAGED_ATTENTION=1`), greedy only, and
  synchronous scheduling only — same as the other methods.
- Acceleration depends entirely on the input having repeated token spans; there
  is no learned drafter to generalize beyond literal repeats.
