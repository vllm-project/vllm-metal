# Speculative Decoding

vllm-metal supports four speculative decoding methods on the paged attention
path. All require synchronous scheduling and greedy sampling.

| | MTP | DFlash | Draft Model | N-gram |
|---|---|---|---|---|
| `--speculative-config` method | `mtp` | `dflash` | `draft_model` | `ngram` |
| Target models | Gemma4 (paged) | mlx_lm llama/qwen-family text models | Any paged-attention model | Any paged-attention model |
| Draft source | MTP assistant checkpoint (reads target KV cache) | Speculators-format parallel drafter (consumes target hidden states) | Separate smaller model (own KV cache) | Prompt/output token history (no model) |
| `num_speculative_tokens` | 1 | The trained block size (7 typical) | Configurable (3–5 typical) | Configurable (3–5 typical) |
| Extra memory | None (reads target KV cache) | Draft weights + small proposer-owned context KV | Second un-budgeted KV cache | None |

All three methods:

- Require `--no-async-scheduling` (vLLM auto-disables async for most spec-decode
  methods; pass it explicitly to be safe).
- Only accelerate greedy requests (`temperature=0`). Non-greedy requests skip
  drafting silently.
- Are lossless under greedy decoding.

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

## DFlash

DFlash ([arXiv 2602.06036](https://arxiv.org/abs/2602.06036)) drafts a whole
block of tokens in **one** forward instead of running a draft model
autoregressively — a good fit for Metal, where per-step overhead is the main
tax on draft-model SD. The drafter is a small stack of target-family decoder
layers; it attends over context K/V projected from the *target's*
intermediate-layer hidden states (`aux_hidden_state_layer_ids`) and fills K
mask tokens non-causally in a single pass. Its context K/V lives in
proposer-owned slabs (~20 KB/token), so there is no second KV cache to budget.

### Serve

```bash
VLLM_METAL_MEMORY_FRACTION=0.62 \
vllm serve Qwen/Qwen3-8B \
  --max-model-len 4096 \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"dflash","model":"RedHatAI/Qwen3-8B-speculator.dflash","num_speculative_tokens":7}'
```

Use a speculators-format DFlash checkpoint, and set `num_speculative_tokens`
to its trained block size (7 for the checkpoint above).

### Limitations

- **Prefix caching must be off** (`--no-enable-prefix-caching`, enforced at
  startup). Cached tokens skip the target forward, so the hidden states DFlash
  projects its context from would never be produced.
- **Fixed block size.** A block-diffusion drafter conditions on its whole mask
  block, so `num_speculative_tokens` must equal the checkpoint's trained K —
  a smaller value is off-distribution and collapses acceptance, unlike the
  free K knob of draft-model SD.

### Performance

On an M5 Pro (64 GB) with Qwen3-8B + RedHatAI/Qwen3-8B-speculator.dflash,
single-stream on natural prompts: **1.19x at the trained K=7** (mean
acceptance length 2.62). Dropping to K=5 or K=3 falls to ~0.78x as acceptance
collapses to ~1.6. Acceptance is strongly domain-dependent — math and code
accept far more than summarization or translation.

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
