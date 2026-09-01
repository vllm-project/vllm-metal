# Speculative Decoding

vllm-metal supports four speculative decoding methods on the paged-attention
path. Use vLLM's [speculative decoding guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
for method behavior and configuration details.

| | MTP | Draft model | N-gram | Grammar-forced |
|---|---|---|---|---|
| `--speculative-config` method | `mtp` | `draft_model` | `ngram` | `custom_class` |
| Target models | Gemma4 | Non-hybrid paged-attention models | Non-hybrid paged-attention models | Non-hybrid paged-attention models |
| Draft source | Matching Gemma4 assistant checkpoint | Separate smaller model | Prompt and output token history | The request's own grammar |
| `num_speculative_tokens` | Configurable (2–3 typical) | Configurable (3–5 typical) | Configurable (3–5 typical) | Configurable (8 typical) |
| Additional model weights | Assistant checkpoint | Draft model | None | None |
| Additional KV cache | None; reads target KV | Second scheduler-managed cache | None | None; one xgrammar matcher per request |

Grammar-forced drafting has a second variant, `ToolSpecProposer`, that adds
retrieval over past invocations to cover the values the grammar cannot force.
See [Retrieval-augmented drafting](#retrieval-augmented-drafting-toolspec).

All four methods currently have these Metal-specific constraints:

- Only greedy requests (`temperature=0`) are drafted. Other requests run
  without speculation.
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

## N-gram

Follow the upstream [N-gram guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/)
for configuration details. N-gram speculation needs no additional model or KV
cache. Its benefit depends on repeated token spans in the request history.

### Example

```bash
VLLM_METAL_USE_PAGED_ATTENTION=1 \
  vllm serve Qwen/Qwen3-8B \
    --max-model-len 2048 \
    --no-async-scheduling \
    --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_min":2,"prompt_lookup_max":3}'
```

## Grammar-forced

Grammar-forced drafting exploits the fact that a request under a JSON schema or a
tool-call structural tag has most of its output decided before the model runs.
The braces, quotes, key names and separators are the *grammar's* choice, not the
model's; only the values are open. This proposer emits the decided part and stops
at every genuine decision point. Like N-gram it loads no model and keeps no KV
cache; unlike N-gram it needs no repetition in the text.

Because vLLM already applies the grammar bitmask to **every** verification row
(not just the bonus row) before `verify_greedy` runs, a drafted token is
grammar-legal by construction. It is not *guaranteed* accepted, though — see
Limitations.

### Why not the in-tree n-gram proposer?

Measured, not argued. On the same structured-output workload, with the same K:

| Model | Batch | baseline TPOT | n-gram | grammar-forced |
|---|---|---|---|---|
| Qwen3-0.6B | 1 | 2.82 ms | 2.90 ms (0.97x) | 1.98 ms (**1.42x**) |
| Qwen3-0.6B | 4 | 6.66 ms | 6.77 ms (0.98x) | 3.68 ms (**1.81x**) |
| Qwen2.5-0.5B-Instruct | 1 | 1.98 ms | 2.07 ms (0.96x) | 1.16 ms (**1.71x**) |
| Qwen2.5-0.5B-Instruct | 4 | 4.65 ms | 4.66 ms (1.00x) | 2.34 ms (**1.98x**) |

n-gram fails here for a structural reason, not just sparse repetition. On one
batch-1 run of 279 output tokens, `NgramProposer.propose()` returned 3 drafts
totalling 24 tokens — but only 2 of those tokens were ever verified, and none
accepted. `Scheduler.update_draft_token_ids` filters drafts through the active
grammar's `validate_tokens()`, and a grammar-blind drafter's history-derived
guesses are mostly illegal at that position, so the engine discards them.
Grammar-forced drafting proposes only tokens the grammar already accepts. Every
arm above reproduced the baseline's token ids exactly.

### Latency

Measured on an M4 Pro. Two different workloads, because they exercise two
different grammars:

**Real tool calling** — OpenAI-API `tools` block, `tool_choice: "auto"`, hermes
parser, so the grammar is a *structural tag* (free text until a `<tool_call>`
trigger, then constrained JSON):

| Model | Speedup | Tokens from drafts | Acceptance | Tool calls parsed |
|---|---|---|---|---|
| Qwen3-0.6B | **1.26x** | 43% | **100%** | 8/8, identical to baseline |

**Constrained JSON generation** — `structured_outputs.json`, a plain JSON schema:

| Model | Batch | Speedup | Tokens from drafts | Acceptance |
|---|---|---|---|---|
| Qwen3-0.6B | 1 | 1.50x | 47% | 100% |
| Qwen3-0.6B | 4 | 1.82x | 49% | 83% |
| Gemma-4 E2B | 1 | 1.58x | 56% | 82% |
| Gemma-4 E2B | 4 | 1.77x | 52% | 71% |
| Gemma-3-4B | 1 | 1.22x | 36% | 67% |

**The gain shrinks as the model grows.** Gemma-3-4B still wins, but at 1.22x
rather than the ~1.5x the sub-2B models show: a wider verification step costs
proportionally more when the forward pass is bigger, and acceptance is lower.
Plan on the smaller end of that range for a model in the multi-billion range.
4B at bf16 is the largest that fits a 24 GB M4 Pro alongside its KV cache
(8.60 GB of weights against 15.26 GB usable), so 8B is untested here.

Tool calling shows the smaller speedup of the two because a structural tag leaves
the model's prose free until the trigger fires, and nothing is drafted there. The
structural tag pays it back elsewhere: its `begin` literal is
`<tool_call>\n{"name": "get_weather", "arguments": ` — 44 characters including
the tool name, forced in one run.

Reproduce with the in-tree report tool, which runs one arm per process and
renders the comparison from the resulting JSONs:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m tools.benchmark.grammar_spec_decode_report \
  run --model Qwen/Qwen3-0.6B --batch-size 1 --output-json run-grammar-bs1.json --grammar
python -m tools.benchmark.grammar_spec_decode_report report run-*.json \
  --output-md grammar-spec-decode-report.md
```

`--arm` selects baseline, the in-tree n-gram proposer, or this one on identical
prompts, checking token-id equality rather than assuming it.

### Example

```bash
VLLM_METAL_USE_PAGED_ATTENTION=1 \
vllm serve Qwen/Qwen3-8B \
  --max-model-len 2048 \
  --no-async-scheduling \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' \
  --speculative-config '{"method":"custom_class","model":"vllm_metal.v1.grammar_proposer.GrammarProposer","num_speculative_tokens":8}'
```

`method: custom_class` is upstream's escape hatch for proposers vLLM does not
ship; `model` carries the dotted path of the class rather than a checkpoint.

Confirm it is active: the server log shows
`Grammar-forced speculative decoding enabled (...)` at startup.

### Tuning

- **`disable_any_whitespace: true` is close to mandatory.** It is the single
  highest-impact setting here. With free whitespace allowed between JSON tokens
  the grammar forces almost nothing, and measured step coverage collapses from
  43% to 10% with every draft shrinking to one token.
- `num_speculative_tokens` (K) — 8 rather than the 3–5 the other methods use.
  Forced runs are long (`", "arguments": {"location": "` is about a dozen
  tokens), and steps where nothing is forced cost nothing because the proposer
  returns no draft at all.
- The backend must be pinned to `xgrammar`. Requests routed to `outlines`,
  `guidance` or `lm-format-enforcer` simply draft nothing.
- For **tool calling** three further settings are load-bearing, and the proposer
  silently drafts nothing without them:
  1. a tool parser that declares a `structural_tag_model` (`hermes`,
     `qwen_3_coder`, `llama`, ... — but *not* Gemma-4, see below);
  2. `VLLM_ENFORCE_STRICT_TOOL_CALLING=1`, or `ToolParser.get_structural_tag`
     returns `None`;
  3. `"strict": true` on at least one tool, or `get_model_structural_tag`
     returns `None` for `tool_choice: "auto"`.
- Do not pass `--reasoning-parser` (see Limitations). For Qwen3 that means
  sending `chat_template_kwargs: {"enable_thinking": false}` instead.

### Limitations

- Paged path only (`VLLM_METAL_USE_PAGED_ATTENTION=1`), greedy only, and
  synchronous scheduling only — same as the other methods.
- **Acceptance is empirical, not guaranteed.** Several *tokenizations* of a
  forced string are legal (after `{` the grammar demands `"name"`, and `"`,
  `"n`, `"na` and `"name` are all legal tokens), so the proposer walks the
  forced string taking the longest matching token at each step. That
  *approximates* the canonical tokenization the model emits but does not
  reproduce it — BPE merges by rank, not by length, and the two can disagree
  (`celsius` walks to `cel|si|us` where the tokenizer produces `c|elsius`).
  Verification is lossless either way, so a miss costs a wider step and nothing
  else.
- **Nothing is drafted for plain chat, for free-form string values, or for the
  model's own choice of which tool to call.** Coverage is concentrated in the
  structure, not the content.
- **Gemma-4's own tool calling is not accelerated by this.** Its parser
  (`gemma4_engine_tool_parser.py`) deliberately skips structured outputs because
  Gemma-4 emits a native `<|tool_call>call:...` syntax, and it declares no
  `structural_tag_model`. No bitmask is produced, so nothing is ever forced.
  Explicit JSON-schema requests to Gemma-4 *are* accelerated.
- Structural tags for `tool_choice: "auto"` only activate when
  `VLLM_ENFORCE_STRICT_TOOL_CALLING=1` is set.
- A configured reasoning parser is refused at construction. While reasoning is
  unfinished the engine stops advancing its grammar matcher, which the worker
  cannot observe, so the two would silently desynchronize.
- **Hybrid GDN models are not supported**, the same as every other Metal
  speculative method: `SpeculativeDecodeController.validate_supported` raises as
  soon as any draft is scheduled. Qwen3.5-0.8B is one such model, so it cannot use
  this (or n-gram, or draft-model) speculative decoding at all.

### Retrieval-augmented drafting (ToolSpec)

The grammar drafts structure and stops at every value, so its ceiling is however
much of the output the schema fixes, and that is corpus-dependent: **37% of
emitted tokens over API-Bank, 50% over BFCL** (Qwen3-0.6B, greedy, measured by
`tools/grammar_determinism.py`, which imports nothing from `vllm_metal` and so
runs against a checkout with no proposer in it). Plan on the lower end — API-Bank
is multi-turn and its arguments carry more free text, which is the more realistic
shape. The rest is argument *content*. `ToolSpecProposer` is the Metal port of
ToolSpec ([arXiv 2604.13519](https://arxiv.org/abs/2604.13519)), which attacks
that half: tool-calling traffic repeats itself, so a value some earlier
invocation produced is a reasonable guess for the one being generated now.

It composes the grammar proposer rather than replacing it. Each finished request
contributes its output to a bounded per-worker memory, keyed by the target's
hidden state at the last prompt token; a later request retrieves the most
similar traces by cosine similarity and n-gram suffix-matches its own output
against them. The grammar drafts first — its draft is legal by construction —
and retrieval fills only the steps the grammar left empty.

Swap the class; everything else is identical to the grammar arm:

```bash
VLLM_METAL_USE_PAGED_ATTENTION=1 \
  vllm serve Qwen/Qwen3-8B \
    --max-model-len 2048 \
    --no-async-scheduling \
    --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' \
    --speculative-config '{"method":"custom_class","model":"vllm_metal.v1.toolspec_proposer.ToolSpecProposer","num_speculative_tokens":8}'
```

Tuned with `VLLM_METAL_TOOLSPEC_CAPACITY` (memory size, default 512),
`_TOP_K` (traces searched per step, 4) and `_NGRAM_MIN`/`_NGRAM_MAX` (suffix
window, 5–7). See [configuration](configuration.md).

Measured on an M4 Pro, Qwen3-0.6B, 32 sequential requests, greedy, K=8. Every
arm reproduces plain greedy token-for-token
(`tools/test_grammar_spec_decode_e2e.py`):

| Workload | baseline TPOT | grammar | +retrieval |
|---|---|---|---|
| API-Bank tool calls | 6.6 ms | 5.3 ms (**1.26x**) | 5.1 ms (**1.31x**) |
| Sonnet under a JSON schema | 7.4 ms | 7.1 ms (1.05x) | 6.7 ms (**1.11x**) |
| Sonnet, no grammar | 7.1 ms | 7.0 ms (1.01x) | 7.1 ms (1.00x) |

Retrieval helps most where the *content* is free but the request is still
structured — the schema'd sonnet row nearly doubles the grammar's gain, because
that workload is mostly free text inside a small envelope. It is measured
sequentially on purpose: the memory holds requests that have already finished,
so within a single batch it is empty by construction, and replaying one batch
would have it retrieve each prompt's own previous output.

The last row is the one to check before enabling this. Unstructured traffic must
not pay for a feature it cannot use, and it does not: zero drafts, and TPOT
within noise of baseline.

#### Limitations

- **Linear drafts, not trees.** ToolSpec drafts a *tree* of candidates and
  verifies it with tree attention. Metal's verify half is linear and shared with
  the MTP and draft-model proposers, so this port takes the single best
  candidate. Part of the paper's headline speedup comes from tree breadth, so
  this does not reproduce it.
- **Retrieval only pays across requests that resemble each other.** Single-turn,
  mutually unrelated traffic stores traces nothing later matches. The searching
  is throttled after a streak of misses (the same back-off
  `NgramProposer` uses), which is what keeps the no-grammar row at 1.00x —
  without it, searching and finding nothing cost about 1%.
- **The memory is shared across requests**: one request's output can seed
  another's draft. Verification is unchanged, so this changes what is *guessed*,
  never what is *emitted*. It is a drafting cache, held per worker and never
  persisted.
- Retrieval acceptance is much lower than the grammar's (27% against 67% on
  API-Bank) and that is expected — a grammar draft is legal by construction, a
  retrieved one is a guess. It still wins because the drafts it does land are
  long (7.0 tokens against 4.4).

## Benchmarking

Use vLLM's benchmark CLI for serving workloads. For a reproducible Gemma4
target-only versus MTP comparison, use the in-tree benchmark:

```bash
python -m tools.benchmark.gemma4_mtp_benchmark --help
```

`tools/README.md` documents the before-and-after commands and the natural-prompt
dataset used for speculative-decoding measurements.
