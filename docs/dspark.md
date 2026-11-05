# DSpark experimental serving

[DSpark](https://github.com/deepseek-ai/DeepSpec) drafts blocks of tokens that the
target model verifies. vLLM Metal supports greedy serving with matched Qwen3
target/draft pairs on a single Apple Silicon device.

This feature is experimental. Fresh target-only and speculative outputs can
differ, and numerical qualification is incomplete. Speedup depends on the model,
workload and concurrency; benchmark your intended setup before deployment.
See [#1005](https://github.com/vllm-project/vllm-metal/issues/1005) for ongoing work.

## Start a server

Use a draft trained for your target, such as
[DeepSpec's Qwen3-4B draft](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7):

```bash
vllm serve mlx-community/Qwen3-4B-4bit \
    --max-model-len 2048 \
    --no-async-scheduling \
    --speculative-config '{
      "method": "dspark",
      "model": "deepseek-ai/dspark_qwen3_4b_block7",
      "num_speculative_tokens": 7
    }'
```

Self-contained Qwen3 checkpoints in the Speculators format are also supported:

```bash
vllm serve Qwen/Qwen3-4B \
    --max-model-len 4096 \
    --no-async-scheduling \
    --speculative-config '{
      "method": "dspark",
      "model": "RedHatAI/Qwen3-4B-speculator.dspark",
      "num_speculative_tokens": 7
    }'
```

See the [Speculators model card](https://huggingface.co/RedHatAI/Qwen3-4B-speculator.dspark)
for the trained pair and evaluation. Other architectures and checkpoints that
omit draft embeddings or output weights are not supported.

Send requests with `temperature=0`. Sampling, penalties, grammar constraints and
sample logprobs fall back to target-only generation. Asynchronous scheduling,
LoRA, TurboQuant and KV offloading are not supported with DSpark. Leave draft
cache precision unset; the target's activation precision must match the draft
checkpoint.

## Tune drafting

- **Draft length:** Set `num_speculative_tokens` between 1 and the checkpoint's
  trained width. Longer drafts can cost more than they save. An optional
  `num_speculative_tokens_per_batch_size` schedule can reduce the width or use
  zero to pause drafting at selected batch sizes.
- **Prefix caching:** Use `--enable-prefix-caching` to reuse prefixes, or
  `--no-enable-prefix-caching` for cold-cache measurements.
- **Markov candidates:** Add `"dspark_draft_topk": 64` to the speculative
  configuration to limit draft candidate scoring. Omit it to use the checkpoint
  setting, or the full vocabulary when no setting is present.
- **Draft weight memory:** Add
  `--additional-config '{"dspark_draft_quantization":"q4"}'` to quantize supported
  draft linear layers at startup. This requires a floating-point checkpoint;
  pre-quantized drafts are not supported.

Candidate limits and Q4 conversion can change acceptance. Measure end-to-end
latency and throughput alongside acceptance; lower memory use alone does not
establish a speedup.

## Adaptive verification

Adaptive verification uses the draft's trained confidence head and a local
timing profile to choose how many draft tokens to send to the target. DSpark
generates the full configured block, then selects a prefix for each request.
The target verifier decides which tokens to accept. A batch-size draft schedule
sets the maximum prefix length; zero pauses drafting at that batch size.

The policy maximizes expected output tokens per measured draft-plus-target
time. Costs are interpolated by mean context length and clamped at the profile's
endpoints. Mixed contexts, unequal request lengths and HTTP overhead can affect
the benefit, so evaluate it on your serving workload.

### Collect a profile

Use the server's Mac, checkpoints, activation precision, draft settings,
verification layout and context limit. The collector covers batch sizes 1
through `max_num_seqs`; startup checks that the profile matches. Regenerate it
after changing model files, hardware, runtime versions or execution settings.

Save the arguments for the fixed-width engine as `llm-arguments.json`:

```json
{
  "model": "mlx-community/Qwen3-4B-4bit",
  "max_model_len": 4096,
  "max_num_seqs": 4,
  "max_num_batched_tokens": 2048,
  "gpu_memory_utilization": 0.5,
  "enable_prefix_caching": false,
  "async_scheduling": false,
  "generation_config": "vllm",
  "speculative_config": {
    "method": "dspark",
    "model": "deepseek-ai/dspark_qwen3_4b_block7",
    "num_speculative_tokens": 7
  }
}
```

Choose a cache allocation that fits your Mac and a draft with a trained
confidence head. Match any draft Q4 conversion and candidate-limit settings
to the server.

Prepare `calibration-inputs.json` as a list of objects with `input_ids` and
`split` (`"fit"` or `"holdout"`). Tokenize representative prompts with the target
tokenizer and intended chat template. Both splits must be nonempty and disjoint
from each other and the serving evaluation. Each prompt plus `--output-cap`
must fit the context limit.

```bash
python tools/benchmark/dspark_adaptive_profile.py \
    --llm-arguments llm-arguments.json \
    --inputs calibration-inputs.json \
    --contexts 128 1024 3072 \
    --output-cap 128 --warmups 2 --repeats 5 \
    --output dspark-profile
```

Run on AC power with enough available memory and no competing GPU job. Each
context point must leave room for the draft block and correction token.
The collector measures full-width drafting and target verification at each
context, batch size and total draft-token budget. It also checks that
observing confidence leaves generated outputs unchanged.

Confidence temperatures are fitted on the `fit` split. Compare raw and
calibrated Brier scores on `holdout` to assess calibration quality. Choose a new
output directory; it will contain `profile.json`, settings, raw observations,
controls, and all timing samples and warmups.

### Serve with the profile

Use the same engine settings and add these entries to the existing
configurations:

```json
{
  "speculative_config": {
    "enable_adaptive_verification": true
  },
  "additional_config": {
    "dspark_adaptive_profile": "/absolute/path/to/dspark-profile/profile.json"
  }
}
```

Merge these entries with the model, method and width settings above. The profile
is loaded at startup, so restart after changing it. Adaptive verification
supports greedy decoding.

Compare adaptive DSpark, fixed-width DSpark and target-only serving with the
same tokenized inputs, settings and repeated warmups. Measure latency,
throughput, output-token counts and per-position acceptance together.

## Validate your setup

Use [the parity tools](tools.md) to compare fresh greedy target-only and
speculative runs. Report exact token matches separately from top-K matches.
Follow [the macOS benchmarking guide](benchmarking-macos.md) and use natural
prompts at the concurrency you expect to serve.

The [DSpark serving benchmark](https://github.com/vllm-project/vllm-metal/blob/main/tools/benchmark/dspark_serving_benchmark.py)
records serving throughput, latency,
acceptance and memory. Keep model revisions, input token IDs, thinking mode,
output limits and cache settings matched across comparisons, and retain
regressions as well as gains.
