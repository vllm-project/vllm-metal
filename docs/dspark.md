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

An optional policy uses the draft's trained confidence head and a local timing
profile to admit promising prefixes for target verification. DSpark generates
the full configured draft block, then chooses how many tokens from each row to
submit. The target verifier still decides which tokens to accept. A batch-size
draft schedule remains an upper bound; zero pauses drafting at that batch size.

The policy estimates expected emitted tokens per measured draft-plus-target
time. It searches every feasible total budget, including zero, because Metal
costs need not increase smoothly with width. Costs are interpolated by mean
context length and clamped at the profile's endpoints. The cost model is an
approximation: heterogeneous contexts, unequal row lengths and HTTP overhead
can change the benefit. Validate end-to-end serving on separate workloads.

### Collect a profile

Use the same Mac, checkpoints, activation precision, draft settings, verification
layout and context limit as the intended server. Include every batch size up
to its `max_num_seqs`. Profiles are local calibration artifacts; startup rejects
a mismatched identity or incomplete cost table. Regenerate a profile after
changing model files, hardware, runtime versions or execution settings.

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

Set the memory allocation for your machine. Use checkpoints with a trained
confidence head. If you use draft Q4 conversion or a candidate limit, include
the same settings here and when serving.

Prepare `calibration-inputs.json` as a JSON list. Each row contains `input_ids`
(the target tokenizer's integer token IDs, including the intended chat template)
and `split` (`"fit"` or `"holdout"`). Use nonempty, disjoint sets of representative
natural prompts. Keep this corpus separate from the later serving evaluation.
Each prompt plus `--output-cap` must fit the context limit.

```bash
python tools/benchmark/dspark_adaptive_profile.py \
    --llm-arguments llm-arguments.json \
    --inputs calibration-inputs.json \
    --contexts 128 1024 3072 \
    --output-cap 128 --warmups 2 --repeats 5 \
    --output dspark-profile
```

Run profiling alone on AC power with enough available memory. The collector
uses an in-process engine to observe conditional confidence and requires exact
control/observer outputs. Both use full-width draft generation, including near
the output cap. It measures the target and full-width draft at each context,
batch size and total admission budget, retaining all samples and warmups.
Context points must leave room for the draft block and correction token.

Temperatures are fitted sequentially to prefix-survival Brier error on the
fitting split, using a fixed grid. Held-out scores are reported without affecting
the fit. Review those scores even when calibration makes them worse. The output
directory includes raw observations, controls, timing samples, settings and
`profile.json`; an existing directory is never replaced.

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

Merge these entries with the model, method and width settings above. Restart the
server after changing the profile; it is loaded once at startup. Greedy-only
restrictions still apply. This native Metal policy does not need CUDA graphs.

Compare fixed-width DSpark and adaptive DSpark with identical tokenized inputs,
server settings and repeated warmups. Also keep target-only controls. Report
latency, throughput, emitted token counts and per-position acceptance together:
a higher accepted/drafted ratio by itself does not establish a speedup.


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
