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

[Sleep mode](configuration.md#sleep-mode) releases both target and draft KV
memory between requests while retaining the model weights. Both prefix caches
start empty after wake; kept requests recompute their prefixes. With
low-precision target logits, this recomputation can change greedy choices near
ties compared with uninterrupted decoding.

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
