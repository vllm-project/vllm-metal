# Experimental PyTorch/MPS backend

Set `VLLM_METAL_MODEL_BACKEND=torch` before starting vLLM to execute upstream
vLLM model implementations on Apple Silicon through PyTorch MPS. The default
remains `mlx`.

This is a compatibility prototype for unquantized dense decoder models. It
does not establish support for every model in vLLM's registry. Models still need
PyTorch-native implementations of all their operations, and this path is not
optimized for throughput.

## Usage

Use the normal vllm-metal development installation, including the macOS vLLM
CPU wheel and its native extension. An `empty` vLLM build is insufficient:
the upstream CPU runner uses its native slot-mapping operation.

Start with a small original Hugging Face checkpoint:

```bash
VLLM_METAL_MODEL_BACKEND=torch vllm serve HuggingFaceTB/SmolLM2-135M \
  --host 127.0.0.1 --dtype float32 --max-model-len 256 \
  --max-num-seqs 2 --max-num-batched-tokens 64 \
  --kv-cache-memory-bytes 16777216
```

The same environment variable selects this backend for `LLM(...)`. Eager
execution and synchronous scheduling are configured by the platform. The model
loader honors upstream checkpoint loading and architecture resolution; no MLX
model conversion is involved.

The default KV budget is capped at 512 MiB and reduced when system or MPS
headroom is lower. `--kv-cache-memory-bytes` overrides that cap within the
available budget. These limits cover KV storage, not total process memory:
weights, activations, and attention scratch space also consume unified memory.
Use small context and batch limits on memory-constrained Macs.

## Reused upstream components

The plugin selects a separate out-of-tree platform before applying the MLX
compatibility patches. Its worker reuses `CPUModelRunner` for request updates,
batching, slot mapping, cache planning/binding, prompt logprobs, and sampling.
The model is instantiated by vLLM's `get_model` and `ModelRegistry`, so model
classes and weight-loading rules stay upstream.

Scheduling tensors and the sampler stay on CPU. A small model wrapper moves
inputs to MPS and returns logits to CPU. Parameters, intermediate activations,
and KV tensors remain on MPS. Standard layers use vLLM's native PyTorch
implementations through the existing out-of-tree custom-op dispatch.

The attention backend consumes upstream `CommonAttentionMetadata`, updates the
paged KV cache, and calls PyTorch scaled dot-product attention per request.
It handles causal prefill, chunked prefill, decode, GQA, prefix reuse, and sliding
windows. It accepts the split-cache layout in vLLM 0.28 and the packed per-layer
layout in current main. The upstream allocator still owns allocation and binding.

This avoids maintaining another scheduler, sampler, model registry, or set of
model implementations. It also establishes a small boundary where optimized
Metal attention could be added later without changing model classes.

## Limits

The initial scope excludes quantization, MoE, SSM/hybrid models, multimodal and
encoder-decoder models, pooling, LoRA, speculative decoding, distributed and
context-parallel execution, KV/weight offloading or transfer, and profiling.
These configurations fail during setup. Attention rejects ALiBi, logit soft
capping, attention sinks, and cross-layer KV sharing. CUDA-specific operations
in other architectures can still fail when their model is loaded or executed.

Attention gathers each request's visible cache into contiguous tensors; GQA
also expands KV heads. This is intentionally a simple correctness path and can
use substantial scratch memory for long contexts. There is no throughput or
full model-matrix parity claim with the MLX backend.

## Validation

```bash
python -m pytest tests/test_pytorch_attention.py \
  tests/test_pytorch_model_runner.py tests/test_pytorch_models.py \
  tests/test_register.py -q
```

The model tests create tiny Llama, Qwen2, and Qwen3 checkpoints locally and
compare greedy tokens against Hugging Face CPU and MLX references. They also
exercise chunked prefill, prefix reuse, logit bias, and prompt/output logprobs.
They run in separate interpreters because vLLM selects its platform once per
process. No checkpoint download is needed. Attention tests cover real MPS
execution in float32, float16, and bfloat16, both cache layouts, and physical
block addressing. Worker tests check the KV budget without allocating it.
