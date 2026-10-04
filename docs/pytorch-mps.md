# PyTorch MPS decision contract

The opt-in MPS path should reuse upstream model and serving logic. Local code
should provide device integration and measured kernel improvements.

## Try the backend

Use the [normal installation](installation.md), then select MPS:

```bash
VLLM_METAL_BACKEND=mps vllm serve Qwen/Qwen3-0.6B --dtype bfloat16
```

Release wheels include both native launchers and the shared Metal shaders;
no compiler is needed. Source checkouts use the [development setup](CONTRIBUTING.md#development-setup).
Selecting MPS defaults to MRv2. Use unquantized FP16/BF16 checkpoints and greedy
requests (`temperature=0`, without logprobs).

## Models

- Use `model_impl="auto"` and vLLM's resolver. Keep model definitions and weight
  loading upstream; do not select a model implementation by its directory name.
- Follow the [hardware-agnostic model direction](https://pytorch.org/blog/hardware-agnostic-models-in-vllm/),
  but do not force `model_impl="transformers"`. In vLLM 0.30, that backend rejects
  layers outside the Transformers attention interface, including Qwen3.5's GDN
  layers. Use its upstream native definition and add MPS state/operator support.
  Revisit this choice as upstream hardware-agnostic coverage grows.
- Extend upstream scheduler, cache, speculative-decoding and omni contracts.
  Continuous batching and duplex are requirements for the corresponding serving
  features, not optional follow-ups to a single-request implementation.

## Kernels: choose the existing integration point

Choose the implementation and its upstream hook separately. Our RMSNorm provider
reuses a PyTorch fused operation through a vLLM IR registration.

| What needs changing? | Use |
| --- | --- |
| PyTorch already implements the operation on MPS | Call that operation, including its fused form. |
| A vLLM IR operator needs an MPS implementation | Register an IR implementation. |
| An existing `CustomOp` needs a different forward implementation | Use `register_oot`; retain the upstream layer and weight loading. |
| Attention metadata, recurrent state, expert execution or quantized weights need device support | Use the corresponding upstream backend interface or `PluggableLayer` hook. An operator registration alone does not supply these contracts. |
| An operation has no suitable implementation | Reuse a shared Metal kernel first; otherwise write a small TileLang kernel behind the same hook. |

- Profile before adding kernels. For fusion across operations, evaluate PyTorch
  compilation; registering an operator does not fuse its callers automatically.
- If upstream has no suitable hook, contribute one. Keep necessary temporary
  adapters narrow and version-specific; avoid copying algorithms or model blocks.
- Reject configurations for concrete missing behavior or incompatible kernel
  contracts. Keep unverified models blank in the support matrix, not in a runtime
  model allowlist. Preserve upstream numerical, cache-lifetime and mutation semantics.

## Performance

Shared and custom Metal kernels should bring MPS performance close to or above
MLX. Prove improvements with the [matched serving comparison](tools.md#mlx-versus-pytorch-mps-performance),
and retain the corresponding operator or model correctness checks.
