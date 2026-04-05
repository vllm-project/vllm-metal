## Why

`vllm-metal` hangs on startup when serving Gemma 4 models because vLLM classifies them as `TransformersMultiModalForCausalLM` (a Transformers fallback architecture), which triggers a conflict between the Transformers initialization path and vllm-metal's MLX runtime, causing a deadlock during distributed setup. `mlx-lm` added native Gemma 4 support on 2026-04-04 (commit `4469ad4`), providing a clean reference implementation that vllm-metal's model runner can route to immediately.

## What Changes

- **Fix model-routing logic** in `vllm_metal/v1/model_runner.py`: when vLLM reports `is_multimodal_model=True` but the underlying `model_type` is `gemma4`, route through `mlx_lm_load()` (text/reasoning path) instead of `mlx_vlm_load()` (vision path). `mlx_lm`'s `gemma4.py` wraps `gemma4_text.py` and handles both modalities.
- **Upgrade mlx-lm dependency** to `>=0.31.2` (git `4469ad4`) which introduces `mlx_lm/models/gemma4.py` and `mlx_lm/models/gemma4_text.py`.
- **Add KV cache dimension extraction** for Gemma 4's mixed sliding-window / full-attention architecture (alternating `sliding_attention` and `full_attention` layers per `sliding_window_pattern`).
- **Add model detection helper** `_should_use_mlx_lm_for_vlm(model_type)` — a registry of multimodal model types that `mlx_lm` handles natively, allowing future models (Gemma 5, etc.) to be onboarded without re-architecting the routing logic.

## Capabilities

### New Capabilities

- `gemma4-routing`: Detect Gemma 4 as an mlx-lm-routable model even when vLLM flags it as multimodal; load and run it via `mlx_lm_load()` with proper KV cache configuration for its sliding-window / full-attention hybrid architecture.

### Modified Capabilities

_(none — this change adds routing logic without altering existing VLM or LLM paths)_

## Impact

- **`vllm_metal/v1/model_runner.py`**: `_is_vlm_model()` and `load_model()` routing logic.
- **`vllm_metal/v1/model_runner.py`**: `_extract_model_args()` / `_resolve_model_dims()` — must handle Gemma 4's `text_config` nesting and mixed layer types.
- **`pyproject.toml`** (vllm-metal): bump `mlx-lm` lower bound to `0.31.2`.
- **No API changes** — the OpenAI-compatible `/v1/chat/completions` endpoint is unaffected.
- **Tested configuration**: `google/gemma-4-E4B-it` (5B, safetensors) on M4 Max 128 GB; the same fix generalises to `gemma-4-31B-it` and `gemma-4-2B-it`.
