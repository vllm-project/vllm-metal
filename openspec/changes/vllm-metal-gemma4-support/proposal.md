## Why

`vllm-metal` hangs on startup when serving Gemma 4 models because vLLM classifies them as `TransformersMultiModalForCausalLM` (a Transformers fallback architecture), which triggers a conflict between the Transformers initialization path and vllm-metal's MLX runtime, causing a deadlock during distributed setup.

Two upstream libraries now provide Gemma 4 implementations:
- **`mlx-lm`** added native Gemma 4 support on 2026-04-04 (commit `4469ad4`), covering text and reasoning inference via `gemma4.py` + `gemma4_text.py`.
- **`mlx-vlm`** has also added Gemma 4 support, enabling the full vision-language path (image + text input).

This changes the routing decision: both backends are now viable, and the preferred path depends on whether vision input is in scope.

## What Changes

- **Fix model-routing logic** in `vllm_metal/v1/model_runner.py`: when vLLM reports `is_multimodal_model=True` and `model_type` is `gemma4`, route through `mlx_vlm_load()` using the now-available mlx-vlm Gemma 4 implementation for full multimodal support. Retain the `_MLX_LM_MULTIMODAL_MODELS` allowlist mechanism as a fallback/override for text-only deployments.
- **Upgrade mlx-lm dependency** to `>=0.31.2` (git `4469ad4`) which introduces `mlx_lm/models/gemma4.py` and `mlx_lm/models/gemma4_text.py`.
- **Install mlx-vlm from local checkout** until a PyPI release with Gemma 4 support is tagged; document alongside the mlx-lm git install in `INSTALL.md` (see D5).
- **Add KV cache dimension extraction** for Gemma 4's mixed sliding-window / full-attention architecture (alternating `sliding_attention` and `full_attention` layers per `sliding_window_pattern`).
- **Add model detection helper** `_should_use_mlx_lm_for_vlm(model_type)` — a registry of multimodal model types that `mlx_lm` handles natively, allowing future text-only overrides without re-architecting the routing logic.

## Capabilities

### New Capabilities

- `gemma4-text`: Serve Gemma 4 text/reasoning requests via `mlx_vlm_load()` with correct KV cache configuration for its sliding-window / full-attention hybrid architecture.
- `gemma4-vision`: Serve Gemma 4 image+text requests via `mlx_vlm_load()` — mlx-vlm's vision tower is fully implemented.
- `gemma4-audio`: Serve Gemma 4 audio+text requests via `mlx_vlm_load()` — mlx-vlm's Conformer audio encoder is fully implemented in `mlx_vlm/models/gemma4/audio.py`; this is an exposure-only change.

### Modified Capabilities

_(none — this change adds routing logic without altering existing VLM or LLM paths)_

## Impact

- **`vllm_metal/v1/model_runner.py`**: `_is_vlm_model()` and `load_model()` routing logic.
- **`vllm_metal/v1/model_runner.py`**: `_extract_model_args()` / `_resolve_model_dims()` — must handle Gemma 4's `text_config` nesting and mixed layer types.
- **`pyproject.toml`** (vllm-metal): bump `mlx-lm` lower bound to `0.31.2`; mlx-vlm installed from local checkout until a Gemma 4-inclusive PyPI release is available.
- **No API changes** — the OpenAI-compatible `/v1/chat/completions` and `/v1/chat/completions` (vision) endpoints are unaffected.
- **Tested configuration**: `google/gemma-4-E4B-it` (5B, safetensors) on M4 Max 128 GB; the same fix generalises to `gemma-4-31B-it` and `gemma-4-2B-it`.
