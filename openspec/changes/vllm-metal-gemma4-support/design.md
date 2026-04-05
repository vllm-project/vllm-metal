## Context

vllm-metal's `MetalModelRunner` in `vllm_metal/v1/model_runner.py` routes model loading through two paths:

- **`mlx_lm_load()`** — for text-only models (`is_multimodal_model = False`)
- **`mlx_vlm_load()`** — for vision-language models (`is_multimodal_model = True`)

Gemma 4 (`google/gemma-4-E4B-it`) is a multimodal model, so vLLM sets `is_multimodal_model = True`. vllm-metal therefore calls `mlx_vlm_load()`, but mlx-vlm has no Gemma 4 implementation. Meanwhile, mlx-lm added Gemma 4 support on 2026-04-04 (`gemma4.py` + `gemma4_text.py`). Additionally, because vLLM core has no native `gemma4` architecture registration, it emits `TransformersMultiModalForCausalLM` as the resolved architecture — a Transformers fallback that conflicts with vllm-metal's MLX runtime during distributed initialisation, causing a deadlock.

**Key source files:**
- `vllm_metal/v1/model_runner.py` — `_is_vlm_model()`, `load_model()`, `_extract_model_args()`, `_resolve_model_dims()`
- `mlx_lm/models/gemma4.py` — multimodal wrapper; delegates to `gemma4_text.Model` and drops vision weights in `sanitize()`
- `mlx_lm/models/gemma4_text.py` — text backbone: mixed `sliding_attention` / `full_attention` layers, `sliding_window_pattern=5`, dual head-dim (`head_dim=256` local, `global_head_dim=512` global), KV-shared layers

## Goals / Non-Goals

**Goals:**
- Gemma 4 models load and serve text inference via `mlx_lm_load()` on vllm-metal without deadlock
- Correct KV cache dimensions extracted for Gemma 4's sliding-window / full-attention architecture
- An extensible per-model-type routing registry so future mlx-lm-backed multimodal models (Gemma 5, etc.) are a one-line addition
- mlx-lm dependency bumped to `>=0.31.2` in `pyproject.toml`

**Non-Goals:**
- Vision / image input support for Gemma 4 (image tokens are stripped by `gemma4.Model.sanitize()`; text-only mode is sufficient for the pearlster use-case and represents a well-scoped first step)
- Audio input support
- Modifications to vLLM core architecture registration (that is upstream work)

## Decisions

### D1: Route by `model_type` string, not by `is_multimodal_model` flag alone

**Choice:** Introduce `_MLX_LM_MULTIMODAL_MODELS: frozenset[str]` — an explicit allowlist of `model_type` values that mlx-lm handles natively despite being multimodal. `load_model()` checks this set before deciding between `mlx_lm_load` and `mlx_vlm_load`.

```python
_MLX_LM_MULTIMODAL_MODELS: frozenset[str] = frozenset({"gemma4"})

def _use_mlx_lm_for_multimodal(self, model_type: str) -> bool:
    return model_type in _MLX_LM_MULTIMODAL_MODELS
```

**Rationale:** An explicit allowlist is conservative and reviewable. The alternative — trying `mlx_lm_load()` first and falling back on `ImportError` — would silently change behaviour for existing VLMs that should use mlx-vlm. A feature-flag per model type makes the routing decision transparent and easy to extend.

**Alternative considered:** Check `mlx_lm.utils._get_classes` at runtime (try importing `mlx_lm.models.<model_type>`). Rejected because it duplicates mlx-lm's internal lookup and would fire for models where mlx-lm has a partial/buggy implementation.

### D2: Extract model dims from nested `text_config` for mlx-lm Gemma 4 path

**Choice:** `_extract_model_args()` already merges `text_config` dicts into top-level `model_args` via `setdefault`. This path already works for Gemma 4 because `mlx_lm.models.gemma4.Model` carries `.args.text_config` as a plain dict. No change needed here — only a test to confirm it.

**Rationale:** The existing merge logic (`tc = self.model_args.get("text_config"); for k, v in tc_dict.items(): self.model_args.setdefault(k, v)`) already handles this. Gemma 4's `text_config` contains `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim` — all fields `_resolve_model_dims()` already reads.

### D3: No special handling for Gemma 4's mixed layer types in `_resolve_model_dims`

**Choice:** Treat Gemma 4 as a standard sliding-window model for KV cache sizing. Use `head_dim` (256, local attention) as the canonical head dimension across all layers; over-allocate slightly for the 20% of global-attention layers that use `global_head_dim=512`.

**Rationale:** The KV cache sizing path already handles per-layer heterogeneity safely (it allocates based on max head dim). Gemma 4's `num_kv_shared_layers=20` layers use an `_OffsetCache` (zero bytes) in mlx-lm, so the over-allocation for those layers is irrelevant. A separate "gemma4-hybrid" path in `_resolve_model_dims` would add complexity without correctness benefit at this stage.

**Trade-off:** Global-attention layers have a larger KV footprint than budgeted, so peak memory may be slightly underestimated. Acceptable for an initial implementation; can be refined if OOM issues appear on smaller-memory Macs.

### D4: Bump mlx-lm to `>=0.31.2` (from git `4469ad4`)

**Choice:** Update the lower-bound in `pyproject.toml`. Once mlx-lm cuts a PyPI release >= 0.31.2, pin to that. Until then, the install script and documentation note the git source.

**Rationale:** 0.31.1 (current PyPI release) has no `gemma4.py`. The Gemma 4 support landed in commit `4469ad4` and will ship as 0.31.2.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| mlx-vlm adds Gemma 4 support and the allowlist becomes stale | `_MLX_LM_MULTIMODAL_MODELS` can be shrunk; mlx-vlm path would be preferred for full multimodal support | 
| Gemma 4's vision weights load into RAM before being stripped by `sanitize()` | Acceptable on M4 Max 128 GB; future work can skip vision weight files at load time via `mlx_lm_load(ignore_patterns=[...])` |
| KV cache under-allocation for global-attention layers | Monitor with `VLLM_METAL_DEBUG=1`; refine `_resolve_model_dims` if OOM is observed |
| mlx-lm 0.31.2 is not on PyPI yet | Pin git commit in install script; update to PyPI release once available |

## Migration Plan

1. Add `_MLX_LM_MULTIMODAL_MODELS` frozenset and `_use_mlx_lm_for_multimodal()` helper to `model_runner.py`
2. Update `load_model()` to consult the helper before choosing mlx-vlm
3. Bump `mlx-lm` lower bound in `pyproject.toml`; update install script if needed
4. Add integration test: `vllm serve google/gemma-4-E4B-it` starts, `/v1/models` returns 200, `/v1/chat/completions` returns a coherent response
5. Open PR to vllm-metal with reference to this change and the mlx-lm commit

**Rollback:** Revert the `_MLX_LM_MULTIMODAL_MODELS` addition — Gemma 4 falls back to the previous (broken) mlx_vlm path. No data migration or API changes.

## Open Questions

1. **mlx-lm 0.31.2 PyPI ETA** — Should the install script pin the git commit, or wait for the PyPI release?
2. **Vision support scope** — For pearlster's use-case, text-only is sufficient. Should the PR to vllm-metal document this limitation clearly, or also prototype image-input routing via `mlx_vlm` once mlx-vlm adds Gemma 4?
3. **`num_kv_shared_layers` handling** — Gemma 4 has 20 KV-shared layers (zero cache bytes). Should `_resolve_model_dims` account for this to produce a tighter memory estimate?
