## Context

vllm-metal's `MetalModelRunner` in `vllm_metal/v1/model_runner.py` routes model loading through two paths:

- **`mlx_lm_load()`** — for text-only models (`is_multimodal_model = False`)
- **`mlx_vlm_load()`** — for vision-language models (`is_multimodal_model = True`)

Gemma 4 (`google/gemma-4-E4B-it`) is a multimodal model, so vLLM sets `is_multimodal_model = True`. vllm-metal therefore calls `mlx_vlm_load()`. Additionally, because vLLM core has no native `gemma4` architecture registration, it emits `TransformersMultiModalForCausalLM` as the resolved architecture — a Transformers fallback that conflicts with vllm-metal's MLX runtime during distributed initialisation, causing a deadlock.

Both upstream libraries now have Gemma 4 support:
- **mlx-lm** added `gemma4.py` + `gemma4_text.py` on 2026-04-04 (commit `4469ad4`), covering text/reasoning inference.
- **mlx-vlm** has also added Gemma 4 support, covering vision (image tokens), and audio (a 12-block Conformer encoder with SSCP front-end, fully implemented in `mlx_vlm/models/gemma4/audio.py` and wired in `gemma4.py` via `audio_tower` + `embed_audio`).

This means the fix is no longer constrained to the mlx-lm workaround path — the standard `mlx_vlm_load()` route can now handle all three modalities (text, vision, audio) for Gemma 4.

**Key source files:**
- `vllm_metal/v1/model_runner.py` — `_is_vlm_model()`, `load_model()`, `_extract_model_args()`, `_resolve_model_dims()`
- `mlx_lm/models/gemma4.py` — multimodal wrapper; delegates to `gemma4_text.Model` and drops vision/audio weights in `sanitize()`
- `mlx_lm/models/gemma4_text.py` — text backbone: mixed `sliding_attention` / `full_attention` layers, `sliding_window_pattern=5`, dual head-dim (`head_dim=256` local, `global_head_dim=512` global), KV-shared layers
- `mlx_vlm/models/gemma4/audio.py` — Conformer audio encoder (SSCP → 12 ConformerBlocks → optional output projection); loaded when `config.audio_config is not None`
- `mlx_vlm/models/gemma4/gemma4.py` — top-level model wiring vision + audio towers into the language backbone

## Goals / Non-Goals

**Goals:**
- Gemma 4 models load and serve text inference on vllm-metal without deadlock
- Gemma 4 models serve vision (image+text) inference via `mlx_vlm_load()` — mlx-vlm's vision tower is fully implemented
- Gemma 4 models serve audio input via `mlx_vlm_load()` — mlx-vlm's `AudioEncoder` (12-block Conformer + SSCP) is fully implemented and wired in; this is purely an exposure change, not new implementation
- Correct KV cache dimensions extracted for Gemma 4's sliding-window / full-attention architecture
- An extensible per-model-type routing registry so future mlx-lm-backed multimodal models (Gemma 5, etc.) are a one-line addition
- mlx-lm dependency bumped to `>=0.31.2` in `pyproject.toml`
- mlx-vlm installed from local checkout until a PyPI release that includes Gemma 4 is available

**Non-Goals:**
- Modifications to vLLM core architecture registration (that is upstream work)

## Decisions

### D1: Prefer `mlx_vlm_load()` for Gemma 4 now that mlx-vlm supports it; retain allowlist for text-only override

**Choice:** With mlx-vlm now supporting Gemma 4, the standard routing (`is_multimodal_model=True` → `mlx_vlm_load()`) works correctly and unlocks vision input. The `_MLX_LM_MULTIMODAL_MODELS` allowlist is retained but Gemma 4 is **removed** from it — it should now flow through the mlx-vlm path by default.

The allowlist remains useful as an escape hatch for future models where mlx-lm support lands before mlx-vlm (the original Gemma 4 scenario), or where operators explicitly want text-only inference:

```python
# Gemma 4 removed — now handled natively by mlx-vlm
_MLX_LM_MULTIMODAL_MODELS: frozenset[str] = frozenset()

def _use_mlx_lm_for_multimodal(self, model_type: str) -> bool:
    return model_type in _MLX_LM_MULTIMODAL_MODELS
```

**Rationale:** mlx-vlm's Gemma 4 implementation provides the full multimodal path (image + text). Keeping Gemma 4 in the mlx-lm allowlist would silently strip vision capability. The conservative choice is to use mlx-vlm now that it is available, and let the allowlist serve as a documented override mechanism for future asymmetric situations.

**Alternative considered:** Keep Gemma 4 in the mlx-lm allowlist (text-only) and add mlx-vlm routing as a follow-on. Rejected because the mlx-vlm path is now unblocked and shipping both text and vision in one change reduces the number of routing transitions the codebase has to undergo.

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

### D5: Install mlx-vlm from local checkout until a PyPI release with Gemma 4 is available

**Choice:** Use `pip install -e <path/to/mlx-vlm>` in the vllm-metal dev environment. The `pyproject.toml` dependency entry stays as `mlx-vlm` with a minimum version comment; the install script handles the local override. Once mlx-vlm cuts a tagged release that includes Gemma 4, switch to the PyPI version.

**Rationale:** Pinning a git commit SHA in `pyproject.toml` is not supported by all package managers and creates fragility. A local editable install is explicit, auditable, and mirrors the mlx-lm approach used during the 0.31.2 pre-release period. The two installs (mlx-lm git, mlx-vlm local) should be documented together in `INSTALL.md` so they can be cut over to PyPI in one step.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| ~~mlx-vlm adds Gemma 4 support and the allowlist becomes stale~~ | **Resolved** — mlx-vlm now supports Gemma 4; Gemma 4 removed from the allowlist; mlx-vlm path is now the default. |
| mlx-vlm Gemma 4 release is not yet on PyPI | Use local editable install (`pip install -e`) per D5; switch to PyPI once tagged. |
| mlx-vlm Gemma 4 implementation has regressions vs. mlx-lm (e.g. memory, correctness) | Run text-only latency check against mlx-lm baseline in integration tests; keep `_MLX_LM_MULTIMODAL_MODELS` override for a quick rollback. |
| Gemma 4's vision weights load into RAM before being stripped | Acceptable on M4 Max 128 GB for text-only use; with mlx-vlm path the weights are actively used so no stripping overhead. |
| KV cache under-allocation for global-attention layers | Monitor with `VLLM_METAL_DEBUG=1`; refine `_resolve_model_dims` if OOM is observed. |
| mlx-lm 0.31.2 is not on PyPI yet | Pin git commit in install script; update to PyPI release once available. |

## Migration Plan

1. Add `_MLX_LM_MULTIMODAL_MODELS` frozenset (initially empty for Gemma 4) and `_use_mlx_lm_for_multimodal()` helper to `model_runner.py`
2. Confirm that `load_model()` routes Gemma 4 through `mlx_vlm_load()` (the default multimodal path) with the mlx-vlm Gemma 4 implementation
3. Bump `mlx-lm` lower bound to `0.31.2` in `pyproject.toml`; install mlx-vlm from local checkout per D5; document both in `INSTALL.md`
4. Add integration tests:
   - **Text**: `vllm serve google/gemma-4-E4B-it`, `/v1/chat/completions` (text-only prompt) returns a coherent response; measure latency vs. mlx-lm baseline
   - **Vision**: `/v1/chat/completions` with an image URL returns a coherent response
   - **Audio**: `/v1/chat/completions` with an audio input returns a coherent response
5. Open PR to vllm-metal with reference to this change and both upstream commits (mlx-lm `4469ad4`, mlx-vlm Gemma 4 commit)

**Rollback:** Add `"gemma4"` back to `_MLX_LM_MULTIMODAL_MODELS` to force text-only mlx-lm routing, or revert both dependency bumps entirely. No data migration or API changes.

## Open Questions

1. ~~**mlx-vlm Gemma 4 PyPI release**~~ — **Resolved by D5**: use local editable install until a tagged release is available.
2. **mlx-lm 0.31.2 PyPI ETA** — Coordinate the two installs (mlx-lm git `4469ad4`, mlx-vlm local) in `INSTALL.md`; switch both to PyPI in one step once released.
3. **`num_kv_shared_layers` memory tightening** — Gemma 4 has `num_kv_shared_layers=20`: these layers reuse the KV cache of a neighbouring layer via an `_OffsetCache` that allocates zero new bytes. `_resolve_model_dims` currently counts all `num_hidden_layers` uniformly, so it over-allocates 20 layer-equivalents of KV memory. In practice this causes vllm-metal to report a smaller usable memory budget than is actually available, which may shrink the effective maximum batch size or context window unnecessarily. Should `_resolve_model_dims` subtract `num_kv_shared_layers * per_layer_kv_bytes` from the estimate to give a tighter, more accurate budget?
4. ~~**Vision regression testing**~~ — **Resolved**: mlx-vlm is the definitive routing path for all Gemma 4 modalities (text, vision, audio). The integration test suite should include a text-only latency check against the mlx-lm baseline to confirm there is no regression; mlx-vlm is retained regardless unless a correctness bug is found.
