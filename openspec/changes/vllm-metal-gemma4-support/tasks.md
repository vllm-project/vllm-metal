## 1. Repository Setup

- [ ] 1.1 Fork `vllm-project/vllm-metal` and create a branch `feat/gemma4-support`
- [ ] 1.2 Bump `mlx-lm` lower bound in `pyproject.toml` to `>= 0.31.2` (or pin git commit `4469ad4` until PyPI release)
- [ ] 1.3 Re-run the install script in a clean venv and confirm `python -c "import mlx_lm.models.gemma4; print('ok')"` exits 0

## 2. Model Routing Logic

- [ ] 2.1 Add `_MLX_LM_MULTIMODAL_MODELS: frozenset[str] = frozenset({"gemma4"})` as a module-level constant in `vllm_metal/v1/model_runner.py`
- [ ] 2.2 Add method `_use_mlx_lm_for_multimodal(self, model_type: str) -> bool` that returns `model_type in _MLX_LM_MULTIMODAL_MODELS`
- [ ] 2.3 In `load_model()`, read `model_type` from `self.model_config.hf_config.model_type` before the `if is_vlm:` branch
- [ ] 2.4 Update the `if is_vlm:` block to: `if is_vlm and not self._use_mlx_lm_for_multimodal(model_type):`; the else branch (`mlx_lm_load`) already handles the Gemma 4 case
- [ ] 2.5 Log the routing decision: `logger.info(f"Loading model: {model_name} (VLM: {is_vlm}, mlx_lm path: {not is_vlm or self._use_mlx_lm_for_multimodal(model_type)})")`

## 3. Model Dimension Extraction

- [ ] 3.1 Confirm that `_extract_model_args()` correctly merges `gemma4.Model.args.text_config` into `self.model_args` by adding a debug log or unit test (the existing `text_config` merge path should already handle this)
- [ ] 3.2 Verify `_resolve_model_dims()` resolves without error for Gemma 4: `num_hidden_layers=35`, `num_attention_heads=8`, `num_key_value_heads=1`, `head_dim=256`
- [ ] 3.3 If `head_dim` is absent from merged args (Gemma 4 stores it explicitly), add a fallback: `head_dim = args.get("head_dim") or args.get("global_head_dim") or (hidden_size // num_attention_heads)`

## 4. Integration Test

- [ ] 4.1 Start server: `vllm serve google/gemma-4-E4B-it --max-model-len 8192 --tool-call-parser functiongemma --enable-auto-tool-choice --host 127.0.0.1 --port 8000`
- [ ] 4.2 Assert `GET /health` returns HTTP 200 within 5 minutes
- [ ] 4.3 Assert `GET /v1/models` returns HTTP 200 with `id: "google/gemma-4-E4B-it"` in the response
- [ ] 4.4 Assert `POST /v1/chat/completions` with `{"model": "google/gemma-4-E4B-it", "messages": [{"role": "user", "content": "Say hello in one sentence."}]}` returns HTTP 200 with a non-empty `choices[0].message.content`
- [ ] 4.5 Assert existing VLM smoke test still passes (e.g., a `Qwen3-VL` or `LLaVA` model still routes through `mlx_vlm_load`)

## 5. Documentation & PR

- [ ] 5.1 Add `gemma4` to the Supported Models list in `README.md` or `docs/`
- [ ] 5.2 Add a note under "Limitations" that image/vision input is not yet supported for Gemma 4 (text-only mode via `gemma4.Model.sanitize()`)
- [ ] 5.3 Reference mlx-lm commit `4469ad4` and this change in the PR description
- [ ] 5.4 Open PR to `vllm-project/vllm-metal` targeting `main`
