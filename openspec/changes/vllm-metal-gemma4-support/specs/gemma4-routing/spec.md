## ADDED Requirements

### Requirement: Gemma 4 loads via mlx_lm on vllm-metal

vllm-metal's `MetalModelRunner` SHALL route models with `model_type == "gemma4"` through `mlx_lm_load()`, regardless of whether vLLM's engine reports `is_multimodal_model = True` for that model.

#### Scenario: Gemma 4 model starts successfully
- **WHEN** `vllm serve google/gemma-4-E4B-it` is invoked with the vllm-metal plugin active
- **THEN** the server SHALL start without deadlock and log `"MLX device set to: Device(gpu, 0)"`
- **THEN** `GET /v1/models` SHALL return HTTP 200 within 5 minutes of process start
- **THEN** `GET /health` SHALL return `{"status": "ok"}`

#### Scenario: Other VLMs still use mlx_vlm
- **WHEN** a model with `model_type` NOT in the mlx-lm allowlist is loaded and `is_multimodal_model = True`
- **THEN** the runner SHALL still call `mlx_vlm_load()` as before (no regression for existing VLMs)

### Requirement: mlx-lm allowlist is a named, extensible frozenset

The set of `model_type` strings that vllm-metal routes through `mlx_lm_load()` despite being multimodal SHALL be declared as a module-level `frozenset[str]` named `_MLX_LM_MULTIMODAL_MODELS` in `vllm_metal/v1/model_runner.py`.

#### Scenario: Adding a future mlx-lm-backed multimodal model
- **WHEN** a new multimodal model type is added to mlx-lm (e.g., `"gemma5"`)
- **THEN** adding its `model_type` string to `_MLX_LM_MULTIMODAL_MODELS` SHALL be sufficient to route it through `mlx_lm_load()` with no other code changes

### Requirement: KV cache dimensions are correctly resolved for Gemma 4

The runner SHALL successfully call `_extract_model_args()` and `_resolve_model_dims()` for a Gemma 4 model loaded via `mlx_lm_load()`, producing valid `num_layers`, `num_kv_heads`, and `head_dim` without raising `ValueError`.

#### Scenario: Gemma 4 model args extracted from nested text_config
- **WHEN** `mlx_lm_load()` returns a `gemma4.Model` whose `.args.text_config` contains `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, and `head_dim`
- **THEN** `_extract_model_args()` SHALL merge those fields into `self.model_args`
- **THEN** `_resolve_model_dims()` SHALL set `self.num_layers`, `self.num_kv_heads`, and `self.head_dim` without raising

#### Scenario: Chat completion returns a valid response
- **WHEN** `POST /v1/chat/completions` is called with `{"messages": [{"role": "user", "content": "Say hello."}]}`
- **THEN** the response SHALL have HTTP 200 and `choices[0].message.content` SHALL be a non-empty string

### Requirement: mlx-lm version constraint updated

`pyproject.toml` in the vllm-metal repository SHALL declare `mlx-lm >= 0.31.2` (or pin the git commit `4469ad4` until 0.31.2 is released on PyPI) to ensure `mlx_lm.models.gemma4` is importable.

#### Scenario: Install succeeds with Gemma 4 model type available
- **WHEN** the install script or `pip install vllm-metal` is run on a fresh environment
- **THEN** `python -c "import mlx_lm.models.gemma4; print('ok')"` SHALL exit 0
