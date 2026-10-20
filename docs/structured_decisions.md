# Structured decisions

Jev-like System 1 decisions return yes/no probabilities, choices, and scores.
See the [upstream guide](https://github.com/vllm-project/vllm/blob/v0.31.0/examples/features/structured_diffusion/README.md)
for `/v1/systemone` request examples, question schemas, and sampling options.

## DiffusionGemma

Start the Metal model server:

```bash
VLLM_METAL_BACKEND=mlx vllm serve mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --served-model-name dgemma --host 127.0.0.1 --port 8000 \
  --diffusion-config '{"canvas_length": 32}' --max-logprobs 32
```

Run the upstream example server in another terminal to expose `/v1/systemone`
on port 8011:

```bash
curl -fL \
  https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/examples/features/structured_diffusion/structured_server.py \
  -o structured_server.py
python structured_server.py \
  --upstream http://127.0.0.1:8000 --model dgemma \
  --tokenizer mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --canvas 32 --canvas-step 32 --host 127.0.0.1 --port 8011
```

Set `--canvas` and `--canvas-step` to the served `canvas_length`; smaller
per-request canvases are unsupported on Metal.
DiffusionGemma currently accepts text input only and uses the MLX backend.
