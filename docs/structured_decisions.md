# Structured decisions

Structured decisions return choices (`choice`), yes/no probabilities (`noul`),
and scores (`score`). With vLLM 0.33.0, `vllm serve` exposes `/v1/systemone`
directly on the model server; no separate proxy is needed.
See the [upstream guide](https://docs.vllm.ai/en/latest/models/structured_decisions_models/)
for question schemas and model-specific options.

## D3 (Decision 3.0)

The MLX backend supports text and JSON states. `vllm-sr/d3-nano` and
`vllm-sr/d3-flash` have been tested on Metal; their checkpoint downloads are
about 4.4 GB and 16.8 GB respectively. Flash has been tested on a 64 GB Mac.
Images and videos are not supported. Checkpoints using `prefix_bidir`, including
`d3-turbo` and `d3-ultra`, are not supported by this integration.

Start the server:

```bash
VLLM_METAL_BACKEND=mlx vllm serve vllm-sr/d3-nano \
  --runner pooling --served-model-name d3 --host 127.0.0.1 \
  --max-model-len 2048 --max-num-batched-tokens 2048 \
  --max-num-seqs 2 --kv-cache-memory-bytes 1073741824
```

For Flash, replace the model ID with `vllm-sr/d3-flash`. These settings are also
available in the [Nano recipe](https://github.com/vllm-project/vllm-metal/blob/main/recipes/overlays/vllm-sr/d3-nano.yaml)
and [Flash recipe](https://github.com/vllm-project/vllm-metal/blob/main/recipes/overlays/vllm-sr/d3-flash.yaml).
The 2,048-token limit includes the state, question, and answer options.
Keep `--max-num-batched-tokens` at least as large as `--max-model-len`.

Ask a question:

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "d3",
    "state": "I was charged twice for my subscription and need a refund.",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
          "billing": "Payments, charges, and refunds",
          "technical": "Software errors and login issues"
        }
      }
    }
  }'
```

The response contains the selected option and its probabilities under
`answers.route`. Use `/v1/systemone`: upstream vLLM 0.33.0 does not enable
`/v1/decisions` for D3 and returns HTTP 501. D3 does not serve chat completions.

Prefix caching and chunked prefill are disabled automatically, as in upstream
vLLM. These checkpoints read the whole prompt bidirectionally, so changing the
question changes the state representations. Each question processes the full
prompt, even when several questions share the same state.

## DiffusionGemma

Start the Metal model server:

```bash
VLLM_METAL_BACKEND=mlx vllm serve mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --served-model-name dgemma --host 127.0.0.1 --port 8000 \
  --diffusion-config '{"canvas_length": 16}' --max-logprobs 32
```

Send the request above to the same `/v1/systemone` endpoint, changing `model`
to `dgemma`. The 16-token canvas matches upstream vLLM 0.33.0's native decision
requests; smaller per-request canvases are unsupported on Metal.
DiffusionGemma currently accepts text input only and uses the MLX backend.
