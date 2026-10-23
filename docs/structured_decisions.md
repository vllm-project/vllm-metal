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

## Laya

Validated checkpoint: `convaiinnovations/laya-typed-decisions`, revision
`e929ae5cf69bc34259cd2f95c9e91145b818b1f0`. Support is text-only
ModernBERT in FP32, FP16, or BF16, with one configured action in `act_costs`
(two action logits).
Other checkpoints, including multilingual variants, remain unvalidated.
Quantization, chunked requests, and output dimension truncation are unsupported.

Load the original checkpoint directly, using the explicit Laya configuration
format and tokenizer mode:

```bash
vllm serve convaiinnovations/laya-typed-decisions \
  --revision e929ae5cf69bc34259cd2f95c9e91145b818b1f0 \
  --config-format laya --tokenizer-mode laya --served-model-name laya \
  --runner pooling --dtype float32 --max-model-len 1024 \
  --enforce-eager --no-enable-prefix-caching
```

Use `--dtype float16` or `--dtype bfloat16` for reduced-precision weights and
encoder activations. Logits and probability calculations retain upstream FP32
handling. Reduced precision can change probabilities and close option rankings;
use FP32 to reproduce the validated CPU FP32 reference. `--dtype auto` follows
vLLM's dtype selection.

An original local checkpoint directory can replace the Hub model ID. Keep its
`encoder/config.json`, `rl_agent_config.json`, `tokenizer/`, and weights in
place. Loading reads these files without creating a converted directory or
rewriting checkpoint files. Both the frontend and Metal backend use the Laya
tokenizer mode, which reads the `tokenizer/` subdirectory. Set the server length
limit to the checkpoint's `max_len` in `rl_agent_config.json`.

For offline use, pass `config_format="laya"` and `tokenizer_mode="laya"` to
`LLM`, together with `runner="pooling"` and `dtype="float32"`, then call
`LLM.encode(..., pooling_task="token_classify")`.

Each prompt contains one question, using the checkpoint tokenizer's layout:

```text
[CLS] <choice|score|noul> question: <instructions> [SEP]
[MASK] option0 [MASK] option1 ... [SEP] <state> [SEP]
```

This short two-option example supplies token IDs directly:

```python
import json
import urllib.request
from transformers import PreTrainedTokenizerFast

tok = PreTrainedTokenizerFast.from_pretrained(
    "convaiinnovations/laya-typed-decisions",
    revision="e929ae5cf69bc34259cd2f95c9e91145b818b1f0",
    subfolder="tokenizer",
)


def encode(text):
    return tok.encode(text, add_special_tokens=False)


ids = [tok.cls_token_id] + encode("choice question: Which team should handle this?")
ids += [tok.sep_token_id, tok.mask_token_id] + encode(" billing")
ids += [tok.mask_token_id] + encode(" technical") + [tok.sep_token_id]
ids += encode("The customer was billed twice.") + [tok.sep_token_id]
request = urllib.request.Request(
    "http://localhost:8000/pooling",
    data=json.dumps(
        {"model": "laya", "task": "token_classify", "input": [ids]}
    ).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    print(json.load(response)["data"][0]["data"])
```

The result has one row per option: `[p_option, p_action0, p_action1]`. Option
probabilities use checkpoint temperature calibration; action probabilities are
repeated on each row. Set `use_activation: false` to return raw logits instead.
Application code selects choice labels, computes expected scores, or reads the
true-option probability for `noul` (false, then true). Action policy and the
Laya Agent/Router APIs are outside this backend.

The server does not render or truncate Laya questions. For structured inputs,
follow the original checkpoint's rendering and token budgets, remove literal
MASK text from instructions/options/state, and keep inputs within the server
length limit. Unknown type tokens fall back to `choice`; no markers return an
empty result. This encoder uses full bidirectional attention without decoder
KV cache or paged attention.
