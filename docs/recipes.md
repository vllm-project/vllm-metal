# Recipes by Mac Memory

Run a model on your own Mac for chat, coding agents and tool use. Pick your
Mac's memory below and the rest of the page follows that choice. To see it,
open the Apple menu, choose **About This Mac**, and look for **Memory**.

!!! note "Estimated starting points"
    Start with the recipe for your Mac's memory. If other apps are open and the
    server does not start, quit them or use a smaller `--max-model-len`.

## Before you start

1. Install vLLM Metal, see [Installation](installation.md).
2. Open **Terminal**: press Command-Space and type "Terminal". You paste each
   command on this page into Terminal and press Return.
3. If you used the install script, turn on vLLM Metal in that window. Do this
   in every new Terminal window:

    ```bash
    source ~/.venv-vllm-metal/bin/activate
    ```

A **token** is the unit a model reads and writes. One token is about three
quarters of a word.

## Start the server

{% for pick in picks %}
=== "{{ pick.memory_tier_gb }} GB"

    {{ pick.description | indent(4) }}

    ```bash
    {{ pick.command | indent(4) }}
    ```

    {{ pick.guide | indent(4) }}

{% endfor %}
What the options do:

- `--host 127.0.0.1` keeps the server to your Mac. See
  [Share with other computers](#share-with-other-computers) to change that.
- `--max-model-len` is the longest conversation in tokens: your messages and
  the replies together.
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder` lets the model
  call tools, which coding agents and other apps need.
- `--reasoning-parser qwen3` puts the model's thinking in a separate field, so
  replies hold only the answer.

The server is ready when Terminal shows `Application startup complete`. Leave
that window open.

## Check that it works

Open a second Terminal window with Command-N and paste:

{% for pick in picks %}
=== "{{ pick.memory_tier_gb }} GB"

    ```bash
    curl {{ pick.base_url }}/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": {{ pick.model_id | tojson }},
           "messages": [{"role": "user", "content": "What is unified memory?"}]}'
    ```

{% endfor %}
The answer is the text after `"content"`. Qwen3.5 thinks before it answers,
which can take a while. To skip the thinking for a faster reply, add
`"chat_template_kwargs": {"enable_thinking": false}` to the request.

## Use it from an app or agent

Apps and coding agents that let you set an OpenAI API address, often called
the base URL, can use the server. Set it to `http://127.0.0.1:8000/v1` and use
the model name from your recipe. If the app asks for an API key, type any
text.

## Share with other computers

To reach the server from other computers on your network, replace
`--host 127.0.0.1` with `--host 0.0.0.0` and add `--api-key` with a secret of
your choice, for example `--api-key my-secret`. Apps then need that key.
Without it, anyone on your network can use the server.

## Stop the server

Click the Terminal window running the server and press Control-C. Run the
same command to start it again. The model is already downloaded.

## If it does not start

Read the error just above the last line in Terminal.

- **"command not found: vllm"**: run the `source` command from
  [Before you start](#before-you-start) in this window.
- **"To serve at least one request with the model's max seq len"**: the
  conversation length does not fit. Replace the number after
  `--max-model-len` with the smaller one the message suggests, or quit other
  large apps and try again.
- **"not enough Metal memory"**: the model does not fit in the memory free
  right now. Quit other large apps and try again. Halving the number after
  `--max-num-batched-tokens` also helps, or use a smaller model.
- **It fits but other apps slow down**: add `--gpu-memory-utilization 0.8` to
  give vLLM a little less memory. The default is 0.92.
- **It stops for about 10 minutes after "init engine", then carries on**:
  vLLM is checking the model on Hugging Face and the connection is slow.
  Once the model is downloaded, start with `HF_HUB_OFFLINE=1` in front of the
  command, for example `HF_HUB_OFFLINE=1 vllm serve ...`.

## Or with llmman

[llmman](https://github.com/llmmanorg/llmman) can start the same models. On a
Mac it uses MLX's own server by default. To use vLLM Metal, run the `source`
{% set example = picks[0] %}
step from [Before you start](#before-you-start), then, for the {{ example.memory_tier_gb }} GB recipe:

```bash
LLMMAN_SAFETENSORS_ENGINE=vllm LLMMAN_CONTEXT_LENGTH={{ example.context }} \
{% if example.batch_tokens %}
  LLMMAN_VLLM_ARGS="--max-num-batched-tokens {{ example.batch_tokens }}" \
{% endif %}
  llmman run hf.co/{{ example.model_id }}
```

`LLMMAN_CONTEXT_LENGTH` is the same as `--max-model-len` above, and
`LLMMAN_VLLM_ARGS` passes other options to vLLM. llmman keeps its own copy of
the model, so the first run downloads it again.

## Reuse long prompts

KV offloading helps when the same long text comes back, such as a long chat
or one document asked about many times. vLLM reloads the work it already did
on that text instead of redoing it. It works with Qwen3 models such as
`mlx-community/Qwen3-8B-4bit`, not with the Qwen3.5 models above. It needs a
vLLM Metal release newer than v0.30.0 or the install script's development
build. See [KV Cache Offloading](kv_offloading.md).
