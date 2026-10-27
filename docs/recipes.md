# Recipes by Mac Memory

Run a model on your own Mac for chat, coding agents and tool use. Pick your
Mac's memory below and the rest of the page follows that choice. To see it,
open the Apple menu, choose **About This Mac**, and look for **Memory**.

!!! note "Estimated starting points"
    The 32 GB and 64 GB settings were first measured on 128 GB Macs with vLLM
    limited to the memory a smaller Mac gives its graphics chip. The 16 GB
    recipe was then checked on a real 16 GB MacBook Pro (M2 Pro). A Mac with
    other apps open may still need a smaller `--max-model-len`. Start here and
    adjust.

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
  `--max-model-len` with the smaller one the message suggests. On a real 16 GB
  Mac this is common if browsers or other large apps are open; try `8192` or
  `4096`.
- **"not enough Metal memory"**: the model does not fit in the memory free
  right now. Quit other large apps and try again. Halving the number after
  `--max-num-batched-tokens` also helps, or use a smaller model.
- **It fits but other apps slow down**: add `--gpu-memory-utilization 0.8` to
  give vLLM a little less memory. The default is 0.92.
- **It stops for about 10 minutes after "init engine", then carries on**:
  vLLM is checking the model on Hugging Face and the connection is slow.
  Once the model is downloaded, start with `HF_HUB_OFFLINE=1` in front of the
  command, for example `HF_HUB_OFFLINE=1 vllm serve ...`.
- **"Found ulimit of 2048"**: raise the open-file limit in that Terminal window
  with `ulimit -n 10240`, then run the serve command again.

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

## How these were tested

You can skip this section.

The 32 GB and 64 GB commands were first run as written with vLLM 0.31 and
vLLM Metal main (October 2026), on a 128 GB M3 Max and a 128 GB M4 Max.
`--gpu-memory-utilization` was added to limit vLLM to the memory a smaller Mac
gives its graphics chip, taken as 2/3 of the Mac's memory. Measured Macs gave
between 67% and 84%. Both Macs gave the same results for those two recipes:

| Recipe | KV cache | Full-length conversations at once | 14,000-token prompt (M4 Max) |
|---|---|---|---|
| 32 GB | 47,786 tokens | 1.5 | 65 s |
| 64 GB | 716,584 tokens | 22 | 12 s |

The 16 GB recipe was then checked on a real 16 GB MacBook Pro (Apple M2 Pro,
macOS 27.2) with vLLM 0.32.0 and vLLM Metal 0.32.0.dev, using the install
script and normal desktop apps open. The earlier simulated 16 GB setting of
`--max-model-len 65536` failed to start: after the 6 GB weights loaded, only a
fraction of a gigabyte remained for KV cache. `--max-model-len 8192` started
and answered a short chat request; larger values such as 16384 worked only when
memory pressure was low. The table above no longer lists a simulated 16 GB KV
figure for that reason.

The llmman command was run on the M4 Max with llmman 0.1.535. Tool calls were
checked with a request that defines one function.

If a recipe does not work on your Mac, please open an issue.
