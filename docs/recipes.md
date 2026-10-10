# Recipes by Mac Memory

vLLM Metal runs an AI chat model on your own Mac. This page gives one tested
setup for each memory size: 16, 32 or 64 GB. Pick yours, run one command, and
you have a chat model that you and your apps can send questions to.

To see your Mac's memory, open the Apple menu, choose **About This Mac**, and
look for **Memory**.

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

## 16 GB Mac

You get the Qwen3.5-9B model, with room for about 49,000 words per
conversation.

```bash
vllm serve mlx-community/Qwen3.5-9B-MLX-4bit --max-model-len 65536 \
  --max-num-batched-tokens 2048
```

- The first run downloads 6.0 GB.
- `--max-model-len 65536` is the longest conversation in tokens: your
  question and the reply together.
- It can work on one conversation of that length at once, or two of half
  that length.
- `--max-num-batched-tokens 2048` makes vLLM read long text in smaller steps,
  so it needs less memory. Without it this model does not fit on 16 GB. It
  did not slow down replies in testing.

## 32 GB Mac

You get the larger Qwen3.5-27B model, with room for about 24,000 words per
conversation.

```bash
vllm serve mlx-community/Qwen3.5-27B-4bit --max-model-len 32768 \
  --max-num-batched-tokens 1024
```

- The first run downloads 16 GB.
- It can work on about one and a half conversations of that length at once.
- It is slow to read long text: about a minute for 10,000 words on an M4 Max.
  For faster replies, use the 16 GB recipe.

## 64 GB Mac

You get Qwen3.5-35B-A3B, with room for about 24,000 words per conversation.

```bash
vllm serve mlx-community/Qwen3.5-35B-A3B-4bit --max-model-len 32768
```

- The first run downloads 20 GB.
- It can work on about 20 conversations of that length at once.
- It is the fastest of the three. Only part of the model runs for each word,
  so it reads 10,000 words in about 12 seconds on an M4 Max.

## Check that it works

The server is ready when Terminal shows `Application startup complete`. Leave
that window open. Open a second Terminal window with Command-N and paste:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-MLX-4bit",
    "messages": [{"role": "user", "content": "What is unified memory?"}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

On a 32 or 64 GB Mac, change `Qwen3.5-9B-MLX-4bit` to the model in your command.
The answer is the text after `"content"`.

Qwen3.5 normally thinks out loud before it answers. `"enable_thinking": false`
turns that off, so the reply is shorter and faster. To see the thinking,
delete that line and the comma at the end of the line above it.

## Use it from a chat app

Apps that let you set an OpenAI API address, often called the base URL, can
use the server. Set it to `http://localhost:8000/v1`. If the app asks for an
API key, type any text.

The server also accepts connections from other computers on your network. To
keep it to your Mac only, add `--host 127.0.0.1` to the command.

## Stop the server

Click the Terminal window running the server and press Control-C. Run the
same command to start it again. The model is already downloaded.

## If it does not start

Read the error just above the last line in Terminal.

- **"command not found: vllm"**: run the `source` command from
  [Before you start](#before-you-start) in this window.
- **"To serve at least one request with the model's max seq len"**: the
  conversation length does not fit. Replace the number after
  `--max-model-len` with the smaller one the message suggests.
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
step from [Before you start](#before-you-start), then:

```bash
LLMMAN_SAFETENSORS_ENGINE=vllm LLMMAN_CONTEXT_LENGTH=65536 \
  LLMMAN_VLLM_ARGS="--max-num-batched-tokens 2048" \
  llmman run hf.co/mlx-community/Qwen3.5-9B-MLX-4bit
```

`LLMMAN_CONTEXT_LENGTH` is the same as `--max-model-len` above, and
`LLMMAN_VLLM_ARGS` passes other options to vLLM. llmman keeps
its own copy of the model, so the first run downloads it again.

## Reuse long prompts

KV offloading helps when the same long text comes back, such as a long chat
or one document asked about many times. vLLM reloads the work it already did
on that text instead of redoing it. It works with Qwen3 models such as
`mlx-community/Qwen3-8B-4bit`, not with the Qwen3.5 models above. It needs a
vLLM Metal release newer than v0.30.0 or the install script's development
build. See
[KV Cache Offloading](kv_offloading.md).

## How these were tested

You can skip this section.

Each recipe was started and sent a request with vLLM 0.31 and vLLM Metal
main (October 2026), on a 128 GB M3 Max and a 128 GB M4 Max. Reading times
were measured on the M4 Max with a 14,000-token prompt. The llmman command was
run on the M4 Max with llmman 0.1.535. `--gpu-memory-utilization`
was set to match the memory a smaller Mac gives its graphics chip, taken as
2/3 of the Mac's memory. Measured Macs gave between 67% and 84%, so a real
Mac of each size should have at least this much. The test Macs had plenty of
free memory, so on a busy Mac fewer conversations fit at once.

If a recipe does not work on your Mac, please open an issue.
