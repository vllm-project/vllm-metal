# Laya typed decisions

Metal runs converted ModernBERT-based Laya checkpoints through the encoder
pooling backend. Real-checkpoint validation covers
`convaiinnovations/laya-typed-decisions`, revision
`e929ae5cf69bc34259cd2f95c9e91145b818b1f0`. Other Laya checkpoints, including
the multilingual mmBERT variant, have not been validated on this backend.
Each prompt describes one `choice`, `score`, or `noul` question.
The `token_classify` result has one row per `[MASK]` option marker:
`[p_option, *p_action]`. Option probabilities use the checkpoint's temperature
calibration. The action distribution is repeated on every row. Setting
`use_activation: false` returns option and action logits instead.

This initial implementation requires `float32`. Quantization, chunked requests,
and output dimension truncation are unsupported. It uses full bidirectional
encoder attention and does not allocate a decoder KV cache.

## Prepare and serve a checkpoint

The converter and example are source-checkout scripts, not installed commands.
After [installing vllm-metal](installation.md), activate its environment and
obtain a checkout matching that installed version. From the checkout root, run
the commands below. A source contributor can use the checkout and environment
from [Development setup](CONTRIBUTING.md#development-setup) instead.

For an installed build, fetch the source scripts with Git (replace
`<installed-version-ref>` with the tag or commit matching your installed build):

```bash
git clone https://github.com/vllm-project/vllm-metal.git
cd vllm-metal
git checkout <installed-version-ref>
```

These commands obtain scripts; they do not install or upgrade the runtime.

Original Laya checkpoints store the encoder config and tokenizer in
subdirectories and the decision configuration in `rl_agent_config.json`.
The current loader expects one root config containing both model components,
plus a root tokenizer. Run this one-time layout conversion explicitly; serving
does not invoke it automatically:

```bash
python tools/convert_laya_checkpoint.py convaiinnovations/laya-typed-decisions \
  /path/to/laya-vllm --revision e929ae5cf69bc34259cd2f95c9e91145b818b1f0

vllm serve /path/to/laya-vllm --runner pooling --dtype float32 \
  --max-model-len 1024 --enforce-eager --no-enable-prefix-caching
```

Use a local original checkpoint directory in place of the repository ID to
convert without downloading. The destination must not already exist.
The converter creates a new output directory, preserves the original weight
bytes through a hard link or copy, and sets `architectures: ["LayaForDecision"]`
plus `laya_config` in the config.
Use `--revision <commit>` to pin a remote checkpoint. Choose `--max-model-len`
to match the checkpoint's `laya_config.max_len` and truncate prompts before
sending them if necessary. The server does not apply Laya's prompt rendering
or state truncation rules.

Prompts must follow the checkpoint tokenizer's format:

```text
[CLS] <type> question: <instructions> [SEP] [MASK] option0 [MASK] option1 ...
[SEP] <state> [SEP]
```

The question type is read from the token after `[CLS]`. Unknown type tokens
fall back to `choice`. A prompt without markers produces an empty result.
Marker-like text in the state also becomes an option marker; build token IDs
with the same rendering rules as the checkpoint's reference implementation.

Call `/pooling` with `task: "token_classify"`. Token IDs can be supplied as
`input` to preserve the reference tokenization. The output supplies probabilities;
application code must convert them into choice labels, expected scores, or
yes/no answers. Laya's Router and `/v1/systemone` API are outside this backend.

## Run and interpret three questions

With the server running, use the converted checkpoint's tokenizer in the same
activated environment:

```bash
python tools/laya_pooling_example.py --model /path/to/laya-vllm
```

The example sends three short structured questions about one customer message,
using token IDs built from each question and the state. It prints:

- `department.answer`: the label with the largest `choice` probability;
- `urgency.expected_score`: the probability-weighted level index, on a 0–2 scale
  for this example's three criteria;
- `refund.p_true`: the second option's probability, with `noul` options ordered
  as false, then true;
- option and action probabilities for each question. Action selection policy is
  left to the application; action probabilities are not an accuracy guarantee.

Change `QUESTIONS` in the script for your use case, or pass `--state "..."` to
change the customer message. Use `--base-url` if the server is not on port 8000.
The server must load the same checkpoint supplied to the example; the HTTP API
does not expose the model's weight hash.

This is a bounded-input example, not an `Agent.system_one` compatibility layer.
It removes literal MASK text from instructions, options, and state so only the
explicit option markers are scored. It rejects options longer than 48 tokens
and questions or states exceeding the checkpoint's token budgets. It does not
implement original Laya truncation, option reordering, conversation handling,
confidence gates, or Router selection. Keep the server's length limit equal to
the checkpoint's `laya_config.max_len` when following this example.

## Validate against original Laya

`tools/check_laya_parity.py` compares the converted FP32 checkpoint with original
Laya 0.4.1 on 26 synthetic short/long questions covering all three question types
and option-count calibration buckets. It records the original token IDs and
weight SHA-256, then checks both probabilities and raw logits through the actual
Metal runner or a running `/pooling` server. This checks implementation parity,
not labeled task accuracy. The default comparison uses `atol=rtol=2e-5` and
also requires matching option argmax. A failed comparison exits nonzero.

Generate the reference in a separate environment containing `laya==0.4.1`, using
an original local checkpoint. Laya is a validation dependency, not a Metal
runtime dependency:

```bash
python tools/check_laya_parity.py --backend reference \
  --model /path/to/original-laya --reference /path/to/laya-reference.json
```

Then, in the vllm-metal environment:

```bash
python tools/check_laya_parity.py --backend metal \
  --model /path/to/laya-vllm --reference /path/to/laya-reference.json \
  --output /path/to/laya-metal-parity.json
```

To check HTTP instead, start the FP32 server above, then run:

```bash
python tools/check_laya_parity.py --backend http \
  --base-url http://127.0.0.1:8000 --model /path/to/laya-vllm \
  --reference /path/to/laya-reference.json --output /path/to/laya-http-parity.json
```

The HTTP server must load the same converted checkpoint supplied as `--model`.
The tool verifies the local checkpoint hash; the API does not expose a weight
hash. Reference capture uses Laya's private `_infer` method, so it refuses other
Laya package versions. Original Laya performs prompt rendering and truncation;
Metal reuses those exact IDs without rendering a second prompt.
