# Text Pooling

Metal V1 has experimental text-only `embed` pooling support for compatible
pooling models. Supported requests run as prefill-only work, return one CPU
L2-normalized embedding tensor per finished request through vLLM's
`pooler_output` contract, and do not sample generation tokens.

It also has experimental text-only `classify` support for original Qwen3
reranker checkpoints that vLLM converts with
`Qwen3ForSequenceClassification`, `classifier_from_token=["no", "yes"]`, and
`is_original_qwen3_reranker=True`. This path returns one scalar score tensor
per request through the same `pooler_output` contract.

## Scope

Current scope is intentionally narrow:

- text `embed` requests with `runner="pooling"` and embedding-capable
  pooler configs (`pooler_config.task` unset or `pooler_config.task="embed"`)
- original Qwen3 reranker `classify` requests with
  `Qwen3ForSequenceClassification`, `classifier_from_token=["no", "yes"]`,
  and `is_original_qwen3_reranker=True`
- decoder-style text models that expose token hidden states through the MLX
  transformer body
- sequence embeddings from the final prompt-token hidden state with LAST
  pooling and L2 normalization on the paged Metal V1 path
- Qwen3 reranker cross-encoder scores from the final prompt-token hidden state,
  using `lm_head` for untied checkpoints or `embed_tokens.as_linear` when word
  embeddings are tied

## Unsupported

The Metal runner rejects these cases with diagnostic errors:

- generic classification heads, generic reranking models, and late interaction
- sequence pooling strategies other than LAST (`MEAN`, `CLS`, `ALL`, `STEP`)
- token-level pooling
- chunked long-input embedding aggregation (`enable_chunked_processing`)
- non-paged pooling execution
- multimodal embeddings and scheduled encoder inputs
- prompt embeddings
- unsafe dimension requests

Direct model-provided embedding tensors are intentionally out of scope for this
MVP. Add that path only after a real model requires it and the output contract
is validated end to end.

## Usage

Set `VLLM_METAL_USE_PAGED_ATTENTION=1` for the current text pooling MVP.

### Offline Embeddings

```python
from vllm import LLM

llm = LLM(
    model="mlx-community/Qwen3-Embedding-0.6B-8bit",
    runner="pooling",
    max_model_len=512,
)
outputs = llm.embed(["hello metal", "semantic search"])
print(len(outputs), len(outputs[0].outputs.embedding))
```

### Embedding Server

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=1 \
VLLM_METAL_MEMORY_FRACTION=auto \
vllm serve mlx-community/Qwen3-Embedding-0.6B-8bit \
  --runner pooling \
  --max-model-len 512
```

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Qwen3-Embedding-0.6B-8bit","input":["hello metal","semantic search"]}'
```

### Offline Qwen3 Reranking

Original Qwen3 reranker checkpoints need vLLM's sequence-classification
overrides. `LLM.score` can format the query/document pair for this checkpoint
without a separate local template file.

```python
from vllm import LLM

llm = LLM(
    model="mku64/Qwen3-Reranker-0.6B-mlx-8Bit",
    revision="ba80418a47fa1c4368a6c2287b0e449904063576",
    runner="pooling",
    max_model_len=512,
    hf_overrides={
        "architectures": ["Qwen3ForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
    },
)
outputs = llm.score(
    ["What is the capital of China?"],
    ["The capital of China is Beijing."],
)
print(outputs[0].outputs.score)
```

### Qwen3 Reranking Server

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=1 \
VLLM_METAL_MEMORY_FRACTION=auto \
vllm serve mku64/Qwen3-Reranker-0.6B-mlx-8Bit \
  --revision ba80418a47fa1c4368a6c2287b0e449904063576 \
  --runner pooling \
  --max-model-len 512 \
  --hf-overrides '{
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": true
  }'
```

```bash
curl http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"text_1":["What is the capital of China?"],"text_2":["The capital of China is Beijing."]}'
```

## Validation

Do not add a model row to [Supported Models](supported_models.md) until a real
`LLM.embed`, `/v1/embeddings`, `LLM.score`, or `/score` smoke passes on Apple
Silicon with the model name, revision, command, and output shape recorded.
