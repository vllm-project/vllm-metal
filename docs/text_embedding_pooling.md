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

BGE-M3 additionally supports `token_classify`, returning one sparse lexical
weight per prompt token after matching boundary BOS and EOS tokens are removed.
Token IDs remain available through vLLM's `/tokenize` endpoint.

## Scope

Current scope is intentionally narrow:

- text `embed` requests with `runner="pooling"` and embedding-capable
  pooler configs (`pooler_config.task` unset or `pooler_config.task="embed"`)
- original Qwen3 reranker `classify` requests with
  `Qwen3ForSequenceClassification`, `classifier_from_token=["no", "yes"]`,
  and `is_original_qwen3_reranker=True`
- decoder-style text models that expose token hidden states through the MLX
  transformer body (LAST pooling)
- encoder embedding checkpoints loaded through optional `mlx-embeddings`
  (`XLMRobertaModel` / `RobertaEmbeddingModel` / `BgeM3EmbeddingModel`), with
  CLS pooling and L2 normalization for dense vectors
- BGE-M3 sparse lexical weights for `mlx-community/bge-m3-mlx-8bit` with an
  explicit `pooler_config.task="token_classify"`
- Qwen3 reranker cross-encoder scores from the final prompt-token hidden state,
  using `lm_head` for untied checkpoints or `embed_tokens.as_linear` when word
  embeddings are tied

Install the encoder path with:

```bash
pip install "vllm-metal[embeddings]"
```

## Runtime flow

BGE-M3 sparse pooling reuses the encoder forward from the dense path, then
applies the official `sparse_linear` head to every token instead of reducing a
request to one sequence vector. The output stays inside vLLM's existing
`pooler_output` contract: one CPU tensor per request, with one lexical weight
per prompt token after matching boundary BOS and EOS tokens are removed.

1. Model-level `PoolerConfig(task="token_classify")` requests the BGE-M3 sparse
   capability at load time; request-level `PoolingParams(task="token_classify")`
   selects it for a call. Other encoder checkpoints continue to expose dense
   `embed` only.
2. For `mlx-community/bge-m3-mlx-8bit`, `EncoderEmbeddingAdapter` loads the MLX
   encoder checkpoint and the official `BAAI/bge-m3` `sparse_linear.pt` head
   from a pinned revision. The head is converted once and retained by the
   adapter; dense-only loads do not fetch it.
3. The paged runner forwards packed encoder requests independently so
   bidirectional attention cannot cross request boundaries. Pooling slices the
   resulting hidden-state pack with the same cumulative sequence boundaries.
4. The packed hidden states run through `sparse_linear` and bias once. Each
   request slice then applies logit calibration and its activation setting;
   matching boundary BOS and EOS rows are removed before the tensor crosses
   from MLX to CPU PyTorch.

Token-wise pooling requires one complete, uncached encoder prefill. A chunked
or resumed request is rejected before the encoder forward rather than returning
weights for only the last chunk. The worker advertises `token_classify` only
after the adapter has loaded a valid BGE-M3 sparse head. Batched complete
requests remain supported and preserve request order.

## Unsupported

The Metal runner rejects these cases with diagnostic errors:

- generic classification heads, generic reranking models, and late interaction
- sequence pooling strategies other than LAST for decoder embed models, and
  other than CLS/LAST for encoder embed models (`MEAN`, `ALL`, `STEP`)
- token-level embedding, combined dense+sparse output, and token classification
  for models other than the supported BGE-M3 checkpoint
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

BGE-M3 sparse lexical weights:

```python
from vllm import LLM
from vllm.config import PoolerConfig

llm = LLM(
    model="mlx-community/bge-m3-mlx-8bit",
    runner="pooling",
    pooler_config=PoolerConfig(task="token_classify"),
    max_model_len=512,
)
outputs = llm.encode(
    ["hello metal", "semantic search"],
    pooling_task="token_classify",
)
print(outputs[0].outputs.data)
```

Dense BGE-M3 / XLM-RoBERTa (requires `vllm-metal[embeddings]`):

```python
from vllm import LLM

llm = LLM(
    model="mlx-community/bge-m3-mlx-8bit",
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

For sparse BGE-M3, pass `--pooler-config '{"task":"token_classify"}'` to
`vllm serve` and call `/pooling` with `"task":"token_classify"`. Use
`/tokenize` when token IDs are needed alongside the returned weights.

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
`LLM.embed`, `/v1/embeddings`, `LLM.encode`, `/pooling`, `LLM.score`, or `/score`
smoke passes on Apple Silicon with the model name, revision, command, output
shape, and any token-ID-to-lexical-weight reference recorded.
