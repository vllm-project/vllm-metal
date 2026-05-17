# Text Embedding Pooling

Metal V1 has experimental text-only `embed` pooling support for compatible
pooling models. Supported requests run as prefill-only work, return one CPU
L2-normalized embedding tensor per finished request through vLLM's
`pooler_output` contract, and do not sample generation tokens.

## Scope

Current scope is intentionally narrow:

- `runner="pooling"` with embedding-capable pooler configs
  (`pooler_config.task` unset or `pooler_config.task="embed"`)
- decoder-style text models that expose token hidden states through the MLX
  transformer body
- sequence embeddings from the final prompt-token hidden state with LAST
  pooling and L2 normalization on the paged Metal V1 path

## Unsupported

The Metal runner rejects these cases with diagnostic errors:

- classification, reranking, and late interaction
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

### Offline Embeddings

Set `VLLM_METAL_USE_PAGED_ATTENTION=1` for this MVP.

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

### OpenAI-Compatible Server

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

## Validation

Do not add a model row to [Supported Models](supported_models.md) until a real
`LLM.embed` or `/v1/embeddings` smoke passes on Apple Silicon with the model
name, revision, command, and output dimension recorded.
