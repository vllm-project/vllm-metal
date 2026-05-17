# SPDX-License-Identifier: Apache-2.0
"""Text embedding pooling helpers for the Metal V1 model runner."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import torch
from vllm.pooling_params import PoolingParams

from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch

_EMBED_POOLER_TASKS = (None, "embed")
_LAST_POOLING = (None, "LAST")


def _model_label(model_config: Any) -> str:
    served = getattr(model_config, "served_model_name", None)
    if isinstance(served, (list, tuple)):
        served = served[0] if served else None
    return str(served or getattr(model_config, "model", "unknown"))


def _hf_config(model_config: Any) -> Any:
    return getattr(model_config, "hf_config", None)


def _architecture_names(model_config: Any) -> tuple[str, ...]:
    candidates: list[str] = []
    hf_config = _hf_config(model_config)
    for source in (model_config, hf_config):
        architectures = getattr(source, "architectures", None)
        if isinstance(architectures, (list, tuple)):
            candidates.extend(str(arch) for arch in architectures)
    return tuple(candidates)


def _pooler_config(model_config: Any) -> Any:
    return getattr(model_config, "pooler_config", None)


def _pooler_task(model_config: Any) -> str | None:
    task = getattr(_pooler_config(model_config), "task", None)
    return str(task) if task is not None else None


def _sequence_pooling_types(model_config: Any) -> tuple[str | None, str | None]:
    pooler_config = _pooler_config(model_config)
    if pooler_config is None:
        return (None, None)
    seq_pooling_type = getattr(pooler_config, "seq_pooling_type", None)
    pooling_type = getattr(pooler_config, "pooling_type", None)
    return (
        str(seq_pooling_type) if seq_pooling_type is not None else None,
        str(pooling_type) if pooling_type is not None else None,
    )


def _unsupported_sequence_pooling_type(model_config: Any) -> str | None:
    for pooling_type in _sequence_pooling_types(model_config):
        if pooling_type not in _LAST_POOLING:
            return pooling_type
    return None


def _pooler_activation_allows_embed(model_config: Any) -> bool:
    pooler_config = _pooler_config(model_config)
    if pooler_config is None:
        return True
    return getattr(pooler_config, "use_activation", None) is not False


def _chunked_processing_enabled(model_config: Any) -> bool:
    return bool(
        getattr(_pooler_config(model_config), "enable_chunked_processing", False)
    )


def _reject_unsupported_pooler_config(model_config: Any) -> None:
    task = _pooler_task(model_config)
    if task not in _EMBED_POOLER_TASKS:
        raise NotImplementedError(
            "Metal embed pooling supports only pooler_config.task unset or "
            f"'embed'; got {task!r} for model={_model_label(model_config)}."
        )

    seq_pooling_type = _unsupported_sequence_pooling_type(model_config)
    if seq_pooling_type is not None:
        raise NotImplementedError(
            "Metal embed pooling currently supports only LAST sequence pooling; "
            f"got {seq_pooling_type!r} for model={_model_label(model_config)}."
        )
    if _chunked_processing_enabled(model_config):
        raise NotImplementedError(
            "Metal embed pooling does not support "
            "pooler_config.enable_chunked_processing=True with LAST pooling; "
            f"model={_model_label(model_config)}."
        )


def _is_decoder_embedding_config(model_config: Any) -> bool:
    architectures = _architecture_names(model_config)
    return any(
        arch.endswith("ForCausalLM")
        or arch.endswith("ForTextEncoding")
        or arch.endswith("EmbeddingModel")
        for arch in architectures
    )


def sequence_model(model: Any) -> Any | None:
    """Return the MLX transformer body when it is available."""
    inner = getattr(model, "model", None)
    return inner if callable(inner) else None


def supports_embed_pooling(model: Any, model_config: Any) -> bool:
    """Return whether this loaded model can use Metal's LAST embed path."""
    if getattr(model_config, "multimodal_config", None) is not None:
        return False
    if _pooler_task(model_config) not in _EMBED_POOLER_TASKS:
        return False
    if _unsupported_sequence_pooling_type(model_config) is not None:
        return False
    if not _pooler_activation_allows_embed(model_config):
        return False
    if _chunked_processing_enabled(model_config):
        return False
    return sequence_model(model) is not None and _is_decoder_embedding_config(
        model_config
    )


def _unsupported_pooling_option(
    pooling_params: PoolingParams,
    model_config: Any,
) -> str | None:
    checks = (
        (
            "late-interaction parameters",
            getattr(pooling_params, "late_interaction_params", None) is not None,
        ),
        (
            "token-level ALL pooling outputs",
            getattr(pooling_params, "requires_token_ids", False),
        ),
        (
            "STEP pooling parameters",
            getattr(pooling_params, "step_tag_id", None) is not None,
        ),
        (
            "returned_token_ids",
            getattr(pooling_params, "returned_token_ids", None) is not None,
        ),
        (
            "extra pooling kwargs",
            bool(getattr(pooling_params, "extra_kwargs", None)),
        ),
        (
            "use_activation=False",
            getattr(pooling_params, "use_activation", None) is False,
        ),
        (
            "embedding-dimension truncation",
            getattr(pooling_params, "dimensions", None) is not None
            or getattr(_pooler_config(model_config), "dimensions", None) is not None,
        ),
    )
    for reason, unsupported in checks:
        if unsupported:
            return reason
    return None


def validate_pooling_params(
    pooling_params: PoolingParams,
    model_config: Any,
) -> None:
    """Validate the narrow text-only LAST `embed` pooling contract."""
    model = _model_label(model_config)
    if getattr(model_config, "runner_type", None) != "pooling":
        raise NotImplementedError(
            "Metal embed pooling requires runner_type='pooling'; got "
            f"{getattr(model_config, 'runner_type', None)!r} for model={model}."
        )
    _reject_unsupported_pooler_config(model_config)
    if not _is_decoder_embedding_config(model_config):
        raise NotImplementedError(
            "Metal embed pooling requires a decoder-style checkpoint; got "
            f"architectures={_architecture_names(model_config)!r} for model="
            f"{model}."
        )

    task = getattr(pooling_params, "task", None)
    if task not in (None, "embed"):
        raise NotImplementedError(
            "Metal pooling supports only text-only task='embed' for now; "
            f"got task={task!r} for model={model}."
        )

    unsupported_option = _unsupported_pooling_option(pooling_params, model_config)
    if unsupported_option is not None:
        raise NotImplementedError(
            f"Metal embed pooling does not support {unsupported_option} "
            f"for model={model}."
        )


def validate_pooling_request(
    new_req: Any,
    model_config: Any,
    *,
    paged_attention_enabled: bool,
) -> None:
    """Validate the request-level contract for Metal text embedding pooling."""
    pooling_params = new_req.pooling_params
    if pooling_params is None:
        return

    validate_pooling_params(pooling_params, model_config)
    if new_req.mm_features:
        raise NotImplementedError(
            "Multimodal pooling inputs are not supported on Metal yet."
        )
    if getattr(new_req, "prompt_embeds", None) is not None:
        raise NotImplementedError(
            "Prompt-embedding pooling inputs are not supported on Metal yet."
        )
    if not paged_attention_enabled:
        raise NotImplementedError(
            "Metal embed pooling currently requires paged attention; "
            "set VLLM_METAL_USE_PAGED_ATTENTION=1."
        )
    if not (new_req.prompt_token_ids or []):
        raise ValueError(
            "Metal embed pooling requires prompt_token_ids for "
            f"request {new_req.req_id!r}."
        )


def forward_sequence_hidden_states(
    model: Any,
    input_ids: mx.array,
    *,
    cache: Any,
    model_config: Any,
) -> mx.array:
    """Run the transformer body and return per-token hidden states."""
    _reject_unsupported_pooler_config(model_config)
    body = sequence_model(model)
    if body is None:
        raise NotImplementedError(
            "Metal embed pooling requires an MLX model with a callable "
            f"'.model' transformer body; model={_model_label(model_config)}; "
            "task='embed'."
        )

    hidden_states = body(input_ids) if cache is None else body(input_ids, cache=cache)

    if not hasattr(hidden_states, "shape") or not hasattr(hidden_states, "dtype"):
        raise ValueError(
            "Metal embed pooling expected MLX hidden states from model body; "
            f"got {type(hidden_states).__name__} for model="
            f"{_model_label(model_config)}."
        )
    return hidden_states


def pooling_dummy_forward_outputs(
    model: Any,
    input_ids: mx.array,
    *,
    model_config: Any,
) -> list[mx.array]:
    """Return warm-up outputs for a pooling model."""
    return [
        forward_sequence_hidden_states(
            model,
            input_ids,
            cache=None,
            model_config=model_config,
        )
    ]


def has_paged_pooling_work(
    prefill_reqs: list[Any],
    decode_reqs: list[Any],
) -> bool:
    """Return whether a paged batch is pure pooling work."""
    pooling_prefills = [pr for pr in prefill_reqs if pr.pooling_params is not None]
    has_pooling_work = bool(pooling_prefills)
    if has_pooling_work and (len(pooling_prefills) != len(prefill_reqs) or decode_reqs):
        raise NotImplementedError(
            "Metal pooling batches cannot mix pooling requests with "
            "generation prefill/decode requests."
        )
    return has_pooling_work


def _normalize_vector(vector: mx.array) -> mx.array:
    norm = mx.sqrt(mx.sum(vector * vector))
    norm = mx.maximum(norm, mx.array(1e-12, dtype=mx.float32))
    return mx.contiguous(vector / norm)


def pool_sequence_embedding(
    hidden_states: mx.array,
    *,
    token_index: int,
    pooling_params: PoolingParams,
    model_config: Any,
) -> torch.Tensor:
    """Return a normalized CPU LAST embedding for one finished request."""
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError(
            "Metal embed pooling expected hidden states with shape "
            f"[1, tokens, hidden], got {hidden_states.shape} "
            f"for model={_model_label(model_config)}."
        )
    if token_index < 0 or token_index >= hidden_states.shape[1]:
        raise ValueError(
            f"Metal embed pooling token index {token_index} is outside hidden "
            f"state shape {hidden_states.shape} for model={_model_label(model_config)}."
        )

    vector = hidden_states[0, token_index, :].astype(mx.float32)
    vector = _normalize_vector(vector)
    tensor = mlx_to_torch(vector, device="cpu", already_contiguous=True)
    return tensor.detach().clone()


def pool_sequence_batch(
    hidden_states: mx.array,
    *,
    token_indices: list[int | None],
    pooling_params: list[PoolingParams],
    model_config: Any,
) -> list[torch.Tensor | None]:
    """Pool a paged prefill batch; unfinished chunks return ``None``."""
    outputs: list[torch.Tensor | None] = []
    for token_index, params in zip(token_indices, pooling_params, strict=True):
        if token_index is None:
            outputs.append(None)
            continue
        outputs.append(
            pool_sequence_embedding(
                hidden_states,
                token_index=token_index,
                pooling_params=params,
                model_config=model_config,
            )
        )
    return outputs


def pool_paged_prefill_batch(
    hidden_states: mx.array,
    *,
    prefill_entries: list[Any],
    cu_seqlens: list[int],
    num_decode_segments: int,
    model_config: Any,
) -> list[torch.Tensor | None]:
    """Pool a paged prefill batch; unfinished chunks return ``None``."""
    mx.eval(hidden_states)

    token_indices: list[int | None] = []
    pooling_params: list[PoolingParams] = []
    for i, entry in enumerate(prefill_entries):
        pooling_params_for_req = entry.prefill.pooling_params
        if pooling_params_for_req is None:
            raise RuntimeError(
                "Paged pooling batch contained a non-pooling prefill request."
            )
        pooling_params.append(pooling_params_for_req)

        if entry.result_mode == "intermediate":
            token_indices.append(None)
        else:
            token_indices.append(cu_seqlens[num_decode_segments + i + 1] - 1)

    return pool_sequence_batch(
        hidden_states,
        token_indices=token_indices,
        pooling_params=pooling_params,
        model_config=model_config,
    )


def finish_paged_pooling_batch(
    batch: Any,
    hidden_states: mx.array,
    *,
    cu_seqlens: list[int],
    num_decode_segments: int,
    model_config: Any,
) -> None:
    """Convert final paged prefill hidden states into batch pooler outputs."""
    pooler_outputs = pool_paged_prefill_batch(
        hidden_states,
        prefill_entries=batch.paged_prefill_entries,
        cu_seqlens=cu_seqlens,
        num_decode_segments=num_decode_segments,
        model_config=model_config,
    )
    for entry, pooler_output in zip(
        batch.paged_prefill_entries, pooler_outputs, strict=True
    ):
        batch.set_output(entry.output_idx, [], None, pooler_output)
