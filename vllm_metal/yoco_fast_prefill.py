# SPDX-License-Identifier: Apache-2.0
"""Gemma4 YOCO fast-prefill helpers.

This module owns the reduced-query metadata contract and the default-off
Gemma4 runtime patch. The metadata helpers intentionally avoid importing MLX or
mlx-lm internals so the indexing contract can be tested in fast CI.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_GEMMA4_MODEL_TYPES = frozenset({"gemma4", "gemma4_text"})
_PATCHED_CALL_ATTR = "_vllm_metal_yoco_fast_prefill_patched"
_ORIGINAL_CALL_ATTR = "_vllm_metal_yoco_fast_prefill_original_call"
_ENABLED_ATTR = "_vllm_metal_yoco_fast_prefill_enabled"
_WARNED_FALLBACK_ATTR = "_vllm_metal_yoco_fast_prefill_warned_fallback"

logger = logging.getLogger(__name__)


def _get_int_model_arg(model_args: Mapping[str, object], key: str) -> int | None:
    value = model_args.get(key, 0)
    if not isinstance(value, int):
        return None
    return value


@dataclass(frozen=True)
class YocoReducedContextMetadata:
    """Reduced-query metadata for YOCO cross-decoder fast prefill.

    ``selected_query_indices`` indexes into the original packed token sequence
    built by ``MetalModelRunner._start_paged_forward``: decode tokens first,
    followed by each prefill chunk. The other fields mirror
    ``PagedAttentionContext`` but describe the reduced cross-decoder query set,
    where every selected query is represented as a length-1 segment.
    """

    selected_query_indices: list[int]
    slot_mapping: list[int]
    block_tables: list[list[int]]
    context_lens: list[int]
    offsets: list[int]
    cu_seqlens: list[int]


def is_yoco_fast_prefill_eligible(
    model_args: Mapping[str, object],
    *,
    use_paged_attention: bool,
) -> bool:
    """Return whether model metadata is eligible for YOCO fast prefill.

    The first implementation target is the Gemma4 text path under paged
    attention.  Keep this predicate intentionally narrow until runtime
    correctness is proven and benchmarked.
    """
    if not use_paged_attention:
        return False

    model_type = model_args.get("model_type")
    if model_type not in _GEMMA4_MODEL_TYPES:
        return False

    num_layers = _get_int_model_arg(model_args, "num_hidden_layers")
    num_shared = _get_int_model_arg(model_args, "num_kv_shared_layers")
    if num_layers is None or num_shared is None:
        return False

    return 0 < num_shared < num_layers


def build_yoco_reduced_context_from_full_metadata(
    *,
    slot_mapping: Sequence[int],
    block_tables: Sequence[Sequence[int]],
    context_lens: Sequence[int],
    offsets: Sequence[int],
    cu_seqlens: Sequence[int],
) -> YocoReducedContextMetadata:
    """Build reduced YOCO metadata from an existing paged-attention context.

    ``prepare_unified`` already computes the full packed metadata used by the
    self-decoder. The YOCO cross-decoder needs the last query from each packed
    segment while preserving that segment's full K/V context. This helper keeps
    the contract close to the actual ``PagedAttentionContext`` fields without
    importing MLX or the runtime context class.
    """
    if not cu_seqlens or int(cu_seqlens[0]) != 0:
        raise ValueError("cu_seqlens must start with 0")

    num_segments = len(cu_seqlens) - 1
    if (
        len(block_tables) != num_segments
        or len(context_lens) != num_segments
        or len(offsets) != num_segments
    ):
        raise ValueError(
            "block_tables, context_lens, and offsets must match cu_seqlens segments"
        )
    selected_query_indices: list[int] = []
    reduced_slot_mapping: list[int] = []
    reduced_block_tables: list[list[int]] = []
    reduced_context_lens: list[int] = []
    reduced_offsets: list[int] = []
    reduced_cu_seqlens: list[int] = [0]

    for segment_idx in range(num_segments):
        q_start = int(cu_seqlens[segment_idx])
        q_end = int(cu_seqlens[segment_idx + 1])
        if q_end <= q_start:
            raise ValueError("cu_seqlens segments must be positive and increasing")
        if q_end > len(slot_mapping):
            raise ValueError("slot_mapping shorter than cu_seqlens")

        segment_len = q_end - q_start
        selected_query_index = q_end - 1
        selected_query_indices.append(selected_query_index)
        reduced_slot_mapping.append(int(slot_mapping[selected_query_index]))
        reduced_block_tables.append(
            [int(block_id) for block_id in block_tables[segment_idx]]
        )
        reduced_context_lens.append(int(context_lens[segment_idx]))
        reduced_offsets.append(int(offsets[segment_idx]) + segment_len - 1)
        reduced_cu_seqlens.append(reduced_cu_seqlens[-1] + 1)

    if int(cu_seqlens[-1]) != len(slot_mapping):
        raise ValueError("slot_mapping length must match final cu_seqlens")

    return YocoReducedContextMetadata(
        selected_query_indices=selected_query_indices,
        slot_mapping=reduced_slot_mapping,
        block_tables=reduced_block_tables,
        context_lens=reduced_context_lens,
        offsets=reduced_offsets,
        cu_seqlens=reduced_cu_seqlens,
    )


def find_gemma4_text_model(model: Any) -> Any | None:
    """Return the mlx-lm Gemma4 text-model object if it matches our contract."""
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
        getattr(getattr(model, "language_model", None), "model", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if _has_gemma4_text_model_contract(candidate):
            return candidate
    return None


def patch_gemma4_yoco_fast_prefill(
    model: Any,
    model_args: Mapping[str, object],
    *,
    use_paged_attention: bool,
) -> bool:
    """Patch an eligible mlx-lm Gemma4 text model for YOCO fast prefill.

    The patch is installed on the model class but guarded by a per-instance
    enable flag. This avoids changing behavior for other Gemma4 instances in
    the same process unless this function is called for them.
    """
    if not is_yoco_fast_prefill_eligible(
        model_args,
        use_paged_attention=use_paged_attention,
    ):
        return False

    text_model = find_gemma4_text_model(model)
    if text_model is None:
        return False

    text_model_cls = type(text_model)
    if not getattr(text_model_cls, _PATCHED_CALL_ATTR, False):
        try:
            original_call = text_model_cls.__call__
        except AttributeError:
            return False
        if not callable(original_call):
            return False

        def _patched_call(
            self,
            inputs=None,
            cache=None,
            input_embeddings=None,
            per_layer_inputs=None,
        ):
            original = getattr(type(self), _ORIGINAL_CALL_ATTR)
            if not getattr(self, _ENABLED_ATTR, False):
                return original(
                    self,
                    inputs,
                    cache=cache,
                    input_embeddings=input_embeddings,
                    per_layer_inputs=per_layer_inputs,
                )
            return _gemma4_text_fast_prefill_call(
                self,
                original,
                inputs=inputs,
                cache=cache,
                input_embeddings=input_embeddings,
                per_layer_inputs=per_layer_inputs,
            )

        setattr(_patched_call, _PATCHED_CALL_ATTR, True)
        setattr(text_model_cls, _ORIGINAL_CALL_ATTR, original_call)
        text_model_cls.__call__ = _patched_call
        setattr(text_model_cls, _PATCHED_CALL_ATTR, True)

    setattr(text_model, _ENABLED_ATTR, True)
    return True


def _has_gemma4_text_model_contract(model: Any) -> bool:
    required_attrs = (
        "config",
        "layers",
        "previous_kvs",
        "embed_tokens",
        "embed_scale",
        "hidden_size_per_layer_input",
        "_get_per_layer_inputs",
        "_project_per_layer_inputs",
        "_make_masks",
        "norm",
    )
    return all(hasattr(model, attr) for attr in required_attrs)


def _first_kv_shared_layer_idx(model: Any) -> int:
    config = model.config
    num_layers = int(getattr(config, "num_hidden_layers", len(model.layers)))
    num_shared = int(getattr(config, "num_kv_shared_layers", 0))
    return num_layers - num_shared


def _call_original(
    original_call: Callable[..., Any],
    model: Any,
    *,
    inputs: Any,
    cache: Any,
    input_embeddings: Any,
    per_layer_inputs: Any,
) -> Any:
    return original_call(
        model,
        inputs,
        cache=cache,
        input_embeddings=input_embeddings,
        per_layer_inputs=per_layer_inputs,
    )


def _warn_fast_prefill_fallback_once(model: Any, reason: str) -> None:
    if getattr(model, _WARNED_FALLBACK_ATTR, False):
        return
    setattr(model, _WARNED_FALLBACK_ATTR, True)
    logger.warning("Gemma4 YOCO fast prefill disabled for this model: %s", reason)


def _gemma4_text_fast_prefill_call(
    model: Any,
    original_call: Callable[..., Any],
    *,
    inputs: Any,
    cache: Any,
    input_embeddings: Any,
    per_layer_inputs: Any,
) -> Any:
    try:
        import mlx.core as mx

        from vllm_metal.paged_attention_common import (
            PagedAttentionContext,
            get_context,
            set_context,
        )
    except ImportError:
        return _call_original(
            original_call,
            model,
            inputs=inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            per_layer_inputs=per_layer_inputs,
        )

    ctx = get_context()
    if ctx is None or ctx.cu_seqlens is None:
        return _call_original(
            original_call,
            model,
            inputs=inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            per_layer_inputs=per_layer_inputs,
        )

    try:
        meta = build_yoco_reduced_context_from_full_metadata(
            slot_mapping=ctx.slot_mapping,
            block_tables=ctx.block_tables,
            context_lens=ctx.context_lens,
            offsets=ctx.offsets,
            cu_seqlens=ctx.cu_seqlens,
        )
    except ValueError as exc:
        _warn_fast_prefill_fallback_once(model, str(exc))
        return _call_original(
            original_call,
            model,
            inputs=inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            per_layer_inputs=per_layer_inputs,
        )

    if not meta.selected_query_indices:
        return _call_original(
            original_call,
            model,
            inputs=inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            per_layer_inputs=per_layer_inputs,
        )

    first_shared = _first_kv_shared_layer_idx(model)
    num_layers = len(model.layers)
    if first_shared <= 0 or first_shared >= num_layers:
        return _call_original(
            original_call,
            model,
            inputs=inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            per_layer_inputs=per_layer_inputs,
        )

    if input_embeddings is None:
        input_embeddings = model.embed_tokens(inputs)
    h = input_embeddings * model.embed_scale

    if model.hidden_size_per_layer_input:
        if per_layer_inputs is None:
            per_layer_inputs = model._get_per_layer_inputs(inputs, input_embeddings)
        per_layer_inputs = model._project_per_layer_inputs(h, per_layer_inputs)

    if per_layer_inputs is not None:
        layer_inputs = [
            per_layer_inputs[:, :, i, :] for i, _ in enumerate(model.layers)
        ]
    else:
        layer_inputs = [None] * num_layers

    if cache is None:
        cache = [None] * num_layers
    else:
        cache = list(cache) + [None] * (num_layers - len(cache))

    masks = model._make_masks(h, cache)
    intermediates = [(None, None)] * num_layers

    for idx in range(first_shared):
        layer = model.layers[idx]
        kvs, offset = intermediates[model.previous_kvs[idx]]
        h, kvs, offset = layer(
            h,
            masks[idx],
            cache[idx],
            per_layer_input=layer_inputs[idx],
            shared_kv=kvs,
            offset=offset,
        )
        intermediates[idx] = (kvs, offset)

    selected = mx.array(meta.selected_query_indices, dtype=mx.int32)
    cross_h = h[:, selected, :]
    reduced_ctx = PagedAttentionContext(
        slot_mapping=meta.slot_mapping,
        block_tables=meta.block_tables,
        context_lens=meta.context_lens,
        offsets=meta.offsets,
        cu_seqlens=meta.cu_seqlens,
        gdn_slot_mapping=ctx.gdn_slot_mapping,
    )

    set_context(reduced_ctx)
    try:
        for idx in range(first_shared, num_layers):
            layer = model.layers[idx]
            kvs, offset = intermediates[model.previous_kvs[idx]]
            layer_input = layer_inputs[idx]
            if layer_input is not None:
                layer_input = layer_input[:, selected, :]
            cross_h, kvs, offset = layer(
                cross_h,
                masks[idx],
                cache[idx],
                per_layer_input=layer_input,
                shared_kv=kvs,
                offset=offset,
            )
            intermediates[idx] = (kvs, offset)
    finally:
        set_context(ctx)

    h[:, selected, :] = cross_h
    return model.norm(h)
