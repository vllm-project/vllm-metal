# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL multimodal adapter for vLLM Metal."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import torch
from vllm.multimodal.inputs import MultiModalKwargsItem

from vllm_metal.multimodal.feature_spec import MultiModalFeatureSpec
from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx


@dataclass(frozen=True)
class Qwen3VLVisionEncodeResult:
    """Vision tower output for one Qwen3-VL multimodal feature."""

    hidden_states: mx.array
    deepstack_visual_embeds: Sequence[mx.array] | None


class Qwen3VLMultimodalAdapter:
    """Model-owned multimodal helpers for the Qwen3-VL execution path."""

    forward_ready: bool = False
    """Closed until parity against ``mlx_vlm.Model.__call__`` passes."""

    _SUPPORTED_EMBEDS_KWARGS: tuple[str, ...] = (
        "inputs_embeds",
        "input_embeddings",
    )

    _DEEPSTACK_KWARGS: tuple[str, ...] = (
        "visual_pos_masks",
        "deepstack_visual_embeds",
    )

    def __init__(
        self,
        *,
        spatial_merge_size: int,
        vision_tower: Any | None = None,
        language_model: Any | None = None,
        embeds_kwarg: str | None = None,
        embed_tokens_fn: Callable[[Any], Any] | None = None,
        supports_deepstack: bool = False,
    ) -> None:
        if spatial_merge_size <= 0:
            raise ValueError(
                f"spatial_merge_size must be positive, got {spatial_merge_size}"
            )
        self._spatial_merge_size = spatial_merge_size
        self._vision_tower = vision_tower
        self._language_model = language_model
        self._embeds_kwarg = embeds_kwarg
        self._embed_tokens_fn = embed_tokens_fn
        self._supports_deepstack = supports_deepstack

    def text_model(self) -> Any:
        """Return the loaded Qwen3-VL language model."""
        return self._language_model

    @classmethod
    def from_loaded_model(cls, model: Any) -> Qwen3VLMultimodalAdapter:
        """Create an adapter from an mlx-vlm Qwen3-VL/Qwen3.5 composite."""
        vision_tower = model.vision_tower
        language_model = model.language_model
        spatial_merge_size = int(model.config.vision_config.spatial_merge_size)
        embeds_kwarg = cls._detect_embeds_kwarg(language_model)
        embed_tokens_fn = cls._resolve_embed_tokens(language_model)
        supports_deepstack = cls._detect_deepstack_kwargs(language_model)
        return cls(
            spatial_merge_size=spatial_merge_size,
            vision_tower=vision_tower,
            language_model=language_model,
            embeds_kwarg=embeds_kwarg,
            embed_tokens_fn=embed_tokens_fn,
            supports_deepstack=supports_deepstack,
        )

    @classmethod
    def _resolve_embed_tokens(cls, language_model: Any) -> Callable[[Any], Any]:
        """Return ``language_model.model.embed_tokens`` or raise.

        Resolving at load time turns any rename/restructure into a clear
        init-time error instead of an attribute-error mid-forward.
        """
        inner = getattr(language_model, "model", None)
        if inner is None:
            raise RuntimeError(
                "language_model.model attribute missing; mlx_vlm version "
                "drift detected.  Expected the bottom-level LM module that "
                "exposes embed_tokens."
            )
        embed_tokens = getattr(inner, "embed_tokens", None)
        if embed_tokens is None or not callable(embed_tokens):
            raise RuntimeError(
                "language_model.model.embed_tokens missing or not callable; "
                "mlx_vlm version drift detected."
            )
        return embed_tokens

    @classmethod
    def _detect_embeds_kwarg(cls, language_model: Any) -> str:
        """Return the embeds keyword accepted by ``language_model.__call__``.

        Sniffing at load time turns a rename (``inputs_embeds`` →
        ``input_embeddings``) into a clear init-time error rather than
        silently wrong attention.
        """
        try:
            sig = inspect.signature(language_model.__call__)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot inspect language_model.__call__ signature: {exc}"
            ) from exc
        params = set(sig.parameters)
        for candidate in cls._SUPPORTED_EMBEDS_KWARGS:
            if candidate in params:
                return candidate
        raise RuntimeError(
            "language_model.__call__ accepts none of "
            f"{cls._SUPPORTED_EMBEDS_KWARGS}; mlx_vlm version drift detected. "
            f"Got parameters: {sorted(params)}"
        )

    @classmethod
    def _detect_deepstack_kwargs(cls, language_model: Any) -> bool:
        """Return True iff the LM declares both deepstack params explicitly.

        Only the explicit form counts; a LM with just ``**kwargs`` would
        silently absorb the arrays without injecting deepstack residuals.
        """
        try:
            sig = inspect.signature(language_model.__call__)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot inspect language_model.__call__ signature: {exc}"
            ) from exc
        explicit = {
            name
            for name, param in sig.parameters.items()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        return all(kw in explicit for kw in cls._DEEPSTACK_KWARGS)

    def embed_tokens(self, input_ids: mx.array) -> mx.array:
        """Return the language model's input embeddings for ``input_ids``.

        Construct via :meth:`from_loaded_model` so the callable is resolved
        at load time; manually built instances must pass ``embed_tokens_fn``.
        """
        if self._embed_tokens_fn is None:
            raise RuntimeError(
                "embed_tokens_fn not resolved; construct via "
                "Qwen3VLMultimodalAdapter.from_loaded_model so the language "
                "model embedding path is sniffed at load time."
            )
        return self._embed_tokens_fn(input_ids)

    def encode_multimodal(
        self,
        features: list[MultiModalFeatureSpec],
    ) -> list[Qwen3VLVisionEncodeResult]:
        """Run the vision tower per feature; return one result per feature.

        Calls ``vision_tower`` directly rather than ``mlx_vlm.Model.__call__``
        so the LM is not re-entered through the top-level VLM dispatch.
        ``deepstack_visual_embeds`` may be ``None`` when the tower does not
        expose per-layer residuals.
        """
        if self._vision_tower is None:
            raise RuntimeError(
                "vision_tower not loaded; encode_multimodal unavailable. "
                "Construct via Qwen3VLMultimodalAdapter.from_loaded_model."
            )

        # Match the cast that mlx_vlm.Model.get_input_embeddings performs
        # before invoking the tower: vLLM's multimodal pipeline supplies
        # pixel_values in fp32, but the loaded MLX vision tower may carry
        # fp16/bf16 weights, so we align dtypes once per encode batch.
        target_dtype = self._vision_tower.patch_embed.proj.weight.dtype

        outputs: list[Qwen3VLVisionEncodeResult] = []
        for feature in features:
            if feature.modality != "image":
                raise ValueError(
                    "encode_multimodal only supports image features; got "
                    f"modality={feature.modality!r}"
                )
            if feature.data is None:
                raise ValueError("feature.data is required for vision encoding")

            pixel_values = self._as_mlx(
                self._feature_value(feature.data, "pixel_values")
            ).astype(target_dtype)
            # vLLM delivers per-feature grid_thw as a 1D ``(3,)`` row, but
            # the mlx_vlm vision tower indexes ``grid_thw[i, 1:]`` and reads
            # ``grid_thw.shape[0]``; reshape back to ``(1, 3)``.
            image_grid_thw = self._as_mlx(
                self._feature_value(feature.data, "image_grid_thw")
            ).reshape(1, 3)

            hidden_states, deepstack_visual_embeds = self._vision_tower(
                pixel_values,
                image_grid_thw,
            )
            outputs.append(
                Qwen3VLVisionEncodeResult(
                    hidden_states=hidden_states,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                )
            )

        return outputs

    def call_lm(
        self,
        input_ids: mx.array,
        inputs_embeds: mx.array,
        cache: list[Any],
        position_ids: mx.array,
        *,
        visual_pos_masks: Any | None = None,
        deepstack_visual_embeds: Any | None = None,
    ) -> Any:
        """Invoke ``language_model`` with runner-built embeds and positions.

        Calls ``language_model`` directly rather than through the top-level
        ``mlx_vlm.Model.__call__``.  Deepstack kwargs are forwarded when
        the LM declares both as explicit parameters.  Receiving non-None
        ``deepstack_visual_embeds`` on a LM without those parameters is a
        mlx-vlm signature mismatch and raises rather than silently running
        with incomplete vision residuals.
        """
        if self._language_model is None:
            raise RuntimeError(
                "language_model not loaded; call_lm unavailable. "
                "Construct via Qwen3VLMultimodalAdapter.from_loaded_model."
            )
        if self._embeds_kwarg is None:
            raise RuntimeError(
                "embeds_kwarg not detected; construct via "
                "Qwen3VLMultimodalAdapter.from_loaded_model so the language "
                "model signature is sniffed at load time."
            )
        extra_kwargs: dict[str, Any] = {}
        if self._supports_deepstack:
            extra_kwargs["visual_pos_masks"] = visual_pos_masks
            extra_kwargs["deepstack_visual_embeds"] = deepstack_visual_embeds
        elif deepstack_visual_embeds is not None:
            raise RuntimeError(
                "deepstack_visual_embeds were produced by the vision tower "
                "but language_model.__call__ does not declare "
                f"{self._DEEPSTACK_KWARGS} as explicit parameters; mlx-vlm "
                "signature mismatch.  Refusing to drop deepstack residuals "
                "silently."
            )
        return self._language_model(
            input_ids,
            cache=cache,
            position_ids=position_ids,
            **{self._embeds_kwarg: inputs_embeds},
            **extra_kwargs,
        )

    @staticmethod
    def _as_mlx(value: Any) -> Any:
        """Return ``value`` as an MLX array, converting from torch when needed.

        Upstream vLLM's multimodal preprocessor stages ``MultiModalKwargsItem``
        fields as torch tensors; vision_tower expects ``mx.array``.
        """
        if isinstance(value, torch.Tensor):
            return torch_to_mlx(value)
        return value

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[mx.array, int]:
        """Return ``((3, 1, seq_len) int32 positions, mrope_position_delta)``.

        Delegates to upstream vLLM's Qwen3-VL M-RoPE helper, materialises
        the batch axis, and returns an MLX array shaped for ``call_lm``.
        """
        if not input_tokens:
            return mx.zeros((3, 1, 0), dtype=mx.int32), 0

        self._validate_image_features(mm_features)

        torch_positions, mrope_position_delta = (
            self._qwen3_vl_cls()._get_mrope_input_positions(
                input_tokens=input_tokens,
                mm_features=mm_features,
                config=self._image_config(),
            )
        )
        llm_positions = torch_positions.cpu().numpy()
        positions = mx.array(llm_positions, dtype=mx.int32)
        return positions[:, None, :], int(mrope_position_delta)

    def _validate_image_features(
        self,
        features: list[MultiModalFeatureSpec],
    ) -> None:
        """Validate the image-only feature shape accepted by the adapter."""
        for feature in sorted(features, key=lambda feature: feature.mm_position.offset):
            modality = feature.modality
            if modality == "video":
                raise NotImplementedError(
                    "Video multimodal features are not yet supported."
                )
            if modality != "image":
                raise ValueError(f"Unsupported modality: {modality}")
            if feature.data is None:
                raise ValueError(
                    "Image feature data is required to read image_grid_thw."
                )

            t, h, w = self._grid_thw(feature.data, "image_grid_thw")
            if t != 1:
                raise ValueError(f"Multi-frame images are not yet supported, got t={t}")

            llm_grid_h = h // self._spatial_merge_size
            llm_grid_w = w // self._spatial_merge_size
            num_grid_tokens = llm_grid_h * llm_grid_w
            num_embeds = feature.mm_position.get_num_embeds()
            if num_embeds != num_grid_tokens:
                raise ValueError(
                    "image_grid_thw implies "
                    f"{num_grid_tokens} multimodal embeddings, got "
                    f"mm_position.get_num_embeds()={num_embeds}"
                )

    def _image_config(self) -> SimpleNamespace:
        """Build the minimal config read by Qwen3-VL's image-only M-RoPE path."""
        # The token-id fields are only read for video; keep sentinels here so the
        # static upstream helper can be reused without constructing a full model.
        return SimpleNamespace(
            video_token_id=-1,
            vision_start_token_id=-1,
            vision_end_token_id=-1,
            vision_config=SimpleNamespace(
                spatial_merge_size=self._spatial_merge_size,
            ),
        )

    @staticmethod
    @cache
    def _qwen3_vl_cls():
        from vllm.model_executor.models.qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

        return Qwen3VLForConditionalGeneration

    @classmethod
    def _grid_thw(cls, data: MultiModalKwargsItem, key: str) -> tuple[int, int, int]:
        values = cls._feature_value(data, key).tolist()
        if len(values) != 3:
            raise ValueError(f"{key} must contain exactly 3 values, got {values}.")
        t, h, w = values
        return int(t), int(h), int(w)

    @staticmethod
    def _feature_value(data: MultiModalKwargsItem, key: str) -> Any:
        return data[key].data
