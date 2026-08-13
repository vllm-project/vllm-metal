# SPDX-License-Identifier: Apache-2.0
"""Pooling request and model-config validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.v1.core.sched.output import NewRequestData

from vllm_metal.v1.pooling.contract import (
    CLASSIFY_TASK,
    EMBED_TASK,
    PoolingBackend,
)

LAST_POOLING = (None, "LAST")
EMBED_CONFIG_TASKS: tuple[PoolingTask | None, ...] = (None, EMBED_TASK)
SUPPORTED_POOLER_TASKS: tuple[PoolingTask | None, ...] = (
    None,
    EMBED_TASK,
    CLASSIFY_TASK,
)


@dataclass(frozen=True, slots=True)
class PoolingConfigView:
    """Small view over vLLM/HF pooling config fields used by Metal."""

    model_config: Any

    @property
    def label(self) -> str:
        served_model_name = self.model_config.served_model_name
        if isinstance(served_model_name, (list, tuple)):
            served_model_name = served_model_name[0] if served_model_name else None
        return str(served_model_name or self.model_config.model)

    @property
    def hf_config(self) -> Any:
        return self.model_config.hf_config

    @property
    def pooler_config(self) -> Any:
        return self.model_config.pooler_config

    @property
    def runner_type(self) -> str | None:
        runner_type = self.model_config.runner_type
        return str(runner_type) if runner_type is not None else None

    @property
    def task(self) -> PoolingTask | None:
        return self.pooler_config.task

    @property
    def architectures(self) -> tuple[str, ...]:
        values = getattr(self.hf_config, "architectures", None)
        if not isinstance(values, (list, tuple)):
            return ()
        return tuple(str(value) for value in values)

    @property
    def has_multimodal_config(self) -> bool:
        return self.model_config.multimodal_config is not None

    @property
    def is_text_only(self) -> bool:
        return not self.has_multimodal_config

    @property
    def unsupported_sequence_pooling_type(self) -> str | None:
        for pooling_type in self.sequence_pooling_types:
            if pooling_type not in LAST_POOLING:
                return pooling_type
        return None

    @property
    def uses_last_pooling(self) -> bool:
        return self.unsupported_sequence_pooling_type is None

    @property
    def sequence_pooling_types(self) -> tuple[str | None, str | None]:
        seq_pooling_type = self.pooler_config.seq_pooling_type
        pooling_type = self.pooler_config.pooling_type
        return (
            str(seq_pooling_type) if seq_pooling_type is not None else None,
            str(pooling_type) if pooling_type is not None else None,
        )

    @property
    def sequence_pooling_type(self) -> str | None:
        for pooling_type in self.sequence_pooling_types:
            if pooling_type is not None:
                return pooling_type
        return None

    @property
    def embed_activation_allowed(self) -> bool:
        return self.pooler_config.use_activation is not False

    @property
    def chunked_processing_enabled(self) -> bool:
        return bool(self.pooler_config.enable_chunked_processing)

    @property
    def has_embedding_dimension_override(self) -> bool:
        return self.pooler_config.dimensions is not None

    @property
    def supports_decoder_embed_config(self) -> bool:
        return (
            self.is_text_only
            and self.task in EMBED_CONFIG_TASKS
            and self.uses_last_pooling
            and self.embed_activation_allowed
            and not self.chunked_processing_enabled
        )

    def reject_unsupported_pooler_config(self) -> None:
        if self.task not in SUPPORTED_POOLER_TASKS:
            raise NotImplementedError(
                "Metal pooling supports only pooler_config.task unset, 'embed', "
                f"or 'classify'; got {self.task!r} for model={self.label}."
            )

        sequence_pooling_type = self.unsupported_sequence_pooling_type
        if sequence_pooling_type is not None:
            raise NotImplementedError(
                "Metal pooling currently supports only LAST sequence pooling; "
                f"got {sequence_pooling_type!r} for model={self.label}."
            )
        if self.chunked_processing_enabled:
            raise NotImplementedError(
                "Metal pooling does not support "
                "pooler_config.enable_chunked_processing=True with LAST pooling; "
                f"model={self.label}."
            )

    def unsupported_pooling_option(self, pooling_params: PoolingParams) -> str | None:
        if pooling_params.late_interaction_params is not None:
            return "late-interaction parameters"
        if pooling_params.requires_token_ids:
            return "token-level ALL pooling outputs"
        if pooling_params.step_tag_id is not None:
            return "STEP pooling parameters"
        if pooling_params.returned_token_ids is not None:
            return "returned_token_ids"
        if pooling_params.extra_kwargs:
            return "extra pooling kwargs"
        if (
            pooling_params.task != CLASSIFY_TASK
            and pooling_params.use_activation is False
        ):
            return "use_activation=False"
        if (
            pooling_params.dimensions is not None
            or self.has_embedding_dimension_override
        ):
            return "embedding-dimension truncation"
        return None


def validate_pooling_request(
    new_req: NewRequestData,
    model_config: Any,
    pooling_backend: PoolingBackend | None,
    paged_attention_enabled: bool,
) -> None:
    pooling_params = new_req.pooling_params
    if pooling_params is None:
        return

    if pooling_backend is None:
        raise RuntimeError("Metal pooling backend is not installed.")

    config = PoolingConfigView(model_config)
    if config.runner_type != "pooling":
        raise NotImplementedError(
            "Metal pooling requires runner_type='pooling'; got "
            f"{config.runner_type!r} for model={config.label}."
        )
    unsupported_option = config.unsupported_pooling_option(pooling_params)
    if unsupported_option is not None:
        raise NotImplementedError(
            f"Metal pooling does not support {unsupported_option} "
            f"for model={config.label}."
        )
    pooling_backend.validate_params(pooling_params)
    if new_req.mm_features:
        raise NotImplementedError(
            "Multimodal pooling inputs are not supported on Metal yet."
        )
    if new_req.prompt_embeds is not None:
        raise NotImplementedError(
            "Prompt-embedding pooling inputs are not supported on Metal yet."
        )
    if (
        pooling_backend.capabilities.requires_paged_attention
        and not paged_attention_enabled
    ):
        raise NotImplementedError(
            "Metal pooling currently requires paged attention; "
            "set VLLM_METAL_USE_PAGED_ATTENTION=1."
        )
    if not (new_req.prompt_token_ids or []):
        raise ValueError(
            f"Metal pooling requires prompt_token_ids for request {new_req.req_id!r}."
        )
