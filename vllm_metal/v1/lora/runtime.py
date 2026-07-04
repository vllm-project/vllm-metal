# SPDX-License-Identifier: Apache-2.0
# LoRA state and per-step routing for the Metal runner.

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

import mlx.core as mx
from vllm.lora.layers import LoRAMapping

from .worker_manager import MetalWorkerLoRAManager

if TYPE_CHECKING:
    import mlx.nn as nn
    from vllm.config.lora import LoRAConfig
    from vllm.lora.request import LoRARequest

logger = logging.getLogger(__name__)


class _LoRAMappingBuilder:
    def __init__(self) -> None:
        self._idx: list[int] = []
        self._prompt: list[int] = []
        self._is_prefill = False

    def add_request(self, lora_id: int | None, num_tokens: int) -> None:
        token_id = 0 if lora_id is None else int(lora_id)
        self._idx += [token_id] * num_tokens
        self._prompt.append(token_id)
        self._is_prefill |= num_tokens > 1

    def build(self) -> LoRAMapping:
        return LoRAMapping(tuple(self._idx), tuple(self._prompt), self._is_prefill)

    def is_empty(self) -> bool:
        return not self._idx


class MetalLoRARuntime:
    """Per-runner owner of LoRA bookkeeping and mapping construction."""

    def __init__(self) -> None:
        self._manager: MetalWorkerLoRAManager | None = None
        self._loaded: dict[int, LoRARequest] = {}

    @property
    def enabled(self) -> bool:
        return self._manager is not None

    def setup(
        self,
        *,
        model: nn.Module,
        lora_config: LoRAConfig | None,
        is_stt: bool,
        paged_attention_enabled: bool,
        speculative_decode_enabled: bool,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        dtype: mx.Dtype,
        max_position_embeddings: int | None,
    ) -> None:
        if lora_config is None:
            return
        if is_stt:
            logger.warning(
                "LoRA is not supported for STT models; ignoring --enable-lora"
            )
            return
        if not paged_attention_enabled:
            raise NotImplementedError(
                "LoRA on Metal requires paged attention. The non-paged "
                "(legacy MLX KV cache) path runs multiple separate forwards "
                "per step, which the step-level Punica routing cannot align. "
                "Enable paged attention (VLLM_METAL_USE_PAGED_ATTENTION=1) "
                "or drop --enable-lora."
            )
        if speculative_decode_enabled:
            raise NotImplementedError(
                "LoRA combined with speculative decode is not supported on "
                "Metal yet: speculative decode forwards multiple draft rows "
                "per decode request, which the current per-request LoRA row "
                "contract does not model. Disable speculative decode or drop "
                "--enable-lora."
            )
        self._manager = MetalWorkerLoRAManager(
            model=model,
            lora_config=lora_config,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            dtype=dtype,
            max_position_embeddings=max_position_embeddings,
        )

    def add_adapter(self, lora_request: LoRARequest) -> bool:
        if self._manager is None:
            logger.warning(
                "add_lora called but --enable-lora was not passed; ignoring."
            )
            return False
        added = self._manager.add_adapter(lora_request)
        if added:
            self._loaded[lora_request.lora_int_id] = lora_request
        return added

    def remove_adapter(self, lora_id: int) -> bool:
        if self._manager is None:
            return False
        removed = self._manager.remove_adapter(lora_id)
        if removed:
            self._loaded.pop(lora_id, None)
        return removed

    def pin_adapter(self, lora_id: int) -> bool:
        if self._manager is None:
            return False
        return self._manager.pin_adapter(lora_id)

    def list_adapters(self) -> set[int]:
        if self._manager is None:
            return set()
        return self._manager.list_adapters()

    def prepare_step(self, routing_entries: Iterable[tuple[int | None, int]]) -> None:
        # Push the per-step active set and token routing to the manager.
        if self._manager is None:
            return
        builder = _LoRAMappingBuilder()
        active_requests: set[LoRARequest] = set()
        for lora_id, num_tokens in routing_entries:
            builder.add_request(lora_id, num_tokens)
            if lora_id is None:
                continue
            req = self._loaded.get(lora_id)
            if req is None:
                raise ValueError(
                    f"LoRA id {lora_id} was routed for this step but is not "
                    f"loaded (loaded ids: {sorted(self._loaded)}). The engine "
                    "must call add_lora before scheduling a request that uses "
                    "it."
                )
            active_requests.add(req)
        mapping = builder.build() if not builder.is_empty() else None
        self._manager.set_active_adapters(active_requests, mapping)
