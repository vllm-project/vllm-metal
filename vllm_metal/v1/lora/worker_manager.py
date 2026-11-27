# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mlx.core as mx
from vllm.lora.layers import LoRAMapping
from vllm.lora.utils import get_adapter_absolute_path

from .model_manager import EncoderLoRATarget, MLXLoRAModelManager
from .peft_loader import LoadedLoRA, default_weights_mapper, load_peft_adapter

if TYPE_CHECKING:
    import mlx.nn as nn
    from vllm.config.lora import LoRAConfig
    from vllm.lora.request import LoRARequest
    from vllm.model_executor.models.utils import WeightsMapper

logger = logging.getLogger(__name__)


class MetalWorkerLoRAManager:
    def __init__(
        self,
        model: nn.Module,
        lora_config: LoRAConfig,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        dtype: mx.Dtype,
        max_position_embeddings: int | None = None,
        encoder: EncoderLoRATarget | None = None,
        weights_mapper: WeightsMapper | None = None,
    ):
        self.lora_config = lora_config
        self.max_position_embeddings = max_position_embeddings
        self.weights_mapper = (
            default_weights_mapper(model) if weights_mapper is None else weights_mapper
        )
        self._mm = MLXLoRAModelManager(
            model,
            lora_config,
            max_num_seqs,
            max_num_batched_tokens,
            dtype,
            encoder=encoder,
        )

    def add_adapter(self, lora_request: LoRARequest) -> bool:
        lora_id = lora_request.lora_int_id
        already_loaded = lora_id in self._mm.list_adapters()
        if already_loaded and not lora_request.load_inplace:
            self._mm.activate_adapter(lora_id)
            return True
        adapter = load_peft_adapter(
            get_adapter_absolute_path(lora_request.lora_path),
            lora_id=lora_id,
            max_position_embeddings=self.max_position_embeddings,
            lora_config=self.lora_config,
            weights_mapper=self.weights_mapper,
        )
        if already_loaded:
            self._mm.replace_adapter(adapter)
            return True
        return self._add_and_activate(adapter)

    def _add_and_activate(self, adapter: LoadedLoRA) -> bool:
        if not self._mm.add_adapter(adapter, require_activation_slot=True):
            return False
        try:
            self._mm.activate_adapter(adapter.lora_id)
        except (RuntimeError, ValueError):
            self._mm.remove_adapter(adapter.lora_id)
            raise
        return True

    def remove_adapter(self, lora_id: int) -> bool:
        return self._mm.remove_adapter(lora_id)

    def pin_adapter(self, lora_id: int) -> bool:
        return self._mm.pin_adapter(lora_id)

    def list_adapters(self) -> set[int]:
        return self._mm.list_adapters()

    def set_encoder_adapter(self, lora_id: int | None) -> None:
        if lora_id is not None:
            self._mm.activate_adapter(lora_id)
        self._mm.set_encoder_adapter(lora_id)

    def set_active_adapters(
        self, lora_requests: set[LoRARequest], mapping: LoRAMapping | None
    ) -> None:
        self._apply(lora_requests)
        if mapping is not None:
            self._mm.set_adapter_mapping(mapping)

    def _apply(self, lora_requests: set[LoRARequest]) -> None:
        requested = {lora_request.lora_int_id for lora_request in lora_requests}
        if len(requested) > self._mm.lora_slots:
            raise RuntimeError(
                f"Number of distinct LoRAs in batch ({len(requested)}) exceeds "
                f"--max-loras ({self._mm.lora_slots})."
            )
        active_now = {a for a in self._mm.lora_index_to_id if a is not None}
        for lora_id in active_now - requested:
            if not self._mm.is_pinned(lora_id):
                self._mm.deactivate_adapter(lora_id)
        for lora_request in sorted(
            lora_requests, key=lambda request: request.lora_int_id
        ):
            lora_id = lora_request.lora_int_id
            if lora_request.load_inplace or lora_id not in self._mm.list_adapters():
                self.add_adapter(lora_request)
            else:
                self._mm.activate_adapter(lora_id)
