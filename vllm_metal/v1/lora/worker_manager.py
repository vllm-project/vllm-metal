# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mlx.core as mx

from .mapping import LoRAMapping
from .model_manager import MLXLoRAModelManager
from .peft_loader import load_peft_adapter

if TYPE_CHECKING:
    import mlx.nn as nn
    from vllm.config.lora import LoRAConfig
    from vllm.lora.request import LoRARequest

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
    ):
        self.lora_config = lora_config
        self.max_position_embeddings = max_position_embeddings
        self._mm = MLXLoRAModelManager(
            model, lora_config, max_num_seqs, max_num_batched_tokens, dtype
        )

    def add_adapter(self, lora_request: LoRARequest) -> bool:
        from vllm.lora.utils import get_adapter_absolute_path

        if lora_request.lora_int_id in self._mm.list_adapters():
            return False
        adapter = load_peft_adapter(
            get_adapter_absolute_path(lora_request.lora_path),
            lora_id=lora_request.lora_int_id,
            max_position_embeddings=self.max_position_embeddings,
            lora_config=self.lora_config,
        )
        if not self._mm.add_adapter(adapter):
            return False
        try:
            self._mm.activate_adapter(lora_request.lora_int_id)
        except ValueError:
            # Slot table full — unwind so add+activate stay atomic.
            self._mm.remove_adapter(lora_request.lora_int_id)
            raise
        return True

    def remove_adapter(self, lora_id: int) -> bool:
        return self._mm.remove_adapter(lora_id)

    def pin_adapter(self, lora_id: int) -> bool:
        return self._mm.pin_adapter(lora_id)

    def list_adapters(self) -> set[int]:
        return self._mm.list_adapters()

    def set_active_adapters(
        self, lora_requests: set[LoRARequest], mapping: LoRAMapping | None
    ) -> None:
        if lora_requests:
            self._apply({r.lora_int_id for r in lora_requests})
        if mapping is not None:
            self._mm.set_adapter_mapping(mapping)

    def _apply(self, requested: set[int]) -> None:
        if len(requested) > self._mm.lora_slots:
            raise RuntimeError(
                f"Number of distinct LoRAs in batch ({len(requested)}) exceeds "
                f"--max-loras ({self._mm.lora_slots})."
            )
        for lid in requested:
            if lid not in self._mm.list_adapters():
                raise RuntimeError(
                    f"LoRA {lid} requested but not loaded — engine should call "
                    "add_lora before scheduling it."
                )
        active_now = {a for a in self._mm.lora_index_to_id if a is not None}
        for lid in active_now - requested:
            self._mm.deactivate_adapter(lid)
        for lid in requested - active_now:
            self._mm.activate_adapter(lid)
