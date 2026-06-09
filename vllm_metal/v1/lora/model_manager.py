# SPDX-License-Identifier: Apache-2.0
"""Slot-table manager"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from vllm.lora.utils import is_in_target_modules

from .layers import MLXLinearWithLoRA, can_wrap
from .mapping import LoRAMapping
from .peft_loader import LoadedLoRA, LoRALayerWeightsMLX
from .punica_wrapper import PunicaWrapperMLX

if TYPE_CHECKING:
    from vllm.config.lora import LoRAConfig

logger = logging.getLogger(__name__)


class MLXLoRAModelManager:
    def __init__(
        self,
        model: nn.Module,
        lora_config: LoRAConfig,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        dtype: mx.Dtype,
    ):
        self.model, self.lora_config = model, lora_config
        self.max_num_seqs, self.max_num_batched_tokens = (
            max_num_seqs,
            max_num_batched_tokens,
        )
        self.dtype = dtype

        self._registered: dict[int, LoadedLoRA] = {}
        self._active: set[int] = set()
        self.lora_index_to_id: list[int | None] = [None] * self.lora_slots

        self.modules: dict[str, MLXLinearWithLoRA] = {}
        self.punica_wrapper = PunicaWrapperMLX(
            max_num_batched_tokens, max_num_seqs, self.lora_slots
        )
        self._last_mapping: LoRAMapping | None = None
        self._wrap_target_modules()

    @property
    def lora_slots(self) -> int:
        return self.lora_config.max_loras

    @property
    def adapter_slots(self) -> int:
        return self.lora_slots

    @property
    def capacity(self) -> int:
        return self.lora_config.max_cpu_loras or self.lora_slots

    def __len__(self) -> int:
        return len(self._registered)

    def _wrap_target_modules(self) -> None:
        targets = self.lora_config.target_modules
        repls = [
            (
                name,
                MLXLinearWithLoRA(
                    m, self.lora_slots, self.lora_config.max_lora_rank, self.dtype
                ),
            )
            for name, m in self.model.named_modules()
            if can_wrap(m) and is_in_target_modules(name, targets)
        ]
        if not repls:
            logger.warning(
                "MLXLoRAModelManager wrapped 0 modules — model has no plain "
                "nn.Linear (quantized models are not supported in v1) or "
                "target_modules excludes everything."
            )
            return
        for name, w in repls:
            w.set_mapping(self.punica_wrapper)
            self.modules[name] = w
        self.model.update_modules(tree_unflatten(repls))
        logger.info("MLXLoRAModelManager wrapped %d Linear modules.", len(repls))

    def add_adapter(self, adapter: LoadedLoRA) -> bool:
        if adapter.lora_id in self._registered:
            return False
        if len(self._registered) >= self.capacity:
            raise RuntimeError(
                f"LoRA capacity {self.capacity} exceeded; raise --max-cpu-loras."
            )
        self._registered[adapter.lora_id] = adapter
        return True

    def replace_adapter(self, adapter: LoadedLoRA) -> None:
        lora_id = adapter.lora_id
        previous = self._registered.get(lora_id)
        if previous is None:
            raise ValueError(f"LoRA adapter {lora_id} is not registered")

        was_active = self.deactivate_adapter(lora_id)
        self._registered[lora_id] = adapter
        try:
            self.activate_adapter(lora_id)
        except Exception:
            # Replacement is transactional: any activation failure keeps the
            # previously loaded adapter visible and active.
            self._registered[lora_id] = previous
            if was_active:
                self.activate_adapter(lora_id)
            raise

    def remove_adapter(self, lora_id: int) -> bool:
        self.deactivate_adapter(lora_id)
        return self._registered.pop(lora_id, None) is not None

    def remove_all_adapters(self) -> None:
        for lid in list(self._active):
            self.deactivate_adapter(lid)
        self._registered.clear()

    def pin_adapter(self, lora_id: int) -> bool:
        return lora_id in self._registered

    def list_adapters(self) -> set[int]:
        return set(self._registered)

    def activate_adapter(self, lora_id: int) -> bool:
        if lora_id in self._active:
            return False
        if lora_id not in self._registered:
            raise ValueError(f"LoRA adapter {lora_id} is not registered")
        adapter = self._registered[lora_id]
        resolved = [
            (w, _lookup_weights_for_module(adapter, name))
            for name, w in self.modules.items()
        ]
        loaded = sum(1 for _, mw in resolved if mw is not None)
        if loaded == 0:
            raise ValueError(
                f"LoRA adapter {lora_id} matched 0 wrapped modules "
                f"(wrapped: {sorted(self.modules)}). The adapter targets "
                "modules this model does not expose under LoRA; check "
                "target_modules / the adapter's base model."
            )
        try:
            slot = next(i for i, sid in enumerate(self.lora_index_to_id) if sid is None)
        except StopIteration as exc:
            raise ValueError(
                f"No free LoRA slots; raise --max-loras (current: {self.lora_slots})."
            ) from exc

        self.lora_index_to_id[slot] = lora_id
        self._active.add(lora_id)
        for w, mw in resolved:
            if mw is None:
                w.reset_lora(slot)
                continue
            # Fold scaling into B once so the kernel stays scale=1.0.
            w.set_lora(slot, mw.lora_a, mw.lora_b * mw.scaling)
        logger.info("Activated LoRA %d in slot %d (%d modules)", lora_id, slot, loaded)
        self._last_mapping = None
        return True

    def deactivate_adapter(self, lora_id: int) -> bool:
        if lora_id not in self._active:
            return False
        try:
            slot = self.lora_index_to_id.index(lora_id)
        except ValueError:
            self._active.discard(lora_id)
            return False
        self.lora_index_to_id[slot] = None
        self._active.discard(lora_id)
        for w in self.modules.values():
            w.reset_lora(slot)
        self._last_mapping = None
        return True

    def set_adapter_mapping(self, mapping: LoRAMapping) -> None:
        if mapping == self._last_mapping:
            return
        self.punica_wrapper.update_metadata(mapping, self.lora_index_to_id)
        self._last_mapping = mapping


def _lookup_weights_for_module(
    adapter: LoadedLoRA, module_name: str
) -> LoRALayerWeightsMLX | None:
    """Match a wrapped module name against the adapter's per-module weights.

    Direct hit first, then a unique trailing-suffix match for adapters trained
    against a different naming prefix (e.g. ``language_model.model.layers...``).
    """
    if (w := adapter.weights.get(module_name)) is not None:
        return w
    suffix = "." + module_name
    matches = [
        (n, w)
        for n, w in adapter.weights.items()
        if n.endswith(suffix) or module_name.endswith("." + n)
    ]
    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        raise ValueError(
            f"LoRA adapter {adapter.lora_id} has ambiguous suffix matches for "
            f"wrapped module {module_name!r}: {sorted(n for n, _ in matches)}. "
            "Rename the adapter weights or narrow target_modules so exactly "
            "one candidate matches."
        )
    return None
