# SPDX-License-Identifier: Apache-2.0
"""Slot-table manager"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from vllm.lora.layers import LoRAMapping
from vllm.lora.utils import is_in_target_modules

from .layers import (
    MLXLinearWithLoRA,
    MLXQuantizedLinearWithLoRA,
    can_wrap,
    can_wrap_qlora,
)
from .peft_loader import LoadedLoRA, LoRALayerWeightsMLX
from .punica_wrapper import PunicaWrapperMLX

if TYPE_CHECKING:
    from vllm.config.lora import LoRAConfig

logger = logging.getLogger(__name__)

_WrappedModule = MLXLinearWithLoRA | MLXQuantizedLinearWithLoRA
_PreparedLoRAWeights = tuple[mx.array, mx.array]
_PreparedModuleUpdate = tuple[_WrappedModule, _PreparedLoRAWeights | None]


class _PreparedAdapterUpdate(NamedTuple):
    module_updates: list[_PreparedModuleUpdate]
    loaded_modules: int


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
        self.lora_index_to_id: list[int | None] = [None] * self.lora_slots

        self.modules: dict[str, _WrappedModule] = {}
        self.punica_wrapper = PunicaWrapperMLX(
            max_num_batched_tokens, max_num_seqs, self.lora_slots
        )
        self._last_mapping: LoRAMapping | None = None
        self._wrap_target_modules()

    @property
    def lora_slots(self) -> int:
        return self.lora_config.max_loras

    @property
    def capacity(self) -> int:
        return self.lora_config.max_cpu_loras or self.lora_slots

    def __len__(self) -> int:
        return len(self._registered)

    def _wrap_target_modules(self) -> None:
        targets = self.lora_config.target_modules
        repls: list[tuple[str, _WrappedModule]] = []
        for name, m in self.model.named_modules():
            if not is_in_target_modules(name, targets):
                continue
            if can_wrap(m):
                repls.append(
                    (
                        name,
                        MLXLinearWithLoRA(
                            m,
                            self.lora_slots,
                            self.lora_config.max_lora_rank,
                            self.dtype,
                        ),
                    )
                )
            elif can_wrap_qlora(m):
                repls.append(
                    (
                        name,
                        MLXQuantizedLinearWithLoRA(
                            m,
                            self.lora_slots,
                            self.lora_config.max_lora_rank,
                            self.dtype,
                        ),
                    )
                )
        if not repls:
            raise RuntimeError(
                "MLXLoRAModelManager found no LoRA target modules to wrap. "
                "LoRAConfig.target_modules may exclude every wrappable module, "
                "or the selected leaves may not expose a supported linear "
                "contract for LoRA/QLoRA wrapping."
            )
        for name, w in repls:
            w.set_mapping(self.punica_wrapper)
            self.modules[name] = w
        self.model.update_modules(tree_unflatten(repls))
        logger.info(
            "MLXLoRAModelManager wrapped %d modules (%d plain, %d quantized).",
            len(repls),
            sum(1 for _, w in repls if isinstance(w, MLXLinearWithLoRA)),
            sum(1 for _, w in repls if isinstance(w, MLXQuantizedLinearWithLoRA)),
        )

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
        if lora_id not in self._registered:
            raise ValueError(f"LoRA adapter {lora_id} is not registered")

        slot = self._slot_for_adapter(lora_id)
        if slot is None:
            slot = self._next_free_slot()
        prepared = self._prepare_adapter_update(adapter, slot)

        self._registered[lora_id] = adapter
        self._commit_adapter_update(lora_id, slot, prepared)

    def remove_adapter(self, lora_id: int) -> bool:
        self.deactivate_adapter(lora_id)
        return self._registered.pop(lora_id, None) is not None

    def remove_all_adapters(self) -> None:
        for lid in [lid for lid in self.lora_index_to_id if lid is not None]:
            self.deactivate_adapter(lid)
        self._registered.clear()

    def pin_adapter(self, lora_id: int) -> bool:
        """Acknowledge upstream pin_lora; Metal has no LoRA cache tier to pin."""
        return lora_id in self._registered

    def list_adapters(self) -> set[int]:
        return set(self._registered)

    def activate_adapter(self, lora_id: int) -> bool:
        if self._slot_for_adapter(lora_id) is not None:
            return False
        if lora_id not in self._registered:
            raise ValueError(f"LoRA adapter {lora_id} is not registered")
        adapter = self._registered[lora_id]
        slot = self._next_free_slot()
        prepared = self._prepare_adapter_update(adapter, slot)
        self._commit_adapter_update(lora_id, slot, prepared)
        logger.info(
            "Activated LoRA %d in slot %d (%d modules)",
            lora_id,
            slot,
            prepared.loaded_modules,
        )
        return True

    def deactivate_adapter(self, lora_id: int) -> bool:
        slot = self._slot_for_adapter(lora_id)
        if slot is None:
            return False
        self.lora_index_to_id[slot] = None
        for w in self.modules.values():
            w.reset_lora(slot)
        self._last_mapping = None
        return True

    def set_adapter_mapping(self, mapping: LoRAMapping) -> None:
        if mapping == self._last_mapping:
            return
        requested = {
            lora_id
            for lora_id in (*mapping.index_mapping, *mapping.prompt_mapping)
            if lora_id != 0
        }
        active = {lora_id for lora_id in self.lora_index_to_id if lora_id is not None}
        missing = sorted(requested - active)
        if missing:
            raise ValueError(
                "LoRA mapping references adapters that are not active in "
                f"LoRA slots: {missing}; slot table: {self.lora_index_to_id}. "
                "Use 0 for no-LoRA tokens."
            )
        self.punica_wrapper.update_metadata(mapping, self.lora_index_to_id)
        self._last_mapping = mapping

    def _slot_for_adapter(self, lora_id: int) -> int | None:
        try:
            return self.lora_index_to_id.index(lora_id)
        except ValueError:
            return None

    def _next_free_slot(self) -> int:
        try:
            return next(i for i, sid in enumerate(self.lora_index_to_id) if sid is None)
        except StopIteration as exc:
            raise ValueError(
                f"No free LoRA slots; raise --max-loras (current: {self.lora_slots})."
            ) from exc

    def _prepare_adapter_update(
        self, adapter: LoadedLoRA, slot: int
    ) -> _PreparedAdapterUpdate:
        updates: list[_PreparedModuleUpdate] = []
        loaded = 0
        for name, module in self.modules.items():
            weights = _lookup_weights_for_module(adapter, name)
            if weights is None:
                updates.append((module, None))
                continue
            loaded += 1
            updates.append(
                (
                    module,
                    module.prepare_lora_weights(
                        slot,
                        weights.lora_a,
                        weights.lora_b * weights.scaling,
                    ),
                )
            )
        if loaded == 0:
            raise ValueError(
                f"LoRA adapter {adapter.lora_id} matched 0 wrapped modules "
                f"(wrapped: {sorted(self.modules)}). The adapter targets "
                "modules this model does not expose under LoRA; check "
                "target_modules / the adapter's base model."
            )
        return _PreparedAdapterUpdate(updates, loaded)

    def _commit_adapter_update(
        self,
        lora_id: int,
        slot: int,
        prepared: _PreparedAdapterUpdate,
    ) -> None:
        for module, weights in prepared.module_updates:
            if weights is None:
                module.reset_lora(slot)
                continue
            lora_a, lora_b = weights
            module.set_prepared_lora(slot, lora_a, lora_b)
        self.lora_index_to_id[slot] = lora_id
        self._last_mapping = None


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
