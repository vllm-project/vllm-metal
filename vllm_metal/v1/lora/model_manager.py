# SPDX-License-Identifier: Apache-2.0
"""Slot-table manager"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from vllm.lora.layers import LoRAMapping
from vllm.lora.utils import is_in_target_modules
from vllm.utils.cache import LRUCache

from .layers import (
    MLXLinearWithLoRA,
    MLXQuantizedLinearWithLoRA,
    can_wrap,
    can_wrap_qlora,
)
from .peft_loader import LoadedLoRA, LoRALayerWeightsMLX
from .punica_wrapper import EncoderPunicaWrapperMLX, PunicaWrapperMLX

if TYPE_CHECKING:
    from vllm.config.lora import LoRAConfig

logger = logging.getLogger(__name__)

_WrappedModule = MLXLinearWithLoRA | MLXQuantizedLinearWithLoRA
_PreparedLoRAWeights = tuple[mx.array, mx.array, int]
_PreparedModuleUpdate = tuple[_WrappedModule, _PreparedLoRAWeights | None]


class _PreparedAdapterUpdate(NamedTuple):
    module_updates: list[_PreparedModuleUpdate]
    loaded_modules: int


@dataclass(frozen=True)
class EncoderLoRATarget:
    """Multimodal encoder (tower and connector) that LoRA may adapt.

    ``prefix`` names the encoder's modules the way adapter weights do after
    the model's weights mapper, e.g. ``visual.`` for ``visual.blocks.0.attn.qkv``.
    """

    module: nn.Module
    prefix: str


class MLXLoRAModelManager:
    def __init__(
        self,
        model: nn.Module,
        lora_config: LoRAConfig,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        dtype: mx.Dtype,
        encoder: EncoderLoRATarget | None = None,
    ):
        self.model, self.lora_config = model, lora_config
        self.encoder = encoder
        self.max_num_seqs, self.max_num_batched_tokens = (
            max_num_seqs,
            max_num_batched_tokens,
        )
        self.dtype = dtype

        if (
            self.lora_config.max_cpu_loras is not None
            and self.lora_config.max_cpu_loras < self.lora_slots
        ):
            raise ValueError(
                "LoRAConfig.max_cpu_loras must be greater than or equal to "
                "LoRAConfig.max_loras. Metal uses max_loras for resident "
                "execution slots and max_cpu_loras for registered adapter "
                "capacity."
            )

        self._registered: LRUCache[int, LoadedLoRA] = LRUCache(self.capacity)
        self._active: LRUCache[int, None] = LRUCache(self.lora_slots)
        self._pinned: set[int] = set()
        self.lora_index_to_id: list[int | None] = [None] * self.lora_slots

        self.modules: dict[str, _WrappedModule] = {}
        self.punica_wrapper = PunicaWrapperMLX(
            max_num_batched_tokens, max_num_seqs, self.lora_slots
        )
        self.encoder_punica_wrapper = EncoderPunicaWrapperMLX()
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
        roots: list[tuple[nn.Module, str, PunicaWrapperMLX | EncoderPunicaWrapperMLX]]
        roots = [(self.model, "", self.punica_wrapper)]
        if self.encoder is not None:
            roots.append(
                (self.encoder.module, self.encoder.prefix, self.encoder_punica_wrapper)
            )
        wrapped: list[_WrappedModule] = []
        for root, prefix, punica_wrapper in roots:
            repls = self._wrappable_modules(root, prefix)
            for name, w in repls:
                w.set_mapping(punica_wrapper)
                self.modules[prefix + name] = w
                wrapped.append(w)
            if repls:
                root.update_modules(tree_unflatten(repls))
        if not wrapped:
            raise RuntimeError(
                "MLXLoRAModelManager found no LoRA target modules to wrap. "
                "LoRAConfig.target_modules may exclude every wrappable module, "
                "or the selected leaves may not expose a supported linear "
                "contract for LoRA/QLoRA wrapping."
            )
        num_plain = sum(1 for w in wrapped if isinstance(w, MLXLinearWithLoRA))
        logger.info(
            "MLXLoRAModelManager wrapped %d modules (%d plain, %d quantized, "
            "%d in the multimodal encoder).",
            len(wrapped),
            num_plain,
            len(wrapped) - num_plain,
            sum(1 for name in self.modules if self._is_encoder_module(name)),
        )

    def _wrappable_modules(
        self, root: nn.Module, prefix: str
    ) -> list[tuple[str, _WrappedModule]]:
        targets = self.lora_config.target_modules
        repls: list[tuple[str, _WrappedModule]] = []
        for name, m in root.named_modules():
            if not is_in_target_modules(prefix + name, targets):
                continue
            if can_wrap(m):
                wrapper_cls: type[_WrappedModule] = MLXLinearWithLoRA
            elif can_wrap_qlora(m):
                wrapper_cls = MLXQuantizedLinearWithLoRA
            else:
                continue
            repls.append(
                (
                    name,
                    wrapper_cls(
                        m,
                        self.lora_slots,
                        self.lora_config.max_lora_rank,
                        self.dtype,
                    ),
                )
            )
        return repls

    def _is_encoder_module(self, name: str) -> bool:
        return self.encoder is not None and name.startswith(self.encoder.prefix)

    def set_encoder_adapter(self, lora_id: int | None) -> None:
        """Route the next encoder calls through ``lora_id`` (``None``: base)."""
        if lora_id is None:
            self.encoder_punica_wrapper.set_slot(None)
            return
        if self.encoder is None:
            raise RuntimeError(
                "Encoder LoRA routing requested but no multimodal encoder "
                "was wrapped; pass --enable-tower-connector-lora."
            )
        slot = self._slot_for_adapter(lora_id)
        if slot is None:
            raise ValueError(
                f"LoRA adapter {lora_id} is not active in a LoRA slot; "
                "activate it before encoding with it."
            )
        self.encoder_punica_wrapper.set_slot(slot)

    def add_adapter(self, adapter: LoadedLoRA) -> bool:
        if adapter.lora_id in self._registered:
            self._registered.touch(adapter.lora_id)
            return False

        # Validate before eviction so a malformed new adapter cannot destroy a
        # working cache entry.
        self._prepare_adapter_update(adapter, 0)

        if len(self._registered) >= self.capacity:
            evicted_id, _ = self._registered.popitem()
            self.deactivate_adapter(evicted_id)
        self._registered[adapter.lora_id] = adapter
        return True

    def replace_adapter(self, adapter: LoadedLoRA) -> None:
        lora_id = adapter.lora_id
        if lora_id not in self._registered:
            raise ValueError(f"LoRA adapter {lora_id} is not registered")

        # Check capacity and validate weights before replacing the cached adapter.
        slot = self._slot_for_adapter(lora_id)
        if slot is None:
            slot = self._next_activation_slot()
        prepared = self._prepare_adapter_update(adapter, slot)

        self._registered[lora_id] = adapter
        self._registered.touch(lora_id)
        if self._slot_for_adapter(lora_id) is None:
            self._activate_prepared(lora_id, slot, prepared)
        else:
            self._commit_adapter_update(lora_id, slot, prepared)
            self._active.touch(lora_id)

    def remove_adapter(self, lora_id: int) -> bool:
        self._pinned.discard(lora_id)
        self.deactivate_adapter(lora_id)
        return self._registered.pop(lora_id, None) is not None

    def remove_all_adapters(self) -> None:
        for lid in [lid for lid in self.lora_index_to_id if lid is not None]:
            self.deactivate_adapter(lid)
        self._registered.clear()
        self._active.clear()
        self._pinned.clear()

    def pin_adapter(self, lora_id: int) -> bool:
        """Pin an adapter in both the registered cache and resident slot cache."""
        if lora_id not in self._registered:
            raise ValueError(f"Pinning failed. LoRA {lora_id} is not registered.")
        if lora_id not in self._active:
            self.activate_adapter(lora_id)
        self._pinned.add(lora_id)
        self._registered.pin(lora_id)
        self._active.pin(lora_id)
        return True

    def is_pinned(self, lora_id: int) -> bool:
        return lora_id in self._pinned

    def list_adapters(self) -> set[int]:
        return set(self._registered)

    def activate_adapter(self, lora_id: int) -> bool:
        if lora_id not in self._registered:
            raise ValueError(f"LoRA adapter {lora_id} is not registered")
        self._registered.touch(lora_id)
        if self._slot_for_adapter(lora_id) is not None:
            self._active.touch(lora_id)
            return False
        adapter = self._registered[lora_id]
        slot = self._next_activation_slot()
        prepared = self._prepare_adapter_update(adapter, slot)
        self._activate_prepared(lora_id, slot, prepared)
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
        self._active.pop(lora_id, None)
        self.lora_index_to_id[slot] = None
        if self.encoder_punica_wrapper.slot == slot:
            self.encoder_punica_wrapper.set_slot(None)
        for w in self.modules.values():
            w.reset_lora(slot)
        self._materialize_slots()
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

    def _next_activation_slot(self) -> int:
        """Return a free slot or the slot owned by the LRU unpinned adapter."""
        slot = next(
            (i for i, lora_id in enumerate(self.lora_index_to_id) if lora_id is None),
            None,
        )
        if slot is not None:
            return slot
        evicted_id = next(
            (
                lora_id
                for lora_id in self._active.order
                if lora_id not in self._active.pinned_items
            ),
            None,
        )
        if evicted_id is None:
            raise RuntimeError(
                "All resident LoRA slots are pinned; cannot activate another adapter."
            )
        evicted_slot = self._slot_for_adapter(evicted_id)
        if evicted_slot is None:
            raise RuntimeError(
                f"Resident LoRA cache is inconsistent: adapter {evicted_id} has no slot."
            )
        return evicted_slot

    def _activate_prepared(
        self,
        lora_id: int,
        slot: int,
        prepared: _PreparedAdapterUpdate,
    ) -> None:
        resident_id = self.lora_index_to_id[slot]
        if resident_id is not None:
            self.deactivate_adapter(resident_id)
        self._commit_adapter_update(lora_id, slot, prepared)
        self._active[lora_id] = None
        if lora_id in self._pinned:
            self._active.pin(lora_id)

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
            lora_a, lora_b = module.prepare_lora_weights(
                slot,
                weights.lora_a,
                weights.lora_b * weights.scaling,
            )
            updates.append(
                (
                    module,
                    (lora_a, lora_b, int(weights.lora_a.shape[0])),
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
            lora_a, lora_b, rank = weights
            module.set_prepared_lora(slot, lora_a, lora_b, rank=rank)
        self._materialize_slots()
        self.lora_index_to_id[slot] = lora_id
        self._last_mapping = None

    def _materialize_slots(self) -> None:
        """Schedule pending slot writes for evaluation now.

        A slot write is a lazy MLX update of the module's stacked weights. A
        wrapped module that does not run before its next write, such as the
        multimodal encoder during text-only steps, would chain every write
        onto the previous one and keep the buffers each write captured alive.
        Under adapter churn that grows the process's live Metal buffer count
        until allocation fails. ``async_eval`` resolves the chain without
        waiting for an in-flight forward.
        """
        mx.async_eval(
            [
                stacked
                for module in self.modules.values()
                for stacked in (module.lora_a_stacked, module.lora_b_stacked)
            ]
        )


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
