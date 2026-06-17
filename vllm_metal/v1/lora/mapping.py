# SPDX-License-Identifier: Apache-2.0
"""Per-step LoRA routing — mirrors ``vllm.lora.layers.LoRAMapping``."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoRAMapping:
    index_mapping: tuple[int, ...]  # one per token (0 = no LoRA)
    prompt_mapping: tuple[int, ...]  # one per request
    is_prefill: bool = False


@dataclass
class LoRAMappingBuilder:
    _idx: list[int] = field(default_factory=list)
    _prompt: list[int] = field(default_factory=list)
    _is_prefill: bool = False

    def add_request(self, lora_id: int | None, num_tokens: int) -> None:
        token_id = 0 if lora_id is None else int(lora_id)
        self._idx += [token_id] * num_tokens
        self._prompt.append(token_id)
        self._is_prefill |= num_tokens > 1

    def build(self) -> LoRAMapping:
        return LoRAMapping(tuple(self._idx), tuple(self._prompt), self._is_prefill)

    def is_empty(self) -> bool:
        return not self._idx
