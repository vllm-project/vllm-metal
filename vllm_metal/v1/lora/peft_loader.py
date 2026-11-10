# SPDX-License-Identifier: Apache-2.0
"""PEFT/Unsloth adapter -> MLX weights via upstream ``PEFTHelper``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mlx.core as mx
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.models.utils import WeightsMapper

if TYPE_CHECKING:
    from vllm.config.lora import LoRAConfig

logger = logging.getLogger(__name__)

__all__ = [
    "LoRALayerWeightsMLX",
    "LoadedLoRA",
    "default_weights_mapper",
    "load_peft_adapter",
]


def default_weights_mapper(model: Any) -> WeightsMapper:
    """Land HF ``*ForConditionalGeneration`` adapter names on the wrapped tree.

    PEFT names the text modules ``model.language_model.<x>``. mlx_lm's VLM
    wrappers hold them at ``language_model.model.<x>``; a text sub-model or a
    text-only model holds them at ``model.<x>``.
    """
    inner = "language_model.model." if hasattr(model, "language_model") else "model."
    return WeightsMapper(orig_to_new_prefix={"model.language_model.": inner})


@dataclass(frozen=True)
class LoRALayerWeightsMLX:
    module_name: str
    rank: int
    lora_a: mx.array
    lora_b: mx.array
    scaling: float


@dataclass
class LoadedLoRA:
    lora_id: int
    rank: int
    weights: dict[str, LoRALayerWeightsMLX]


def load_peft_adapter(
    adapter_path: str | Path,
    *,
    lora_id: int,
    max_position_embeddings: int | None = None,
    lora_config: LoRAConfig | None = None,
    weights_mapper: WeightsMapper | None = None,
) -> LoadedLoRA:
    adapter_path = Path(adapter_path)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"LoRA adapter path is not a directory: {adapter_path}")
    safetensors_path = adapter_path / "adapter_model.safetensors"
    if not safetensors_path.is_file():
        raise FileNotFoundError(
            f"Missing adapter_model.safetensors in {adapter_path} "
            "(only PEFT safetensors format is supported on Metal)."
        )

    helper = PEFTHelper.from_local_dir(str(adapter_path), max_position_embeddings)
    if lora_config is not None:
        helper.validate_legal(lora_config)
    raw = cast(dict[str, mx.array], mx.load(str(safetensors_path)))
    # Preserve eager loading: later in-place adapter updates must not change
    # tensors already returned to the adapter manager.
    mx.eval(raw)

    pairs: dict[str, dict[str, mx.array]] = {}
    peft_modules: dict[str, str] = {}
    for name, tensor in raw.items():
        try:
            module, is_a = parse_fine_tuned_lora_name(name, weights_mapper)
            peft_module, _ = parse_fine_tuned_lora_name(name)
        except ValueError:
            logger.debug("Skipping unrecognized LoRA tensor: %s", name)
            continue
        pairs.setdefault(module, {})["lora_a" if is_a else "lora_b"] = tensor
        peft_modules[module] = peft_module

    weights: dict[str, LoRALayerWeightsMLX] = {}
    for module, pair in pairs.items():
        if "lora_a" not in pair or "lora_b" not in pair:
            present = "lora_a" if "lora_a" in pair else "lora_b"
            missing = "lora_b" if present == "lora_a" else "lora_a"
            peft_key = f"lora_{missing[-1].upper()}"  # lora_a -> lora_A
            raise ValueError(
                f"LoRA adapter at {adapter_path}: module {module!r} has "
                f"{present} but no matching {missing} tensor. Either the "
                f"{missing} tensor is absent from adapter_model.safetensors, "
                "or its key does not follow PEFT naming "
                f"('base_model.model.<module>.{peft_key}.weight'). "
                "Both lora_a and lora_b are required per adapted module."
            )
        # PEFT patterns match checkpoint names, before weights mapping.
        _, scaling = helper.get_rank_and_scaling(peft_modules[module])
        weights[module] = LoRALayerWeightsMLX(
            module_name=module,
            rank=int(pair["lora_a"].shape[0]),
            lora_a=pair["lora_a"],
            lora_b=pair["lora_b"],
            scaling=scaling,
        )

    if not weights:
        raise ValueError(
            f"No usable LoRA tensors in {adapter_path} "
            "(expected PEFT keys like 'base_model.model.<...>.lora_A.weight')."
        )
    return LoadedLoRA(lora_id=lora_id, rank=helper.r, weights=weights)
