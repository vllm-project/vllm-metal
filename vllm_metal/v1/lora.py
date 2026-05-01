# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest

logger = logging.getLogger(__name__)

_PEFT_LORA_A_MARKER = ".lora_A."
_PEFT_LORA_B_MARKER = ".lora_B."
_MLX_LORA_A_SUFFIX = ".lora_a"
_MLX_LORA_B_SUFFIX = ".lora_b"


@dataclass(frozen=True)
class LoRAWeights:
    """Weights for one adapted module in MLX LoRA orientation."""

    module_name: str
    lora_a: Any
    lora_b: Any
    rank: int
    scale: float
    dropout: float = 0.0


@dataclass
class LoadedLoRAAdapter:
    """Loaded adapter metadata and tensors."""

    lora_id: int
    name: str
    path: Path
    weights: list[LoRAWeights]
    pinned: bool = False


class MetalLoRAManager:
    """Manage single-active-adapter LoRA execution for MLX models.

    vLLM's CUDA path can batch multiple LoRA IDs by routing each request through
    specialized kernels.  The Metal/MLX path mutates the module graph, so one
    adapter may be active for a forward pass.  The runner enforces that every
    scheduled request in a batch uses the same adapter ID before activating it.
    """

    def __init__(self, runner: Any) -> None:
        self._runner = runner
        self._adapters: dict[int, LoadedLoRAAdapter] = {}
        self._active_lora_id: int | None = None

    @property
    def active_lora_id(self) -> int | None:
        return self._active_lora_id

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load an Unsloth/PEFT or mlx-lm adapter into the registry."""
        lora_id = _request_lora_id(lora_request)
        if lora_id in self._adapters:
            logger.info("LoRA adapter %s is already loaded", lora_id)
            return True

        adapter_path = _request_lora_path(lora_request)
        if adapter_path is None:
            logger.warning("LoRA request %s has no local adapter path", lora_id)
            return False
        if not adapter_path.is_dir():
            logger.warning("LoRA adapter path does not exist: %s", adapter_path)
            return False

        weights = _load_adapter_weights(adapter_path, self._module_names())
        if not weights:
            logger.warning("No compatible LoRA weights found in %s", adapter_path)
            return False

        self._adapters[lora_id] = LoadedLoRAAdapter(
            lora_id=lora_id,
            name=getattr(lora_request, "lora_name", str(lora_id)),
            path=adapter_path,
            weights=weights,
        )
        logger.info(
            "Loaded LoRA adapter %s from %s (%d modules)",
            lora_id,
            adapter_path,
            len(weights),
        )
        return True

    def remove_lora(self, lora_id: int) -> bool:
        """Remove a loaded adapter, deactivating it first if needed."""
        adapter = self._adapters.get(lora_id)
        if adapter is None:
            return False
        if adapter.pinned:
            logger.warning("Cannot remove pinned LoRA adapter %s", lora_id)
            return False
        if self._active_lora_id == lora_id:
            self.activate_lora(None)
        del self._adapters[lora_id]
        return True

    def pin_lora(self, lora_id: int) -> bool:
        adapter = self._adapters.get(lora_id)
        if adapter is None:
            return False
        adapter.pinned = True
        return True

    def list_loras(self) -> set[int]:
        return set(self._adapters)

    def activate_lora(self, lora_id: int | None) -> None:
        """Make the requested adapter active for subsequent model forwards."""
        if lora_id == self._active_lora_id:
            return
        if lora_id is not None and lora_id not in self._adapters:
            raise RuntimeError(f"LoRA adapter {lora_id} is not loaded")

        self._restore_base_modules()
        self._active_lora_id = None

        if lora_id is None:
            return

        adapter = self._adapters[lora_id]
        self._apply_adapter(adapter)
        self._active_lora_id = lora_id

    def validate_uniform_lora(self, lora_ids: set[int | None]) -> int | None:
        """Return the single scheduled LoRA ID, or fail on mixed batches."""
        if len(lora_ids) > 1:
            raise NotImplementedError(
                "vllm-metal currently supports one LoRA adapter per scheduled "
                "forward pass. Avoid mixing base-model and LoRA requests, or "
                "requests with different LoRA IDs, in the same batch."
            )
        lora_id = next(iter(lora_ids), None)
        if lora_id is not None and lora_id not in self._adapters:
            raise RuntimeError(f"LoRA adapter {lora_id} is not loaded")
        return lora_id

    def _forward_model(self) -> Any:
        return self._runner._forward_model

    def _module_names(self) -> set[str]:
        model = self._forward_model()
        if model is None:
            return set()
        return {name for name, _module in model.named_modules()}

    def _restore_base_modules(self) -> None:
        """Replace active LoRA wrappers with their original base modules."""
        model = self._forward_model()
        if model is None:
            return

        from mlx.utils import tree_unflatten

        replacements: list[tuple[str, Any]] = []
        for name, module in model.named_modules():
            if hasattr(module, "linear") and hasattr(module, "lora_a"):
                replacements.append((name, module.linear))
            elif hasattr(module, "embedding") and hasattr(module, "lora_a"):
                replacements.append((name, module.embedding))

        if replacements:
            model.update_modules(tree_unflatten(replacements))

    def _apply_adapter(self, adapter: LoadedLoRAAdapter) -> None:
        model = self._forward_model()
        if model is None:
            raise RuntimeError("Model must be loaded before activating LoRA")

        from mlx.utils import tree_unflatten

        modules = dict(model.named_modules())
        replacements: list[tuple[str, Any]] = []
        for weight in adapter.weights:
            module = modules.get(weight.module_name)
            if module is None:
                raise RuntimeError(
                    f"LoRA target module {weight.module_name!r} is not present"
                )
            lora_module = _wrap_module_with_lora(
                module,
                rank=weight.rank,
                scale=weight.scale,
                dropout=weight.dropout,
            )
            lora_module.lora_a = weight.lora_a
            lora_module.lora_b = weight.lora_b
            replacements.append((weight.module_name, lora_module))

        model.update_modules(tree_unflatten(replacements))


def _request_lora_id(lora_request: LoRARequest) -> int:
    lora_id = getattr(lora_request, "lora_int_id", None)
    if lora_id is None:
        lora_id = getattr(lora_request, "lora_id", None)
    if lora_id is None:
        raise ValueError("LoRARequest does not expose a numeric LoRA ID")
    return int(lora_id)


def _request_lora_path(lora_request: LoRARequest) -> Path | None:
    for attr in ("lora_local_path", "lora_path", "path"):
        value = getattr(lora_request, attr, None)
        if value:
            return Path(value)
    return None


def get_lora_id_from_request_data(request_data: Any) -> int | None:
    """Extract a LoRA ID from vLLM scheduler request data, if present."""
    lora_request = getattr(request_data, "lora_request", None)
    if lora_request is None:
        return None
    return _request_lora_id(lora_request)


def _load_adapter_weights(adapter_path: Path, module_names: set[str]) -> list[LoRAWeights]:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_path}")
    with config_path.open("r") as f:
        config = json.load(f)

    if _is_peft_config(config):
        return _load_peft_adapter(adapter_path, config, module_names)
    return _load_mlx_lm_adapter(adapter_path, config, module_names)


def _is_peft_config(config: dict[str, Any]) -> bool:
    return "peft_type" in config or "base_model_name_or_path" in config


def _adapter_tensor_file(adapter_path: Path, preferred: str) -> Path:
    path = adapter_path / preferred
    if path.is_file():
        return path
    candidates = sorted(adapter_path.glob("*.safetensors"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No safetensors adapter file found in {adapter_path}")


def _read_safetensors(path: Path) -> dict[str, Any]:
    import mlx.core as mx
    from safetensors import safe_open

    tensors: dict[str, Any] = {}
    with safe_open(path, framework="np") as f:
        for key in f.keys():
            tensors[key] = mx.array(f.get_tensor(key))
    return tensors


def _load_peft_adapter(
    adapter_path: Path,
    config: dict[str, Any],
    module_names: set[str],
) -> list[LoRAWeights]:
    tensors = _read_safetensors(_adapter_tensor_file(adapter_path, "adapter_model.safetensors"))
    rank = int(config.get("r", 0) or 0)
    alpha = float(config.get("lora_alpha", rank or 1))
    dropout = float(config.get("lora_dropout", 0.0) or 0.0)
    scale = alpha / (math.sqrt(rank) if config.get("use_rslora", False) else rank)

    weights: list[LoRAWeights] = []
    for a_key, a_value in tensors.items():
        if _PEFT_LORA_A_MARKER not in a_key:
            continue
        b_key = a_key.replace(_PEFT_LORA_A_MARKER, _PEFT_LORA_B_MARKER)
        if b_key not in tensors:
            continue

        raw_module_name = a_key.split(_PEFT_LORA_A_MARKER, 1)[0]
        module_name = _resolve_module_name(raw_module_name, module_names)
        if module_name is None:
            logger.warning("Skipping unmatched PEFT LoRA module: %s", raw_module_name)
            continue

        lora_a = a_value.T
        lora_b = tensors[b_key].T
        resolved_rank = int(lora_a.shape[1])
        weights.append(
            LoRAWeights(
                module_name=module_name,
                lora_a=lora_a,
                lora_b=lora_b,
                rank=resolved_rank,
                scale=scale,
                dropout=dropout,
            )
        )

    return weights


def _load_mlx_lm_adapter(
    adapter_path: Path,
    config: dict[str, Any],
    module_names: set[str],
) -> list[LoRAWeights]:
    tensors = _read_safetensors(_adapter_tensor_file(adapter_path, "adapters.safetensors"))
    params = config.get("lora_parameters", {})
    rank = int(params.get("rank", 8))
    scale = float(params.get("scale", 20.0))
    dropout = float(params.get("dropout", 0.0))

    weights: list[LoRAWeights] = []
    for a_key, a_value in tensors.items():
        if not a_key.endswith(_MLX_LORA_A_SUFFIX):
            continue
        base_key = a_key[: -len(_MLX_LORA_A_SUFFIX)]
        b_key = f"{base_key}{_MLX_LORA_B_SUFFIX}"
        if b_key not in tensors:
            continue
        module_name = _resolve_module_name(base_key, module_names)
        if module_name is None:
            logger.warning("Skipping unmatched mlx-lm LoRA module: %s", base_key)
            continue
        weights.append(
            LoRAWeights(
                module_name=module_name,
                lora_a=a_value,
                lora_b=tensors[b_key],
                rank=int(a_value.shape[1]),
                scale=scale,
                dropout=dropout,
            )
        )
    return weights


def _resolve_module_name(raw_name: str, module_names: set[str]) -> str | None:
    """Map PEFT/Unsloth checkpoint names to MLX model module names."""
    candidates = [raw_name]
    prefixes = (
        "base_model.model.",
        "base_model.",
        "language_model.model.",
        "language_model.",
        "model.model.",
        "model.",
    )
    for prefix in prefixes:
        if raw_name.startswith(prefix):
            candidates.append(raw_name[len(prefix) :])

    for candidate in candidates:
        if candidate in module_names:
            return candidate

    # Prefer exact suffix matches and reject ambiguous mappings.
    suffix_matches: list[str] = []
    for candidate in candidates:
        suffix_matches.extend(
            module_name
            for module_name in module_names
            if module_name.endswith(f".{candidate}")
        )
        suffix_matches.extend(
            module_name
            for module_name in module_names
            if candidate.endswith(f".{module_name}")
        )
    unique = sorted(set(suffix_matches), key=len)
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        logger.warning("Ambiguous LoRA target %s matched %s", raw_name, unique[:8])
    return None


def _wrap_module_with_lora(module: Any, *, rank: int, scale: float, dropout: float) -> Any:
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear

    if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
        return LoRALinear.from_base(module, r=rank, scale=scale, dropout=dropout)
    if isinstance(module, (nn.Embedding, nn.QuantizedEmbedding)):
        return LoRAEmbedding.from_base(module, r=rank, scale=scale, dropout=dropout)
    if hasattr(module, "to_lora"):
        return module.to_lora(r=rank, scale=scale, dropout=dropout)
    raise TypeError(f"Module of type {type(module).__name__} cannot be wrapped with LoRA")
