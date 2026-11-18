# SPDX-License-Identifier: Apache-2.0
"""AWQ load owner for vllm-metal.

Encapsulates the entry-point preflight and the ``mlx_lm.utils.load_model``
invocation with a normalized ``model_config={"quantization_config": ...}``
kwarg. ``ModelLifecycle`` delegates the entire AWQ branch to instances of
this class so quantization policy stays cohesive in one place rather than
leaking into the generic loader flow.

GPTQ checkpoints are explicitly rejected at the entry point. mlx-lm's
``_transform_awq_weights`` accepts ``quant_method="gptq"`` at the
transform level, but GPTQ is not part of the v1 support claim and is
deferred to a follow-up PR once a real GPTQ checkpoint is validated
end-to-end. Until then the loader does not silently take ownership of
GPTQ checkpoints.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from mlx_lm.utils import load_model as mlx_lm_load_model

from vllm_metal.quant.awq_config import (
    UnsupportedQuantizationConfigError,
    normalize_quant_config,
)
from vllm_metal.utils import resolve_mlx_lm_model_path


class AWQQuantLoader:
    """Owner for the AWQ load flow.

    Construct via :meth:`for_model`, which inspects the checkpoint's
    ``config.json`` and returns ``None`` when the checkpoint is not AWQ.
    Lifecycle dispatches the quantized branch to this owner so quantization
    policy does not bleed into generic loader code paths.
    """

    def __init__(self, normalized_quant_config: Mapping[str, Any]) -> None:
        # `normalize_quant_config` already canonicalized aliases and
        # rejected unsupported variants; stash a kwarg dict ready to hand
        # to ``mlx_lm.utils.load_model(model_config=...)``.
        self._mlx_lm_model_config: dict[str, Any] = {
            "quantization_config": dict(normalized_quant_config)
        }

    @classmethod
    def for_model(
        cls, model_name: str, *, revision: str | None = None
    ) -> AWQQuantLoader | None:
        """Detect AWQ in ``model_name``'s ``config.json`` (local dir or HF
        Hub) and return a configured loader. Returns ``None`` when the
        checkpoint has no quantization config or uses a quant method
        unrelated to AWQ.

        GPTQ checkpoints raise :class:`UnsupportedQuantizationConfigError`
        rather than returning ``None``: the loader does not silently take
        ownership of a quantized checkpoint that is outside the v1
        support claim.

        Raises:
            UnsupportedQuantizationConfigError: AWQ but outside v1 scope,
                or GPTQ (not yet validated for vllm-metal).
        """
        raw_qc = cls._read_raw_quantization_config(model_name, revision=revision)
        if raw_qc is None:
            return None
        quant_method = raw_qc.get("quant_method")
        if quant_method == "gptq":
            raise UnsupportedQuantizationConfigError(
                "GPTQ checkpoints are not yet validated for vllm-metal; "
                "v1 only supports AWQ. Use an AWQ release of this "
                "checkpoint, or re-quantize with AWQ."
            )
        if quant_method != "awq":
            return None
        return cls(normalize_quant_config(raw_qc))

    def load(self, model_path: str, *, revision: str | None = None) -> Any:
        """Run ``mlx_lm.utils.load_model`` with the normalized quant config.

        The model keeps the dtype mlx-lm loads it in, the checkpoint's
        scales dtype (typically fp16), like any other MLX-LM checkpoint;
        the runtime dtype sets the KV cache. Casting the other floating
        params to a different runtime dtype makes MLX promote every
        quantized layer's output to float32.

        ``model_path`` is the (possibly compatibility-adapted) path that
        the lifecycle resolves via ``_mlx_lm_compatible_model_path``; the
        owner does not duplicate that path discovery.
        """
        resolved_path = resolve_mlx_lm_model_path(model_path, revision=revision)
        model, _ = mlx_lm_load_model(
            resolved_path, model_config=self._mlx_lm_model_config
        )
        return model

    # -- private helpers (owned by the loader, not module-level) -----------

    @staticmethod
    def _read_raw_quantization_config(
        model_name: str, *, revision: str | None = None
    ) -> Mapping[str, Any] | None:
        """Read ``quantization_config`` from the model's ``config.json``
        without invoking ``mlx_lm.utils.load_model``. Returns ``None`` if the field is
        absent (the checkpoint genuinely is not quantized) or the local
        directory exists but has no ``config.json``.

        Mirrors ``mlx_lm.utils.load_model``'s fallback to
        ``text_config.quantization_config`` for wrapper / multimodal
        configs. Without this, multimodal AWQ checkpoints that nest the
        quant config under ``text_config`` would skip the preflight
        entirely while mlx_lm itself would still apply the transform.

        Hub-fetch errors (``HfHubHTTPError``, ``HFValidationError``,
        ``OSError``) are intentionally NOT caught here. A transient Hub
        failure or a malformed repo id should surface to the caller, not
        silently demote an AWQ checkpoint to the generic loader path,
        which would bypass the preflight. The local-directory branch above
        is the single legitimate "no config" case and remains a silent
        ``None``.
        """
        model_path = Path(model_name)
        if model_path.is_dir():
            config_path = model_path / "config.json"
            if not config_path.is_file():
                return None
        else:
            config_path = Path(
                hf_hub_download(model_name, "config.json", revision=revision)
            )
        with open(config_path, encoding="utf-8") as fid:
            config = json.load(fid)
        qc = config.get("quantization_config")
        if isinstance(qc, dict):
            return qc
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            nested = text_config.get("quantization_config")
            if isinstance(nested, dict):
                return nested
        return None
