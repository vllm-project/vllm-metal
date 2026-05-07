# SPDX-License-Identifier: Apache-2.0
"""AWQ load owner for vllm-metal.

Encapsulates the entry-point preflight, the ``mlx_lm.load`` invocation
with a normalized ``model_config={"quantization_config": ...}`` kwarg,
the dtype-scoped cache key, and the post-load alignment of non-quantized
floating params. ``ModelLifecycle`` delegates the entire AWQ branch to
instances of this class so quantization policy stays cohesive in one
place rather than leaking into the generic loader flow.

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

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten
from mlx_lm import load as mlx_lm_load
from vllm.logger import init_logger

from vllm_metal.quant.awq_config import (
    UnsupportedQuantizationConfigError,
    normalize_quant_config,
)

logger = init_logger(__name__)


# AWQ-transform output buffer names. Their dtype is owned by the upstream
# transform (typically fp16) and must not be touched by the post-load
# alignment, even on MLX-quantize-protocol leaves.
_AWQ_QUANT_BUFFER_NAMES = ("scales", "biases")


class AWQQuantLoader:
    """Owner for the AWQ load flow.

    Construct via :meth:`for_model`, which inspects the checkpoint's
    ``config.json`` and returns ``None`` when the checkpoint is not AWQ.
    Lifecycle dispatches the quantized branch to this owner so the
    dtype-scoped cache key and the post-load alignment policy do not
    bleed into generic loader code paths.
    """

    def __init__(self, normalized_quant_config: Mapping[str, Any]) -> None:
        # `normalize_quant_config` already canonicalized aliases and
        # rejected unsupported variants; stash a kwarg dict ready to hand
        # to ``mlx_lm.load(model_config=...)``.
        self._mlx_lm_model_config: dict[str, Any] = {
            "quantization_config": dict(normalized_quant_config)
        }

    @classmethod
    def for_model(cls, model_name: str) -> AWQQuantLoader | None:
        """Detect AWQ in ``model_name``'s ``config.json`` (local dir or HF
        Hub) and return a configured loader. Returns ``None`` when the
        checkpoint has no quantization config or uses a quant method
        unrelated to AWQ.

        GPTQ checkpoints raise :class:`UnsupportedQuantizationConfigError`
        rather than returning ``None``: the loader does not silently take
        ownership of a quantized checkpoint that is outside the v1
        support claim, and falling through to the generic loader would
        bypass the dtype-alignment contract this owner enforces.

        Raises:
            UnsupportedQuantizationConfigError: AWQ but outside v1 scope,
                or GPTQ (not yet validated for vllm-metal).
        """
        raw_qc = cls._read_raw_quantization_config(model_name)
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

    @staticmethod
    def cache_key(model_name: str, *, target_dtype: Any) -> tuple[str, str]:
        """Cache key for an AWQ load.

        ``target_dtype`` is encoded into the loader segment because the
        post-load alignment mutates the model in place: a model first
        loaded with bf16 and later requested as fp16 must NOT be served
        from cache, since the cached object would carry the wrong dtype
        on its non-quantized floating params. Encoding the dtype inside
        the loader segment keeps the cache key shape identical to the
        generic ``_generation_cache_key`` (a 2-tuple), so dtype scoping
        is a property of this owner rather than of the generic cache.

        Static so lifecycle can compute the speculative AWQ cache key
        before deciding whether to invoke detection (which involves an
        HF Hub config fetch on cache miss).
        """
        return (model_name, f"mlx_lm-awq:{target_dtype}")

    def load(
        self,
        model_path: str,
        *,
        target_dtype: Any,
        tokenizer_config: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        """Run ``mlx_lm.load`` with the normalized quant config, then align
        non-quantized floating params to ``target_dtype``.

        ``model_path`` is the (possibly compatibility-adapted) path that
        the lifecycle resolves via ``_mlx_lm_compatible_model_path``; the
        owner does not duplicate that path discovery.
        """
        model, tokenizer = mlx_lm_load(
            model_path,
            tokenizer_config=dict(tokenizer_config) if tokenizer_config else None,
            model_config=self._mlx_lm_model_config,
        )
        n_cast = self._align_non_quantized_dtypes(model, target_dtype)
        logger.info(
            "AWQ load: aligned %d non-quantized floating params to %s",
            n_cast,
            target_dtype,
        )
        return model, tokenizer

    # -- private helpers (owned by the loader, not module-level) -----------

    @staticmethod
    def _read_raw_quantization_config(model_name: str) -> Mapping[str, Any] | None:
        """Read ``quantization_config`` from the model's ``config.json``
        without invoking ``mlx_lm.load``. Returns ``None`` if the field is
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
        which would bypass the dtype-alignment contract this owner
        enforces. The local-directory branch above is the single
        legitimate "no config" case and remains a silent ``None``.
        """
        model_path = Path(model_name)
        if model_path.is_dir():
            config_path = model_path / "config.json"
            if not config_path.is_file():
                return None
        else:
            config_path = Path(hf_hub_download(model_name, "config.json"))
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

    @staticmethod
    def _is_mlx_quantized_module(module: Any) -> bool:
        """Whether ``module`` follows the MLX quantize protocol.

        The protocol's defining surface is the instance-level ``bits`` and
        ``group_size`` attributes set during construction. This duck-typed
        check covers ``mlx.nn.QuantizedLinear`` /
        ``mlx.nn.QuantizedEmbedding`` together with the mlx_lm peers
        ``QuantizedSwitchLinear`` (MoE) and ``QuantizedMultiLinear`` (MLA),
        none of which subclass ``nn.QuantizedLinear``. An ``isinstance``
        check would silently misclassify those as plain modules and let
        their AWQ-transform ``scales`` / ``biases`` be cast to the runtime
        dtype. A separate import dependency on the mlx_lm peers would make
        the alignment fragile across mlx_lm versions; the protocol
        attributes are stable.
        """
        return hasattr(module, "bits") and hasattr(module, "group_size")

    @staticmethod
    def _align_non_quantized_dtypes(model: Any, target_dtype: Any) -> int:
        """Cast floating-dtype params on leaf modules to ``target_dtype``.
        Returns the number of cast tensors.

        The AWQ transform produces ``scales`` and ``biases`` parameters at
        the transform's dtype (typically fp16) for every MLX-quantize-protocol
        leaf: ``nn.QuantizedLinear``, ``nn.QuantizedEmbedding``, and the
        mlx_lm MoE / MLA peers. Those buffers are exempt from alignment.

        Surrounding floating params follow the engine's runtime dtype:
        layer norms, embeddings (when not quantized), and the quantized
        layer's own ordinary ``bias`` (Qwen2 q/k/v projections, MoE
        per-expert biases, etc.). Without aligning the regular ``bias``
        the projection emits mixed-dtype activations into a bf16 KV cache
        / sampler.
        """
        # `tree_flatten` is overloaded `list[tuple[str, Any]] | dict[str, Any]`
        # depending on the `destination` kwarg; with `destination=None`
        # (default) it returns the list. Narrow at runtime so mypy can
        # unpack the tuples.
        leaves = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
        assert isinstance(leaves, list)

        n_cast = 0
        for _path, module in leaves:
            is_quantized = AWQQuantLoader._is_mlx_quantized_module(module)
            updates = {}
            for name, value in module.parameters().items():
                if is_quantized and name in _AWQ_QUANT_BUFFER_NAMES:
                    continue
                dtype = getattr(value, "dtype", None)
                if dtype is None:
                    continue
                if not mx.issubdtype(dtype, mx.floating):
                    continue
                if dtype == target_dtype:
                    continue
                updates[name] = value.astype(target_dtype)
            if updates:
                module.update(updates)
                n_cast += len(updates)
        return n_cast
