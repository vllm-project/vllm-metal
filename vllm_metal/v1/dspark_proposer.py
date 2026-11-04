# SPDX-License-Identifier: Apache-2.0
"""Experimental greedy DSpark serving with scheduler-owned draft KV."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from vllm.logger import init_logger

from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.config import (
    DSPARK_DRAFT_QUANTIZATION_KEY,
    DSPARK_DRAFT_QUANTIZATION_Q4,
    DSPARK_Q4_GROUP_SIZE,
)
from vllm_metal.v1.block_draft_proposer import BlockDraftProposer, DraftForward
from vllm_metal.v1.dspark import DSparkModel, load_dspark
from vllm_metal.v1.dspark_adaptive import AdaptiveVerificationProfile
from vllm_metal.v1.dspark_paged import DSparkPagedCache
from vllm_metal.v1.spec_decode import SpeculativeDecodeController

if TYPE_CHECKING:
    from vllm_metal.v1.model_runner import MetalModelRunner

logger = init_logger(__name__)


class DSparkProposer(BlockDraftProposer):
    """Predict from every slot of an anchor plus K-1 masks, then verify on target.

    The shared lifecycle commits verified target features even while drafting
    is paused. DSpark owns its embeddings and sequential Markov prediction head;
    Optional calibrated prefix admission reduces the target's verification work.
    """

    name = "DSpark"
    extra_slots = 0

    def __init__(
        self,
        model: DSparkModel,
        *,
        num_draft_tokens: int,
        controller: SpeculativeDecodeController,
        draft_topk: int | None = None,
        enable_prefix_caching: bool = False,
        adaptive_profile: AdaptiveVerificationProfile | None = None,
    ) -> None:
        model.validate_draft_topk(draft_topk, model.config.backbone.vocab_size)
        super().__init__(
            model.backbone,
            num_draft_tokens=num_draft_tokens,
            controller=controller,
            enable_prefix_caching=enable_prefix_caching,
        )
        self.draft_model = model
        self.draft_topk = draft_topk
        if adaptive_profile is not None and model.confidence_head is None:
            raise ValueError("Adaptive DSpark requires a trained confidence head")
        self.adaptive_profile = adaptive_profile

    @classmethod
    def build(cls, runner: MetalModelRunner) -> DSparkProposer:
        spec = runner.vllm_config.speculative_config
        assert spec is not None and spec.draft_model_config is not None
        if (
            spec.draft_sample_method != "greedy"
            or spec.rejection_sample_method != "standard"
        ):
            raise NotImplementedError(
                "DSpark on Metal requires greedy drafting and standard verification "
                "for fixed target logits"
            )
        hf_config = spec.draft_model_config.hf_config
        draft_topk = spec.dspark_draft_topk
        if draft_topk is None:
            draft_topk = getattr(hf_config, "dspark_draft_topk", None)
        if draft_topk is not None:
            DSparkModel.validate_draft_topk(draft_topk, hf_config.vocab_size)
        if (
            spec.quantization is not None
            or spec.draft_model_config.quantization is not None
        ):
            raise NotImplementedError(
                "DSpark on Metal requires an unquantized draft checkpoint"
            )
        if spec.kv_cache_dtype not in (None, "auto"):
            requested_dtype = {
                "float16": mx.float16,
                "bfloat16": mx.bfloat16,
            }.get(spec.kv_cache_dtype)
            if requested_dtype is None or requested_dtype != runner.kv_cache_dtype:
                raise NotImplementedError(
                    "DSpark on Metal requires draft KV in the target activation precision"
                )
        path = cls._checkpoint_path(runner)
        additional = runner.vllm_config.additional_config
        draft_quantization = (
            additional.get(DSPARK_DRAFT_QUANTIZATION_KEY)
            if isinstance(additional, dict)
            else None
        )
        model = load_dspark(
            path,
            target_config=runner.model_config.hf_config.to_dict(),
            draft_quantization=draft_quantization,
        )
        profile_path = (
            additional.get("dspark_adaptive_profile")
            if isinstance(additional, dict)
            else None
        )
        adaptive_profile = None
        if spec.enable_adaptive_verification:
            if not isinstance(profile_path, str) or not profile_path:
                raise ValueError("Adaptive DSpark requires dspark_adaptive_profile")
            adaptive_profile = AdaptiveVerificationProfile.load(
                profile_path,
                identity=cls.profile_identity(
                    runner, path, draft_topk, draft_quantization
                ),
                width=spec.num_speculative_tokens,
                max_num_seqs=runner.scheduler_config.max_num_seqs,
            )
        elif profile_path is not None:
            raise ValueError(
                "dspark_adaptive_profile requires enable_adaptive_verification"
            )
        if model.backbone.fc.weight.dtype != runner.kv_cache_dtype:
            raise NotImplementedError(
                "DSpark on Metal requires matching target and draft activation precision"
            )
        if draft_quantization == DSPARK_DRAFT_QUANTIZATION_Q4:
            logger.info(
                "DSpark draft linears use affine Q4 (group_size=%d); "
                "activations and KV remain %s",
                DSPARK_Q4_GROUP_SIZE,
                model.embed_tokens.weight.dtype,
            )
        proposer = cls(
            model,
            num_draft_tokens=spec.num_speculative_tokens,
            controller=runner._spec_decode_controller,
            draft_topk=draft_topk,
            enable_prefix_caching=runner.vllm_config.cache_config.enable_prefix_caching,
            adaptive_profile=adaptive_profile,
        )
        proposer.max_model_len = min(
            proposer.max_model_len, spec.draft_model_config.max_model_len
        )
        return proposer

    @staticmethod
    def profile_identity(
        runner: MetalModelRunner,
        draft_path: str,
        draft_topk: int | None,
        draft_quantization: str | None,
    ) -> dict[str, object]:
        from vllm_metal.utils import get_model_download_path

        target = get_model_download_path(
            runner.model_config.model, revision=runner.model_config.revision
        )
        spec = runner.vllm_config.speculative_config
        assert spec is not None
        return {
            "target": str(Path(target).resolve()),
            "draft": str(Path(draft_path).resolve()),
            "activation_dtype": str(runner.kv_cache_dtype),
            "draft_topk": draft_topk,
            "draft_quantization": draft_quantization,
            "num_draft_tokens": spec.num_speculative_tokens,
            "max_model_len": runner.model_config.max_model_len,
            "block_size": runner.vllm_config.cache_config.block_size,
            "mlx": version("mlx"),
            "vllm": version("vllm"),
            "device": mx.device_info()["device_name"],
            "merge_verify_windows": runner.merge_verify_windows,
        }

    def _draft_width(self, limit: int) -> int:
        # Bidirectional block predictions depend on width. Admission trims the
        # trained block after generation, never changes its hidden states.
        return self.num_draft_tokens if self.adaptive_profile is not None else limit

    def _admit_drafts(
        self,
        confidence: mx.array | None,
        rows: Sequence[tuple[Sequence[int], int]],
        limits: list[int],
    ) -> list[int]:
        if self.adaptive_profile is None:
            return limits
        assert confidence is not None
        return self.adaptive_profile.admit(
            confidence.tolist(), limits, sum(end for _, end in rows) / len(rows)
        )

    def _make_cache(
        self, storage: KVCacheStorage, *, max_model_len: int
    ) -> DSparkPagedCache:
        return DSparkPagedCache(
            self.draft_model, storage, self.layer_names, max_model_len=max_model_len
        )

    def _compile_draft(self, width: int) -> DraftForward:
        assert isinstance(self.cache, DSparkPagedCache)
        draft = self.cache.compile_draft(
            num_draft_tokens=width,
            draft_topk=self.draft_topk,
            corrected_logits=False,
        )

        def forward(
            anchors: mx.array, rows: Sequence[tuple[Sequence[int], int]]
        ) -> tuple[mx.array, mx.array | None]:
            tokens, _, confidence = draft(anchors, rows)
            return tokens, confidence if self.adaptive_profile is not None else None

        return forward

    def _profile_draft(self, anchors: mx.array, features: Sequence[mx.array]) -> None:
        mx.eval(
            self.draft_model.draft(
                anchors,
                features,
                num_draft_tokens=self.num_draft_tokens,
                draft_topk=self.draft_topk,
                corrected_logits=False,
            )
        )
