# SPDX-License-Identifier: Apache-2.0
"""Experimental greedy DSpark serving with scheduler-owned draft KV."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import mlx.core as mx

from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.v1.block_draft_proposer import BlockDraftProposer, DraftForward
from vllm_metal.v1.dspark import DSparkModel, load_dspark
from vllm_metal.v1.dspark_paged import DSparkPagedCache
from vllm_metal.v1.spec_decode import SpeculativeDecodeController

if TYPE_CHECKING:
    from vllm_metal.v1.model_runner import MetalModelRunner


class DSparkProposer(BlockDraftProposer):
    """Predict from every slot of an anchor plus K-1 masks, then verify on target.

    The shared lifecycle commits verified target features even while drafting
    is paused. DSpark owns its embeddings and sequential Markov prediction head;
    confidence does not change the scheduler's proposal or verification budget.
    """

    name = "DSpark"
    extra_slots = 0

    def __init__(
        self,
        model: DSparkModel,
        *,
        num_draft_tokens: int,
        controller: SpeculativeDecodeController,
    ) -> None:
        super().__init__(
            model.backbone,
            num_draft_tokens=num_draft_tokens,
            controller=controller,
        )
        self.draft_model = model

    @classmethod
    def build(cls, runner: MetalModelRunner) -> DSparkProposer:
        spec = runner.vllm_config.speculative_config
        assert spec is not None and spec.draft_model_config is not None
        if (
            spec.enable_adaptive_verification
            or spec.draft_sample_method != "greedy"
            or spec.rejection_sample_method != "standard"
            or spec.dspark_draft_topk is not None
            or getattr(spec.draft_model_config.hf_config, "dspark_draft_topk", None)
            is not None
        ):
            raise NotImplementedError(
                "DSpark on Metal requires greedy drafting and standard verification "
                "without adaptive verification or dspark_draft_topk"
            )
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
        model = load_dspark(path, target_config=runner.model_config.hf_config.to_dict())
        if model.backbone.fc.weight.dtype != runner.kv_cache_dtype:
            raise NotImplementedError(
                "DSpark on Metal requires matching target and draft activation precision"
            )
        proposer = cls(
            model,
            num_draft_tokens=spec.num_speculative_tokens,
            controller=runner._spec_decode_controller,
        )
        proposer.max_model_len = min(
            proposer.max_model_len, spec.draft_model_config.max_model_len
        )
        return proposer

    def _make_cache(
        self, storage: KVCacheStorage, *, max_model_len: int
    ) -> DSparkPagedCache:
        return DSparkPagedCache(
            self.draft_model, storage, self.layer_names, max_model_len=max_model_len
        )

    def _compile_draft(self, width: int) -> DraftForward:
        assert isinstance(self.cache, DSparkPagedCache)
        draft = self.cache.compile_draft(num_draft_tokens=width)
        # The paged adapter has already applied the sequential Markov head.
        # Return its IDs without enabling confidence-based truncation.
        return lambda anchors, rows: draft(anchors, rows)[0]

    def _profile_draft(self, anchors: mx.array, features: Sequence[mx.array]) -> None:
        mx.eval(
            self.draft_model.draft(
                anchors, features, num_draft_tokens=self.num_draft_tokens
            )
        )
