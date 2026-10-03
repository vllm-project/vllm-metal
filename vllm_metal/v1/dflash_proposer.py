# SPDX-License-Identifier: Apache-2.0
"""Experimental synchronous DFlash serving with scheduler-owned draft KV."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import mlx.core as mx

from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.v1.block_draft_proposer import BlockDraftProposer, DraftForward
from vllm_metal.v1.dflash import DFlashModel, load_dflash
from vllm_metal.v1.dflash_paged import DFlashPagedCache
from vllm_metal.v1.spec_decode import SpeculativeDecodeController

if TYPE_CHECKING:
    from vllm_metal.v1.model_runner import MetalModelRunner


class DFlashProposer(BlockDraftProposer):
    """DFlash predicts K tokens from an anchor followed by K mask slots."""

    name = "DFlash"
    extra_slots = 1

    def __init__(
        self,
        model: DFlashModel,
        *,
        num_draft_tokens: int,
        embed: Callable[[mx.array], mx.array],
        project: Callable[[mx.array], mx.array],
        controller: SpeculativeDecodeController,
    ) -> None:
        super().__init__(
            model, num_draft_tokens=num_draft_tokens, controller=controller
        )
        self.embed, self.project = embed, project

    @classmethod
    def build(cls, runner: MetalModelRunner) -> DFlashProposer:
        path = cls._checkpoint_path(runner)
        spec = runner.vllm_config.speculative_config
        assert spec is not None and spec.draft_model_config is not None
        model = load_dflash(path, target_config=runner.model_config.hf_config.to_dict())
        if model.fc.weight.dtype != runner.kv_cache_dtype:
            raise NotImplementedError(
                "DFlash on Metal requires matching target and draft activation precision"
            )
        target = runner._forward_model
        # DFlash is qualified only for native Qwen3; preserve its tied/untied
        # and quantized head without registering borrowed target parameters.
        project = (
            target.model.embed_tokens.as_linear
            if target.args.tie_word_embeddings
            else target.lm_head
        )
        proposer = cls(
            model,
            num_draft_tokens=spec.num_speculative_tokens,
            embed=runner._target_input_embeddings,
            project=project,
            controller=runner._spec_decode_controller,
        )
        proposer.max_model_len = min(
            proposer.max_model_len, spec.draft_model_config.max_model_len
        )
        return proposer

    def _make_cache(
        self, storage: KVCacheStorage, *, max_model_len: int
    ) -> DFlashPagedCache:
        return DFlashPagedCache(
            self.model, storage, self.layer_names, max_model_len=max_model_len
        )

    def _compile_draft(self, width: int) -> DraftForward:
        assert isinstance(self.cache, DFlashPagedCache)
        draft = self.cache.compile_draft(
            num_draft_tokens=width, embed=self.embed, project=self.project
        )
        return lambda anchors, rows: mx.argmax(draft(anchors, rows), axis=-1)

    def _profile_draft(self, anchors: mx.array, features: Sequence[mx.array]) -> None:
        mx.eval(
            self.model.draft_logits(
                anchors,
                features,
                num_draft_tokens=self.num_draft_tokens,
                embed=self.embed,
                project=self.project,
            )
        )
