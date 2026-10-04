# SPDX-License-Identifier: Apache-2.0
"""Shared committed-feature lifecycle for synchronous block drafting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec
from vllm.v1.outputs import DraftTokenIds

from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.metal.constants import KERNEL_BLOCK_SIZES
from vllm_metal.pytorch_backend.tensor_bridge import MLX_TO_TORCH_DTYPE
from vllm_metal.utils import get_model_download_path
from vllm_metal.v1.block_draft_paged import BlockDraftPagedCache
from vllm_metal.v1.dflash import DFlashModel, DFlashTargetCapture
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import PagedDecodeSegment, SpeculativeDecodeController

if TYPE_CHECKING:
    from vllm_metal.v1.model_runner import MetalModelRunner


DraftForward = Callable[[mx.array, Sequence[tuple[Sequence[int], int]]], mx.array]


class BlockDraftProposer(ABC):
    """Commit target features, then draft one bidirectional anchor/mask block.

    Accepted draft tokens still need their target features projected next step:
    temporary block KV is never equivalent to committed context KV. Prefix
    caching is deliberately unsupported until that lifecycle is qualified.
    """

    name: str
    # DFlash predicts after its anchor; DSpark predicts from its anchor.
    extra_slots: int

    def __init__(
        self,
        model: DFlashModel,
        *,
        num_draft_tokens: int,
        controller: SpeculativeDecodeController,
    ) -> None:
        model._validate_num_draft_tokens(num_draft_tokens, extra_slots=self.extra_slots)
        if model.fc.weight.dtype not in (mx.float16, mx.bfloat16):
            raise ValueError(f"{self.name} serving requires an FP16 or BF16 checkpoint")
        self.model = model
        self.max_model_len = model.config.max_position_embeddings
        self.num_draft_tokens = num_draft_tokens
        self.controller = controller
        self.layer_names = tuple(
            f"{self.name.lower()}_layers.{i}.self_attn"
            for i in range(model.config.num_hidden_layers)
        )
        self.cache: BlockDraftPagedCache | None = None
        self._drafts: dict[int, DraftForward] = {}
        self._group_index = 0
        self._valid_ends: dict[str, int] = {}

    @classmethod
    def _checkpoint_path(cls, runner: MetalModelRunner) -> str:
        config = runner.vllm_config
        spec = config.speculative_config
        assert spec is not None and spec.draft_model_config is not None
        if config.cache_config.enable_prefix_caching:
            raise NotImplementedError(
                f"{cls.name} on Metal requires --no-enable-prefix-caching"
            )
        if (
            runner.is_hybrid
            or runner.is_mla
            or runner._is_vlm
            or runner._is_pooling
            or config.lora_config is not None
            or config.parallel_config.tensor_parallel_size != 1
            or config.parallel_config.pipeline_parallel_size != 1
        ):
            raise NotImplementedError(
                f"{cls.name} on Metal requires a single-device Qwen3 text target without LoRA"
            )
        from vllm_metal.config import get_config

        if get_config().turboquant:
            raise NotImplementedError(
                f"{cls.name} on Metal does not support TurboQuant"
            )
        if config.cache_config.block_size not in KERNEL_BLOCK_SIZES:
            raise NotImplementedError(
                f"{cls.name} on Metal requires a native attention block size"
            )
        path = get_model_download_path(
            spec.draft_model_config.model, revision=spec.draft_model_config.revision
        )
        if not Path(path).is_dir():
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                path,
                revision=spec.draft_model_config.revision,
                allow_patterns=[
                    "config.json",
                    "*.safetensors",
                    "*.safetensors.index.json",
                ],
            )
        return path

    def target_capture(self, target: nn.Module) -> DFlashTargetCapture:
        return DFlashTargetCapture(target, self.model.config)

    def kv_specs(self, block_size: int) -> dict[str, KVCacheSpec]:
        cfg = self.model.config
        return {
            name: FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=cfg.num_key_value_heads,
                head_size=cfg.head_dim,
                dtype=MLX_TO_TORCH_DTYPE[self.model.fc.weight.dtype],
            )
            for name in self.layer_names
        }

    def bind_cache(
        self, storage: KVCacheStorage, *, group_index: int, max_model_len: int
    ) -> None:
        self.cache = self._make_cache(
            storage, max_model_len=min(max_model_len, self.max_model_len)
        )
        self._group_index = group_index
        self._drafts.clear()
        self._valid_ends.clear()

    def needs_target_hidden_states(
        self, decode_segments: Sequence[PagedDecodeSegment], *, has_final_prefill: bool
    ) -> bool:
        return False

    def release_requests(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            self._valid_ends.pop(req_id, None)

    def profile_warmup(self, runner: MetalModelRunner, tokens: mx.array) -> None:
        """Run the target capture forward and profile the drafter's buffers."""
        captured = runner._target_forward(
            tokens, logits_indices=runner._profile_logits_indices(tokens)
        )
        mx.eval(captured.logits, *captured.aux_hidden_states)
        self.profile(captured.aux_hidden_states, runner.scheduler_config.max_num_seqs)

    def profile(self, features: Sequence[mx.array], max_num_seqs: int) -> None:
        """Include captured-feature projection and block drafting in profiling."""
        model = self.model
        context = model.hidden_norm(model.fc(mx.concatenate(features, axis=-1)))
        mx.eval(context)
        for layer in model.layers:
            mx.eval(layer.self_attn.k_proj(context), layer.self_attn.v_proj(context))
        # Prefix storage is accounted for by KV specs. A one-token prefix
        # profiles the fixed block's activations and prediction heads.
        cfg = model.config
        prefix = [
            mx.zeros((max_num_seqs, 1, cfg.hidden_size), dtype=context.dtype)
            for _ in cfg.target_layer_ids
        ]
        self._profile_draft(mx.zeros((max_num_seqs,), dtype=mx.int32), prefix)

    @abstractmethod
    def _make_cache(
        self, storage: KVCacheStorage, *, max_model_len: int
    ) -> BlockDraftPagedCache: ...

    @abstractmethod
    def _compile_draft(self, width: int) -> DraftForward: ...

    @abstractmethod
    def _profile_draft(
        self, anchors: mx.array, features: Sequence[mx.array]
    ) -> None: ...

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        cache = self.cache
        if cache is None:
            raise RuntimeError(f"{self.name} must bind scheduler KV before proposing")
        width = ctx.num_speculative_tokens
        if type(width) is not int or not 0 <= width <= self.num_draft_tokens:
            raise ValueError(
                f"{self.name} draft width exceeds the configured token budget"
            )
        features = ctx.target_aux_hidden_states
        if not features or any(f.shape[0] != ctx.cu_seqlens[-1] for f in features):
            raise RuntimeError(
                f"{self.name} requires target features for every packed input row"
            )
        spans: list[tuple[Sequence[int], int, int]] = []
        slices: list[slice] = []
        updates: dict[str, int] = {}

        def ingest(
            req_id: str,
            groups: Sequence[Sequence[int]],
            start: int,
            count: int,
            row: int,
        ) -> None:
            end = min(start + count, cache.max_model_len)
            if start >= end:
                return
            if start != self._valid_ends.get(req_id, 0):
                raise RuntimeError(
                    f"{self.name} committed context is discontinuous for {req_id!r}"
                )
            spans.append((groups[self._group_index], start, end - start))
            slices.append(slice(row, row + end - start))
            updates[req_id] = end

        for segment, tokens in zip(
            ctx.decode_segments, ctx.decode_token_ids, strict=True
        ):
            # The anchor plus accepted drafts have committed target features;
            # the last sampled token becomes the NEXT block's anchor.
            ingest(
                segment.req_id,
                segment.block_ids,
                segment.cache_start_pos,
                len(tokens),
                segment.start_row,
            )
        for i, prefill in enumerate(ctx.prefill_reqs):
            ingest(
                prefill.req_id,
                prefill.block_ids,
                prefill.start_pos,
                len(prefill.token_ids),
                ctx.cu_seqlens[ctx.num_decode_segments + i],
            )
        if spans:
            cache.write_context(
                [mx.concatenate([f[s] for s in slices]) for f in features], spans
            )
            self._valid_ends.update(updates)

        eligible = self.controller.draft_eligible_requests(
            ctx.decode_reqs,
            ctx.decode_token_ids,
            ctx.prefill_reqs,
            ctx.prefill_result_modes,
            ctx.request_states,
        )
        req_ids, anchors, rows = [], [], []
        if width:
            for req_id, state in eligible:
                # This stage does not qualify scheduler-invalid grammar drafts.
                # Keep constrained requests on the target's grammar sampler.
                if state.sampling_params.structured_outputs is not None:
                    continue
                end = self._valid_ends.get(req_id, 0)
                if end + width + self.extra_slots > cache.max_model_len:
                    continue
                if end != len(state.token_ids) - 1:
                    raise RuntimeError(
                        f"{self.name} anchor does not follow committed target features"
                    )
                req_ids.append(req_id)
                anchors.append(state.token_ids[-1])
                rows.append((state.block_ids[self._group_index], end))
        if not rows:
            mx.eval(*cache.storage.buffers)
            return None
        # The scheduler selects the next width from its batch-size schedule.
        # Keep a compiled callable for each encountered nonzero width. K=0
        # still commits features above, so later drafting needs no replay.
        if width not in self._drafts:
            self._drafts[width] = self._compile_draft(width)
        tokens = self._drafts[width](mx.array(anchors, dtype=mx.int32), rows)
        mx.eval(tokens, *cache.storage.buffers)
        return DraftTokenIds(req_ids=req_ids, draft_token_ids=tokens.tolist())
