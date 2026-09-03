# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, HiddenStateCacheSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_metal.attention.runtime.hybrid import HybridPagedAttentionRuntime
from vllm_metal.v1.qwen_mtp_paged import (
    QwenMTPAttentionSpec,
    QwenMTPBoundaryHiddenCache,
)
from vllm_metal.v1.spec_decode import SpeculativeDecodeController


class TestQwenMTPAttentionSpec:
    def test_reports_dense_key_value_page_bytes(self) -> None:
        # Four logical latent heads represent K+V for two physical MTP KV heads.
        spec = QwenMTPAttentionSpec(
            block_size=4,
            num_kv_heads=4,
            head_size=8,
            dtype=torch.float16,
        )
        assert isinstance(spec, HiddenStateCacheSpec)
        assert spec.page_size_bytes == 4 * 2 * (8 + 8) * 2
        assert KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager
        assert KVCacheSpecRegistry.get_uniform_type_base_spec(spec) is type(spec)

    def test_cache_only_group_does_not_fragment_target_groups(self) -> None:
        mtp = QwenMTPAttentionSpec(
            block_size=4,
            num_kv_heads=4,
            head_size=8,
            dtype=torch.float16,
        )
        target = FullAttentionSpec(
            block_size=4,
            num_kv_heads=2,
            head_size=8,
            dtype=torch.float16,
        )
        # Production inserts the distinct cache-only MTP spec first. That makes
        # vLLM's early uniformity probe use the custom registered base, then its
        # cache-only extraction path removes MTP before target group sizing.
        groups = get_kv_cache_groups(
            SimpleNamespace(
                scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
                speculative_config=SimpleNamespace(use_eagle=lambda: True),
            ),
            {
                "mtp.layers.0.self_attn": mtp,
                "layers.0.self_attn": target,
                "layers.1.self_attn": target,
                "layers.2.self_attn": target,
                "layers.3.self_attn": target,
            },
        )

        assert len(groups) == 2
        target_group = next(
            group for group in groups if "layers.0.self_attn" in group.layer_names
        )
        mtp_group = next(
            group for group in groups if "mtp.layers.0.self_attn" in group.layer_names
        )
        assert target_group.layer_names == [
            "layers.0.self_attn",
            "layers.1.self_attn",
            "layers.2.self_attn",
            "layers.3.self_attn",
        ]
        assert mtp_group.layer_names == ["mtp.layers.0.self_attn"]
        assert (
            target_group.kv_cache_spec.page_size_bytes
            == mtp_group.kv_cache_spec.page_size_bytes
        )


class TestQwenMTPBoundaryHiddenCache:
    def test_store_read_copy_and_reuse_follow_target_blocks(self) -> None:
        cache = QwenMTPBoundaryHiddenCache(
            num_blocks=8,
            block_size=4,
            hidden_size=3,
            dtype=mx.float32,
        )
        assert cache.bytes_per_block == 3 * mx.float32.size + mx.uint8.size
        hidden = mx.array(
            [
                [10.0, 11.0, 12.0],
                [20.0, 21.0, 22.0],
                [30.0, 31.0, 32.0],
                [40.0, 41.0, 42.0],
            ],
            dtype=mx.float32,
        )
        # Scheduler slots 8..11 complete block 2. Only its final hidden state
        # becomes a durable warm-prefix checkpoint.
        cache.store([8, 9, 10, 11], hidden)
        mx.eval(cache.cache, cache.valid)

        value = cache.read([0, 1, 2], token_position=11)
        np.testing.assert_array_equal(np.array(value), [[40.0, 41.0, 42.0]])

        cache.copy_blocks([(2, 5)])
        mx.eval(cache.cache, cache.valid)
        copied = cache.read([0, 1, 5], token_position=11)
        np.testing.assert_array_equal(np.array(copied), [[40.0, 41.0, 42.0]])

        # Beginning a new lifetime for physical block 5 invalidates the old
        # occupant's checkpoint until that new block is complete.
        cache.store([20], mx.array([[90.0, 91.0, 92.0]], dtype=mx.float32))
        mx.eval(cache.valid)
        with pytest.raises(RuntimeError, match="no valid boundary-hidden"):
            cache.read([0, 1, 5], token_position=11)

    def test_missing_unaligned_or_out_of_range_boundary_fails_closed(self) -> None:
        cache = QwenMTPBoundaryHiddenCache(
            num_blocks=2,
            block_size=4,
            hidden_size=2,
            dtype=mx.float16,
        )
        with pytest.raises(RuntimeError, match="not target-block aligned"):
            cache.read([0, 1], token_position=6)
        with pytest.raises(RuntimeError, match="missing the boundary-hidden block"):
            cache.read([0], token_position=7)
        with pytest.raises(RuntimeError, match="out of range"):
            cache.read([0, 3], token_position=7)
        with pytest.raises(RuntimeError, match="no valid boundary-hidden"):
            cache.read([0, 1], token_position=7)
        with pytest.raises(RuntimeError, match="slot is out of range"):
            cache.store([8], mx.zeros((1, 2), dtype=mx.float16))


class _FakeQwenMTPState:
    ready = True

    def __init__(self) -> None:
        self.copy_calls = []
        self.hidden_calls = []
        self.pair_calls = []

    def copy_blocks(self, copies):
        self.copy_calls.append(list(copies))

    def store_target_hidden(self, ctx, hidden_states):
        self.hidden_calls.append((ctx, hidden_states))

    def read_boundary_hidden(self, block_ids, token_position):
        self.hidden_calls.append((block_ids, token_position))
        return mx.ones((1, 4), dtype=mx.float32)

    def run_pairs(self, **kwargs):
        self.pair_calls.append(kwargs)
        return 17

    def extend_forward_eval_outputs(self, outputs):
        outputs.append(mx.array([1], dtype=mx.int32))


class TestHybridRuntimeQwenMTPContract:
    def _runtime(self) -> HybridPagedAttentionRuntime:
        runtime = HybridPagedAttentionRuntime.__new__(HybridPagedAttentionRuntime)
        runtime._qwen_mtp_state = _FakeQwenMTPState()
        runtime._mamba_cache_mode = "align"
        return runtime

    def test_capability_is_exposed_only_for_ready_align_state(self) -> None:
        runtime = self._runtime()
        assert runtime.qwen_mtp_ready
        assert runtime.supports_hybrid_speculative_decode()
        runtime._mamba_cache_mode = "none"
        assert not runtime.supports_hybrid_speculative_decode()

    def test_runtime_proxies_boundary_and_pair_operations(self) -> None:
        runtime = self._runtime()
        hidden = runtime.qwen_mtp_boundary_hidden([[1, 2], [3, 4]], 7)
        assert hidden.shape == (1, 4)
        draft = runtime.qwen_mtp_run_pairs(
            hidden_rows=mx.ones((1, 4)),
            next_token_ids=[9],
            block_ids_by_group=[[1, 2], [3, 4]],
            start_pos=7,
        )
        assert draft == 17
        assert runtime._qwen_mtp_state.pair_calls[0]["start_pos"] == 7


class TestHybridSpeculativeValidationGate:
    @staticmethod
    def _scheduler_output():
        return SimpleNamespace(
            scheduled_spec_decode_tokens={"r0": [10]},
            num_invalid_spec_tokens={},
            num_scheduled_tokens={"r0": 2},
        )

    @staticmethod
    def _decode_reqs():
        return [
            (
                "r0",
                SimpleNamespace(
                    sampling_params=SamplingParams(temperature=0),
                ),
            )
        ]

    def test_hybrid_verification_fails_without_transactional_runtime(self) -> None:
        with pytest.raises(NotImplementedError, match="transactional paged Qwen MTP"):
            SpeculativeDecodeController().validate_supported(
                self._scheduler_output(),
                self._decode_reqs(),
                paged_attention_enabled=True,
                is_hybrid=True,
                speculative_config=SimpleNamespace(),
                hybrid_speculative_ready=False,
            )

    def test_hybrid_verification_opens_with_transactional_runtime(self) -> None:
        SpeculativeDecodeController().validate_supported(
            self._scheduler_output(),
            self._decode_reqs(),
            paged_attention_enabled=True,
            is_hybrid=True,
            speculative_config=SimpleNamespace(),
            hybrid_speculative_ready=True,
        )
