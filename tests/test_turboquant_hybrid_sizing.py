# SPDX-License-Identifier: Apache-2.0
"""Hybrid + TurboQuant sizing must match the runtime's packed KV layout.

Regression tests for #468: ``HybridPagedAttentionRuntime`` compresses the
SDPA layers when TurboQuant is enabled, but every scheduler-visible sizing
path (KV cache specs, planner per-block bytes, one-sequence estimates,
hybrid block-size alignment) used the uncompressed fp16 math, so
``check_enough_kv_cache_memory`` rejected contexts that fit and the paged
planner budgeted ~3-4x fewer blocks than the packed cache needs.
"""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import torch
from vllm.model_executor.models import ModelRegistry
from vllm.v1.kv_cache_interface import MambaSpec

from tests.stub_runner import make_stub_runner
from vllm_metal.config import MetalConfig
from vllm_metal.platform import MetalPlatform
from vllm_metal.v1.cache_policy import (
    TurboQuantAttentionSpec,
    _turboquant_page_size_bytes,
)

BLOCK_SIZE = 1056
NUM_LAYERS = 8
SDPA_LAYERS = [3, 7]
KV_HEADS = 4
HEAD_DIM = 128
K_QUANT = "q4_0"
V_QUANT = "q4_0"

TQ_PAGE = _turboquant_page_size_bytes(
    block_size=BLOCK_SIZE,
    num_kv_heads=KV_HEADS,
    head_dim=HEAD_DIM,
    k_quant=K_QUANT,
    v_quant=V_QUANT,
)


def _hybrid_runner():
    return make_stub_runner(
        model_args={"full_attention_interval": 4},
        num_layers=NUM_LAYERS,
        full_attention_interval=4,
        sdpa_layer_indices=set(SDPA_LAYERS),
        num_sdpa_layers=len(SDPA_LAYERS),
        num_linear_layers=NUM_LAYERS - len(SDPA_LAYERS),
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        kv_cache_dtype=mx.float16,
        cache_config=SimpleNamespace(
            block_size=BLOCK_SIZE, mamba_page_size_padded=None
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        linear_conv_kernel_dim=4,
        linear_conv_dim=1024,
        linear_num_v_heads=16,
        linear_value_head_dim=64,
        linear_key_head_dim=64,
    )


def _tq_config() -> MetalConfig:
    return MetalConfig(
        memory_fraction=-1.0,
        use_mlx=False,
        mlx_device="gpu",
        debug=False,
        use_paged_attention=True,
        turboquant=True,
        k_quant=K_QUANT,
        v_quant=V_QUANT,
    )


class TestHybridTurboQuantCachePolicy:
    def test_sdpa_specs_report_packed_pages(self) -> None:
        runner = _hybrid_runner()
        with patch("vllm_metal.v1.cache_policy.get_config", return_value=_tq_config()):
            specs = runner._cache_policy.get_kv_cache_spec()

        for idx in SDPA_LAYERS:
            spec = specs[f"layers.{idx}.self_attn"]
            assert isinstance(spec, TurboQuantAttentionSpec)
            assert spec.page_size_bytes == TQ_PAGE
        for idx in set(range(NUM_LAYERS)) - set(SDPA_LAYERS):
            assert isinstance(specs[f"layers.{idx}.linear_attn"], MambaSpec)

    def test_per_block_bytes_use_packed_pages(self) -> None:
        runner = _hybrid_runner()
        with patch("vllm_metal.v1.cache_policy.get_config", return_value=_tq_config()):
            per_block = runner._cache_policy.get_cache_block_size_bytes()

        assert per_block == len(SDPA_LAYERS) * TQ_PAGE

    def test_one_sequence_estimate_includes_linear_state(self) -> None:
        runner = _hybrid_runner()
        max_model_len = 2048
        with patch("vllm_metal.v1.cache_policy.get_config", return_value=_tq_config()):
            estimate = runner._cache_policy.estimate_one_sequence_kv_bytes(
                max_model_len=max_model_len, block_size=BLOCK_SIZE
            )
        linear_bytes = runner._cache_policy.linear_cache_bytes_per_slot()

        aligned_tokens = -(-max_model_len // BLOCK_SIZE) * BLOCK_SIZE
        sdpa_bytes = len(SDPA_LAYERS) * _turboquant_page_size_bytes(
            block_size=aligned_tokens,
            num_kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            k_quant=K_QUANT,
            v_quant=V_QUANT,
        )
        assert estimate == sdpa_bytes + linear_bytes


class TestTurboQuantHybridAlignment:
    """MetalPlatform re-aligns hybrid block size with packed page math."""

    # Large GDN state so the fp16-aligned block is too small for the
    # packed page: (3*8192 + 32*128*128) * 2 bytes = 1,097,728 per request.
    _MAMBA_SHAPES = ((3, 8192), (32, 128, 128))
    _MAMBA_PAGE = (3 * 8192 + 32 * 128 * 128) * 2
    _TQ_PAGE_1_TOKEN = _turboquant_page_size_bytes(
        block_size=1,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        k_quant=K_QUANT,
        v_quant=V_QUANT,
    )

    def _vllm_config(self, *, block_size: int) -> SimpleNamespace:
        return SimpleNamespace(
            model_config=SimpleNamespace(
                is_hybrid=True,
                architecture="StubHybridForCausalLM",
                get_num_kv_heads=lambda parallel_config: KV_HEADS,
                get_head_size=lambda: HEAD_DIM,
            ),
            cache_config=SimpleNamespace(
                block_size=block_size,
                mamba_cache_mode="none",
                mamba_block_size=None,
                user_specified_mamba_block_size=False,
                mamba_page_size_padded=block_size * 2 * KV_HEADS * HEAD_DIM * 2,
            ),
            parallel_config=SimpleNamespace(),
        )

    def _stub_model_cls(self) -> SimpleNamespace:
        return SimpleNamespace(
            get_mamba_state_shape_from_config=lambda vllm_config: self._MAMBA_SHAPES,
            get_mamba_state_dtype_from_config=lambda vllm_config: (
                torch.float16,
                torch.float16,
            ),
        )

    def test_realign_grows_block_and_pads_mamba_to_packed_page(
        self, monkeypatch
    ) -> None:
        # Simulate the post-super() state: fp16 alignment picked
        # 16 * cdiv(mamba_page, 16 * fp16_page_1_token) = 544 tokens.
        vllm_config = self._vllm_config(block_size=544)
        monkeypatch.setattr("vllm_metal.platform.get_config", lambda: _tq_config())
        monkeypatch.setattr(
            ModelRegistry,
            "resolve_model_cls",
            lambda architecture, model_config: (self._stub_model_cls(), None),
        )

        MetalPlatform._realign_hybrid_block_size_for_turboquant(vllm_config)

        expected_block = 16 * -(-self._MAMBA_PAGE // (16 * self._TQ_PAGE_1_TOKEN))
        assert vllm_config.cache_config.block_size == expected_block
        expected_page = expected_block * self._TQ_PAGE_1_TOKEN
        assert expected_page >= self._MAMBA_PAGE
        assert vllm_config.cache_config.mamba_page_size_padded == expected_page

    def test_realign_noop_without_turboquant(self, monkeypatch) -> None:
        vllm_config = self._vllm_config(block_size=544)
        before_padded = vllm_config.cache_config.mamba_page_size_padded
        config = _tq_config()
        config.turboquant = False
        monkeypatch.setattr("vllm_metal.platform.get_config", lambda: config)

        MetalPlatform._realign_hybrid_block_size_for_turboquant(vllm_config)

        assert vllm_config.cache_config.block_size == 544
        assert vllm_config.cache_config.mamba_page_size_padded == before_padded

    def test_realign_noop_for_non_hybrid(self, monkeypatch) -> None:
        vllm_config = self._vllm_config(block_size=16)
        vllm_config.model_config.is_hybrid = False
        monkeypatch.setattr("vllm_metal.platform.get_config", lambda: _tq_config())

        MetalPlatform._realign_hybrid_block_size_for_turboquant(vllm_config)

        assert vllm_config.cache_config.block_size == 16
