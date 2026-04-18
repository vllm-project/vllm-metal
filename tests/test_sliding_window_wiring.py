# SPDX-License-Identifier: Apache-2.0
"""End-to-end wiring tests for per-layer sliding window enforcement.

Verifies the full chain from model config to Metal KV cache:

    config.layer_types + config.sliding_window
      -> DefaultModelAdapter.build_sliding_window_per_layer
      -> MetalModelRunner.sliding_window_per_layer
      -> ModelCachePolicy._build_mha_backend (slice to num_cache_layers)
      -> MHAPagedAttentionBackend (kwarg)
      -> MetalPagedKVCache.sliding_window_per_layer

Kernel-level correctness (that a ``sliding_window`` value actually
masks out-of-window tokens) is separately validated by
``test_metal_unified_attn`` in ``test_metal_unified_attention.py``;
both the production ``paged_attention_primitive`` and the test helper
``metal_unified_attention`` dispatch to the same
``paged_attention_v2_online`` kernel (see ``paged_ops.cpp``).
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from tests.stub_runner import make_stub_runner
from vllm_metal.v1.model_adapter import DefaultModelAdapter

# --- Test parameters (kept small for fast execution) ---
_SLIDING_WINDOW = 64
_BLOCK_SIZE = 16
_NUM_BLOCKS = 2
_NUM_KV_HEADS = 4
_HEAD_DIM = 64
_KV_CACHE_DTYPE = mx.bfloat16

# Sentinel used by the Metal kernel to mean "no window enforcement".
# Must match ``MetalPagedKVCache`` default and the kernel-side
# ``if (sliding_window >= 0)`` guard in ``pagedattention.metal``.
_NO_WINDOW = -1


# ===
# Non-YOCO: every model layer has its own cache entry.
# ===


class TestNonYocoSlidingWindowWiring:
    """Config -> cache with identity layer-to-cache mapping."""

    def test_gemma4_like_config_lands_in_kv_cache(self) -> None:
        """Mixed sliding/full config produces matching per-layer list in cache."""
        # Arrange: 4-layer model mixing sliding and full attention, mirroring
        # Gemma4's ``layer_types`` pattern.
        layer_types = [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
        ]
        num_layers = len(layer_types)
        expected = [
            _SLIDING_WINDOW,
            _NO_WINDOW,
            _SLIDING_WINDOW,
            _SLIDING_WINDOW,
        ]
        model_args = {
            "layer_types": layer_types,
            "sliding_window": _SLIDING_WINDOW,
            "num_hidden_layers": num_layers,
        }
        adapter = DefaultModelAdapter()
        sw_list = adapter.build_sliding_window_per_layer(model_args, num_layers)

        runner = make_stub_runner(
            model_args=model_args,
            num_layers=num_layers,
            num_kv_cache_layers=num_layers,
            num_kv_heads=_NUM_KV_HEADS,
            head_dim=_HEAD_DIM,
            kv_cache_dtype=_KV_CACHE_DTYPE,
            cache_config=SimpleNamespace(block_size=_BLOCK_SIZE),
            sliding_window_per_layer=sw_list,
            _yoco_cache_mapping=None,
        )

        # Act: build the backend, which constructs the KV cache with the
        # per-layer window list sliced by the cache policy.
        backend = runner.build_paged_attention_backend(block_size=_BLOCK_SIZE)
        backend.initialize(num_blocks=_NUM_BLOCKS)

        # Assert
        assert backend._cache is not None
        assert backend._cache.sliding_window_per_layer == expected

    def test_model_without_sliding_window_defaults_to_disabled(self) -> None:
        """Non-Gemma config (Qwen3/Llama) yields all-disabled windows.

        Regression guard: adding the sliding-window plumbing must not
        accidentally enforce a window on models that never had one.
        """
        # Arrange: no ``layer_types`` or ``sliding_window`` in the config.
        num_layers = 4
        model_args = {"num_hidden_layers": num_layers}

        adapter = DefaultModelAdapter()
        sw_list = adapter.build_sliding_window_per_layer(model_args, num_layers)
        assert sw_list is None

        runner = make_stub_runner(
            model_args=model_args,
            num_layers=num_layers,
            num_kv_cache_layers=num_layers,
            num_kv_heads=_NUM_KV_HEADS,
            head_dim=_HEAD_DIM,
            kv_cache_dtype=_KV_CACHE_DTYPE,
            cache_config=SimpleNamespace(block_size=_BLOCK_SIZE),
            sliding_window_per_layer=sw_list,
            _yoco_cache_mapping=None,
        )

        # Act
        backend = runner.build_paged_attention_backend(block_size=_BLOCK_SIZE)
        backend.initialize(num_blocks=_NUM_BLOCKS)

        # Assert: all disabled (preserves pre-PR behaviour for non-sliding models)
        assert backend._cache is not None
        assert backend._cache.sliding_window_per_layer == [_NO_WINDOW] * num_layers


# ===
# YOCO: shared cache layers indexed via ``cache_idx_map``.
# ===


class TestYocoSlidingWindowWiring:
    """YOCO shared layers resolve to the correct reference-layer window."""

    def test_shared_layer_cache_idx_retrieves_correct_window(self) -> None:
        """Every model layer (unique + shared) resolves to the right window.

        Gemma4 YOCO: the last ``num_kv_shared_layers`` layers do not own a
        cache entry; instead their ``cache_idx_map`` points back to the
        most recent unique layer of the same attention type.  In
        ``sdpa_forward`` the retrieval key is
        ``kv_cache.sliding_window_per_layer[cache_idx]``, so for correctness
        each shared layer must resolve to a window matching its own type.
        """
        # Arrange: 5 layers, last 2 shared with earlier same-type unique layers.
        # Unique segment: indices 0..2; shared segment: indices 3..4.
        layer_types = [
            "sliding_attention",  # unique, cache 0  -> sliding
            "full_attention",  # unique, cache 1  -> full
            "sliding_attention",  # unique, cache 2  -> sliding
            "full_attention",  # shared, cache_idx = 1 (last full)
            "sliding_attention",  # shared, cache_idx = 2 (last sliding)
        ]
        num_layers = len(layer_types)
        num_kv_shared = 2
        num_unique = num_layers - num_kv_shared
        model_args = {
            "layer_types": layer_types,
            "sliding_window": _SLIDING_WINDOW,
            "num_hidden_layers": num_layers,
            "num_kv_shared_layers": num_kv_shared,
        }

        adapter = DefaultModelAdapter()
        sw_list = adapter.build_sliding_window_per_layer(model_args, num_layers)
        yoco = adapter.build_yoco_cache_mapping(model_args)
        assert yoco is not None
        _, cache_idx_map = yoco

        runner = make_stub_runner(
            model_args=model_args,
            num_layers=num_layers,
            num_kv_cache_layers=num_unique,
            num_kv_heads=_NUM_KV_HEADS,
            head_dim=_HEAD_DIM,
            kv_cache_dtype=_KV_CACHE_DTYPE,
            cache_config=SimpleNamespace(block_size=_BLOCK_SIZE),
            sliding_window_per_layer=sw_list,
            _yoco_cache_mapping=yoco,
        )

        # Act
        backend = runner.build_paged_attention_backend(block_size=_BLOCK_SIZE)
        backend.initialize(num_blocks=_NUM_BLOCKS)

        # Assert #1: cache stores one entry per unique layer, matching the
        # identity segment of the YOCO map.
        cache = backend._cache
        assert cache is not None
        assert cache.sliding_window_per_layer == [
            _SLIDING_WINDOW,
            _NO_WINDOW,
            _SLIDING_WINDOW,
        ]

        # Assert #2: every model layer (unique + shared) retrieves a window
        # matching its own attention type -- this is the invariant that
        # ``sdpa_forward`` relies on when indexing by ``cache_idx``.
        for model_layer_idx, lt in enumerate(layer_types):
            cache_idx = cache_idx_map[model_layer_idx]
            retrieved = cache.sliding_window_per_layer[cache_idx]
            expected = _SLIDING_WINDOW if lt == "sliding_attention" else _NO_WINDOW
            assert retrieved == expected, (
                f"layer {model_layer_idx} ({lt}): "
                f"cache_idx={cache_idx}, got {retrieved}, expected {expected}"
            )
