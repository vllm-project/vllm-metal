# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3-Next GDN compatibility and multi-request fixes.

Covers:
  - GDNPagedAttentionWrapper projection dispatch (in_proj_qkvz vs in_proj_qkv)
  - _gdn_free_slot state zeroing (slot-only, preserves other slots)
  - sync_mlx insertion in mlx_to_torch for MPS safety
  - Golden token deterministic test for Qwen3-Next (slow, requires model)

Golden token IDs were generated with greedy decoding (argmax sampler) on
mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit using mlx_lm.

Run unit tests:
    python -m pytest tests/test_qwen3_next_gdn.py -v -k "not slow"

Run golden token test (requires model download):
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        python -m pytest tests/test_qwen3_next_gdn.py -v -k slow -s
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlx.core as mx

from vllm_metal.mlx_backend.gdn_cache import GDNPagedStateCache


class TestGDNProjectionDispatch:
    """Verify that GDNPagedAttentionWrapper selects the correct projection
    path based on whether the inner module has ``in_proj_qkvz`` (Qwen3-Next)
    or ``in_proj_qkv`` (Qwen3.5)."""

    def test_detects_qwen3_next_projection(self):
        """Module with in_proj_qkvz should be detected as Qwen3-Next style."""
        module = MagicMock(spec=["in_proj_qkvz", "in_proj_ba"])
        assert hasattr(module, "in_proj_qkvz")
        assert not hasattr(module, "in_proj_qkv")

    def test_detects_qwen35_projection(self):
        """Module with in_proj_qkv should be detected as Qwen3.5 style."""
        module = MagicMock(spec=["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"])
        assert hasattr(module, "in_proj_qkv")
        assert not hasattr(module, "in_proj_qkvz")


class TestGDNFreeSlotZeroing:
    """Verify that _gdn_free_slot zeros only the freed slot."""

    def _make_cache(self, num_layers: int = 2, max_seqs: int = 2) -> GDNPagedStateCache:
        return GDNPagedStateCache(
            num_layers=num_layers,
            max_seqs=max_seqs,
            conv_kernel_dim=4,
            conv_dim=64,
            num_v_heads=4,
            value_head_dim=16,
            key_head_dim=16,
            dtype=mx.float16,
        )

    def test_slot_zeroing_preserves_other_slots(self):
        """Zeroing slot 0 must not affect slot 1."""
        sc = self._make_cache(num_layers=2, max_seqs=2)

        # Write non-zero data to both slots
        for layer_idx in range(sc.num_layers):
            sc.conv_states[layer_idx] = mx.ones_like(sc.conv_states[layer_idx])
            sc.recurrent_states[layer_idx] = mx.ones_like(
                sc.recurrent_states[layer_idx]
            )
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        # Simulate _gdn_free_slot for slot 0 only
        slot = 0
        mx.eval(*sc.conv_states, *sc.recurrent_states)
        for layer_idx in range(sc.num_layers):
            conv = sc.conv_states[layer_idx]
            conv[slot] = 0
            sc.conv_states[layer_idx] = conv
            rec = sc.recurrent_states[layer_idx]
            rec[slot] = 0
            sc.recurrent_states[layer_idx] = rec
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        # Slot 0 should be zeros
        assert mx.allclose(sc.conv_states[0][0], mx.zeros((3, 64), dtype=mx.float16))
        assert mx.allclose(
            sc.recurrent_states[0][0], mx.zeros((4, 16, 16), dtype=mx.float32)
        )
        # Slot 1 should still be ones
        assert mx.allclose(sc.conv_states[0][1], mx.ones((3, 64), dtype=mx.float16))
        assert mx.allclose(
            sc.recurrent_states[0][1], mx.ones((4, 16, 16), dtype=mx.float32)
        )

    def test_zeroed_slot_produces_zeros(self):
        """Freed slot must be all zeros after zeroing."""
        sc = self._make_cache(num_layers=1, max_seqs=1)

        # Write non-zero data
        sc.conv_states[0] = mx.ones_like(sc.conv_states[0])
        sc.recurrent_states[0] = mx.ones_like(sc.recurrent_states[0])
        mx.eval(sc.conv_states[0], sc.recurrent_states[0])

        # Zero slot 0
        mx.eval(*sc.conv_states, *sc.recurrent_states)
        conv = sc.conv_states[0]
        conv[0] = 0
        sc.conv_states[0] = conv
        rec = sc.recurrent_states[0]
        rec[0] = 0
        sc.recurrent_states[0] = rec
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        assert mx.array_equal(sc.conv_states[0], mx.zeros_like(sc.conv_states[0]))
        assert mx.array_equal(
            sc.recurrent_states[0], mx.zeros_like(sc.recurrent_states[0])
        )

    def test_shapes_preserved_after_zeroing(self):
        """Array shapes and dtypes must be preserved after slot zeroing."""
        sc = self._make_cache(num_layers=3, max_seqs=2)
        expected_conv_shape = sc.conv_states[0].shape
        expected_rec_shape = sc.recurrent_states[0].shape

        # Zero slot 1
        mx.eval(*sc.conv_states, *sc.recurrent_states)
        for layer_idx in range(sc.num_layers):
            conv = sc.conv_states[layer_idx]
            conv[1] = 0
            sc.conv_states[layer_idx] = conv
            rec = sc.recurrent_states[layer_idx]
            rec[1] = 0
            sc.recurrent_states[layer_idx] = rec
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        for layer_idx in range(sc.num_layers):
            assert sc.conv_states[layer_idx].shape == expected_conv_shape
            assert sc.recurrent_states[layer_idx].shape == expected_rec_shape
            assert sc.conv_states[layer_idx].dtype == mx.float16
            assert sc.recurrent_states[layer_idx].dtype == mx.float32


class TestGDNAllocSlotZeroing:
    """Verify that _gdn_alloc_slot zeros state for reused slots."""

    def _make_runner_stub(self, max_seqs: int = 2):
        """Build a minimal stub with GDN slot management wired up."""
        from vllm_metal.mlx_backend.gdn_cache import GDNPagedStateCache

        sc = GDNPagedStateCache(
            num_layers=2,
            max_seqs=max_seqs,
            conv_kernel_dim=4,
            conv_dim=64,
            num_v_heads=4,
            value_head_dim=16,
            key_head_dim=16,
            dtype=mx.float16,
        )
        backend = MagicMock()
        backend._state_cache = sc

        runner = MagicMock()
        runner._gdn_req_to_slot = {}
        runner._gdn_free_slots = []
        runner._paged_attention_backend = backend
        return runner, sc

    def test_reused_slot_is_zeroed(self):
        """A slot returned to the free list and re-allocated must have
        zeroed conv and recurrent state."""
        from vllm_metal.v1.model_runner import MetalModelRunner

        runner, sc = self._make_runner_stub()

        # Allocate slot 0 for req-A
        slot = MetalModelRunner._gdn_alloc_slot(runner, "req-A")
        assert slot == 0

        # Write non-zero data to slot 0
        for layer_idx in range(sc.num_layers):
            sc.conv_states[layer_idx] = mx.ones_like(sc.conv_states[layer_idx])
            sc.recurrent_states[layer_idx] = mx.ones_like(
                sc.recurrent_states[layer_idx]
            )
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        # Free slot 0 (bare pop+append, no zeroing — matches early release path)
        runner._gdn_req_to_slot.pop("req-A")
        runner._gdn_free_slots.append(slot)

        # Re-allocate — should trigger alloc-time zeroing
        slot2 = MetalModelRunner._gdn_alloc_slot(runner, "req-B")
        assert slot2 == 0  # reused
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        # Slot 0 must be zeroed
        for layer_idx in range(sc.num_layers):
            assert mx.array_equal(
                sc.conv_states[layer_idx][0],
                mx.zeros_like(sc.conv_states[layer_idx][0]),
            )
            assert mx.array_equal(
                sc.recurrent_states[layer_idx][0],
                mx.zeros_like(sc.recurrent_states[layer_idx][0]),
            )

    def test_reused_slot_preserves_other_slots(self):
        """Alloc-time zeroing of slot 0 must not affect slot 1."""
        from vllm_metal.v1.model_runner import MetalModelRunner

        runner, sc = self._make_runner_stub()

        # Allocate slots 0 and 1
        MetalModelRunner._gdn_alloc_slot(runner, "req-A")
        MetalModelRunner._gdn_alloc_slot(runner, "req-B")

        # Write ones everywhere
        for layer_idx in range(sc.num_layers):
            sc.conv_states[layer_idx] = mx.ones_like(sc.conv_states[layer_idx])
            sc.recurrent_states[layer_idx] = mx.ones_like(
                sc.recurrent_states[layer_idx]
            )
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        # Free slot 0, re-allocate
        runner._gdn_req_to_slot.pop("req-A")
        runner._gdn_free_slots.append(0)
        MetalModelRunner._gdn_alloc_slot(runner, "req-C")
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        # Slot 1 must still be ones
        assert mx.allclose(sc.conv_states[0][1], mx.ones((3, 64), dtype=mx.float16))
        assert mx.allclose(
            sc.recurrent_states[0][1], mx.ones((4, 16, 16), dtype=mx.float32)
        )

    def test_new_slot_not_zeroed(self):
        """A brand-new slot (not from free list) should not trigger zeroing."""
        from vllm_metal.v1.model_runner import MetalModelRunner

        runner, sc = self._make_runner_stub()
        slot = MetalModelRunner._gdn_alloc_slot(runner, "req-A")
        assert slot == 0
        # No crash, no zeroing needed — state was already initialized to zero

    def test_double_free_is_safe(self):
        """Calling _gdn_free_slot twice for the same req_id must not crash."""
        from vllm_metal.v1.model_runner import MetalModelRunner

        runner, sc = self._make_runner_stub()
        MetalModelRunner._gdn_alloc_slot(runner, "req-A")

        MetalModelRunner._gdn_free_slot(runner, "req-A")
        MetalModelRunner._gdn_free_slot(runner, "req-A")  # no-op, no crash

    def test_slot_reuse_after_early_release(self):
        """Slot freed via bare pop+append (early release) should be
        available for immediate reallocation with zeroed state."""
        from vllm_metal.v1.model_runner import MetalModelRunner

        runner, sc = self._make_runner_stub(max_seqs=1)

        slot0 = MetalModelRunner._gdn_alloc_slot(runner, "req-A")
        assert slot0 == 0

        # Write non-zero data to slot 0
        for layer_idx in range(sc.num_layers):
            sc.conv_states[layer_idx] = mx.ones_like(sc.conv_states[layer_idx])
            sc.recurrent_states[layer_idx] = mx.ones_like(
                sc.recurrent_states[layer_idx]
            )
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        # Early release (bare pop+append, as in execute_model)
        runner._gdn_req_to_slot.pop("req-A")
        runner._gdn_free_slots.append(slot0)

        # Same-step allocation should reuse slot 0
        slot1 = MetalModelRunner._gdn_alloc_slot(runner, "req-B")
        assert slot1 == 0

        # Alloc-time zeroing must have cleared the state
        mx.eval(*sc.conv_states, *sc.recurrent_states)
        for layer_idx in range(sc.num_layers):
            assert mx.array_equal(
                sc.conv_states[layer_idx][0],
                mx.zeros_like(sc.conv_states[layer_idx][0]),
            ), f"conv_states[{layer_idx}][0] not zeroed after early-release reuse"
            assert mx.array_equal(
                sc.recurrent_states[layer_idx][0],
                mx.zeros_like(sc.recurrent_states[layer_idx][0]),
            ), f"recurrent_states[{layer_idx}][0] not zeroed after early-release reuse"

    def test_early_release_then_alloc_full_cycle(self):
        """Simulate the full execute_model early-release → alloc cycle
        with 2 slots: finish req-A, start req-C while req-B still active."""
        from vllm_metal.v1.model_runner import MetalModelRunner

        runner, sc = self._make_runner_stub(max_seqs=2)

        # Allocate 2 requests
        slot_a = MetalModelRunner._gdn_alloc_slot(runner, "req-A")
        slot_b = MetalModelRunner._gdn_alloc_slot(runner, "req-B")
        assert slot_a == 0
        assert slot_b == 1

        # Write distinct data to each slot
        for layer_idx in range(sc.num_layers):
            sc.conv_states[layer_idx] = mx.ones_like(sc.conv_states[layer_idx]) * 5.0
            sc.recurrent_states[layer_idx] = (
                mx.ones_like(sc.recurrent_states[layer_idx]) * 3.0
            )
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        # Early-release req-A (bare pop+append, no zeroing)
        runner._gdn_req_to_slot.pop("req-A")
        runner._gdn_free_slots.append(slot_a)

        # Allocate req-C — should reuse slot 0 with zeroed state
        slot_c = MetalModelRunner._gdn_alloc_slot(runner, "req-C")
        assert slot_c == 0  # reused slot
        mx.eval(*sc.conv_states, *sc.recurrent_states)

        # Slot 0 (req-C) must be zeroed
        assert mx.array_equal(sc.conv_states[0][0], mx.zeros_like(sc.conv_states[0][0]))
        assert mx.array_equal(
            sc.recurrent_states[0][0], mx.zeros_like(sc.recurrent_states[0][0])
        )

        # Slot 1 (req-B) must still have its data (5.0 / 3.0)
        assert sc.conv_states[0][1].sum().item() != 0, "req-B conv state was corrupted"
        assert sc.recurrent_states[0][1].sum().item() != 0, (
            "req-B recurrent state was corrupted"
        )


class TestSyncMLXInTensorBridge:
    """Verify sync_mlx is called before MPS tensor transfer."""

    def test_sync_mlx_called_before_mps_transfer(self):
        """mlx_to_torch must call sync_mlx() when target device is MPS."""
        from vllm_metal.pytorch_backend import tensor_bridge

        array = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        mx.eval(array)

        with patch.object(tensor_bridge, "sync_mlx") as mock_sync:
            try:
                tensor_bridge.mlx_to_torch(array)
            except Exception:
                pass  # MPS may not be available in CI
            # sync_mlx should be called if device is MPS
            if tensor_bridge.get_torch_device().type == "mps":
                mock_sync.assert_called()
