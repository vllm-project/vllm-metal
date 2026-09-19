# SPDX-License-Identifier: Apache-2.0
"""CPU model of the native GDN fallback's out-of-graph buffer writes."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import vllm_metal.attention.impls.linear as attention_linear


@pytest.mark.parametrize("output_dtype", [np.float32, np.float16])
def test_fallback_completes_native_writes_before_output_and_state_reads(
    monkeypatch: pytest.MonkeyPatch, output_dtype: Any
) -> None:
    """A materialized array remains available while native work is queued.

    This models the completion boundary, not the GDN arithmetic. Returning from
    the wrapper must make both downstream output math and the next state read
    observe native writes. If the completion fence is removed, eval is still a
    no-op, the returned output contains zeros, and the pool remains unchanged.
    NumPy handles every array operation here; no native kernel/GPU is run.
    """
    pending: list[Callable[[], None]] = []
    pool = np.zeros((3, 1, 2, 2), dtype=np.float32)
    pool[0] = -11  # Unscheduled state must survive the completion boundary.

    def synchronize() -> None:
        while pending:
            pending.pop(0)()

    def available_eval(*arrays: Any) -> None:
        # These buffers were allocated/materialized before the native launch;
        # graph evaluation has no dependency edge to its out-of-graph writes.
        assert all(isinstance(array, np.ndarray) for array in arrays)

    fake_mx = SimpleNamespace(
        array=np.array,
        zeros=np.zeros,
        contiguous=np.ascontiguousarray,
        result_type=np.result_type,
        int32=np.int32,
        eval=available_eval,
        synchronize=synchronize,
    )

    class QueuedNativeOps:
        def gdn_linear_attention(self, *args: Any) -> None:
            state_pool, slots, output = args[5], args[7], args[8]
            assert state_pool is pool

            # The native encoder has recorded work but has not completed it.
            # Keep references as the real encoder keeps native temporaries.
            def complete() -> None:
                for i, slot in enumerate(slots):
                    state_pool[slot] = 7 + i
                    output[i] = 3 + i

            pending.append(complete)

    monkeypatch.setattr(attention_linear, "mx", fake_mx)
    monkeypatch.setattr(attention_linear, "get_ops", QueuedNativeOps)
    wrapper = SimpleNamespace(
        _inner=SimpleNamespace(
            num_k_heads=1, num_v_heads=1, head_k_dim=2, head_v_dim=2
        ),
        _gdn_cache_idx=0,
        _gdn_state_cache=SimpleNamespace(
            recurrent_states=[pool], apply_pending_recurrent_state=lambda _: None
        ),
    )
    state = SimpleNamespace(
        total_tokens=2,
        cu_seqlens=[0, 1, 2],
        slot_ids=[2, 1],
        x=np.ones((1, 2, 2), dtype=output_dtype),
    )
    qkv = np.ones((1, 2, 1, 2), dtype=output_dtype)
    gates = np.ones((1, 2, 1), dtype=output_dtype)

    output = attention_linear.GDNPagedAttentionWrapper._run_recurrent_fallback(
        wrapper, qkv, qkv, qkv, gates, gates, state
    )

    # Consume immediately, with no test-side synchronize/eval. The cast inside
    # the wrapper must also occur after completion, including a dtype change.
    projected = output.reshape(2, 2) @ np.array([2, 3], dtype=output_dtype)
    next_state_read = pool[state.slot_ids].sum(axis=(1, 2, 3))
    np.testing.assert_array_equal(projected, np.array([15, 20], dtype=output_dtype))
    np.testing.assert_array_equal(next_state_read, np.array([28, 32], dtype=np.float32))
    np.testing.assert_array_equal(pool[0], np.full((1, 2, 2), -11, dtype=np.float32))
    assert output.dtype == output_dtype
    assert not pending
