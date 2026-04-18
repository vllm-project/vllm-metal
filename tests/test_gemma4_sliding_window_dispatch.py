# SPDX-License-Identifier: Apache-2.0
"""End-to-end kernel-dispatch verification for Gemma4 sliding window.

Spies on the Metal ``paged_attention_primitive`` op during a real vLLM
inference and asserts the per-layer ``sliding_window`` values reach the
kernel with the correct split between sliding-attention and
full-attention layers.  Complements ``test_sliding_window_wiring.py``
which exercises the same wiring without running the Metal kernel.

Gemma4 E2B is used (not 31B) because:
- E2B has uniform KV heads, so the paged-attention guard in
  ``require_uniform_kv_heads`` admits it today.  31B support unblocks
  after upstream PR #279 lands.
- E2B has YOCO (``num_kv_shared_layers=20``), so every model forward
  exercises the shared-layer cache-index retrieval -- shared layers
  must still receive a sliding_window matching their own attention
  type, via ``cache_idx_map``.

Run:
    GEMMA4_MODEL_PATH=/path/to/gemma-4-E2B-it \\
        pytest tests/test_gemma4_sliding_window_dispatch.py -v -s -m slow
"""

from __future__ import annotations

import os
from collections import Counter

import pytest

MODEL_ENV = "GEMMA4_MODEL_PATH"

# --- Gemma4 E2B text_config invariants ---
# Verified against mlx-community/gemma-4-E2B-it config.json.  Every 5th
# layer (indices 4, 9, 14, 19, 24, 29, 34) is ``full_attention``; the
# other 28 are ``sliding_attention``.
_E2B_SLIDING_WINDOW = 512
_E2B_NUM_SLIDING_LAYERS = 28
_E2B_NUM_FULL_LAYERS = 7
_E2B_TOTAL_LAYERS = _E2B_NUM_SLIDING_LAYERS + _E2B_NUM_FULL_LAYERS

_NO_WINDOW = -1

# Position of ``sliding_window`` in ``paged_attention_primitive``'s
# positional signature (see ``attention_sdpa.py:489-510``).
_SLIDING_WINDOW_ARG_INDEX = 11

# Ratio tolerance: layer_types is a config constant, but prefill and
# decode may dispatch slightly different counts across forwards, so we
# accept a 1% slack.
_RATIO_TOLERANCE = 0.01


@pytest.fixture(scope="module")
def kernel_sliding_window_log() -> list[int]:
    """Run one Gemma4 inference with a spy on ``paged_attention_primitive``.

    Returns the list of ``sliding_window`` ints passed to every kernel
    dispatch during the inference.  Skips if the model path env var is
    unset.
    """
    model_path = os.environ.get(MODEL_ENV)
    if not model_path:
        pytest.skip(f"{MODEL_ENV} not set -- skipping slow kernel dispatch test")
    if not os.path.isdir(model_path):
        pytest.skip(f"{MODEL_ENV}={model_path} is not a directory")

    with pytest.MonkeyPatch.context() as mp:
        # Single-process mode for determinism (mirrors test_gemma4_golden.py).
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        # Install the spy BEFORE importing vllm or creating the LLM --
        # ``get_ops()`` caches the module the first time it runs, so the
        # patch must sit on the attribute before any forward pass starts.
        from vllm_metal.metal import get_ops

        ops = get_ops()
        orig_fn = ops.paged_attention_primitive
        captured: list[int] = []

        def spy(*args, **kwargs):
            sw = (
                args[_SLIDING_WINDOW_ARG_INDEX]
                if len(args) > _SLIDING_WINDOW_ARG_INDEX
                else kwargs.get("sliding_window")
            )
            captured.append(sw)
            return orig_fn(*args, **kwargs)

        mp.setattr(ops, "paged_attention_primitive", spy)

        from vllm import LLM, SamplingParams

        llm = LLM(model=model_path, max_model_len=512, max_num_seqs=1)
        sp = SamplingParams(temperature=0, max_tokens=5, ignore_eos=True)
        llm.generate(["The capital of France is"], sp)

        return captured


@pytest.mark.slow
class TestGemma4KernelReceivesPerLayerSlidingWindow:
    """Kernel-level assertions on the sliding_window values dispatched."""

    def test_only_expected_window_values_appear(
        self, kernel_sliding_window_log: list[int]
    ) -> None:
        """No stray values leak from wiring errors."""
        # Act
        unexpected = {
            w
            for w in kernel_sliding_window_log
            if w not in (_E2B_SLIDING_WINDOW, _NO_WINDOW)
        }
        # Assert
        assert not unexpected, (
            f"kernel received unexpected sliding_window values: {unexpected}"
        )

    def test_both_sliding_and_full_layers_dispatch(
        self, kernel_sliding_window_log: list[int]
    ) -> None:
        """``sliding_window=512`` and ``-1`` both appear."""
        # Act
        counts = Counter(kernel_sliding_window_log)
        # Assert
        assert counts[_E2B_SLIDING_WINDOW] > 0, (
            "sliding layers never received their window -- enforcement is "
            "not reaching the kernel"
        )
        assert counts[_NO_WINDOW] > 0, (
            "full layers never received -1 -- they may be incorrectly getting a window"
        )

    def test_ratio_matches_layer_types_config(
        self, kernel_sliding_window_log: list[int]
    ) -> None:
        """Sliding/full dispatch ratio matches the 28:7 layer_types split.

        Each model forward dispatches one kernel call per layer (YOCO
        shared layers reuse the cache index of their same-type reference,
        so the retrieved ``sliding_window`` still matches their own
        attention type).  The aggregate ratio is therefore fixed by
        ``layer_types`` and not stochastic.
        """
        # Act
        counts = Counter(kernel_sliding_window_log)
        sliding = counts[_E2B_SLIDING_WINDOW]
        full = counts[_NO_WINDOW]
        total = sliding + full
        assert total > 0
        actual_sliding = sliding / total
        actual_full = full / total
        expected_sliding = _E2B_NUM_SLIDING_LAYERS / _E2B_TOTAL_LAYERS
        expected_full = _E2B_NUM_FULL_LAYERS / _E2B_TOTAL_LAYERS
        # Assert
        assert actual_sliding == pytest.approx(
            expected_sliding, abs=_RATIO_TOLERANCE
        ), f"sliding ratio {actual_sliding:.3f} != expected {expected_sliding:.3f}"
        assert actual_full == pytest.approx(expected_full, abs=_RATIO_TOLERANCE), (
            f"full ratio {actual_full:.3f} != expected {expected_full:.3f}"
        )
