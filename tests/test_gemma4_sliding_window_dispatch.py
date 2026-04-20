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
import re
from collections import Counter
from dataclasses import dataclass

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

# The bug only manifests once the prompt exceeds Gemma4 E2B's 512-token
# sliding window. Build a prompt with a comfortable margin so the kernel
# must actually decide whether to enforce the window.
_LONG_CONTEXT_TOKEN_MARGIN = 128
_LONG_CONTEXT_MIN_TOKENS = _E2B_SLIDING_WINDOW + _LONG_CONTEXT_TOKEN_MARGIN
_MAX_MODEL_LEN = 1024
_MAX_TOKENS = 1
_PROMPT_FRAGMENT = "The capital of France is Paris. "

# Ratio tolerance: layer_types is a config constant, but prefill and
# decode may dispatch slightly different counts across forwards, so we
# accept a 1% slack.
_RATIO_TOLERANCE = 0.01

_NB_PARAM_RE = re.compile(r"([A-Za-z_]\w*)\s*:")


@dataclass(frozen=True)
class _KernelDispatch:
    sliding_window: int
    max_seq_len: int


def _nanobind_param_indices(fn, *names: str) -> dict[str, int]:
    """Resolve parameter positions from nanobind's runtime signature metadata."""
    overloads = getattr(fn, "__nb_signature__", ())
    if not overloads:
        raise RuntimeError("paged_attention_primitive is missing __nb_signature__")

    signature_text = overloads[0][0]
    params_text = signature_text.partition("(")[2].rpartition(")")[0]
    param_names = _NB_PARAM_RE.findall(params_text)

    indices: dict[str, int] = {}
    for name in names:
        if name not in param_names:
            raise RuntimeError(
                f"parameter {name!r} not found in nanobind signature: {signature_text}"
            )
        indices[name] = param_names.index(name)
    return indices


def _get_call_arg(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    param_indices: dict[str, int],
    name: str,
) -> object:
    """Read a native-op argument by name from positional/keyword call data."""
    index = param_indices[name]
    if len(args) > index:
        return args[index]
    if name in kwargs:
        return kwargs[name]
    raise RuntimeError(f"paged_attention_primitive call missing {name!r}")


def _build_long_prompt(tokenizer) -> str:
    """Return a prompt whose tokenized length exceeds Gemma4's window size."""
    fragments: list[str] = []
    for _ in range(512):
        fragments.append(_PROMPT_FRAGMENT)
        prompt = "".join(fragments)
        token_count = len(tokenizer.encode(text=prompt, add_special_tokens=False))
        if token_count > _LONG_CONTEXT_MIN_TOKENS:
            return prompt
    raise AssertionError(
        "failed to construct a prompt longer than Gemma4's sliding window"
    )


@pytest.fixture(scope="module")
def kernel_dispatch_log() -> list[_KernelDispatch]:
    """Run one Gemma4 inference with a spy on ``paged_attention_primitive``.

    Returns the ``sliding_window`` and ``max_seq_len`` seen by every
    kernel dispatch during one long-context inference. Skips if the
    model path env var is unset.
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
        param_indices = _nanobind_param_indices(
            orig_fn, "sliding_window", "max_seq_len"
        )
        captured: list[_KernelDispatch] = []

        def spy(*args, **kwargs):
            captured.append(
                _KernelDispatch(
                    sliding_window=int(
                        _get_call_arg(args, kwargs, param_indices, "sliding_window")
                    ),
                    max_seq_len=int(
                        _get_call_arg(args, kwargs, param_indices, "max_seq_len")
                    ),
                )
            )
            return orig_fn(*args, **kwargs)

        mp.setattr(ops, "paged_attention_primitive", spy)

        from vllm import LLM, SamplingParams

        llm = LLM(model=model_path, max_model_len=_MAX_MODEL_LEN, max_num_seqs=1)
        prompt = _build_long_prompt(llm.get_tokenizer())
        sp = SamplingParams(temperature=0, max_tokens=_MAX_TOKENS, ignore_eos=True)
        llm.generate([prompt], sp)

        return captured


@pytest.mark.slow
class TestGemma4KernelReceivesPerLayerSlidingWindow:
    """Kernel-level assertions on the sliding_window values dispatched."""

    def test_only_expected_window_values_appear(
        self, kernel_dispatch_log: list[_KernelDispatch]
    ) -> None:
        """No stray values leak from wiring errors."""
        # Act
        unexpected = {
            dispatch.sliding_window
            for dispatch in kernel_dispatch_log
            if dispatch.sliding_window not in (_E2B_SLIDING_WINDOW, _NO_WINDOW)
        }
        # Assert
        assert not unexpected, (
            f"kernel received unexpected sliding_window values: {unexpected}"
        )

    def test_both_sliding_and_full_layers_dispatch(
        self, kernel_dispatch_log: list[_KernelDispatch]
    ) -> None:
        """``sliding_window=512`` and ``-1`` both appear."""
        # Act
        counts = Counter(dispatch.sliding_window for dispatch in kernel_dispatch_log)
        # Assert
        assert counts[_E2B_SLIDING_WINDOW] > 0, (
            "sliding layers never received their window -- enforcement is "
            "not reaching the kernel"
        )
        assert counts[_NO_WINDOW] > 0, (
            "full layers never received -1 -- they may be incorrectly getting a window"
        )

    def test_kernel_sees_context_longer_than_the_window(
        self, kernel_dispatch_log: list[_KernelDispatch]
    ) -> None:
        """The regression test must actually exercise long-context behavior."""
        max_seen = max(dispatch.max_seq_len for dispatch in kernel_dispatch_log)
        assert max_seen > _E2B_SLIDING_WINDOW, (
            f"long-context path was not exercised: max_seq_len={max_seen}, "
            f"sliding_window={_E2B_SLIDING_WINDOW}"
        )

    def test_ratio_matches_layer_types_config(
        self, kernel_dispatch_log: list[_KernelDispatch]
    ) -> None:
        """Sliding/full dispatch ratio matches the 28:7 layer_types split.

        Each model forward dispatches one kernel call per layer (YOCO
        shared layers reuse the cache index of their same-type reference,
        so the retrieved ``sliding_window`` still matches their own
        attention type).  The aggregate ratio is therefore fixed by
        ``layer_types`` and not stochastic.
        """
        # Act
        counts = Counter(dispatch.sliding_window for dispatch in kernel_dispatch_log)
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
