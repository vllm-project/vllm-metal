# SPDX-License-Identifier: Apache-2.0
"""Tests for chunked prefill equivalence with single-pass prefill."""

import pytest
from mlx_lm import load as mlx_lm_load
from vllm.sampling_params import SamplingParams

from vllm_metal.v1.model_runner import MetalModelRunner

TEST_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def model_runner():
    """Create a MetalModelRunner with the test model."""
    try:
        model, tokenizer = mlx_lm_load(TEST_MODEL)
    except Exception as e:
        pytest.skip(f"Could not load test model {TEST_MODEL}: {e}")

    runner = MetalModelRunner.__new__(MetalModelRunner)
    runner.model = model
    runner.tokenizer = tokenizer
    runner._is_vlm = False
    runner._request_states = {}
    runner._rust_state_manager = None
    runner._sampler = None
    runner.device = "cpu"

    return runner


class TestPrefillChunkedEquivalence:
    """Test that _prefill_chunked produces same output as _prefill_single."""

    def test_single_chunk_equals_single_pass(self, model_runner):
        """Single chunk covering full prompt should equal _prefill_single."""
        token_ids = model_runner.tokenizer.encode("Hello, how are you?")
        sampling_params = SamplingParams(temperature=0.0)

        token_single, _ = model_runner._prefill_single(
            req_id="single",
            token_ids=token_ids,
            sampling_params=sampling_params,
        )

        model_runner._request_states.clear()

        token_chunked, _ = model_runner._prefill_chunked(
            req_id="chunked",
            token_ids=token_ids,
            num_computed=0,
            num_scheduled=len(token_ids),
            sampling_params=sampling_params,
        )

        assert token_single == token_chunked

    def test_two_chunks_equals_single_pass(self, model_runner):
        """Two chunks should produce same output as single pass."""
        token_ids = model_runner.tokenizer.encode(
            "The quick brown fox jumps over the lazy dog."
        )
        sampling_params = SamplingParams(temperature=0.0)

        token_single, _ = model_runner._prefill_single(
            req_id="single",
            token_ids=token_ids,
            sampling_params=sampling_params,
        )

        model_runner._request_states.clear()

        mid = len(token_ids) // 2

        # First chunk - should return None
        token1, _ = model_runner._prefill_chunked(
            req_id="chunked",
            token_ids=token_ids,
            num_computed=0,
            num_scheduled=mid,
            sampling_params=sampling_params,
        )
        assert token1 is None

        # Second chunk - should return token
        token2, _ = model_runner._prefill_chunked(
            req_id="chunked",
            token_ids=token_ids,
            num_computed=mid,
            num_scheduled=len(token_ids) - mid,
            sampling_params=sampling_params,
        )

        assert token_single == token2
