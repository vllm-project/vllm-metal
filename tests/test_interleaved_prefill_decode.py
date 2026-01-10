# SPDX-License-Identifier: Apache-2.0
"""Test interleaved prefill + decode produces correct output."""

import pytest
from mlx_lm import load as mlx_lm_load
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

from vllm_metal.v1.model_runner import MetalModelRunner

TEST_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def model_runner():
    """Create a MetalModelRunner with the test model."""
    try:
        model, tokenizer = mlx_lm_load(TEST_MODEL)
    except Exception as e:
        pytest.skip(f"Could not load test model: {e}")

    runner = MetalModelRunner.__new__(MetalModelRunner)
    runner.model = model
    runner.tokenizer = tokenizer
    runner._is_vlm = False
    runner._request_states = {}
    runner._rust_state_manager = None
    runner._sampler = None
    runner.device = "cpu"

    return runner


def make_new_request(req_id, token_ids, num_computed=0):
    """Helper to create NewRequestData."""
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=token_ids,
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=num_computed,
        lora_request=None,
    )


def make_scheduler_output(new_reqs=None, cached_req_ids=None, num_scheduled=None):
    """Helper to create SchedulerOutput."""
    new_reqs = new_reqs or []
    cached_req_ids = cached_req_ids or []
    num_scheduled = num_scheduled or {}

    cached = CachedRequestData(
        req_ids=cached_req_ids,
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in cached_req_ids],
        all_token_ids={},
        new_block_ids=[None for _ in cached_req_ids],
        num_computed_tokens=[0 for _ in cached_req_ids],
        num_output_tokens=[0 for _ in cached_req_ids],
    )

    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=sum(num_scheduled.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


class TestInterleavedPrefillDecode:
    """Test that interleaved prefill + decode works correctly."""

    def test_interleaved_prefill_and_decode(self, model_runner):
        """Simulate: prefill A, then interleaved (prefill B + decode A)."""
        model_runner._request_states.clear()

        tok = model_runner.tokenizer
        prompt_a = tok.encode("Hello world")
        prompt_b = tok.encode("How are you today?")

        # Step 1: Prefill request A (full)
        out1 = model_runner.execute_model(
            make_scheduler_output(
                new_reqs=[make_new_request("A", prompt_a)],
                num_scheduled={"A": len(prompt_a)},
            )
        )
        token_a1 = out1.sampled_token_ids[0][0]

        # Step 2: Interleaved - prefill B (full) + decode A
        out2 = model_runner.execute_model(
            make_scheduler_output(
                new_reqs=[make_new_request("B", prompt_b)],
                cached_req_ids=["A"],
                num_scheduled={"B": len(prompt_b), "A": 1},
            )
        )

        # Should have outputs for both B (prefill) and A (decode)
        assert len(out2.sampled_token_ids) == 2
        token_b1 = out2.sampled_token_ids[out2.req_id_to_index["B"]][0]
        token_a2 = out2.sampled_token_ids[out2.req_id_to_index["A"]][0]

        # Verify tokens are valid (non-zero integers)
        assert isinstance(token_a1, int) and token_a1 > 0
        assert isinstance(token_a2, int) and token_a2 > 0
        assert isinstance(token_b1, int) and token_b1 > 0

    def test_interleaved_equals_sequential(self, model_runner):
        """Interleaved execution should produce same tokens as sequential."""
        tok = model_runner.tokenizer
        prompt_a = tok.encode("The sky is")
        prompt_b = tok.encode("Water is")

        # === Sequential execution ===
        model_runner._request_states.clear()

        # Prefill A
        seq_out1 = model_runner.execute_model(
            make_scheduler_output(
                new_reqs=[make_new_request("A", prompt_a)],
                num_scheduled={"A": len(prompt_a)},
            )
        )
        seq_token_a1 = seq_out1.sampled_token_ids[0][0]

        # Decode A
        seq_out2 = model_runner.execute_model(
            make_scheduler_output(
                cached_req_ids=["A"],
                num_scheduled={"A": 1},
            )
        )
        seq_token_a2 = seq_out2.sampled_token_ids[0][0]

        # Prefill B
        seq_out3 = model_runner.execute_model(
            make_scheduler_output(
                new_reqs=[make_new_request("B", prompt_b)],
                num_scheduled={"B": len(prompt_b)},
            )
        )
        seq_token_b1 = seq_out3.sampled_token_ids[0][0]

        # === Interleaved execution ===
        model_runner._request_states.clear()

        # Prefill A
        int_out1 = model_runner.execute_model(
            make_scheduler_output(
                new_reqs=[make_new_request("A", prompt_a)],
                num_scheduled={"A": len(prompt_a)},
            )
        )
        int_token_a1 = int_out1.sampled_token_ids[0][0]

        # Interleaved: Prefill B + Decode A
        int_out2 = model_runner.execute_model(
            make_scheduler_output(
                new_reqs=[make_new_request("B", prompt_b)],
                cached_req_ids=["A"],
                num_scheduled={"B": len(prompt_b), "A": 1},
            )
        )
        int_token_b1 = int_out2.sampled_token_ids[int_out2.req_id_to_index["B"]][0]
        int_token_a2 = int_out2.sampled_token_ids[int_out2.req_id_to_index["A"]][0]

        # === Compare ===
        assert seq_token_a1 == int_token_a1, "First token A mismatch"
        assert seq_token_a2 == int_token_a2, "Second token A mismatch"
        assert seq_token_b1 == int_token_b1, "First token B mismatch"
