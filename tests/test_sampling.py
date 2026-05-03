# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal sampling with vLLM Sampler integration."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.outputs import LogprobsLists
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from tests.stub_runner import make_stub_runner
from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
from vllm_metal.v1.model_runner import (
    MetalModelRunner,
    RequestState,
    _create_request_generator,
    _ExecutionBatch,
)
from vllm_metal.v1.sampling_batch import SamplingBatch, sample_from_logits

VOCAB_SIZE = 1024
MAX_NUM_PROMPT_TOKENS = 64
NUM_OUTPUT_TOKENS = 20


def create_prompt_tokens_tensor(
    prompt_token_ids: list[list[int]],
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Create padded tensor from variable-length prompt token lists."""
    return make_tensor_with_pad(
        prompt_token_ids,
        pad=vocab_size,
        device=device,
        dtype=torch.int64,
        pin_memory=False,
    )


def create_fake_logits_mlx(batch_size: int, vocab_size: int) -> mx.array:
    """Create uniform logits as MLX array."""
    logits = mx.full((batch_size, vocab_size), 1e-2)
    mx.eval(logits)
    return logits


def create_sampling_metadata(
    batch_size: int,
    vocab_size: int,
    device: torch.device,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
) -> SamplingMetadata:
    """Create SamplingMetadata matching vLLM's structure."""
    output_token_ids = [
        np.random.randint(0, vocab_size, size=NUM_OUTPUT_TOKENS).tolist()
        for _ in range(batch_size)
    ]
    prompt_token_ids = [
        np.random.randint(
            0, vocab_size, size=np.random.randint(1, MAX_NUM_PROMPT_TOKENS)
        ).tolist()
        for _ in range(batch_size)
    ]

    # Determine if all greedy or all random
    all_greedy = temperature < 1e-5
    all_random = not all_greedy

    return SamplingMetadata(
        temperature=None
        if all_greedy
        else torch.full((batch_size,), temperature, device=device),
        all_greedy=all_greedy,
        all_random=all_random,
        top_p=None
        if top_p is None
        else torch.full((batch_size,), top_p, device=device),
        top_k=None
        if top_k is None
        else torch.full((batch_size,), top_k, dtype=torch.int, device=device),
        generators={},
        max_num_logprobs=None,
        prompt_token_ids=create_prompt_tokens_tensor(
            prompt_token_ids, vocab_size, device
        ),
        output_token_ids=output_token_ids,
        frequency_penalties=torch.zeros(batch_size, device=device),
        presence_penalties=torch.zeros(batch_size, device=device),
        repetition_penalties=torch.ones(batch_size, device=device),
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


class TestMLXToTorchSampling:
    """Test MLX logits -> torch conversion -> vLLM Sampler."""

    @pytest.mark.parametrize("batch_size", [1, 2, 32])
    def test_greedy_sampling_with_vllm_sampler(self, batch_size: int) -> None:
        """Test greedy sampling using vLLM's Sampler with MLX logits."""
        # Create MLX logits with known maximum
        logits_mlx = create_fake_logits_mlx(batch_size, VOCAB_SIZE)
        # Set a clear maximum at different positions per batch
        logits_list = []
        for i in range(batch_size):
            target_idx = (i * 7) % VOCAB_SIZE  # Different max per request
            row = mx.full((VOCAB_SIZE,), 1e-2)
            row = row.at[target_idx].add(10.0)
            logits_list.append(row)
        logits_mlx = mx.stack(logits_list)
        mx.eval(logits_mlx)

        # Convert to torch
        logits_torch = mlx_to_torch(logits_mlx, device="cpu")

        # Create sampling metadata (greedy)
        metadata = create_sampling_metadata(
            batch_size, VOCAB_SIZE, torch.device("cpu"), temperature=0.0
        )

        # Use vLLM's Sampler
        sampler = Sampler()
        output = sampler.forward(logits_torch, metadata)

        # Verify correct tokens sampled
        for i in range(batch_size):
            expected_idx = (i * 7) % VOCAB_SIZE
            assert output.sampled_token_ids[i, 0].item() == expected_idx

    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
    def test_temperature_sampling(self, batch_size: int, temperature: float) -> None:
        """Test temperature-based sampling with vLLM Sampler."""
        logits_mlx = create_fake_logits_mlx(batch_size, VOCAB_SIZE)
        mx.eval(logits_mlx)

        logits_torch = mlx_to_torch(logits_mlx, device="cpu")
        metadata = create_sampling_metadata(
            batch_size, VOCAB_SIZE, torch.device("cpu"), temperature=temperature
        )

        sampler = Sampler()
        output = sampler.forward(logits_torch, metadata)

        # Should produce valid token IDs
        assert output.sampled_token_ids.shape == (batch_size, 1)
        assert (output.sampled_token_ids >= 0).all()
        assert (output.sampled_token_ids < VOCAB_SIZE).all()

    @pytest.mark.parametrize("top_k", [1, 10, 50])
    def test_top_k_sampling(self, top_k: int) -> None:
        """Test top-k sampling limits candidates."""
        batch_size = 4
        logits_mlx = mx.random.normal((batch_size, VOCAB_SIZE))
        mx.eval(logits_mlx)

        logits_torch = mlx_to_torch(logits_mlx, device="cpu")
        metadata = create_sampling_metadata(
            batch_size,
            VOCAB_SIZE,
            torch.device("cpu"),
            temperature=1.0,
            top_k=top_k,
        )

        sampler = Sampler()
        output = sampler.forward(logits_torch, metadata)

        # With top_k=1, should always pick the maximum
        if top_k == 1:
            expected = torch.argmax(logits_torch, dim=-1)
            assert (output.sampled_token_ids.squeeze() == expected).all()

    @pytest.mark.parametrize("top_p", [0.1, 0.5, 0.9])
    def test_top_p_sampling(self, top_p: float) -> None:
        """Test top-p (nucleus) sampling."""
        batch_size = 4
        # Create skewed logits
        logits_mlx = mx.zeros((batch_size, VOCAB_SIZE))
        # Set first token to have high probability
        logits_list = []
        for _ in range(batch_size):
            row = mx.zeros((VOCAB_SIZE,))
            row = row.at[0].add(10.0)
            logits_list.append(row)
        logits_mlx = mx.stack(logits_list)
        mx.eval(logits_mlx)

        logits_torch = mlx_to_torch(logits_mlx, device="cpu")
        metadata = create_sampling_metadata(
            batch_size,
            VOCAB_SIZE,
            torch.device("cpu"),
            temperature=1.0,
            top_p=top_p,
        )

        sampler = Sampler()
        output = sampler.forward(logits_torch, metadata)

        # With very low top_p and skewed distribution, should mostly pick token 0
        if top_p <= 0.1:
            assert (output.sampled_token_ids == 0).sum() >= batch_size // 2


class TestBatchedPerRequestSampling:
    """Test batched sampling with different params per request."""

    def test_mixed_greedy_and_random(self) -> None:
        """Test batch with some greedy, some random requests."""
        batch_size = 4
        # Create logits with different maxima per request
        logits_list = []
        for i in range(batch_size):
            row = mx.full((VOCAB_SIZE,), 1e-2)
            row = row.at[i * 10].add(10.0)
            logits_list.append(row)
        logits_mlx = mx.stack(logits_list)
        mx.eval(logits_mlx)

        logits_torch = mlx_to_torch(logits_mlx, device="cpu")

        # Create per-request temperatures (half greedy, half random)
        temperatures = torch.tensor([0.0, 1.0, 0.0, 1.0])

        metadata = SamplingMetadata(
            temperature=temperatures,
            all_greedy=False,
            all_random=False,
            top_p=None,
            top_k=None,
            generators={i: torch.Generator().manual_seed(42 + i) for i in [1, 3]},
            max_num_logprobs=None,
            prompt_token_ids=None,
            output_token_ids=[[] for _ in range(batch_size)],
            frequency_penalties=torch.zeros(batch_size),
            presence_penalties=torch.zeros(batch_size),
            repetition_penalties=torch.ones(batch_size),
            no_penalties=True,
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )

        sampler = Sampler()
        output = sampler.forward(logits_torch, metadata)

        # Greedy requests (0, 2) should pick the max
        assert output.sampled_token_ids[0, 0].item() == 0
        assert output.sampled_token_ids[2, 0].item() == 20


class TestV1SeededSamplingGenerator:
    def test_seeded_sampling_generator_advances_across_decode_steps(self) -> None:
        """Seeded sampling should reuse (and advance) a per-request generator."""

        def uniform_logits_model(vocab_size: int):
            def _model(input_ids: mx.array, cache=None) -> mx.array:  # noqa: ANN001
                batch_size = int(input_ids.shape[0])
                return mx.zeros((batch_size, 1, vocab_size), dtype=mx.float32)

            return _model

        vocab_size = 32
        runner = make_stub_runner(
            model_args={"vocab_size": vocab_size},
            _sampler=Sampler(),
            model=uniform_logits_model(vocab_size),
        )

        sp = SamplingParams(temperature=1.0, seed=123)
        generator = _create_request_generator(runner.device, sp)
        assert generator is not None

        state = RequestState(
            token_ids=[1],
            prompt_len=1,
            cache=[],
            sampling_params=sp,
            generator=generator,
        )

        before = generator.get_state()
        runner._batched_decode([("r1", state)])
        after_first = generator.get_state()
        runner._batched_decode([("r1", state)])
        after_second = generator.get_state()

        assert not torch.equal(after_first, before)
        assert not torch.equal(after_second, after_first)


class TestV1SamplingBatch:
    def test_can_use_native_greedy_requires_greedy_without_filters(self) -> None:
        assert SamplingBatch.can_use_native_greedy([SamplingParams(temperature=0.0)])
        assert not SamplingBatch.can_use_native_greedy(
            [SamplingParams(temperature=0.8)]
        )
        # vLLM normalizes top-k/top-p back to no-op under greedy decoding.
        assert SamplingBatch.can_use_native_greedy(
            [SamplingParams(temperature=0.0, top_k=5)]
        )
        assert SamplingBatch.can_use_native_greedy(
            [SamplingParams(temperature=0.0, top_p=0.9)]
        )
        assert not SamplingBatch.can_use_native_greedy(
            [SamplingParams(temperature=0.8, top_k=5)]
        )
        assert not SamplingBatch.can_use_native_greedy(
            [SamplingParams(temperature=0.8, top_p=0.9)]
        )
        assert not SamplingBatch.can_use_native_greedy(
            [SamplingParams(temperature=0.0, presence_penalty=0.5)]
        )
        assert not SamplingBatch.can_use_native_greedy(
            [SamplingParams(temperature=0.0, repetition_penalty=1.1)]
        )
        assert not SamplingBatch.can_use_native_greedy(
            [SamplingParams(temperature=0.0, logprobs=1)]
        )

    def test_can_use_native_greedy_requires_every_request_to_match(self) -> None:
        assert SamplingBatch.can_use_native_greedy(
            [SamplingParams(temperature=0.0), SamplingParams(temperature=0.0)]
        )
        assert not SamplingBatch.can_use_native_greedy(
            [SamplingParams(temperature=0.0), SamplingParams(temperature=0.3)]
        )
        assert not SamplingBatch.can_use_native_greedy(
            [SamplingParams(temperature=0.0), SamplingParams(temperature=0.8, top_k=4)]
        )

    def test_sampling_metadata_uses_requested_logprobs(self) -> None:
        batch = SamplingBatch(
            [SamplingParams(temperature=0.0, logprobs=5)],
            [[1, 2, 3]],
            [[]],
            vocab_size=VOCAB_SIZE,
            device=torch.device("cpu"),
        )

        metadata = batch.make_sampling_metadata()

        assert metadata.max_num_logprobs == 5

    def test_sample_from_logits_returns_logprobs_for_greedy_request(self) -> None:
        logits = mx.array([[0.0, 1.0, 4.0, 2.0]], dtype=mx.float32)
        batch = SamplingBatch(
            [SamplingParams(temperature=0.0, logprobs=2)],
            [[1, 2, 3]],
            [[]],
            vocab_size=4,
            device=torch.device("cpu"),
        )

        result = sample_from_logits(logits, batch, Sampler(), torch.device("cpu"))

        assert result.token_ids == [2]
        assert result.logprobs is not None
        assert result.logprobs.logprob_token_ids.shape == (1, 3)
        assert result.logprobs.logprob_token_ids[0, 0] == 2
        assert result.logprobs.sampled_token_ranks.tolist() == [1]

    def test_model_runner_output_keeps_logprobs_slot_alignment(self) -> None:
        batch = _ExecutionBatch()
        batch.add_output("intermediate", [])
        batch.add_output(
            "decode",
            [7],
            LogprobsLists(
                logprob_token_ids=np.array([[7, 7, 3]], dtype=np.int32),
                logprobs=np.array([[-0.1, -0.1, -2.0]], dtype=np.float32),
                sampled_token_ranks=np.array([1], dtype=np.int32),
            ),
        )

        output = MetalModelRunner._build_output(batch)

        assert output.logprobs is not None
        assert output.logprobs.logprob_token_ids.shape == (2, 3)
        assert output.logprobs.slice_request(1, 1).logprob_token_ids[0, 0] == 7

    def test_logprobs_flow_through_request_output_processing(self) -> None:
        sampling_params = SamplingParams(
            temperature=0.0,
            logprobs=2,
            max_tokens=1,
            detokenize=False,
        )
        request = EngineCoreRequest(
            request_id="req-1",
            prompt_token_ids=[1, 2, 3],
            mm_features=None,
            sampling_params=sampling_params,
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            external_req_id="req-1",
        )
        output_processor = OutputProcessor(tokenizer=None, log_stats=False)
        output_processor.add_request(request, prompt=None)

        result = output_processor.process_outputs(
            [
                EngineCoreOutput(
                    request_id="req-1",
                    new_token_ids=[7],
                    new_logprobs=LogprobsLists(
                        logprob_token_ids=np.array([[7, 5, 3]], dtype=np.int32),
                        logprobs=np.array([[-0.1, -1.0, -2.0]], dtype=np.float32),
                        sampled_token_ranks=np.array([1], dtype=np.int32),
                    ),
                    finish_reason=FinishReason.LENGTH,
                )
            ]
        )

        assert not result.reqs_to_abort
        [request_output] = result.request_outputs
        assert isinstance(request_output, RequestOutput)
        [completion_output] = request_output.outputs
        assert completion_output.token_ids == [7]
        assert completion_output.logprobs is not None
        assert completion_output.logprobs[0] is not None
        assert completion_output.logprobs[0][7].logprob == pytest.approx(-0.1)


class TestV1SamplingMetadataLogitsProcessors:
    @staticmethod
    def _make_runner(vocab_size: int = 32) -> MetalModelRunner:
        return make_stub_runner(model_args={"vocab_size": vocab_size})

    def test_make_sampling_metadata_uses_runner_logitsprocs(self) -> None:
        runner = self._make_runner()
        expected_logitsprocs = LogitsProcessors()
        runner._logitsprocs = expected_logitsprocs

        metadata = runner._make_sampling_metadata(
            sampling_params_list=[SamplingParams(temperature=0.0)],
            prompt_token_id_lists=[[]],
            output_token_id_lists=[[]],
        )

        assert metadata.logitsprocs is expected_logitsprocs


class TestV1PenaltyTokenAccounting:
    """Regression tests for prompt vs output token accounting in penalties."""

    @staticmethod
    def _fixed_logits_model(vocab_size: int, token_a: int, token_b: int):
        """Return a dummy MLX model that always prefers token_a over token_b."""

        def _model(input_ids: mx.array, cache=None) -> mx.array:  # noqa: ANN001
            batch_size = int(input_ids.shape[0])
            logits = mx.zeros((batch_size, 1, vocab_size), dtype=mx.float32)
            logits = logits.at[:, :, token_a].add(10.0)
            logits = logits.at[:, :, token_b].add(9.5)
            return logits

        return _model

    @staticmethod
    def _make_runner(vocab_size: int) -> MetalModelRunner:
        return make_stub_runner(
            model_args={"vocab_size": vocab_size},
            _sampler=Sampler(),
        )

    def test_presence_penalty_does_not_apply_to_prompt_tokens(self) -> None:
        """Presence/frequency penalties should apply only to generated tokens."""
        vocab_size = 64
        prompt_token = 5
        alternative_token = 7
        already_generated_token = 11

        runner = self._make_runner(vocab_size)
        runner.model = self._fixed_logits_model(
            vocab_size, prompt_token, alternative_token
        )

        # If prompt tokens were incorrectly treated as output tokens, a presence penalty
        # would demote `prompt_token` and flip greedy selection to `alternative_token`.
        sp = SamplingParams(
            temperature=0.0,
            presence_penalty=1.0,
            repetition_penalty=1.0,
            frequency_penalty=0.0,
        )
        state = RequestState(
            # `token_ids` stores prompt + already-generated output tokens.
            # `prompt_len` splits them so presence_penalty is applied to output-only.
            token_ids=[prompt_token, already_generated_token],
            prompt_len=1,
            cache=[],
            sampling_params=sp,
            generated_tokens=0,
        )

        next_tokens = runner._sequential_decode([("r1", state)]).token_ids

        assert next_tokens == [prompt_token]

    def test_frequency_penalty_does_not_apply_to_prompt_tokens(self) -> None:
        """Frequency penalty should apply only to generated tokens."""
        vocab_size = 64
        prompt_token = 5
        alternative_token = 7
        already_generated_token = 11

        runner = self._make_runner(vocab_size)
        runner.model = self._fixed_logits_model(
            vocab_size, prompt_token, alternative_token
        )

        sp = SamplingParams(
            temperature=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            frequency_penalty=1.0,
        )
        state = RequestState(
            token_ids=[prompt_token, already_generated_token],
            prompt_len=1,
            cache=[],
            sampling_params=sp,
            generated_tokens=0,
        )

        next_tokens = runner._sequential_decode([("r1", state)]).token_ids

        assert next_tokens == [prompt_token]

    def test_presence_penalty_applies_to_output_tokens(self) -> None:
        """Presence penalty should apply to generated tokens."""
        vocab_size = 64
        repeated_token = 5
        alternative_token = 7

        runner = self._make_runner(vocab_size)

        logits = torch.zeros((1, vocab_size), dtype=torch.float32)
        logits[0, repeated_token] = 10.0
        logits[0, alternative_token] = 9.5

        sp = SamplingParams(
            temperature=0.0,
            presence_penalty=1.0,
            repetition_penalty=1.0,
            frequency_penalty=0.0,
        )
        metadata = runner._make_sampling_metadata(
            sampling_params_list=[sp],
            prompt_token_id_lists=[[]],
            # Mock: the model has already generated `repeated_token`, so presence_penalty
            # should demote it and flip greedy selection to `alternative_token`.
            output_token_id_lists=[[repeated_token]],
        )

        output = runner._sampler.forward(logits, metadata)

        assert int(output.sampled_token_ids[0, 0].item()) == alternative_token

    def test_frequency_penalty_applies_to_output_tokens(self) -> None:
        """Frequency penalty should apply to generated tokens."""
        vocab_size = 64
        repeated_token = 5
        alternative_token = 7

        runner = self._make_runner(vocab_size)

        logits = torch.zeros((1, vocab_size), dtype=torch.float32)
        logits[0, repeated_token] = 10.0
        logits[0, alternative_token] = 9.5

        sp = SamplingParams(
            temperature=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            frequency_penalty=1.0,
        )
        metadata = runner._make_sampling_metadata(
            sampling_params_list=[sp],
            prompt_token_id_lists=[[]],
            # Mock: the model has already generated `repeated_token` once, so a positive
            # frequency_penalty should reduce its probability and pick `alternative_token`.
            output_token_id_lists=[[repeated_token]],
        )

        output = runner._sampler.forward(logits, metadata)

        assert int(output.sampled_token_ids[0, 0].item()) == alternative_token

    def test_repetition_penalty_applies_to_prompt_tokens(self) -> None:
        """Repetition penalty should consider prompt+output tokens (vLLM semantics)."""
        vocab_size = 64
        prompt_token = 5
        alternative_token = 7

        runner = self._make_runner(vocab_size)

        # Model "prefers" prompt_token, but repetition penalty should demote it because
        # it already appears in the prompt.
        logits = torch.zeros((1, vocab_size), dtype=torch.float32)
        logits[0, prompt_token] = 10.0
        logits[0, alternative_token] = 9.5

        sp = SamplingParams(
            temperature=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.2,
            frequency_penalty=0.0,
        )
        metadata = runner._make_sampling_metadata(
            sampling_params_list=[sp],
            # Mock: the prompt already contains `prompt_token`, so repetition_penalty
            # should demote it and prefer `alternative_token`.
            prompt_token_id_lists=[[prompt_token]],
            output_token_id_lists=[[]],
        )

        output = runner._sampler.forward(logits, metadata)

        assert int(output.sampled_token_ids[0, 0].item()) == alternative_token

    def test_repetition_penalty_applies_to_output_tokens(self) -> None:
        """Repetition penalty should consider output tokens as well."""
        vocab_size = 64
        repeated_token = 5
        alternative_token = 7

        runner = self._make_runner(vocab_size)

        logits = torch.zeros((1, vocab_size), dtype=torch.float32)
        logits[0, repeated_token] = 10.0
        logits[0, alternative_token] = 9.5

        sp = SamplingParams(
            temperature=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.2,
            frequency_penalty=0.0,
        )
        metadata = runner._make_sampling_metadata(
            sampling_params_list=[sp],
            prompt_token_id_lists=[[]],
            # Mock: `repeated_token` is already in the generated output, so repetition_penalty
            # should demote it and select `alternative_token`.
            output_token_id_lists=[[repeated_token]],
        )

        output = runner._sampler.forward(logits, metadata)

        assert int(output.sampled_token_ids[0, 0].item()) == alternative_token
