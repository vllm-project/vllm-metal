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

from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
from vllm_metal.v1 import sampling_batch
from vllm_metal.v1.model_runner import (
    PrefillRequest,
    RequestState,
    _create_request_generator,
    _ExecutionBatch,
)
from vllm_metal.v1.sampling_batch import (
    SamplingBatch,
    sample_decode_tokens,
    sample_from_logits,
)

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

        vocab_size = 32
        sampler = Sampler()

        sp = SamplingParams(temperature=1.0, seed=123)
        generator = _create_request_generator(sp)
        assert generator is not None

        state = RequestState(
            token_ids=[1],
            prompt_len=1,
            sampling_params=sp,
            generator=generator,
        )

        before = generator.get_state()
        sample_decode_tokens(
            mx.zeros((1, 1, vocab_size), dtype=mx.float32),
            [("r1", state)],
            num_decode=1,
            vocab_size=vocab_size,
            sampler=sampler,
        )
        after_first = generator.get_state()
        sample_decode_tokens(
            mx.zeros((1, 1, vocab_size), dtype=mx.float32),
            [("r1", state)],
            num_decode=1,
            vocab_size=vocab_size,
            sampler=sampler,
        )
        after_second = generator.get_state()

        assert not torch.equal(after_first, before)
        assert not torch.equal(after_second, after_first)


class TestSamplerDevicePolicy:
    """Regression tests for #622: torch sampling must never run on MPS.

    ``exponential_()`` on MPS can return exact zeros, which the vLLM
    sampler's Gumbel-style division turns into inf/NaN rows whose argmax
    lands on an arbitrary vocab id.
    """

    def test_sampling_metadata_tensors_live_on_cpu(self) -> None:
        batch = SamplingBatch(
            [
                SamplingParams(temperature=0.8, top_k=20, top_p=0.9),
                SamplingParams(temperature=1.0, presence_penalty=0.5),
                SamplingParams(temperature=0.7, allowed_token_ids=[1, 2]),
            ],
            [[1, 2], [3], [4]],
            [[], [], []],
            vocab_size=VOCAB_SIZE,
        )

        metadata = batch.make_sampling_metadata()

        assert metadata.temperature is not None
        assert metadata.temperature.device.type == "cpu"
        assert metadata.top_k is not None
        assert metadata.top_k.device.type == "cpu"
        assert metadata.top_p is not None
        assert metadata.top_p.device.type == "cpu"
        assert metadata.prompt_token_ids is not None
        assert metadata.prompt_token_ids.device.type == "cpu"
        assert metadata.frequency_penalties.device.type == "cpu"
        assert metadata.presence_penalties.device.type == "cpu"
        assert metadata.repetition_penalties.device.type == "cpu"
        assert metadata.allowed_token_ids_mask is not None
        assert metadata.allowed_token_ids_mask.device.type == "cpu"

    def test_seeded_generator_lives_on_cpu(self) -> None:
        generator = _create_request_generator(SamplingParams(temperature=1.0, seed=7))

        assert generator is not None
        assert generator.device.type == "cpu"

    def test_bridged_logits_reach_sampler_on_cpu(self) -> None:
        logits = mx.array([[0.0, 1.0, 4.0, 2.0]], dtype=mx.float32)
        batch = SamplingBatch(
            [SamplingParams(temperature=0.8, top_k=2)],
            [[1]],
            [[]],
            vocab_size=4,
        )
        seen_devices: list[torch.device] = []

        class SpySampler(Sampler):
            def forward(self, logits_torch, sampling_metadata):  # noqa: ANN001
                seen_devices.append(logits_torch.device)
                return super().forward(logits_torch, sampling_metadata)

        result = sample_from_logits(logits, batch, SpySampler())

        assert [d.type for d in seen_devices] == ["cpu"]
        assert 0 <= result.token_ids[0] < 4


class TestV1SamplingBatch:
    @staticmethod
    def _batch(params_list: list[SamplingParams]) -> SamplingBatch:
        return SamplingBatch(
            params_list,
            [[1]] * len(params_list),
            [[]] * len(params_list),
            vocab_size=VOCAB_SIZE,
        )

    def test_prefill_requests_share_one_sampler_batch(self, monkeypatch) -> None:
        logits = mx.arange(24, dtype=mx.float16).reshape(1, 6, 4)
        greedy_params = SamplingParams(temperature=0.0, logprobs=1)
        random_params = SamplingParams(temperature=0.7, top_k=3, logprobs=1)
        generator = torch.Generator().manual_seed(42)
        prefill_reqs = [
            PrefillRequest("p0", [10, 11], greedy_params, [[0]], None, 2, 0, None),
            PrefillRequest(
                "p1", [21, 22], random_params, [[1]], generator, 3, 1, [20, 21, 22]
            ),
        ]
        sampler_calls: list[tuple[mx.array, SamplingBatch]] = []

        def capture_batch(
            logits_2d: mx.array,
            batch: SamplingBatch,
            *_args: object,
        ) -> sampling_batch._SamplingResult:
            sampler_calls.append((logits_2d, batch))
            return sampling_batch._SamplingResult([7] * len(batch.sampling_params_list))

        monkeypatch.setattr(sampling_batch, "sample_from_logits", capture_batch)

        result = sampling_batch.sample_prefill_tokens(
            logits,
            prefill_reqs,
            cu_seqlens=[0, 2, 4, 6],
            num_decode=1,
            sampler=Sampler(),
            vocab_size=4,
        )

        assert len(sampler_calls) == 1
        sampled_logits, batch = sampler_calls[0]
        assert sampled_logits.tolist() == [
            [12.0, 13.0, 14.0, 15.0],
            [20.0, 21.0, 22.0, 23.0],
        ]
        assert sampled_logits.dtype == mx.float16
        assert batch.sampling_params_list == [greedy_params, random_params]
        assert batch.prompt_token_id_lists == [[10, 11], [20, 21, 22]]
        assert batch.output_token_id_lists == [[], []]
        assert batch.generators == {1: generator}
        assert result.token_ids == [7, 7]

    def test_empty_prefill_batch_skips_sampling(self) -> None:
        result = sampling_batch.sample_prefill_tokens(
            mx.zeros((1, 0, 4)),
            [],
            cu_seqlens=[0],
            num_decode=0,
            sampler=Sampler(),
            vocab_size=4,
        )

        assert result == sampling_batch._SamplingResult([])

    def test_can_use_native_greedy_requires_greedy_without_filters(self) -> None:
        assert self._batch([SamplingParams(temperature=0.0)]).can_use_native_greedy()
        assert not self._batch(
            [SamplingParams(temperature=0.8)]
        ).can_use_native_greedy()
        # vLLM normalizes top-k/top-p back to no-op under greedy decoding.
        assert self._batch(
            [SamplingParams(temperature=0.0, top_k=5)]
        ).can_use_native_greedy()
        assert self._batch(
            [SamplingParams(temperature=0.0, top_p=0.9)]
        ).can_use_native_greedy()
        assert not self._batch(
            [SamplingParams(temperature=0.8, top_k=5)]
        ).can_use_native_greedy()
        assert not self._batch(
            [SamplingParams(temperature=0.8, top_p=0.9)]
        ).can_use_native_greedy()
        assert not self._batch(
            [SamplingParams(temperature=0.0, presence_penalty=0.5)]
        ).can_use_native_greedy()
        assert not self._batch(
            [SamplingParams(temperature=0.0, repetition_penalty=1.1)]
        ).can_use_native_greedy()
        assert not self._batch(
            [SamplingParams(temperature=0.0, logprobs=1)]
        ).can_use_native_greedy()
        assert not self._batch(
            [SamplingParams(temperature=0.0, logprob_token_ids=[1, 2])]
        ).can_use_native_greedy()
        assert not self._batch(
            [SamplingParams(temperature=0.0, allowed_token_ids=[1, 2])]
        ).can_use_native_greedy()
        bad_sp = SamplingParams(temperature=0.0)
        bad_sp._bad_words_token_ids = [[99]]
        assert not self._batch([bad_sp]).can_use_native_greedy()

    def test_allowed_token_ids_constrains_greedy_sampling(self) -> None:
        """Greedy with allowed_token_ids must fall back and the constraint works."""
        logits = mx.array([[10.0, 1.0, 5.0, 1.0]], dtype=mx.float32)
        batch = SamplingBatch(
            [SamplingParams(temperature=0.0, allowed_token_ids=[2, 3])],
            [[1, 2, 3]],
            [[]],
            vocab_size=4,
        )
        result = sample_from_logits(logits, batch, Sampler())
        assert result.token_ids[0] in [2, 3]

    def test_allowed_token_ids_mixed_batch(self) -> None:
        """Mixed batch: try one constrained, one unconstrained."""
        logits = mx.array(
            [[10.0, 1.0, 5.0, 1.0], [1.0, 10.0, 1.0, 1.0]], dtype=mx.float32
        )
        batch = SamplingBatch(
            [
                SamplingParams(temperature=0.0, allowed_token_ids=[2, 3]),
                SamplingParams(temperature=0.0),
            ],
            [[1, 2], [3, 4]],
            [[], []],
            vocab_size=4,
        )
        result = sample_from_logits(logits, batch, Sampler())
        assert result.token_ids[0] in [2, 3]
        assert result.token_ids[1] == 1  # unconstrained, should pick highest logit

    def test_mixed_disabled_and_enabled_top_k(self) -> None:
        """Mixed top_k=0 (disabled) and top_k=64 rows must not crash.

        Regression for issue #646: the disabled row reached vLLM's sampler as
        raw 0, which the PyTorch top-k path interprets as a gather index of
        vocab_size (out of bounds). It must carry the vocab_size sentinel
        instead, matching GPUInputBatch. Exercises the real Sampler.forward()
        path via sample_from_logits (native greedy is unavailable for a mixed
        batch).
        """
        top_k = 64
        logits = mx.arange(2 * VOCAB_SIZE, dtype=mx.float32).reshape(2, VOCAB_SIZE)
        batch = SamplingBatch(
            [
                SamplingParams(temperature=0.0),
                SamplingParams(temperature=1.0, top_k=top_k, top_p=0.95),
            ],
            [[1], [2]],
            [[], []],
            vocab_size=VOCAB_SIZE,
        )
        # Disabled row must be normalized to the vocab_size sentinel.
        metadata = batch.make_sampling_metadata()
        assert metadata.top_k is not None
        assert metadata.top_k.tolist() == [VOCAB_SIZE, top_k]

        result = sample_from_logits(logits, batch, Sampler())
        # Greedy row samples from the full vocab: argmax of row 0.
        assert result.token_ids[0] == VOCAB_SIZE - 1
        # Top-k row stays inside the top-64 candidates (values 960..1023).
        assert VOCAB_SIZE - top_k <= result.token_ids[1] < VOCAB_SIZE

    def test_min_p_narrows_candidates_before_top_p(self) -> None:
        """Native sampling applies min_p before top_p."""
        probs = [0.5, 0.4, 0.1]
        logits = mx.array([[float(np.log(p)) for p in probs]], dtype=mx.float32)
        params = [SamplingParams(temperature=1.0, top_p=0.55, min_p=0.25)]
        assert SamplingBatch.params_allow_native_random(params)

        keys = iter([mx.random.key(seed) for seed in range(64)])
        sampled = {
            int(
                SamplingBatch.native_decode_tokens(
                    logits,
                    params,
                    vocab_size=len(probs),
                    next_key=lambda: next(keys),
                ).item()
            )
            for _ in range(64)
        }
        assert sampled == {0}

    def test_min_p_masks_after_temperature_on_torch_path(self) -> None:
        """Torch sampling applies min_p after temperature."""
        sampler = Sampler()
        sampled = set()
        for seed in range(40):
            logits = mx.array([[4.0, 2.0, 0.0, -2.0]], dtype=mx.float32)
            sp = SamplingParams(temperature=2.0, min_p=0.3, seed=seed)
            assert not SamplingBatch.params_allow_native_random([sp])
            batch = SamplingBatch(
                [sp],
                [[1]],
                [[]],
                vocab_size=4,
                generators={0: torch.Generator().manual_seed(seed)},
            )
            metadata = batch.make_sampling_metadata(torch.zeros(1, 4))
            assert len(list(metadata.logitsprocs.argmax_invariant)) == 1

            sampled.add(sample_from_logits(logits, batch, sampler).token_ids[0])
        assert sampled == {0, 1}

    def test_bad_words_blocks_greedy_token(self) -> None:
        """Greedy + bad_words_token_ids must fall back and block the banned token."""
        logits = mx.array([[10.0, 1.0, 5.0, 1.0]], dtype=mx.float32)
        sp = SamplingParams(temperature=0.0)
        sp._bad_words_token_ids = [[0]]  # ban token 0 (the highest logit)
        batch = SamplingBatch(
            [sp],
            [[1, 2, 3]],
            [[]],
            vocab_size=4,
        )
        result = sample_from_logits(logits, batch, Sampler())
        assert result.token_ids[0] != 0  # token 0 should be blocked

    def test_can_use_native_greedy_requires_every_request_to_match(self) -> None:
        assert self._batch(
            [SamplingParams(temperature=0.0), SamplingParams(temperature=0.0)]
        ).can_use_native_greedy()
        assert not self._batch(
            [SamplingParams(temperature=0.0), SamplingParams(temperature=0.3)]
        ).can_use_native_greedy()
        assert not self._batch(
            [SamplingParams(temperature=0.0), SamplingParams(temperature=0.8, top_k=4)]
        ).can_use_native_greedy()

    def test_sampling_metadata_skips_penalty_allocation_when_no_penalties(
        self,
    ) -> None:
        """When the batch has no penalties, the per-request penalty tensors
        should be zero-length placeholders rather than ``batch_size`` allocations.

        Both ``Sampler.apply_penalties`` and ``RejectionSampler`` short-circuit
        on ``no_penalties=True`` before reading these fields, so skipping the
        allocations is safe and removes 3 redundant ``torch.tensor(...)`` calls
        per decode step on hot paths.
        """
        batch = SamplingBatch(
            [
                SamplingParams(temperature=0.0),
                SamplingParams(temperature=0.8),
                SamplingParams(temperature=1.0, top_k=10),
            ],
            [[1, 2, 3], [4, 5], [6]],
            [[], [], []],
            vocab_size=VOCAB_SIZE,
        )
        assert batch.no_penalties

        metadata = batch.make_sampling_metadata()

        assert metadata.no_penalties is True
        assert metadata.frequency_penalties.numel() == 0
        assert metadata.presence_penalties.numel() == 0
        assert metadata.repetition_penalties.numel() == 0
        assert metadata.frequency_penalties.dtype == torch.float32
        assert metadata.presence_penalties.dtype == torch.float32
        assert metadata.repetition_penalties.dtype == torch.float32

    def test_sampling_metadata_allocates_penalty_tensors_when_penalized(
        self,
    ) -> None:
        """Regression: ensure penalty tensors are still built per-request when
        any request opts into a non-default penalty value.
        """
        batch = SamplingBatch(
            [
                SamplingParams(temperature=0.0),
                SamplingParams(temperature=0.8, frequency_penalty=0.5),
                SamplingParams(temperature=0.8, repetition_penalty=1.2),
            ],
            [[1], [2], [3]],
            [[], [], []],
            vocab_size=VOCAB_SIZE,
        )
        assert not batch.no_penalties

        metadata = batch.make_sampling_metadata()

        assert metadata.no_penalties is False
        assert metadata.frequency_penalties.tolist() == pytest.approx([0.0, 0.5, 0.0])
        assert metadata.presence_penalties.tolist() == pytest.approx([0.0, 0.0, 0.0])
        assert metadata.repetition_penalties.tolist() == pytest.approx([1.0, 1.0, 1.2])

    def test_sampling_metadata_uses_requested_logprobs(self) -> None:
        batch = SamplingBatch(
            [SamplingParams(temperature=0.0, logprobs=5)],
            [[1, 2, 3]],
            [[]],
            vocab_size=VOCAB_SIZE,
        )

        metadata = batch.make_sampling_metadata()

        assert metadata.max_num_logprobs == 5

    def test_sampling_metadata_rejects_full_vocab_logprobs(self) -> None:
        batch = SamplingBatch(
            [SamplingParams(temperature=0.0, logprobs=-1)],
            [[1, 2, 3]],
            [[]],
            vocab_size=VOCAB_SIZE,
        )

        with pytest.raises(
            NotImplementedError,
            match="Metal runner does not support logprobs=-1 yet",
        ):
            batch.make_sampling_metadata()

    def test_sampling_metadata_rejects_mixed_full_vocab_logprobs(self) -> None:
        batch = SamplingBatch(
            [
                SamplingParams(temperature=0.0, logprobs=-1),
                SamplingParams(temperature=0.0, logprobs=2),
            ],
            [[1, 2, 3], [4, 5]],
            [[], []],
            vocab_size=VOCAB_SIZE,
        )

        with pytest.raises(
            NotImplementedError,
            match="Metal runner does not support logprobs=-1 yet",
        ):
            batch.make_sampling_metadata()

    def test_sample_from_logits_returns_logprobs_for_greedy_request(self) -> None:
        logits = mx.array([[0.0, 1.0, 4.0, 2.0]], dtype=mx.float32)
        batch = SamplingBatch(
            [SamplingParams(temperature=0.0, logprobs=2)],
            [[1, 2, 3]],
            [[]],
            vocab_size=4,
        )

        result = sample_from_logits(logits, batch, Sampler())

        assert result.token_ids == [2]
        assert result.logprobs is not None
        assert result.logprobs.logprob_token_ids.shape == (1, 3)
        assert result.logprobs.logprob_token_ids[0, 0] == 2
        assert result.logprobs.sampled_token_ranks.tolist() == [1]

    def test_sample_from_logits_returns_requested_token_logprobs(self) -> None:
        logits = mx.array([[0.0, 1.0, 4.0, 2.0]], dtype=mx.float32)
        batch = SamplingBatch(
            [SamplingParams(temperature=0.0, logprob_token_ids=[0, 3])],
            [[1, 2, 3]],
            [[]],
            vocab_size=4,
        )

        result = sample_from_logits(logits, batch, Sampler())

        assert result.token_ids == [2]
        assert result.logprobs is not None
        assert result.logprobs.logprob_token_ids.tolist() == [[2, 0, 3]]
        expected = torch.log_softmax(torch.tensor([0.0, 1.0, 4.0, 2.0]), dim=0)
        assert result.logprobs.logprobs[0].tolist() == pytest.approx(
            expected[[2, 0, 3]].tolist()
        )
        assert result.logprobs.sampled_token_ranks.tolist() == [1]

    def test_empty_requested_token_list_does_not_request_logprobs(self) -> None:
        params = SamplingParams(temperature=0.0, logprob_token_ids=[])
        batch = SamplingBatch([params], [[1, 2, 3]], [[]], vocab_size=4)

        assert params.num_logprobs is None
        assert batch.can_use_native_greedy()
        assert batch.make_sampling_metadata().logprob_token_ids is None
        assert (
            sample_from_logits(
                mx.array([[0.0, 1.0, 4.0, 2.0]], dtype=mx.float32),
                batch,
                Sampler(),
            ).logprobs
            is None
        )

    @pytest.mark.parametrize("specific_row", [0, 1])
    def test_mixed_specific_and_topk_logprobs_preserve_each_row(
        self, specific_row: int
    ) -> None:
        raw_logits = [[0.0, 1.0, 4.0, 2.0], [5.0, 1.0, 3.0, 2.0]]
        logits = mx.array(raw_logits, dtype=mx.float32)
        specific_params = SamplingParams(temperature=0.0, logprob_token_ids=[0, 3])
        topk_params = SamplingParams(temperature=0.0, logprobs=2)
        sampling_params = [
            specific_params if row == specific_row else topk_params for row in range(2)
        ]
        batch = SamplingBatch(
            sampling_params,
            [[1, 2], [3, 4]],
            [[], []],
            vocab_size=4,
        )

        topk_ids = [[2, 3], [0, 2]]
        metadata = batch.make_sampling_metadata(torch.tensor(raw_logits))
        assert metadata.max_num_logprobs is None
        assert metadata.logprob_token_ids == {
            specific_row: [0, 3],
            1 - specific_row: topk_ids[1 - specific_row],
        }

        result = sample_from_logits(logits, batch, Sampler())

        assert result.token_ids == [2, 0]
        assert result.logprobs is not None
        expected_ids = []
        for row, sampled_token_id in enumerate(result.token_ids):
            requested_ids = [0, 3] if row == specific_row else topk_ids[row]
            expected_ids.append([sampled_token_id, *requested_ids])
        assert result.logprobs.logprob_token_ids.tolist() == expected_ids
        expected = torch.log_softmax(
            torch.tensor([[0.0, 1.0, 4.0, 2.0], [5.0, 1.0, 3.0, 2.0]]),
            dim=1,
        )
        for row, token_ids in enumerate(expected_ids):
            assert result.logprobs.logprobs[row].tolist() == pytest.approx(
                expected[row, token_ids].tolist()
            )
        assert result.logprobs.sampled_token_ranks.tolist() == [1, 1]

    def test_mixed_logprob_metadata_requires_logits(self) -> None:
        batch = SamplingBatch(
            [
                SamplingParams(temperature=0.0, logprob_token_ids=[0, 3]),
                SamplingParams(temperature=0.0, logprobs=2),
            ],
            [[1, 2], [3, 4]],
            [[], []],
            vocab_size=4,
        )

        with pytest.raises(
            ValueError,
            match="Logits are required when a batch mixes top-k and specific-token",
        ):
            batch.make_sampling_metadata()

    def test_same_row_specific_logprobs_override_topk_without_logits(self) -> None:
        batch = SamplingBatch(
            [
                SamplingParams(
                    temperature=0.0,
                    logprobs=2,
                    logprob_token_ids=[0, 3],
                )
            ],
            [[1, 2]],
            [[]],
            vocab_size=4,
        )

        metadata = batch.make_sampling_metadata()

        assert metadata.max_num_logprobs is None
        assert metadata.logprob_token_ids == {0: [0, 3]}

    def test_mixed_sampled_token_only_and_specific_logprobs_need_no_logits(
        self,
    ) -> None:
        batch = SamplingBatch(
            [
                SamplingParams(temperature=0.0, logprob_token_ids=[0, 3]),
                SamplingParams(temperature=0.0, logprobs=0),
            ],
            [[1, 2], [3, 4]],
            [[], []],
            vocab_size=4,
        )

        metadata = batch.make_sampling_metadata()

        assert metadata.max_num_logprobs is None
        assert metadata.logprob_token_ids == {0: [0, 3], 1: []}

    def test_mixed_topk_logprobs_preserve_batch_tie_breaking(self) -> None:
        batch = SamplingBatch(
            [
                SamplingParams(temperature=0.0, logprob_token_ids=[0]),
                SamplingParams(temperature=0.0, logprobs=2),
                SamplingParams(temperature=0.0, logprobs=5),
            ],
            [[1], [2], [3]],
            [[], [], []],
            vocab_size=8,
        )
        logits = torch.tensor(
            [
                [0.0] * 8,
                [-1.0, -1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0],
                list(range(8)),
            ]
        )

        metadata = batch.make_sampling_metadata(logits)

        assert metadata.max_num_logprobs is None
        assert metadata.logprob_token_ids == {
            0: [0],
            1: [5, 7],
            2: [7, 6, 5, 4, 3],
        }

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

        output = batch.to_model_runner_output()

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


class TestV1PenaltyTokenAccounting:
    """Regression tests for prompt vs output token accounting in penalties."""

    @staticmethod
    def _fixed_logits(vocab_size: int, token_a: int, token_b: int) -> mx.array:
        logits = mx.zeros((1, 1, vocab_size), dtype=mx.float32)
        logits = logits.at[:, :, token_a].add(10.0)
        return logits.at[:, :, token_b].add(9.5)

    def test_presence_penalty_does_not_apply_to_prompt_tokens(self) -> None:
        """Presence/frequency penalties should apply only to generated tokens."""
        vocab_size = 64
        prompt_token = 5
        alternative_token = 7
        already_generated_token = 11

        logits = self._fixed_logits(vocab_size, prompt_token, alternative_token)

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
            sampling_params=sp,
            generated_tokens=0,
        )

        next_tokens = sample_decode_tokens(
            logits,
            [("r1", state)],
            num_decode=1,
            vocab_size=vocab_size,
            sampler=Sampler(),
        ).token_ids

        assert next_tokens == [prompt_token]

    def test_frequency_penalty_does_not_apply_to_prompt_tokens(self) -> None:
        """Frequency penalty should apply only to generated tokens."""
        vocab_size = 64
        prompt_token = 5
        alternative_token = 7
        already_generated_token = 11

        logits = self._fixed_logits(vocab_size, prompt_token, alternative_token)

        sp = SamplingParams(
            temperature=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            frequency_penalty=1.0,
        )
        state = RequestState(
            token_ids=[prompt_token, already_generated_token],
            prompt_len=1,
            sampling_params=sp,
            generated_tokens=0,
        )

        next_tokens = sample_decode_tokens(
            logits,
            [("r1", state)],
            num_decode=1,
            vocab_size=vocab_size,
            sampler=Sampler(),
        ).token_ids

        assert next_tokens == [prompt_token]

    def test_presence_penalty_applies_to_output_tokens(self) -> None:
        """Presence penalty should apply to generated tokens."""
        vocab_size = 64
        repeated_token = 5
        alternative_token = 7

        logits = torch.zeros((1, vocab_size), dtype=torch.float32)
        logits[0, repeated_token] = 10.0
        logits[0, alternative_token] = 9.5

        sp = SamplingParams(
            temperature=0.0,
            presence_penalty=1.0,
            repetition_penalty=1.0,
            frequency_penalty=0.0,
        )
        metadata = SamplingBatch(
            sampling_params_list=[sp],
            prompt_token_id_lists=[[]],
            # Mock: the model has already generated `repeated_token`, so presence_penalty
            # should demote it and flip greedy selection to `alternative_token`.
            output_token_id_lists=[[repeated_token]],
            vocab_size=vocab_size,
        ).make_sampling_metadata()

        output = Sampler().forward(logits, metadata)

        assert int(output.sampled_token_ids[0, 0].item()) == alternative_token

    def test_frequency_penalty_applies_to_output_tokens(self) -> None:
        """Frequency penalty should apply to generated tokens."""
        vocab_size = 64
        repeated_token = 5
        alternative_token = 7

        logits = torch.zeros((1, vocab_size), dtype=torch.float32)
        logits[0, repeated_token] = 10.0
        logits[0, alternative_token] = 9.5

        sp = SamplingParams(
            temperature=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            frequency_penalty=1.0,
        )
        metadata = SamplingBatch(
            sampling_params_list=[sp],
            prompt_token_id_lists=[[]],
            # Mock: the model has already generated `repeated_token` once, so a positive
            # frequency_penalty should reduce its probability and pick `alternative_token`.
            output_token_id_lists=[[repeated_token]],
            vocab_size=vocab_size,
        ).make_sampling_metadata()

        output = Sampler().forward(logits, metadata)

        assert int(output.sampled_token_ids[0, 0].item()) == alternative_token

    def test_repetition_penalty_applies_to_prompt_tokens(self) -> None:
        """Repetition penalty should consider prompt+output tokens (vLLM semantics)."""
        vocab_size = 64
        prompt_token = 5
        alternative_token = 7

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
        metadata = SamplingBatch(
            sampling_params_list=[sp],
            # Mock: the prompt already contains `prompt_token`, so repetition_penalty
            # should demote it and prefer `alternative_token`.
            prompt_token_id_lists=[[prompt_token]],
            output_token_id_lists=[[]],
            vocab_size=vocab_size,
        ).make_sampling_metadata()

        output = Sampler().forward(logits, metadata)

        assert int(output.sampled_token_ids[0, 0].item()) == alternative_token

    def test_repetition_penalty_applies_to_output_tokens(self) -> None:
        """Repetition penalty should consider output tokens as well."""
        vocab_size = 64
        repeated_token = 5
        alternative_token = 7

        logits = torch.zeros((1, vocab_size), dtype=torch.float32)
        logits[0, repeated_token] = 10.0
        logits[0, alternative_token] = 9.5

        sp = SamplingParams(
            temperature=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.2,
            frequency_penalty=0.0,
        )
        metadata = SamplingBatch(
            sampling_params_list=[sp],
            prompt_token_id_lists=[[]],
            # Mock: `repeated_token` is already in the generated output, so repetition_penalty
            # should demote it and select `alternative_token`.
            output_token_id_lists=[[repeated_token]],
            vocab_size=vocab_size,
        ).make_sampling_metadata()

        output = Sampler().forward(logits, metadata)

        assert int(output.sampled_token_ids[0, 0].item()) == alternative_token
