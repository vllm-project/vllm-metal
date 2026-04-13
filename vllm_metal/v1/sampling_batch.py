# SPDX-License-Identifier: Apache-2.0
"""Internal sampling batch ownership and token sampling for Metal v1.

Pure functions: logits in, token IDs out.  No model runner state accessed.
"""

from collections.abc import Sequence

import mlx.core as mx
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch

GREEDY_TEMPERATURE_EPS = 1e-5


class SamplingBatch:
    """Sampling-side batch owner for ``MetalModelRunner``.

    This is an interim extraction that keeps sampling policy and
    ``SamplingMetadata`` construction out of ``model_runner.py`` while the
    runner is being slimmed down.

    Today it owns only the sampling-side state for one step. As more per-step
    batch state moves out of ``model_runner.py``, this should evolve into a
    fuller ``MetalInputBatch``-style owner that can absorb request indexing,
    token views, generators, logits processor ownership, and metadata refresh.
    """

    def __init__(
        self,
        sampling_params_list: Sequence[SamplingParams],
        prompt_token_id_lists: Sequence[list[int]],
        output_token_id_lists: Sequence[list[int]],
        *,
        vocab_size: int,
        device: torch.device,
        logitsprocs: LogitsProcessors | None = None,
        generators: dict[int, torch.Generator] | None = None,
    ) -> None:
        batch_size = len(sampling_params_list)
        if len(prompt_token_id_lists) != batch_size:
            raise ValueError(
                "Expected prompt token ids for each request in the batch "
                f"(len(prompt_token_id_lists)={len(prompt_token_id_lists)} "
                f"!= batch_size={batch_size})."
            )
        if len(output_token_id_lists) != batch_size:
            raise ValueError(
                "Expected output token ids for each request in the batch "
                f"(len(output_token_id_lists)={len(output_token_id_lists)} "
                f"!= batch_size={batch_size})."
            )

        self.sampling_params_list = list(sampling_params_list)
        self.prompt_token_id_lists = list(prompt_token_id_lists)
        self.output_token_id_lists = list(output_token_id_lists)
        self.vocab_size = vocab_size
        self.device = device
        self.logitsprocs = logitsprocs or LogitsProcessors()
        self.generators = {} if generators is None else generators
        self.all_greedy = all(
            sampling_params.temperature < GREEDY_TEMPERATURE_EPS
            for sampling_params in self.sampling_params_list
        )
        self.all_random = not self.all_greedy and all(
            sampling_params.temperature >= GREEDY_TEMPERATURE_EPS
            for sampling_params in self.sampling_params_list
        )
        self.no_top_p = all(
            sampling_params.top_p == 1.0
            for sampling_params in self.sampling_params_list
        )
        self.no_top_k = all(
            sampling_params.top_k <= 0 for sampling_params in self.sampling_params_list
        )
        self.no_penalties = all(
            sampling_params.frequency_penalty == 0.0
            and sampling_params.presence_penalty == 0.0
            and sampling_params.repetition_penalty == 1.0
            for sampling_params in self.sampling_params_list
        )

    @staticmethod
    def can_use_native_greedy(
        sampling_params_list: Sequence[SamplingParams],
    ) -> bool:
        """Return whether MLX argmax matches the requested sampling behavior."""
        return all(
            sampling_params.temperature < GREEDY_TEMPERATURE_EPS
            and sampling_params.top_k <= 0
            and sampling_params.top_p == 1.0
            and sampling_params.frequency_penalty == 0.0
            and sampling_params.presence_penalty == 0.0
            and sampling_params.repetition_penalty == 1.0
            for sampling_params in sampling_params_list
        )

    def _make_temperature(self) -> torch.Tensor | None:
        if self.all_greedy:
            return None

        return torch.tensor(
            [
                sampling_params.temperature
                for sampling_params in self.sampling_params_list
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def _make_top_p(self) -> torch.Tensor | None:
        if self.no_top_p:
            return None

        return torch.tensor(
            [sampling_params.top_p for sampling_params in self.sampling_params_list],
            dtype=torch.float32,
            device=self.device,
        )

    def _make_top_k(self) -> torch.Tensor | None:
        if self.no_top_k:
            return None

        return torch.tensor(
            [sampling_params.top_k for sampling_params in self.sampling_params_list],
            dtype=torch.int32,
            device=self.device,
        )

    def _make_prompt_token_ids(self) -> torch.Tensor | None:
        if self.no_penalties:
            return None

        return make_tensor_with_pad(
            self.prompt_token_id_lists,
            pad=self.vocab_size,
            device=self.device,
            dtype=torch.int64,
            pin_memory=False,
        )

    def _make_penalty_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frequency_penalties = torch.tensor(
            [
                sampling_params.frequency_penalty
                for sampling_params in self.sampling_params_list
            ],
            dtype=torch.float32,
            device=self.device,
        )
        presence_penalties = torch.tensor(
            [
                sampling_params.presence_penalty
                for sampling_params in self.sampling_params_list
            ],
            dtype=torch.float32,
            device=self.device,
        )
        repetition_penalties = torch.tensor(
            [
                sampling_params.repetition_penalty
                for sampling_params in self.sampling_params_list
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return frequency_penalties, presence_penalties, repetition_penalties

    def make_sampling_metadata(self) -> SamplingMetadata:
        """Create vLLM ``SamplingMetadata`` for this batch."""
        (
            frequency_penalties,
            presence_penalties,
            repetition_penalties,
        ) = self._make_penalty_tensors()

        return SamplingMetadata(
            temperature=self._make_temperature(),
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=self._make_top_p(),
            top_k=self._make_top_k(),
            generators=self.generators,
            max_num_logprobs=None,
            prompt_token_ids=self._make_prompt_token_ids(),
            output_token_ids=self.output_token_id_lists,
            frequency_penalties=frequency_penalties,
            presence_penalties=presence_penalties,
            repetition_penalties=repetition_penalties,
            no_penalties=self.no_penalties,
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=self.logitsprocs,
        )


# ---------------------------------------------------------------------------
# Pure sampling functions
# ---------------------------------------------------------------------------


def _mlx_greedy_sample(logits: mx.array) -> mx.array:
    """Native MLX greedy sampling — avoids PyTorch round-trip."""
    return mx.argmax(logits, axis=-1)


def sample_from_logits(
    logits_2d: mx.array,
    batch: SamplingBatch,
    sampler: Sampler,
    device: torch.device,
) -> list[int]:
    """Sample tokens from pre-sliced 2D logits ``(batch_size, vocab)``.

    Single entry point for all sampling paths.  Chooses native MLX greedy
    when possible, otherwise bridges to the vLLM torch sampler.
    """
    if batch.all_greedy and batch.no_top_k and batch.no_top_p and batch.no_penalties:
        tokens = _mlx_greedy_sample(logits_2d)
        mx.eval(tokens)
        if tokens.ndim == 0:
            return [int(tokens.item())]
        return tokens.tolist()  # type: ignore[return-value]

    mx.eval(logits_2d)
    logits_torch = mlx_to_torch(logits_2d.astype(mx.float32), device=device)
    metadata = batch.make_sampling_metadata()
    output = sampler.forward(logits_torch, metadata)
    return output.sampled_token_ids[:, 0].tolist()


def sample_decode_tokens(
    logits: mx.array,
    decode_reqs: list[tuple[str, object]],
    num_decode: int,
    sampler: Sampler,
    device: torch.device,
    *,
    vocab_size: int,
    logitsprocs: LogitsProcessors | None = None,
) -> list[int]:
    """Sample one token per decode request from evaluated logits.

    Args:
        logits: Full logits array, shape ``(1, total_tokens, vocab)``.
        decode_reqs: ``(req_id, RequestState)`` pairs for decode requests.
        num_decode: Number of decode requests (prefix of the token dimension).
        sampler: vLLM Sampler instance.
        device: PyTorch device for the torch bridge path.
        vocab_size: Model vocabulary size.
        logitsprocs: Optional logits processors.

    Returns:
        List of sampled token IDs, one per decode request.
    """
    if not decode_reqs:
        return []

    decode_logits = logits[0, :num_decode, :]  # (num_decode, vocab)

    sampling_params_list = [state.sampling_params for _, state in decode_reqs]
    prompt_token_ids_list = [
        state.token_ids[: state.prompt_len] for _, state in decode_reqs
    ]
    output_tokens_list = [
        state.token_ids[state.prompt_len :] for _, state in decode_reqs
    ]
    generators = {
        i: state.generator
        for i, (_, state) in enumerate(decode_reqs)
        if state.generator is not None
    }

    batch = SamplingBatch(
        sampling_params_list,
        prompt_token_ids_list,
        output_tokens_list,
        vocab_size=vocab_size,
        device=device,
        logitsprocs=logitsprocs,
        generators=generators,
    )
    return sample_from_logits(decode_logits, batch, sampler, device)


def sample_prefill_tokens(
    logits: mx.array,
    prefill_reqs: list,
    cu_seqlens: list[int],
    num_decode: int,
    sampler: Sampler,
    device: torch.device,
    *,
    vocab_size: int,
    logitsprocs: LogitsProcessors | None = None,
) -> list[int]:
    """Sample one token per prefill request from the last logit position.

    Args:
        logits: Full logits array, shape ``(1, total_tokens, vocab)``.
        prefill_reqs: List of ``PrefillRequest`` objects.
        cu_seqlens: Cumulative sequence lengths for logit position lookup.
        num_decode: Number of decode requests (offset into cu_seqlens).
        sampler: vLLM Sampler instance.
        device: PyTorch device for the torch bridge path.
        vocab_size: Model vocabulary size.
        logitsprocs: Optional logits processors.

    Returns:
        List of sampled token IDs, one per prefill request.
    """
    prefill_next_tokens: list[int] = []
    for j, pr in enumerate(prefill_reqs):
        last_idx = cu_seqlens[num_decode + j + 1] - 1
        last_logits = logits[0, last_idx : last_idx + 1, :]  # (1, vocab)

        if pr.full_prompt_token_ids is not None:
            prompt_len = len(pr.full_prompt_token_ids)
        elif pr.prompt_len is not None:
            prompt_len = pr.prompt_len
        else:
            prompt_len = len(pr.token_ids)

        prompt_for_meta = (
            pr.full_prompt_token_ids
            if pr.full_prompt_token_ids is not None
            else pr.token_ids
        )
        generators = {} if pr.generator is None else {0: pr.generator}

        batch = SamplingBatch(
            [pr.sampling_params],
            [prompt_for_meta[:prompt_len]],
            [prompt_for_meta[prompt_len:]],
            vocab_size=vocab_size,
            device=device,
            logitsprocs=logitsprocs,
            generators=generators,
        )
        [next_token] = sample_from_logits(last_logits, batch, sampler, device)
        prefill_next_tokens.append(next_token)

    return prefill_next_tokens
