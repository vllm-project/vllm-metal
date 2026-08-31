# SPDX-License-Identifier: Apache-2.0
"""Internal sampling batch ownership and token sampling for Metal v1.

Pure functions: logits in, token IDs out.  No model runner state accessed.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import mlx.core as mx
import numpy as np
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.outputs import LogprobsLists
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch

GREEDY_TEMPERATURE_EPS = 1e-5
_EMPTY_LOGITSPROCS = LogitsProcessors()


@dataclass(frozen=True)
class _SamplingResult:
    """Sampled token ids plus optional vLLM logprobs rows."""

    token_ids: list[int]
    logprobs: LogprobsLists | None = None


class SamplingBatch:
    """Sampling-side batch owner for ``MetalModelRunner``.

    This is an interim extraction that keeps sampling policy and
    ``SamplingMetadata`` construction out of ``model_runner.py`` while the
    runner is being slimmed down.

    Today it owns only the sampling-side state for one step.
    """

    # The torch sampler always runs on CPU. ``Tensor.exponential_()`` on MPS
    # can return exact zeros, and the Gumbel-style ``probs.div(q).argmax()``
    # in vLLM's sampler maps a zero draw to an inf/NaN row whose argmax picks
    # an arbitrary vocab id (issue #622).
    SAMPLER_DEVICE: ClassVar[torch.device] = torch.device("cpu")

    def __init__(
        self,
        sampling_params_list: Sequence[SamplingParams],
        prompt_token_id_lists: Sequence[list[int]],
        output_token_id_lists: Sequence[list[int]],
        *,
        vocab_size: int,
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

    @property
    def max_num_logprobs(self) -> int | None:
        """Return the batch-wide sample-logprobs request, if any."""
        requested = [
            sampling_params.logprobs
            for sampling_params in self.sampling_params_list
            if sampling_params.logprobs is not None
        ]
        if any(num_logprobs == -1 for num_logprobs in requested):
            raise NotImplementedError("Metal runner does not support logprobs=-1 yet")
        return max(requested) if requested else None

    @property
    def needs_logprobs(self) -> bool:
        return self.max_num_logprobs is not None

    @staticmethod
    def merge_logprobs_rows(
        rows: Sequence[LogprobsLists | None],
    ) -> LogprobsLists | None:
        """Merge per-request sample logprobs from sampling calls."""
        present_rows = [row for row in rows if row is not None]
        if not present_rows:
            return None

        max_width = max(row.logprob_token_ids.shape[1] for row in present_rows)
        token_rows: list[np.ndarray] = []
        logprob_rows: list[np.ndarray] = []
        rank_rows: list[np.ndarray] = []

        for row in rows:
            if row is None:
                token_rows.append(np.zeros((1, max_width), dtype=np.int32))
                logprob_rows.append(
                    np.full((1, max_width), float("-inf"), dtype=np.float32)
                )
                rank_rows.append(np.zeros((1,), dtype=np.int32))
                continue

            token_ids = row.logprob_token_ids
            logprobs = row.logprobs
            if token_ids.shape[1] < max_width:
                pad_width = max_width - token_ids.shape[1]
                token_ids = np.pad(
                    token_ids,
                    ((0, 0), (0, pad_width)),
                    mode="constant",
                    constant_values=0,
                )
                logprobs = np.pad(
                    logprobs,
                    ((0, 0), (0, pad_width)),
                    mode="constant",
                    constant_values=float("-inf"),
                )

            token_rows.append(token_ids.astype(np.int32, copy=False))
            logprob_rows.append(logprobs.astype(np.float32, copy=False))
            rank_rows.append(row.sampled_token_ranks.astype(np.int32, copy=False))

        return LogprobsLists(
            logprob_token_ids=np.concatenate(token_rows, axis=0),
            logprobs=np.concatenate(logprob_rows, axis=0),
            sampled_token_ranks=np.concatenate(rank_rows, axis=0),
            cu_num_generated_tokens=None,
        )

    def can_use_native_greedy(self) -> bool:
        """Return whether MLX argmax matches the requested sampling behavior."""
        return self.params_allow_native_greedy(self.sampling_params_list)

    @staticmethod
    def params_allow_native_greedy(
        sampling_params_list: Sequence[SamplingParams],
    ) -> bool:
        """Whether MLX argmax matches *sampling_params_list* exactly.

        Single source of truth for the native-greedy decision so callers that
        gate before constructing a :class:`SamplingBatch` (e.g. the decode
        pipeline) cannot drift from the sampling-time check.
        """
        return all(
            sampling_params.temperature < GREEDY_TEMPERATURE_EPS
            and sampling_params.top_k <= 0
            and sampling_params.top_p == 1.0
            and sampling_params.frequency_penalty == 0.0
            and sampling_params.presence_penalty == 0.0
            and sampling_params.repetition_penalty == 1.0
            and sampling_params.logprobs is None
            and not sampling_params.allowed_token_ids
            and not sampling_params.bad_words_token_ids
            for sampling_params in sampling_params_list
        )

    @staticmethod
    def params_allow_native_random(
        sampling_params_list: Sequence[SamplingParams],
    ) -> bool:
        """Whether MLX categorical sampling matches *sampling_params_list*.

        Mirror of :meth:`params_allow_native_greedy` for the non-greedy case:
        every request must use plain temperature/top-k/top-p sampling, with
        one shared ``(top_k, top_p)`` across the batch so a single mask graph
        covers every row. Seeded requests keep the torch path, whose
        per-request ``torch.Generator`` contract MLX keys do not reproduce.
        Options the platform rejects outright (``min_p``, ``logit_bias``)
        are not re-checked here.
        """
        if not sampling_params_list:
            return False
        shared_top_k = len({sp.top_k for sp in sampling_params_list}) == 1
        shared_top_p = len({sp.top_p for sp in sampling_params_list}) == 1
        return (
            shared_top_k
            and shared_top_p
            and all(
                sp.temperature >= GREEDY_TEMPERATURE_EPS
                and sp.seed is None
                and sp.frequency_penalty == 0.0
                and sp.presence_penalty == 0.0
                and sp.repetition_penalty == 1.0
                and sp.logprobs is None
                and sp.logprob_token_ids is None
                and not sp.allowed_token_ids
                and not sp.bad_words_token_ids
                for sp in sampling_params_list
            )
        )

    @classmethod
    def native_decode_tokens(
        cls,
        logits_2d: mx.array,
        sampling_params_list: Sequence[SamplingParams],
        *,
        vocab_size: int,
        next_key: Callable[[], mx.array] | None,
    ) -> mx.array:
        """Build the lazy native token graph for a deferred decode batch.

        Owns the sampling policy for the decode pipeline's deferred step:
        greedy argmax when the batch allows it, otherwise the temperature/
        top-k/top-p categorical graph over the vocab-sliced logits (unlike
        argmax, categorical can select a padded lm_head column). The caller
        owns evaluation and RNG state; ``next_key`` is consumed only on the
        random path. Raises on batches neither native path admits — the
        pipeline gate must not let those reach here.
        """
        if cls.params_allow_native_greedy(sampling_params_list):
            return mlx_greedy_tokens(logits_2d)
        if next_key is not None and cls.params_allow_native_random(
            sampling_params_list
        ):
            return cls._native_random_tokens(
                logits_2d[..., :vocab_size], sampling_params_list, next_key()
            )
        key_state = "live" if next_key is not None else "absent"
        raise RuntimeError(
            "Deferred sampling requires a native-greedy or native-random "
            f"eligible batch (native sample key {key_state}, random-eligible="
            f"{cls.params_allow_native_random(sampling_params_list)}) "
            "— gate desync."
        )

    @staticmethod
    def _top_k_top_p_masked_logits(
        scaled_logits: mx.array,
        top_k: int,
        top_p: float,
    ) -> mx.array:
        """Mask temperature-scaled logits to the top-k/top-p candidate set.

        Mask semantics match vLLM's ``apply_top_k_top_p``: ties at the top-k
        threshold survive, while top-p masks sorted positions individually
        (boundary ties do NOT all survive; which tied token survives follows
        sort order). The leading sorted position carries zero leading mass,
        so every valid ``top_p > 0`` keeps at least one candidate.
        Non-candidates become ``-inf``.
        """
        vocab_size = int(scaled_logits.shape[-1])
        if 0 < top_k < vocab_size:
            kth_largest = mx.min(
                mx.topk(scaled_logits, k=top_k, axis=-1), axis=-1, keepdims=True
            )
            scaled_logits = mx.where(
                scaled_logits < kth_largest, -mx.inf, scaled_logits
            )
        if top_p < 1.0:
            order_desc = mx.argsort(scaled_logits, axis=-1)[..., ::-1]
            sorted_desc = mx.take_along_axis(scaled_logits, order_desc, axis=-1)
            sorted_probs = mx.softmax(sorted_desc, axis=-1)
            leading_mass = mx.cumsum(sorted_probs, axis=-1) - sorted_probs
            keep = leading_mass < top_p
            masked_sorted = mx.where(keep, sorted_desc, -mx.inf)
            scaled_logits = mx.put_along_axis(
                mx.full(scaled_logits.shape, -mx.inf, dtype=scaled_logits.dtype),
                order_desc,
                masked_sorted,
                axis=-1,
            )
        return scaled_logits

    @classmethod
    def _native_random_tokens(
        cls,
        logits_2d: mx.array,
        sampling_params_list: Sequence[SamplingParams],
        key: mx.array,
    ) -> mx.array:
        """Lazy temperature/top-k/top-p token ids, one per row.

        Only valid for batches that pass :meth:`params_allow_native_random`:
        per-row temperature with one shared ``(top_k, top_p)``.
        """
        temperatures = mx.array(
            [sp.temperature for sp in sampling_params_list], dtype=mx.float32
        )
        scaled = logits_2d.astype(mx.float32) / temperatures[:, None]
        masked = cls._top_k_top_p_masked_logits(
            scaled,
            sampling_params_list[0].top_k,
            sampling_params_list[0].top_p,
        )
        return mx.random.categorical(masked, axis=-1, key=key)

    def _make_temperature(self) -> torch.Tensor | None:
        if self.all_greedy:
            return None

        return torch.tensor(
            [
                sampling_params.temperature
                for sampling_params in self.sampling_params_list
            ],
            dtype=torch.float32,
            device=self.SAMPLER_DEVICE,
        )

    def _make_top_p(self) -> torch.Tensor | None:
        if self.no_top_p:
            return None

        return torch.tensor(
            [sampling_params.top_p for sampling_params in self.sampling_params_list],
            dtype=torch.float32,
            device=self.SAMPLER_DEVICE,
        )

    def _make_top_k(self) -> torch.Tensor | None:
        if self.no_top_k:
            return None

        # Match vLLM's per-row convention: top_k <= 0 (or >= vocab_size)
        # disables top-k, and the sentinel is vocab_size, not 0. The PyTorch
        # top-k path computes a gather index of vocab_size - top_k, so a 0
        # sentinel indexes out of bounds (issue #646). GPUInputBatch and the
        # v1 sampler states normalize the same way.
        return torch.tensor(
            [
                sampling_params.top_k
                if 0 < sampling_params.top_k < self.vocab_size
                else self.vocab_size
                for sampling_params in self.sampling_params_list
            ],
            dtype=torch.int32,
            device=self.SAMPLER_DEVICE,
        )

    def _make_prompt_token_ids(self) -> torch.Tensor | None:
        if self.no_penalties:
            return None

        return make_tensor_with_pad(
            self.prompt_token_id_lists,
            pad=self.vocab_size,
            device=self.SAMPLER_DEVICE,
            dtype=torch.int64,
            pin_memory=False,
        )

    def _make_penalty_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build per-request penalty tensors.

        When the batch has no penalties, the vLLM ``Sampler`` and
        ``RejectionSampler`` short-circuit on ``no_penalties=True`` before
        touching these tensors, so we can return zero-length placeholders and
        avoid allocating ``batch_size`` tensors three times every step.
        """
        if self.no_penalties:
            empty = torch.empty(0, dtype=torch.float32, device=self.SAMPLER_DEVICE)
            return empty, empty, empty

        frequency_penalties = torch.tensor(
            [
                sampling_params.frequency_penalty
                for sampling_params in self.sampling_params_list
            ],
            dtype=torch.float32,
            device=self.SAMPLER_DEVICE,
        )
        presence_penalties = torch.tensor(
            [
                sampling_params.presence_penalty
                for sampling_params in self.sampling_params_list
            ],
            dtype=torch.float32,
            device=self.SAMPLER_DEVICE,
        )
        repetition_penalties = torch.tensor(
            [
                sampling_params.repetition_penalty
                for sampling_params in self.sampling_params_list
            ],
            dtype=torch.float32,
            device=self.SAMPLER_DEVICE,
        )
        return frequency_penalties, presence_penalties, repetition_penalties

    def _make_allowed_token_ids_mask(self) -> torch.Tensor | None:
        """Build allowed_token_ids_mask from SamplingParams.

        Mask convention: True -> disallowed.
        Unconstrained request keeps all-False rows so they can sample any token.
        """
        if not any(sp.allowed_token_ids for sp in self.sampling_params_list):
            return None
        mask = torch.zeros(
            len(self.sampling_params_list),
            self.vocab_size,
            dtype=torch.bool,
            device=self.SAMPLER_DEVICE,
        )
        for i, sp in enumerate(self.sampling_params_list):
            if sp.allowed_token_ids:
                mask[i] = True
                mask[i, sp.allowed_token_ids] = False
        return mask

    def _make_bad_words_token_ids(self) -> dict[int, list[list[int]]]:
        """Build bad_words_token_ids from SamplingParams."""
        result: dict[int, list[list[int]]] = {}
        for i, sp in enumerate(self.sampling_params_list):
            if sp.bad_words_token_ids:
                result[i] = sp.bad_words_token_ids
        return result

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
            max_num_logprobs=self.max_num_logprobs,
            prompt_token_ids=self._make_prompt_token_ids(),
            output_token_ids=self.output_token_id_lists,
            frequency_penalties=frequency_penalties,
            presence_penalties=presence_penalties,
            repetition_penalties=repetition_penalties,
            no_penalties=self.no_penalties,
            allowed_token_ids_mask=self._make_allowed_token_ids_mask(),
            bad_words_token_ids=self._make_bad_words_token_ids(),
            logitsprocs=_EMPTY_LOGITSPROCS,
            logprob_token_ids=None,
        )


# ---------------------------------------------------------------------------
# Pure sampling functions
# ---------------------------------------------------------------------------


def mlx_greedy_tokens(logits_2d: mx.array) -> mx.array:
    """Lazy native-greedy token ids for pre-sliced 2D logits.

    Pure graph construction — callers own evaluation, which lets the decode
    pipeline submit the argmax asynchronously and defer the sync one step.
    """
    return mx.argmax(logits_2d, axis=-1)


def sample_from_logits(
    logits_2d: mx.array,
    batch: SamplingBatch,
    sampler: Sampler,
) -> _SamplingResult:
    """Sample tokens from pre-sliced 2D logits ``(batch_size, vocab)``.

    Single entry point for all sampling paths.  Chooses native MLX greedy
    when possible, otherwise bridges to the vLLM torch sampler on
    ``SamplingBatch.SAMPLER_DEVICE``. Requests that need sample logprobs must
    use the vLLM sampler so ``ModelRunnerOutput`` can satisfy the OpenAI
    serving contract.
    """
    if batch.can_use_native_greedy() and not batch.needs_logprobs:
        tokens = mlx_greedy_tokens(logits_2d)
        mx.eval(tokens)
        if tokens.ndim == 0:
            return _SamplingResult([int(tokens.item())])
        return _SamplingResult(tokens.tolist())  # type: ignore[arg-type]

    mx.eval(logits_2d)
    logits_torch = mlx_to_torch(
        logits_2d.astype(mx.float32), device=SamplingBatch.SAMPLER_DEVICE
    )
    metadata = batch.make_sampling_metadata()
    output = sampler.forward(logits_torch, metadata)
    logprobs = (
        output.logprobs_tensors.tolists()
        if output.logprobs_tensors is not None
        else None
    )
    return _SamplingResult(output.sampled_token_ids[:, 0].tolist(), logprobs)


def sample_decode_tokens(
    logits: mx.array,
    decode_reqs: list[tuple[str, object]],
    num_decode: int,
    sampler: Sampler,
    *,
    vocab_size: int,
) -> _SamplingResult:
    """Sample one token per decode request from evaluated logits.

    Args:
        logits: Full logits array, shape ``(1, total_tokens, vocab)``.
        decode_reqs: ``(req_id, RequestState)`` pairs for decode requests.
        num_decode: Number of decode requests (prefix of the token dimension).
        sampler: vLLM Sampler instance.
        vocab_size: Model vocabulary size.
    Returns:
        Sampled token IDs and optional logprobs, one row per decode request.
    """
    if not decode_reqs:
        return _SamplingResult([])

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
        generators=generators,
    )
    return sample_from_logits(decode_logits, batch, sampler)


def sample_prefill_tokens(
    logits: mx.array,
    prefill_reqs: list,
    cu_seqlens: list[int],
    num_decode: int,
    sampler: Sampler,
    *,
    vocab_size: int,
) -> _SamplingResult:
    """Sample one token per prefill request from the last logit position.

    Args:
        logits: Full logits array, shape ``(1, total_tokens, vocab)``.
        prefill_reqs: List of ``PrefillRequest`` objects.
        cu_seqlens: Cumulative sequence lengths for logit position lookup.
        num_decode: Number of decode requests (offset into cu_seqlens).
        sampler: vLLM Sampler instance.
        vocab_size: Model vocabulary size.
    Returns:
        Sampled token IDs and optional logprobs, one row per prefill request.
    """
    if not prefill_reqs:
        return _SamplingResult([])

    prompt_token_id_lists: list[list[int]] = []
    output_token_id_lists: list[list[int]] = []
    for pr in prefill_reqs:
        token_ids = (
            pr.full_prompt_token_ids
            if pr.full_prompt_token_ids is not None
            else pr.token_ids
        )
        prompt_len = pr.prompt_len if pr.prompt_len is not None else len(token_ids)
        prompt_token_id_lists.append(token_ids[:prompt_len])
        output_token_id_lists.append(token_ids[prompt_len:])

    last_logits = mx.stack(
        [
            logits[0, cu_seqlens[num_decode + j + 1] - 1, :]
            for j in range(len(prefill_reqs))
        ]
    )
    batch = SamplingBatch(
        [pr.sampling_params for pr in prefill_reqs],
        prompt_token_id_lists,
        output_token_id_lists,
        vocab_size=vocab_size,
        generators={
            j: pr.generator
            for j, pr in enumerate(prefill_reqs)
            if pr.generator is not None
        },
    )
    return sample_from_logits(last_logits, batch, sampler)
