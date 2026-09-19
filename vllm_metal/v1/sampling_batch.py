# SPDX-License-Identifier: Apache-2.0
"""Sampling batch ownership and token sampling for the Metal runners.

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
from vllm_metal.v1.logits_processors import BatchMinPLogitsProcessor

GREEDY_TEMPERATURE_EPS = 1e-5
_EMPTY_LOGITSPROCS = LogitsProcessors()


@dataclass(frozen=True)
class _SamplingResult:
    """Sampled token ids plus optional vLLM logprobs rows."""

    token_ids: list[int]
    logprobs: LogprobsLists | None = None


class SamplingBatch:
    """Sampling-side state for one step of one batch.

    Both the generation runner and the one-shot STT decode build one of these
    per step and hand it to :func:`sample_from_logits`.
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
        return any(
            sampling_params.num_logprobs is not None
            for sampling_params in self.sampling_params_list
        )

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
            and sampling_params.num_logprobs is None
            and not sampling_params.allowed_token_ids
            and not sampling_params.bad_words_token_ids
            for sampling_params in sampling_params_list
        )

    @staticmethod
    def params_allow_native_random(
        sampling_params_list: Sequence[SamplingParams],
    ) -> bool:
        """Whether MLX categorical sampling matches *sampling_params_list*.

        Requests must use plain temperature/top-k/top-p/min-p sampling; the
        mask is built per row, so rows may mix greedy (temperature 0) with
        random sampling and differ in top-k/top-p/min-p. Seeded requests
        stay on the torch path.
        """
        if not sampling_params_list:
            return False
        return all(
            sp.seed is None
            and sp.frequency_penalty == 0.0
            and sp.presence_penalty == 0.0
            and sp.repetition_penalty == 1.0
            and sp.num_logprobs is None
            and not sp.allowed_token_ids
            and not sp.bad_words_token_ids
            for sp in sampling_params_list
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
        min_p: float = 0.0,
    ) -> mx.array:
        """Mask temperature-scaled logits to the native candidate set.

        Batch-wide scalar form of :meth:`_sorted_candidate_logits`, scattered
        back into vocab order (used by the mask-parity tests).
        """
        rows = int(scaled_logits.shape[0])
        sorted_desc, sorted_idx = SamplingBatch._sorted_candidate_logits(
            scaled_logits, [top_k] * rows, [top_p] * rows, [min_p] * rows
        )
        return mx.put_along_axis(
            mx.full(scaled_logits.shape, -mx.inf, dtype=scaled_logits.dtype),
            sorted_idx,
            sorted_desc,
            axis=-1,
        )

    @staticmethod
    def _sorted_candidate_logits(
        scaled_logits: mx.array,
        top_k: Sequence[int],
        top_p: Sequence[float],
        min_p: Sequence[float],
    ) -> tuple[mx.array, mx.array]:
        """Per-row top-k/min-p/top-p masking in descending-sorted order.

        ``top_k``/``top_p``/``min_p`` hold one host value per row
        (``top_k <= 0`` means no top-k), so the graph stays lazy: nothing
        here evaluates an array. Returns ``(sorted_logits, sorted_token_ids)``:
        every row's candidate logits in descending order with masked positions
        at ``-inf``, and the vocab ids at those positions. When every row has a
        top-k, only the ``max(top_k)`` largest logits per row are partitioned
        out and sorted; the rest of the vocab is never a candidate, so this is
        exact except for ties straddling that boundary.
        """
        vocab_size = int(scaled_logits.shape[-1])
        effective_k = [k if 0 < k < vocab_size else vocab_size for k in top_k]
        k_max = max(effective_k)
        if k_max < vocab_size:
            cand_idx = mx.argpartition(-scaled_logits, kth=k_max - 1, axis=-1)[
                ..., :k_max
            ]
            cand = mx.take_along_axis(scaled_logits, cand_idx, axis=-1)
            order = mx.argsort(-cand, axis=-1)
            sorted_desc = mx.take_along_axis(cand, order, axis=-1)
            sorted_idx = mx.take_along_axis(cand_idx, order, axis=-1)
        else:
            sorted_idx = mx.argsort(scaled_logits, axis=-1)[..., ::-1]
            sorted_desc = mx.take_along_axis(scaled_logits, sorted_idx, axis=-1)

        if any(k < vocab_size for k in effective_k):
            # Value threshold (not rank) so boundary ties survive like vLLM's
            # top-k mask keeps every logit >= the k-th largest.
            kth_pos = mx.array([k - 1 for k in effective_k], dtype=mx.int32)[:, None]
            kth_value = mx.take_along_axis(sorted_desc, kth_pos, axis=-1)
            sorted_desc = mx.where(sorted_desc < kth_value, -mx.inf, sorted_desc)
        if any(p > 0.0 for p in min_p):
            min_p_col = mx.array(min_p, dtype=mx.float32)[:, None]
            sorted_probs = mx.softmax(sorted_desc, axis=-1)
            keep = sorted_probs >= min_p_col * sorted_probs[..., :1]
            sorted_desc = mx.where(keep, sorted_desc, -mx.inf)
        if any(p < 1.0 for p in top_p):
            top_p_col = mx.array(top_p, dtype=mx.float32)[:, None]
            sorted_probs = mx.softmax(sorted_desc, axis=-1)
            leading_mass = mx.cumsum(sorted_probs, axis=-1) - sorted_probs
            sorted_desc = mx.where(leading_mass < top_p_col, sorted_desc, -mx.inf)
        return sorted_desc, sorted_idx

    @classmethod
    def _native_random_tokens(
        cls,
        logits_2d: mx.array,
        sampling_params_list: Sequence[SamplingParams],
        key: mx.array,
    ) -> mx.array:
        """Lazy per-row temperature/top-k/top-p/min-p token ids.

        Greedy rows (temperature below ``GREEDY_TEMPERATURE_EPS``) become a
        top-1 candidate set at unit temperature, so they resolve to the argmax
        inside the same categorical draw as the random rows.
        """
        temperatures: list[float] = []
        top_k: list[int] = []
        top_p: list[float] = []
        min_p: list[float] = []
        for sp in sampling_params_list:
            if sp.temperature < GREEDY_TEMPERATURE_EPS:
                temperatures.append(1.0)
                top_k.append(1)
                top_p.append(1.0)
                min_p.append(0.0)
            else:
                temperatures.append(sp.temperature)
                top_k.append(sp.top_k)
                top_p.append(sp.top_p)
                min_p.append(sp.min_p)
        scaled = (
            logits_2d.astype(mx.float32)
            / mx.array(temperatures, dtype=mx.float32)[:, None]
        )
        sorted_desc, sorted_idx = cls._sorted_candidate_logits(
            scaled, top_k, top_p, min_p
        )
        position = mx.random.categorical(sorted_desc, axis=-1, key=key)
        return mx.take_along_axis(sorted_idx, position[:, None], axis=-1)[:, 0]

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

    def _make_logprob_args(
        self,
        logits: torch.Tensor | None,
    ) -> tuple[int | None, dict[int, list[int]] | None]:
        """Build mutually exclusive logprob arguments for vLLM's sampler.

        vLLM's compatibility sampler treats any ``logprob_token_ids`` mapping
        as a batch-wide override of ``max_num_logprobs``. When a batch mixes
        both request types, materialize the ordinary rows' raw-logit top-k IDs
        into the mapping and disable the batch-wide top-k argument.
        """
        max_num_logprobs = self.max_num_logprobs
        token_ids_by_row: dict[int, list[int]] = {}
        for i, sampling_params in enumerate(self.sampling_params_list):
            if sampling_params.logprob_token_ids:
                token_ids_by_row[i] = sampling_params.logprob_token_ids
        if not token_ids_by_row:
            return max_num_logprobs, None

        topk_requests = [
            (i, sampling_params.logprobs)
            for i, sampling_params in enumerate(self.sampling_params_list)
            if i not in token_ids_by_row and sampling_params.logprobs is not None
        ]
        max_topk = max((num_logprobs for _, num_logprobs in topk_requests), default=0)
        if max_topk > 0:
            if logits is None:
                raise ValueError(
                    "Logits are required when a batch mixes top-k and "
                    "specific-token logprobs."
                )
            topk_token_ids = torch.topk(logits, max_topk, dim=-1).indices
            for i, num_logprobs in topk_requests:
                token_ids_by_row[i] = topk_token_ids[i, :num_logprobs].tolist()
        else:
            for i, _ in topk_requests:
                token_ids_by_row[i] = []
        return None, token_ids_by_row

    def _make_logitsprocs(self) -> LogitsProcessors:
        min_p_vals = [sp.min_p for sp in self.sampling_params_list]
        if not any(min_p > 0.0 for min_p in min_p_vals):
            return _EMPTY_LOGITSPROCS
        return LogitsProcessors(
            [
                BatchMinPLogitsProcessor(
                    torch.tensor(
                        min_p_vals,
                        dtype=torch.float32,
                        device=self.SAMPLER_DEVICE,
                    )
                )
            ]
        )

    def make_sampling_metadata(
        self, logits: torch.Tensor | None = None
    ) -> SamplingMetadata:
        """Create vLLM ``SamplingMetadata`` for this batch."""
        (
            frequency_penalties,
            presence_penalties,
            repetition_penalties,
        ) = self._make_penalty_tensors()
        max_num_logprobs, logprob_token_ids = self._make_logprob_args(logits)

        return SamplingMetadata(
            temperature=self._make_temperature(),
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=self._make_top_p(),
            top_k=self._make_top_k(),
            generators=self.generators,
            max_num_logprobs=max_num_logprobs,
            prompt_token_ids=self._make_prompt_token_ids(),
            output_token_ids=self.output_token_id_lists,
            frequency_penalties=frequency_penalties,
            presence_penalties=presence_penalties,
            repetition_penalties=repetition_penalties,
            no_penalties=self.no_penalties,
            allowed_token_ids_mask=self._make_allowed_token_ids_mask(),
            bad_words_token_ids=self._make_bad_words_token_ids(),
            logitsprocs=self._make_logitsprocs(),
            logprob_token_ids=logprob_token_ids,
        )


# ---------------------------------------------------------------------------
# Pure sampling functions
# ---------------------------------------------------------------------------


def create_request_generator(
    sampling_params: SamplingParams,
) -> torch.Generator | None:
    """Create a per-request generator for seeded sampling.

    vLLM uses a per-request generator only when an explicit seed is provided.
    For unseeded sampling, vLLM relies on the global RNG state.
    """
    if sampling_params.seed is None:
        return None
    if sampling_params.temperature < GREEDY_TEMPERATURE_EPS:
        return None
    generator = torch.Generator(device=SamplingBatch.SAMPLER_DEVICE)
    generator.manual_seed(sampling_params.seed)
    return generator


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

    The bridged tensor aliases ``logits_2d``, so a penalized call rewrites the
    caller's array; callers pass logits they do not read again.
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
    metadata = batch.make_sampling_metadata(logits_torch)
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
