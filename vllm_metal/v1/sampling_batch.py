# SPDX-License-Identifier: Apache-2.0
"""Sampling batch ownership and token sampling for the Metal runners.

Pure functions: logits in, token IDs out.  No model runner state accessed.
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

import mlx.core as mx
import numpy as np
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, SamplerOutput
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitBiasLogitsProcessor,
    LogitsProcessor,
    LogitsProcessors,
    MinTokensLogitsProcessor,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_metal.tensor_bridge import mlx_to_torch
from vllm_metal.v1.logits_processors import BatchMinPLogitsProcessor
from vllm_metal.v1.native_logprobs import gather_native_logprobs

GREEDY_TEMPERATURE_EPS = 1e-5
_EMPTY_LOGITSPROCS = LogitsProcessors()


@dataclass(frozen=True)
class _SamplingResult:
    """Sampled token ids plus optional vLLM logprobs rows."""

    token_ids: list[int]
    logprobs: LogprobsLists | None = None


@dataclass(frozen=True)
class _NativeSample:
    token_ids: mx.array
    processed_logits: mx.array | None = None


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

    @property
    def output_lengths(self) -> list[int]:
        return [len(token_ids) for token_ids in self.output_token_id_lists]

    def generators_are_seeded(self) -> bool:
        """Whether every torch generator belongs to a seeded row.

        Seeded rows get their generator from :func:`create_request_generator`;
        on the native path the seed-derived MLX key replaces it. Any other
        generator is an explicit caller RNG that only torch can honor.
        """
        return all(
            0 <= i < len(self.sampling_params_list)
            and self.sampling_params_list[i].seed is not None
            for i in self.generators
        )

    def can_use_native_greedy(self) -> bool:
        """Return whether MLX argmax matches the requested sampling behavior."""
        return self.params_allow_native_greedy(
            self.sampling_params_list, self.output_lengths
        )

    @staticmethod
    def params_allow_native_greedy(
        sampling_params_list: Sequence[SamplingParams],
        output_lengths: Sequence[int] | None = None,
        *,
        allow_logprobs: bool = False,
    ) -> bool:
        """Whether MLX argmax matches *sampling_params_list* exactly.

        Single source of truth for the native-greedy decision so callers that
        gate before constructing a :class:`SamplingBatch` (e.g. the decode
        pipeline) cannot drift from the sampling-time check.
        ``output_lengths`` is each row's generated-token count; see
        :meth:`_min_tokens_pending`.
        ``allow_logprobs`` is for the synchronous path, which carries scores
        alongside tokens. Deferred callers leave it false.
        """
        return not SamplingBatch._min_tokens_pending(
            sampling_params_list, output_lengths
        ) and all(
            sampling_params.temperature < GREEDY_TEMPERATURE_EPS
            and sampling_params.top_k <= 0
            and sampling_params.top_p == 1.0
            and sampling_params.frequency_penalty == 0.0
            and sampling_params.presence_penalty == 0.0
            and sampling_params.repetition_penalty == 1.0
            and (
                sampling_params.num_logprobs is None
                or (allow_logprobs and sampling_params.logprobs != -1)
            )
            and not sampling_params.allowed_token_ids
            and not sampling_params.bad_words_token_ids
            and not sampling_params.logit_bias
            for sampling_params in sampling_params_list
        )

    @staticmethod
    def params_allow_native_random(
        sampling_params_list: Sequence[SamplingParams],
        output_lengths: Sequence[int] | None = None,
        *,
        allow_seed: bool = False,
        allow_logprobs: bool = False,
    ) -> bool:
        """Whether the native MLX sampler supports *sampling_params_list*.

        Requests must use plain temperature/top-k/top-p/min-p sampling; the
        mask is built per row, so rows may mix greedy (temperature 0) with
        random sampling and differ in top-k/top-p/min-p. Requests with sample
        logprobs require ``allow_logprobs``; structured outputs and
        ``logit_bias`` stay on the torch path.
        ``output_lengths`` is as in :meth:`params_allow_native_greedy`.
        Seeded requests are admitted only with ``allow_seed``: the synchronous
        sample path derives a per-request key from the seed and the row's
        output length (see :func:`native_sampling_key`), while the deferred decode
        pipeline keeps them out.
        """
        if not sampling_params_list:
            return False
        return not SamplingBatch._min_tokens_pending(
            sampling_params_list, output_lengths
        ) and all(
            (allow_seed or sp.seed is None)
            and sp.frequency_penalty == 0.0
            and sp.presence_penalty == 0.0
            and sp.repetition_penalty == 1.0
            and (sp.num_logprobs is None or (allow_logprobs and sp.logprobs != -1))
            and sp.structured_outputs is None
            and not sp.allowed_token_ids
            and not sp.bad_words_token_ids
            and not sp.logit_bias
            for sp in sampling_params_list
        )

    @staticmethod
    def _min_tokens_pending(
        sampling_params_list: Sequence[SamplingParams],
        output_lengths: Sequence[int] | None,
    ) -> bool:
        """Whether any row still needs vLLM's min-tokens processor.

        The processor masks stop tokens only while a row has generated fewer
        than ``min_tokens``, so rows past the minimum can sample natively
        again. ``None`` counts every row as having generated nothing.
        """
        if output_lengths is None:
            output_lengths = [0] * len(sampling_params_list)
        return any(
            output_length < sampling_params.min_tokens
            for sampling_params, output_length in zip(
                sampling_params_list, output_lengths, strict=True
            )
        )

    @classmethod
    def native_sample_tokens(
        cls,
        logits_2d: mx.array,
        sampling_params_list: Sequence[SamplingParams],
        *,
        vocab_size: int,
        next_native_rng_key: Callable[[], mx.array] | None,
        output_lengths: Sequence[int],
        allow_seed: bool = False,
    ) -> mx.array:
        """Build tokens only; deferred decode does not yet carry logprobs."""
        return cls._native_sample(
            logits_2d,
            sampling_params_list,
            vocab_size=vocab_size,
            next_native_rng_key=next_native_rng_key,
            output_lengths=output_lengths,
            allow_seed=allow_seed,
        ).token_ids

    @classmethod
    def _native_sample(
        cls,
        logits_2d: mx.array,
        sampling_params_list: Sequence[SamplingParams],
        *,
        vocab_size: int,
        next_native_rng_key: Callable[[], mx.array] | None,
        output_lengths: Sequence[int],
        allow_seed: bool = False,
        allow_logprobs: bool = False,
        return_processed_logits: bool = False,
    ) -> _NativeSample:
        """Build the lazy native token graph for an eligible batch.

        ``output_lengths`` holds the tokens already generated per row. With
        ``allow_seed`` each seeded row draws from :func:`native_sampling_key`
        instead of the shared step key; without it, seeded rows are rejected
        as a gate desync.

        Uses greedy argmax when the batch allows it. Otherwise the shared Metal
        sampler draws from the first ``vocab_size`` logits to exclude padded
        LM-head columns. The caller owns evaluation and RNG state.
        ``next_native_rng_key`` advances the engine counter only when an unseeded
        random row needs a draw.
        Raises on batches neither native path admits — the caller must not
        let those reach here.
        """
        if cls.params_allow_native_greedy(
            sampling_params_list, output_lengths, allow_logprobs=allow_logprobs
        ):
            return _NativeSample(
                mlx_greedy_tokens(logits_2d),
                logits_2d.astype(mx.float32) if return_processed_logits else None,
            )
        random_eligible = cls.params_allow_native_random(
            sampling_params_list,
            output_lengths,
            allow_seed=allow_seed,
            allow_logprobs=allow_logprobs,
        )
        if next_native_rng_key is not None and random_eligible:
            row_keys = None
            if any(sp.seed is not None for sp in sampling_params_list):
                row_keys = [
                    None if sp.seed is None else native_sampling_key(sp.seed, n)
                    for sp, n in zip(sampling_params_list, output_lengths, strict=True)
                ]
            needs_engine_key = any(
                sp.seed is None and sp.temperature >= GREEDY_TEMPERATURE_EPS
                for sp in sampling_params_list
            )
            return cls._native_random_sample(
                logits_2d[..., :vocab_size],
                sampling_params_list,
                next_native_rng_key() if needs_engine_key else None,
                row_keys=row_keys,
                return_processed_logits=return_processed_logits,
            )
        key_state = "live" if next_native_rng_key is not None else "absent"
        raise RuntimeError(
            "Deferred sampling requires a native-greedy or native-random "
            f"eligible batch (native sample key {key_state}, random-eligible="
            f"{random_eligible}) — gate desync."
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
        at ``-inf``, and the vocab ids at those positions.

        The whole vocabulary is sorted, not just the ``max(top_k)`` largest
        logits: top-k masks by value, so a row whose k-th largest logit is
        tied keeps every token sharing that value, and a candidate set cut to
        ``max(top_k)`` would drop the ones past it.
        """
        vocab_size = int(scaled_logits.shape[-1])
        effective_k = [k if 0 < k < vocab_size else vocab_size for k in top_k]
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
        key: mx.array | None,
        *,
        row_keys: Sequence[mx.array | None] | None = None,
    ) -> mx.array:
        """Build a token-only draw without retaining processed scores."""
        return cls._native_random_sample(
            logits_2d, sampling_params_list, key, row_keys=row_keys
        ).token_ids

    @classmethod
    def _native_random_sample(
        cls,
        logits_2d: mx.array,
        sampling_params_list: Sequence[SamplingParams],
        key: mx.array | None,
        *,
        row_keys: Sequence[mx.array | None] | None = None,
        return_processed_logits: bool = False,
    ) -> _NativeSample:
        """Lazy per-row temperature/top-k/top-p/min-p token ids.

        Rows with an entry in ``row_keys`` (seeded requests) draw over their
        own row with an MLX key derived from the seed and output position.
        Draws are reproducible for identical logits, sampling params, seed
        and position. End-to-end tokens also require a compatible execution
        environment and batching: batch shape can change logits numerics,
        and native and torch-fallback paths use different random streams.
        Other rows share one batched draw with ``key``. Its derivation is
        stateless, but the runner keeps a global counter, so unseeded draws
        depend on scheduling, sub-batch order and batch composition and are
        reproducible only for a fixed request schedule.
        ``key`` may be ``None`` if no unseeded row needs a random draw.
        Greedy rows use zero temperature and break ties by vocabulary ID.
        """
        from vllm_metal.metal import get_ops

        temperatures: list[float] = []
        top_k: list[int] = []
        top_p: list[float] = []
        min_p: list[float] = []
        for sp in sampling_params_list:
            if sp.temperature < GREEDY_TEMPERATURE_EPS:
                temperatures.append(0.0)
                top_k.append(0)
                top_p.append(1.0)
                min_p.append(0.0)
            else:
                temperatures.append(sp.temperature)
                top_k.append(sp.top_k)
                top_p.append(sp.top_p)
                min_p.append(sp.min_p)
        # Keep request-key ownership above the lazy primitive. Seeded rows use
        # their own key; unseeded rows retain their position in the batch draw.
        fallback_key = key if key is not None else mx.zeros((2,), dtype=mx.uint32)
        row_keys = row_keys if row_keys is not None else [None] * len(temperatures)
        keys = mx.stack([k if k is not None else fallback_key for k in row_keys])
        seeds = keys.view(mx.int64).reshape(-1)
        positions = mx.array(
            [0 if k is not None else i for i, k in enumerate(row_keys)], dtype=mx.int64
        )
        values = logits_2d.astype(mx.float32)
        token_ids = None
        if any(
            0 < k < logits_2d.shape[1] or p != 1.0 or m != 0.0
            for k, p, m in zip(top_k, top_p, min_p, strict=True)
        ):
            # Filters consume scaled logits. Preserve zero for the final greedy draw.
            scales = mx.array(
                [t if t != 0 else 1.0 for t in temperatures], dtype=mx.float32
            )
            values, token_ids = cls._sorted_candidate_logits(
                values / scales[:, None], top_k, top_p, min_p
            )
            temperatures = [1.0 if t != 0 else 0.0 for t in temperatures]
        tokens = get_ops().gumbel_sample(
            values,
            seeds,
            positions,
            mx.array(temperatures, dtype=mx.float32),
            token_ids,
        )
        processed = None
        if return_processed_logits:
            scales = mx.array([t if t != 0 else 1.0 for t in temperatures])
            processed = values / scales[:, None]
            if token_ids is not None:
                processed = mx.put_along_axis(
                    mx.full(processed.shape, -mx.inf), token_ids, processed, axis=-1
                )
        return _NativeSample(tokens, processed)

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

    def _split_logprob_requests(
        self,
    ) -> tuple[dict[int, list[int]], list[tuple[int, int]]]:
        """Split rows into specific-token requests and top-k requests.

        A row that sets ``logprob_token_ids`` counts only as specific-token,
        matching vLLM, which prefers the specific IDs over ``logprobs``.
        """
        token_ids_by_row: dict[int, list[int]] = {}
        for i, sampling_params in enumerate(self.sampling_params_list):
            if sampling_params.logprob_token_ids:
                token_ids_by_row[i] = list(sampling_params.logprob_token_ids)
        topk_requests = [
            (i, sampling_params.logprobs)
            for i, sampling_params in enumerate(self.sampling_params_list)
            if i not in token_ids_by_row and sampling_params.logprobs is not None
        ]
        return token_ids_by_row, topk_requests

    @property
    def mixes_topk_and_specific_logprobs(self) -> bool:
        """Whether rows asking for top-k logprobs share the batch with rows
        asking for specific token IDs.

        Such a batch needs :meth:`gather_mixed_logprobs` after sampling.
        """
        token_ids_by_row, topk_requests = self._split_logprob_requests()
        return bool(token_ids_by_row) and any(
            num_logprobs > 0 for _, num_logprobs in topk_requests
        )

    def _make_logprob_args(self) -> tuple[int | None, dict[int, list[int]] | None]:
        """Build mutually exclusive logprob arguments for vLLM's sampler.

        vLLM's compatibility sampler treats any ``logprob_token_ids`` mapping
        as a batch-wide override of ``max_num_logprobs``. A batch that
        :attr:`mixes_topk_and_specific_logprobs` asks for the full-vocab
        logprobs (``-1``) instead; :meth:`gather_mixed_logprobs` then picks
        each row's token IDs from them, so top-k rows rank the values
        ``--logprobs-mode`` selects.
        """
        max_num_logprobs = self.max_num_logprobs
        if self.mixes_topk_and_specific_logprobs:
            return -1, None
        token_ids_by_row, topk_requests = self._split_logprob_requests()
        if not token_ids_by_row:
            return max_num_logprobs, None
        for i, _ in topk_requests:
            token_ids_by_row[i] = []
        return None, token_ids_by_row

    def gather_mixed_logprobs(
        self,
        sampler: Sampler,
        output: SamplerOutput,
    ) -> LogprobsTensors:
        """Gather per-row logprobs for a batch that mixes both request types.

        ``output`` comes from sampling with ``max_num_logprobs=-1``, so its
        logprobs are the full-vocab tensor: raw or processed, per the
        sampler's logprobs mode.
        """
        if output.logprobs_tensors is None:
            raise RuntimeError(
                "Sampler returned no logprobs for a mixed logprobs batch."
            )
        logprobs = output.logprobs_tensors.logprobs
        sampled = output.sampled_token_ids[:, 0]
        token_ids_by_row, topk_requests = self._split_logprob_requests()
        max_topk = max(num_logprobs for _, num_logprobs in topk_requests)
        topk_token_ids = torch.topk(logprobs, max_topk, dim=-1).indices
        for i, num_logprobs in topk_requests:
            token_ids_by_row[i] = topk_token_ids[i, :num_logprobs].tolist()
        tensors = sampler.gather_specific_token_logprobs(
            logprobs, token_ids_by_row, sampled.long()
        )
        if tensors is None:
            raise RuntimeError(
                "Sampler gathered no logprobs for a mixed logprobs batch."
            )
        return tensors

    def _make_logitsprocs(self) -> LogitsProcessors:
        processors: list[LogitsProcessor] = []
        min_p_vals = [sp.min_p for sp in self.sampling_params_list]
        if any(min_p > 0.0 for min_p in min_p_vals):
            processors.append(
                BatchMinPLogitsProcessor(
                    torch.tensor(
                        min_p_vals,
                        dtype=torch.float32,
                        device=self.SAMPLER_DEVICE,
                    )
                )
            )
        upstream_processors = [
            processor_cls
            for processor_cls, needed in (
                (
                    LogitBiasLogitsProcessor,
                    any(sp.logit_bias for sp in self.sampling_params_list),
                ),
                (
                    MinTokensLogitsProcessor,
                    self._min_tokens_pending(
                        self.sampling_params_list, self.output_lengths
                    ),
                ),
            )
            if needed
        ]
        if upstream_processors:
            # These vLLM processors track rows of a persistent batch; adding
            # every row rebuilds that state for this step's batch.
            batch_update = BatchUpdate(
                batch_size=len(self.sampling_params_list),
                removed=(),
                added=list(
                    zip(
                        range(len(self.sampling_params_list)),
                        self.sampling_params_list,
                        self.prompt_token_id_lists,
                        self.output_token_id_lists,
                        strict=True,
                    )
                ),
                moved=(),
            )
            for processor_cls in upstream_processors:
                processor = processor_cls(None, self.SAMPLER_DEVICE, False)
                processor.update_state(batch_update)
                processors.append(processor)
        if not processors:
            return _EMPTY_LOGITSPROCS
        return LogitsProcessors(processors)

    def make_sampling_metadata(self) -> SamplingMetadata:
        """Create vLLM ``SamplingMetadata`` for this batch."""
        (
            frequency_penalties,
            presence_penalties,
            repetition_penalties,
        ) = self._make_penalty_tensors()
        max_num_logprobs, logprob_token_ids = self._make_logprob_args()

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


def native_sampling_key(
    seed: int, step: int, *, domain: Literal["seed", "engine"] = "seed"
) -> mx.array:
    """Stateless MLX key from a seed and zero-based sampling position.

    Seeded rows use the request seed and output length; unseeded rows share
    the engine seed and runner's native sampling counter in the "engine"
    domain. The default "seed" domain preserves seeded request keys.
    """
    digest = hashlib.blake2b(
        f"vllm-metal-{domain}:{int(seed)}:{int(step)}".encode(),
        digest_size=8,
    ).digest()
    return mx.random.key(int.from_bytes(digest, "little"))


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
    *,
    next_native_rng_key: Callable[[], mx.array] | None = None,
) -> _SamplingResult:
    """Sample tokens from pre-sliced 2D logits ``(batch_size, vocab)``.

    Single entry point for all sampling paths. Chooses native MLX greedy
    when possible, or eligible native random sampling with a live ``next_native_rng_key``.
    Otherwise bridges to the vLLM torch sampler on
    ``SamplingBatch.SAMPLER_DEVICE``. Eligible sample logprobs are gathered on
    MLX; only the compact result is materialized for ``ModelRunnerOutput``.

    The bridged tensor aliases ``logits_2d``, so a penalized call rewrites the
    caller's array; callers pass logits they do not read again.
    """
    native_greedy = SamplingBatch.params_allow_native_greedy(
        batch.sampling_params_list, batch.output_lengths, allow_logprobs=True
    )
    native_random = (
        next_native_rng_key is not None
        and batch.generators_are_seeded()
        and SamplingBatch.params_allow_native_random(
            batch.sampling_params_list,
            batch.output_lengths,
            allow_seed=True,
            allow_logprobs=True,
        )
    )
    if native_greedy or native_random:
        processed_mode = batch.needs_logprobs and sampler.logprobs_mode in (
            "processed_logits",
            "processed_logprobs",
        )
        result = SamplingBatch._native_sample(
            logits_2d[..., : batch.vocab_size],
            batch.sampling_params_list,
            vocab_size=batch.vocab_size,
            next_native_rng_key=next_native_rng_key,
            output_lengths=batch.output_lengths,
            allow_seed=True,
            allow_logprobs=True,
            return_processed_logits=batch.needs_logprobs and processed_mode,
        )
        logprobs = None
        if batch.needs_logprobs:
            scoring_logits = (
                result.processed_logits
                if processed_mode
                else logits_2d[..., : batch.vocab_size]
            )
            assert scoring_logits is not None
            specific_ids, _ = batch._split_logprob_requests()
            gathered = gather_native_logprobs(
                scoring_logits,
                result.token_ids,
                [sp.logprobs or 0 for sp in batch.sampling_params_list],
                logits_mode=sampler.logprobs_mode in ("raw_logits", "processed_logits"),
                requested_token_ids=specific_ids,
            )
            logprobs = gathered.tolists()
        mx.eval(result.token_ids)
        return _SamplingResult(result.token_ids.reshape(-1).tolist(), logprobs)

    mx.eval(logits_2d)
    logits_torch = mlx_to_torch(
        logits_2d.astype(mx.float32), device=SamplingBatch.SAMPLER_DEVICE
    )
    metadata = batch.make_sampling_metadata()
    output = sampler.forward(logits_torch, metadata)
    logprobs_tensors = (
        batch.gather_mixed_logprobs(sampler, output)
        if batch.mixes_topk_and_specific_logprobs
        else output.logprobs_tensors
    )
    logprobs = logprobs_tensors.tolists() if logprobs_tensors is not None else None
    return _SamplingResult(output.sampled_token_ids[:, 0].tolist(), logprobs)


def sample_decode_tokens(
    logits: mx.array,
    decode_reqs: list[tuple[str, object]],
    num_decode: int,
    sampler: Sampler,
    *,
    vocab_size: int,
    next_native_rng_key: Callable[[], mx.array] | None = None,
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
    return sample_from_logits(
        decode_logits, batch, sampler, next_native_rng_key=next_native_rng_key
    )


def sample_prefill_tokens(
    logits: mx.array,
    prefill_reqs: list,
    cu_seqlens: list[int],
    num_decode: int,
    sampler: Sampler,
    *,
    vocab_size: int,
    next_native_rng_key: Callable[[], mx.array] | None = None,
) -> _SamplingResult:
    """Sample from the last logit position of each completed prefill.

    Args:
        logits: Full logits array, shape ``(1, total_tokens, vocab)``.
        prefill_reqs: List of ``PrefillRequest`` objects.
        cu_seqlens: Cumulative sequence lengths for logit position lookup.
        num_decode: Number of decode requests (offset into cu_seqlens).
        sampler: vLLM Sampler instance.
        vocab_size: Model vocabulary size.
    Returns:
        Sampled token IDs and optional logprobs, aligned to ``prefill_reqs``.
        Intermediate chunks have an unused zero token placeholder; they must
        not consume RNG state or emit sample logprobs.
    """
    final_indices = [
        i for i, pr in enumerate(prefill_reqs) if pr.prompt_len is not None
    ]
    if not final_indices:
        return _SamplingResult([0] * len(prefill_reqs))
    final_reqs = [prefill_reqs[i] for i in final_indices]

    prompt_token_id_lists: list[list[int]] = []
    output_token_id_lists: list[list[int]] = []
    for pr in final_reqs:
        token_ids = (
            pr.full_prompt_token_ids
            if pr.full_prompt_token_ids is not None
            else pr.token_ids
        )
        prompt_len = pr.prompt_len if pr.prompt_len is not None else len(token_ids)
        prompt_token_id_lists.append(token_ids[:prompt_len])
        output_token_id_lists.append(token_ids[prompt_len:])

    last_logits = mx.stack(
        [logits[0, cu_seqlens[num_decode + j + 1] - 1, :] for j in final_indices]
    )
    batch = SamplingBatch(
        [pr.sampling_params for pr in final_reqs],
        prompt_token_id_lists,
        output_token_id_lists,
        vocab_size=vocab_size,
        generators={
            j: pr.generator
            for j, pr in enumerate(final_reqs)
            if pr.generator is not None
        },
    )
    result = sample_from_logits(
        last_logits, batch, sampler, next_native_rng_key=next_native_rng_key
    )
    if len(final_reqs) == len(prefill_reqs):
        return result

    # Restore the packed prefill order consumed by postprocessing and the
    # drafter. Only completed rows own actual sampled tokens and logprobs.
    token_ids = [0] * len(prefill_reqs)
    logprobs_rows: list[LogprobsLists | None] = [None] * len(prefill_reqs)
    for sampled_index, prefill_index in enumerate(final_indices):
        token_ids[prefill_index] = result.token_ids[sampled_index]
        if result.logprobs is not None:
            logprobs_rows[prefill_index] = result.logprobs.slice_request(
                sampled_index, 1
            )
    return _SamplingResult(token_ids, SamplingBatch.merge_logprobs_rows(logprobs_rows))
