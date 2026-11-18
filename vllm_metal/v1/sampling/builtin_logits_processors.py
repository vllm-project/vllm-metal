# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from vllm b46907cce0f7 (== v0.33.0), Apache-2.0:
# vllm/v1/sample/logits_processor/builtin.py,
# which vllm-project/vllm#63373 removes. Local modifications:
#   * only the pieces SamplingBatch drives: LogitBiasLogitsProcessor,
#     MinTokensLogitsProcessor and process_dict_updates; BatchUpdate and
#     MoveDirectionality live with the interface in
#     vllm_metal.v1.logits_processors.
#   * the processors drop the LogitsProcessor ABC base; they satisfy the
#     vllm_metal.v1.logits_processors.LogitsProcessor protocol structurally.
#   * MinTokensLogitsProcessor.apply_with_spec_decode is omitted (unused here).
#   * process_dict_updates uses a PEP 695 type parameter for the repo linter.
"""Built-in logit-bias and min-tokens processors for the vendored sampler."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch
from vllm import SamplingParams
from vllm.utils.torch_utils import async_tensor_h2d

from vllm_metal.v1.logits_processors import BatchUpdate, MoveDirectionality

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class LogitBiasLogitsProcessor:
    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.biases: dict[int, dict[int, float]] = {}

        self.bias_tensor: torch.Tensor = torch.tensor(())
        self.logits_slice = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )

    def is_argmax_invariant(self) -> bool:
        """Logit bias can rebalance token probabilities and change the
        outcome of argmax in greedy sampling."""
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        needs_update = process_dict_updates(
            self.biases, batch_update, lambda params, _, __: params.logit_bias or None
        )

        # Update tensors if needed.
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            biases: list[float] = []
            for req, lb in self.biases.items():
                reqs.extend([req] * len(lb))
                tok_ids.extend(lb.keys())
                biases.extend(lb.values())

            self.bias_tensor = self._device_tensor(biases, torch.float32)
            self.logits_slice = (
                self._device_tensor(reqs, torch.int32),
                self._device_tensor(tok_ids, torch.int32),
            )

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return async_tensor_h2d(data, device=self.device, dtype=dtype)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits


class MinTokensLogitsProcessor:
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        # index -> (min_toks, output_token_ids, stop_token_ids, uses_structured_output)
        self.device = device
        self.min_toks: dict[int, tuple[int, Sequence[int], set[int], bool]] = {}

        # (req_idx_tensor, stop_token_id_tensor)
        self.logits_slice: tuple[torch.Tensor, torch.Tensor] = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )

        # Structured-output stop-token logits that may need to be restored.
        self.restore_logits_slice: tuple[torch.Tensor, torch.Tensor] = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )

        self.neg_inf_tensor = torch.tensor(
            -float("inf"), dtype=torch.float32, device=self.device
        )

    def is_argmax_invariant(self) -> bool:
        """By censoring stop tokens, min-tokens can change the outcome
        of the argmax operation in greedy sampling."""
        return False

    @staticmethod
    def add_request(
        params: SamplingParams, _: list[int] | None, output_tok_ids: list[int]
    ) -> tuple[int, Sequence[int], set[int], bool] | None:
        min_tokens = params.min_tokens
        if not min_tokens or len(output_tok_ids) >= min_tokens:
            return None
        return (
            min_tokens,
            output_tok_ids,
            params.all_stop_token_ids,
            params.structured_outputs is not None,
        )

    def update_state(self, batch_update: BatchUpdate | None):
        needs_update = process_dict_updates(
            self.min_toks, batch_update, self.add_request
        )
        if self.min_toks:
            # Check for any requests that have attained their min tokens.
            to_remove = tuple(
                index
                for index, (min_toks, out_tok_ids, _, _) in self.min_toks.items()
                if len(out_tok_ids) >= min_toks
            )
            if to_remove:
                needs_update = True
                for index in to_remove:
                    del self.min_toks[index]

        # Update tensors if needed.
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            restore_reqs: list[int] = []
            restore_tok_ids: list[int] = []
            for req, (
                _,
                _,
                stop_tok_ids,
                uses_structured_output,
            ) in self.min_toks.items():
                reqs.extend([req] * len(stop_tok_ids))
                tok_ids.extend(stop_tok_ids)
                if uses_structured_output:
                    restore_reqs.extend([req] * len(stop_tok_ids))
                    restore_tok_ids.extend(stop_tok_ids)

            self.logits_slice = (
                self._device_tensor(reqs, torch.int32),
                self._device_tensor(tok_ids, torch.int32),
            )
            self.restore_logits_slice = (
                self._device_tensor(restore_reqs, torch.int32),
                self._device_tensor(restore_tok_ids, torch.int32),
            )

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return async_tensor_h2d(data, device=self.device, dtype=dtype)

    def _mask_stop_token_logits(
        self,
        logits: torch.Tensor,
        logits_slice: tuple[torch.Tensor, torch.Tensor],
        restore_logits_slice: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        restore_rows, restore_toks = restore_logits_slice
        # Advanced indexing already returns a copy, so this survives the
        # in-place masking below.
        stop_logits = logits[restore_logits_slice]
        logits.index_put_(logits_slice, self.neg_inf_tensor)

        # Stop-token entries for the same row are emitted together, so this
        # checks each row once without sorting the index tensor.
        unique_rows, inverse_indices = torch.unique_consecutive(
            restore_rows, return_inverse=True
        )
        row_needs_restore = torch.isneginf(logits[unique_rows]).all(dim=-1)
        restore_mask = row_needs_restore[inverse_indices] & torch.isfinite(stop_logits)
        restore_slice = (
            restore_rows[restore_mask],
            restore_toks[restore_mask],
        )
        logits.index_put_(restore_slice, stop_logits[restore_mask])

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            # Inhibit stop tokens for requests which have not reached min length.
            if self.restore_logits_slice[0].numel() == 0:
                logits.index_put_(self.logits_slice, self.neg_inf_tensor)
            else:
                self._mask_stop_token_logits(
                    logits, self.logits_slice, self.restore_logits_slice
                )
        return logits


def process_dict_updates[T](
    req_entries: dict[int, T],
    batch_update: BatchUpdate | None,
    new_state: Callable[[SamplingParams, list[int] | None, list[int]], T | None],
) -> bool:
    """Utility function to update dict state for sparse LogitsProcessors."""
    if not batch_update:
        # Nothing to do.
        return False

    updated = False
    for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
        if (state := new_state(params, prompt_tok_ids, output_tok_ids)) is not None:
            req_entries[index] = state
            updated = True
        elif req_entries.pop(index, None) is not None:
            updated = True

    if req_entries:
        # Process removed requests.
        for index in batch_update.removed:
            if req_entries.pop(index, None):
                updated = True

        # Process moved requests, unidirectional (a->b) and
        # swapped (a<->b)
        for a_index, b_index, direct in batch_update.moved:
            a_entry = req_entries.pop(a_index, None)
            b_entry = req_entries.pop(b_index, None)
            if a_entry is not None:
                req_entries[b_index] = a_entry
                updated = True
            if b_entry is not None:
                updated = True
                if direct == MoveDirectionality.SWAP:
                    req_entries[a_index] = b_entry

    return updated
