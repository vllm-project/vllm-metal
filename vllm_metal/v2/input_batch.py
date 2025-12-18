# SPDX-License-Identifier: Apache-2.0
"""Metal-compatible input batch preparation functions.

These functions replace Triton kernels with PyTorch/MLX implementations
for the Metal backend.
"""

import torch


def prepare_prefill_inputs(
    num_scheduled_tokens_per_request: torch.Tensor,
    token_indices: torch.Tensor,
    position_ids: torch.Tensor,
    slot_mapping: torch.Tensor,
    batch_indices: torch.Tensor,
    prefill_positions_cpu: torch.Tensor,
    input_token_ids: torch.Tensor,
    input_positions: torch.Tensor,
    input_slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
) -> int:
    """Prepare inputs for prefill phase using PyTorch operations.

    This replaces the Triton kernel version with pure PyTorch.

    Args:
        num_scheduled_tokens_per_request: Number of tokens per request.
        token_indices: Token indices for each position.
        position_ids: Position IDs for each token.
        slot_mapping: Slot mapping for KV cache.
        batch_indices: Batch index for each token.
        prefill_positions_cpu: CPU tensor for position computation.
        input_token_ids: Output tensor for token IDs.
        input_positions: Output tensor for positions.
        input_slot_mapping: Output tensor for slot mapping.
        query_start_loc: Output tensor for query start locations.
        seq_lens: Output tensor for sequence lengths.

    Returns:
        Total number of tokens prepared.
    """
    num_requests = num_scheduled_tokens_per_request.shape[0]
    total_tokens = 0

    # Compute cumulative positions for query_start_loc
    query_start_loc[0] = 0
    for i in range(num_requests):
        num_tokens = int(num_scheduled_tokens_per_request[i])
        query_start_loc[i + 1] = query_start_loc[i] + num_tokens
        seq_lens[i] = num_tokens

        # Copy data for this request
        if num_tokens > 0:
            start = total_tokens
            end = start + num_tokens

            # Get indices for this request
            request_token_indices = token_indices[start:end]
            request_positions = position_ids[start:end]
            request_slots = slot_mapping[start:end]

            # Copy to output tensors
            input_token_ids[start:end] = request_token_indices
            input_positions[start:end] = request_positions
            input_slot_mapping[start:end] = request_slots

            total_tokens += num_tokens

    return total_tokens


def prepare_pos_seq_lens(
    num_scheduled_tokens_per_request: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    positions_cpu: torch.Tensor,
    input_positions: torch.Tensor,
    seq_lens: torch.Tensor,
) -> int:
    """Prepare position and sequence length tensors using PyTorch.

    Args:
        num_scheduled_tokens_per_request: Tokens per request.
        seq_lens_tensor: Input sequence lengths.
        positions_cpu: CPU positions tensor.
        input_positions: Output positions tensor.
        seq_lens: Output sequence lengths tensor.

    Returns:
        Total number of tokens.
    """
    num_requests = num_scheduled_tokens_per_request.shape[0]
    total_tokens = 0

    for i in range(num_requests):
        num_tokens = int(num_scheduled_tokens_per_request[i])
        seq_len = int(seq_lens_tensor[i])
        seq_lens[i] = seq_len

        if num_tokens > 0:
            # Generate positions for this sequence
            start_pos = seq_len - num_tokens
            positions = torch.arange(
                start_pos,
                seq_len,
                dtype=input_positions.dtype,
                device=input_positions.device,
            )
            input_positions[total_tokens : total_tokens + num_tokens] = positions
            total_tokens += num_tokens

    return total_tokens


def combine_sampled_and_draft_tokens(
    sampled_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_accepted: torch.Tensor,
    output_token_ids: torch.Tensor,
    output_num_tokens: torch.Tensor,
    bonus_token_ids: torch.Tensor | None = None,
) -> int:
    """Combine sampled and draft tokens using PyTorch.

    This is used for speculative decoding to combine the accepted
    draft tokens with the newly sampled tokens.

    Args:
        sampled_token_ids: Newly sampled token IDs.
        draft_token_ids: Draft token IDs to verify.
        num_accepted: Number of accepted draft tokens per sequence.
        output_token_ids: Output tensor for combined tokens.
        output_num_tokens: Output tensor for token counts.
        bonus_token_ids: Optional bonus tokens.

    Returns:
        Total number of output tokens.
    """
    batch_size = sampled_token_ids.shape[0]
    total_tokens = 0

    for i in range(batch_size):
        n_accepted = int(num_accepted[i])

        # Copy accepted draft tokens
        if n_accepted > 0:
            output_token_ids[i, :n_accepted] = draft_token_ids[i, :n_accepted]

        # Add sampled token after accepted drafts
        output_token_ids[i, n_accepted] = sampled_token_ids[i, 0]

        # Count total tokens for this sequence
        output_num_tokens[i] = n_accepted + 1
        total_tokens += n_accepted + 1

        # Add bonus token if provided
        if bonus_token_ids is not None:
            output_token_ids[i, n_accepted + 1] = bonus_token_ids[i]
            output_num_tokens[i] += 1
            total_tokens += 1

    return total_tokens


def post_update(
    req_ids: list,
    num_tokens_per_request: torch.Tensor,
    finished: torch.Tensor,
    output_token_ids: torch.Tensor,
    output_num_tokens: torch.Tensor,
) -> dict:
    """Post-process batch update using PyTorch.

    Args:
        req_ids: Request IDs.
        num_tokens_per_request: Tokens per request.
        finished: Whether each request is finished.
        output_token_ids: Generated token IDs.
        output_num_tokens: Number of tokens per request.

    Returns:
        Dictionary mapping request ID to token list.
    """
    results = {}

    for i, req_id in enumerate(req_ids):
        n_tokens = int(output_num_tokens[i])
        tokens = output_token_ids[i, :n_tokens].tolist()
        results[req_id] = tokens

    return results
