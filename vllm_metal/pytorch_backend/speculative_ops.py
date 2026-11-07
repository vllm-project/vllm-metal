# SPDX-License-Identifier: Apache-2.0
"""MPS tensor operations for upstream MRv2's greedy speculative decoding.

The model, auxiliary-state extraction, proposal loop, verifier bookkeeping and
cache management remain upstream. These replace its Triton-only device helpers.
"""

import torch

from vllm_metal.pytorch_backend.input_ops import token_rows


def prepare_dflash_inputs(
    buffers,
    query_slots,
    context_positions,
    context_slots,
    sample_indices,
    sample_pos,
    sample_mapping,
    temperature,
    seeds,
    batch,
    sampled,
    rejected,
    last,
    next_prefill,
    input_temperature,
    input_seeds,
    block_table,
    block_size,
    cp_rank,
    cp_size,
    cp_interleave,
    mask_token,
    query_width,
    num_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    sample_from_anchor=False,
    num_reqs_override=None,
):
    n = batch.num_reqs if num_reqs_override is None else num_reqs_override
    total = int(batch.num_scheduled_tokens[:n].sum())
    cu = batch.query_start_loc[: n + 1]
    ends = cu[1:] - rejected[:n]
    offsets, rows = token_rows(cu, total)

    def slots(positions, request_rows):
        blocks = block_table[
            request_rows, (positions // block_size).clamp_max(block_table.shape[1] - 1)
        ].long()
        return torch.where(
            blocks != 0, blocks * block_size + positions % block_size, -1
        )

    valid = offsets < ends[rows]
    positions = torch.where(valid, batch.positions[:total], 0)
    context_positions[:total].copy_(positions)
    context_slots[:total] = torch.where(valid, slots(positions, rows), -1)

    request_slots = batch.idx_mapping[:n].long()
    anchor = torch.where(
        sampled[:n] > 0,
        last.reshape(-1)[request_slots],
        next_prefill.reshape(-1)[request_slots],
    )
    columns = torch.arange(query_width, device=cu.device)
    query_positions = batch.positions[(ends - 1).long(), None] + 1 + columns
    count = n * query_width
    buffers.input_ids[:count] = torch.where(
        columns == 0, anchor[:, None], mask_token
    ).flatten()
    buffers.positions[:count].copy_(
        query_positions.clamp_max(max_model_len - 1).flatten()
    )
    buffers.input_ids[count:].zero_()
    buffers.positions[count:].zero_()
    buffers.is_padding[:count].zero_()
    buffers.is_padding[count:].fill_(True)
    query_slots[:count] = slots(
        query_positions, torch.arange(n, device=cu.device)[:, None]
    ).flatten()
    query_slots[count:].fill_(-1)
    buffers.query_start_loc.copy_(
        torch.arange(max_num_reqs + 1, device=cu.device).clamp_max(n) * query_width
    )
    buffers.seq_lens[:n] = (query_positions[:, -1] + 1).clamp_max(max_model_len)
    buffers.seq_lens[n:].zero_()

    first = 0 if sample_from_anchor else 1
    num_sample = n * num_steps
    indices = torch.arange(count, device=cu.device).view(n, query_width)
    sample_indices[:num_sample].copy_(indices[:, first:].flatten())
    sample_pos[:num_sample].copy_(
        (query_positions[:, first:] + sample_from_anchor).flatten()
    )
    sample_mapping[:num_sample].copy_(
        request_slots[:, None].expand(n, num_steps).flatten()
    )
    sample_indices[num_sample:].zero_()
    sample_pos[num_sample:].zero_()
    sample_mapping[num_sample:].fill_(-1)
    temperature[request_slots] = input_temperature[request_slots]
    seeds[request_slots] = input_seeds[request_slots]


def prepare_prefill_inputs(
    last_indices, step, buffers, batch, sampled, rejected, last, lookahead, max_num_reqs
):
    n, total = batch.num_reqs, batch.num_tokens
    cu = batch.query_start_loc
    offsets, rows = token_rows(cu, total)
    slots = batch.idx_mapping.long()
    ends = cu[1:] - rejected - 1
    next_token = torch.where(sampled > 0, last[slots, 0], lookahead[0, slots])
    shifted = batch.input_ids[(offsets + 1).clamp_max(total - 1)]
    buffers.input_ids[:total] = torch.where(
        offsets == ends[rows], next_token[rows], shifted
    )
    buffers.positions[:total].copy_(batch.positions[:total])
    buffers.query_start_loc[: n + 1].copy_(cu)
    buffers.query_start_loc[n + 1 :].fill_(total)
    buffers.seq_lens[:n].copy_(batch.seq_lens[:n])
    buffers.seq_lens[n:].zero_()
    last_indices[:n].copy_(ends)
    last_indices[n:].zero_()
    step.zero_()
    return last_indices


def prepare_decode_inputs(
    tokens,
    target_lens,
    rejected,
    buffers,
    sample_positions,
    max_model_len,
    max_num_reqs,
    advance_draft_positions=True,
):
    n = tokens.numel()
    buffers.input_ids[:n].copy_(tokens)
    sample_positions[:n].add_(1)
    lengths = target_lens[:n] - rejected
    if advance_draft_positions:
        buffers.positions[:n].add_(1).clamp_max_(max_model_len - 1)
        lengths = (lengths + 1).clamp_max(max_model_len)
    buffers.seq_lens[:n].copy_(lengths)
    buffers.seq_lens[n:].zero_()
    buffers.query_start_loc.copy_(
        torch.arange(
            max_num_reqs + 1, device=tokens.device, dtype=torch.int32
        ).clamp_max(n)
    )


def update_draft_inputs(
    tokens,
    step,
    hidden,
    output,
    next_hidden,
    buffers,
    sample_positions,
    n,
    max_model_len,
    num_steps,
    advance_draft_positions=True,
):
    output[:n].scatter_(1, step.expand(n, 1), tokens[:, None])
    more = step < num_steps - 1
    buffers.input_ids[:n] = torch.where(more, tokens, buffers.input_ids[:n])
    next_hidden[:n] = torch.where(more, hidden[:n], next_hidden[:n])
    sample_positions[:n].add_(more.to(sample_positions.dtype))
    if advance_draft_positions:
        buffers.positions[:n].add_(more).clamp_max_(max_model_len - 1)
        buffers.seq_lens[:n].add_(more).clamp_max_(max_model_len)


def rejection_sample(
    logits,
    draft_logits,
    draft_tokens,
    cu,
    pos,
    mapping,
    expanded_mapping,
    local_pos,
    temperature,
    seed,
    num_steps,
    synthetic_conditional_rates=None,
    **kwargs,
):
    # Greedy verification accepts only the consecutive matching draft prefix,
    # then emits the target's correction (or its bonus token if all matched).
    cols = torch.arange(num_steps + 1, device=logits.device)
    rows = (cu[:-1, None] + cols).long().clamp_max(logits.shape[0] - 1)
    target = logits.argmax(-1)[rows]
    proposed = draft_tokens[(rows[:, :-1] + 1).clamp_max(logits.shape[0] - 1)]
    matches = (target[:, :-1] == proposed) & (cols[:-1] < cu.diff()[:, None] - 1)
    count = matches.to(torch.int32).cumprod(1).sum(1).to(torch.int32) + 1
    return torch.where(cols < count[:, None], target, -1), count


def gather_draft_sampled(
    output_tokens,
    output_positions,
    input_ids,
    positions,
    logits_indices,
    request_slots,
    local_positions,
    prefill_lengths,
    num_logits,
    BLOCK_SIZE,  # noqa: N803 - upstream kernel launch keyword
):
    indices = logits_indices[:num_logits].long()
    slots = request_slots[:num_logits].long()
    local = local_positions[:num_logits]
    pos = positions[indices]
    # Draft slots before the first generated token contain scheduler padding.
    placeholder = (local > 0) & (pos - local < prefill_lengths[slots])
    output_tokens[:num_logits].copy_(input_ids[indices].masked_fill(placeholder, -1))
    output_positions[:num_logits].copy_(pos)


def flatten_sampled(
    output, sampled, sampled_stride, num_sampled, cu_num_logits, num_warps=1
):
    offsets, rows = token_rows(cu_num_logits, output.numel())
    columns = (offsets - cu_num_logits[rows]).long()
    values = sampled[rows, columns.clamp_max(sampled.shape[1] - 1)]
    output.copy_(torch.where(columns < num_sampled[rows], values, output))
