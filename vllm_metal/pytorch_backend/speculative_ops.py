# SPDX-License-Identifier: Apache-2.0
"""MPS tensor operations for upstream MRv2's greedy EAGLE3 path.

The model, auxiliary-state extraction, proposal loop, verifier bookkeeping and
cache management remain upstream. These replace its Triton-only device helpers.
"""

import torch

from vllm_metal.pytorch_backend.input_ops import token_rows


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
