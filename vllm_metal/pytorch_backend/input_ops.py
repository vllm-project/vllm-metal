# SPDX-License-Identifier: Apache-2.0
"""PyTorch implementations of MRV2's input/state kernels for MPS."""

import torch

from vllm_metal.pytorch_backend.runtime import cpu_mirror


def token_rows(cu, size):
    offsets = torch.arange(size, device=cu.device, dtype=torch.int64)
    rows = torch.bucketize(offsets, cu[1:].contiguous(), right=True)
    return offsets, rows.clamp_max(cu.numel() - 2)


def prepare_prefill_inputs(ids, lookahead, mapping, cu, all_ids, prefill, computed):
    offsets, rows = token_rows(cu, ids.numel())
    slots = mapping[rows].long()
    positions = computed[slots].long() + offsets - cu[rows]
    values = all_ids[slots, positions.clamp_max(all_ids.shape[1] - 1)]
    active = (offsets < cu[-1]) & (computed[slots] < prefill[slots])
    ids.copy_(torch.where(active, values, ids))
    slots = mapping.long()
    next_pos = computed[slots].long() + cu[1:] - cu[:-1]
    for i in range(lookahead.shape[0]):
        p = next_pos + i
        lookahead[i, slots] = torch.where(
            p < prefill[slots], all_ids[slots, p.clamp_max(all_ids.shape[1] - 1)], 0
        )


def prepare_pos_seq_lens(mapping, cu, computed, pos, seq_lens):
    offsets, rows = token_rows(cu, pos.numel())
    pos.copy_(computed[mapping[rows].long()].long() + offsets - cu[rows])
    n = mapping.numel()
    seq_lens[:n] = computed[mapping.long()] + cu[1:] - cu[:-1]
    seq_lens[n:].zero_()


def combine_sampled_and_draft_tokens(
    ids,
    mapping,
    last,
    cu,
    lengths,
    prefill,
    drafts,
    cu_logits,
    num_logits,
    num_new_sampled_tokens=1,
):
    if drafts.shape[1] or num_new_sampled_tokens != 1:
        raise NotImplementedError("MPS draft-token input kernels are not implemented")
    indices = (cu[1:] - 1).long()
    slots = mapping.long()
    ids[indices] = torch.where(
        lengths[: mapping.numel()] > prefill[slots], last[slots, 0], ids[indices]
    ).to(ids.dtype)
    return indices


def expand_idx_mapping(mapping, total, cu_logits, max_expand_len):
    offsets, rows = token_rows(cu_logits, total)
    return mapping[rows], (offsets - cu_logits[rows]).to(torch.int32)


def get_num_sampled_and_rejected(count, lengths, cu_logits, mapping, prefill):
    active = lengths[: mapping.numel()] >= prefill[mapping.long()]
    count = torch.where(active, count, 0)
    return count, torch.where(active, cu_logits[1:] - cu_logits[:-1] - count, 0)


def post_update(
    mapping, computed, last, bins, tokens, count, rejected, cu, all_ids, total
):
    if tokens.shape[1] != 1:
        raise NotImplementedError("MPS speculative writeback is not implemented")
    valid_rows = cpu_mirror(mapping).numpy() >= 0
    if not valid_rows.all():
        keep = torch.tensor(valid_rows.nonzero()[0], device=mapping.device)
        if keep.numel() == 0:
            return
        mapping, tokens = mapping[keep], tokens[keep]
        count, rejected = count[keep], rejected[keep]
        if cu is not None:
            lengths = (cu[1:] - cu[:-1])[keep]
            cu = torch.cat((cu.new_zeros(1), lengths.cumsum(0)))
    slots = mapping.long()
    valid = slots >= 0
    slots = slots.clamp_min(0)
    active = valid & (count > 0)
    old_total = total[slots].long()
    ids = tokens[:, 0]
    # A request has at most one row in the ordinary decode batch.
    last[slots, 0] = torch.where(active, ids, last[slots, 0])
    write_pos = old_total.clamp_max(all_ids.shape[1] - 1)
    all_ids[slots, write_pos] = torch.where(
        active, ids.to(all_ids.dtype), all_ids[slots, write_pos]
    )
    total[slots] = (old_total + torch.where(active, count, 0)).to(total.dtype)
    delta = -rejected if cu is None else cu[1:] - cu[:-1] - rejected
    computed[slots] += torch.where(valid, delta, 0).to(computed.dtype)
    if bins is not None:
        bins[slots, ids] += active.to(bins.dtype)


def post_update_num_computed_tokens(mapping, computed, cu):
    computed[mapping.long()] += (cu[1:] - cu[:-1]).to(computed.dtype)


def gather_block_tables(self, mapping, num_reqs_padded, out=None, out_ptrs=None):
    padded = num_reqs_padded
    out = self.input_block_tables if out is None else out
    n = mapping.numel()
    for src, dst in zip(self.block_tables, out, strict=True):
        dst[:n].copy_(src.gpu.index_select(0, mapping.long()))
        dst[n:padded].zero_()
    return tuple(t[:padded] for t in out)


def compute_slot_mappings(self, mapping, cu, positions, num_tokens_padded, out=None):
    padded = num_tokens_padded
    out = self.slot_mappings if out is None else out
    offsets, rows = token_rows(cu, positions.numel())
    slots = mapping[rows].long()
    for i, table in enumerate(self.block_tables):
        block_size = self.kernel_block_sizes[i]
        physical = table.gpu[slots, (positions // block_size).long()]
        values = physical.long() * block_size + positions % block_size
        out[i, : positions.numel()] = torch.where(offsets < cu[-1], values, -1)
        out[i, positions.numel() : padded].fill_(-1)
    return out[:, :padded]


def make_block_pointer_tensor(self, tensors):
    # Tensor operations use tensor references; the upstream descriptor remains
    # on the host because MPS does not support uint64 device tensors.
    return torch.tensor([tensor.data_ptr() for tensor in tensors], dtype=torch.uint64)


def install():
    from vllm.v1.worker.gpu import block_table, input_batch, model_runner
    from vllm.v1.worker.gpu.sample import sampler

    for name in (
        "prepare_prefill_inputs",
        "prepare_pos_seq_lens",
        "combine_sampled_and_draft_tokens",
        "expand_idx_mapping",
        "post_update",
        "post_update_num_computed_tokens",
    ):
        setattr(input_batch, name, globals()[name])
        setattr(model_runner, name, globals()[name])
    input_batch.get_num_sampled_and_rejected = get_num_sampled_and_rejected
    sampler.get_num_sampled_and_rejected = get_num_sampled_and_rejected
    block_table.BlockTables.gather_block_tables = gather_block_tables
    block_table.BlockTables.compute_slot_mappings = compute_slot_mappings
    block_table.BlockTables._make_ptr_tensor = make_block_pointer_tensor
