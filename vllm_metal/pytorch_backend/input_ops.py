# SPDX-License-Identifier: Apache-2.0
"""Temporary Torch replacements for vLLM 0.32 MRv2 input/state kernels on MPS.

Upstream launches Triton kernels without a Torch fallback at these entry points.
Replace the private bindings in bindings.install() as upstream kernel/component
hooks become usable
(vllm-project/vllm#43048 and #51212). Registration alone does not remove these
implementations; prefer shared device-neutral fallbacks when they are available.
"""

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
    if drafts.shape[1]:
        offsets, rows = token_rows(cu_logits, num_logits)
        local = offsets - cu_logits[rows]
        widths = cu_logits.diff()
        starts = cu[1:] - widths
        indices = (starts[rows] + local).long()
        slots = mapping[rows].long()
        first_pos = lengths[rows] - widths[rows]
        is_draft = local >= num_new_sampled_tokens
        draft_col = (local - num_new_sampled_tokens).clamp(0, drafts.shape[1] - 1)
        values = torch.where(is_draft, drafts[slots, draft_col], last[slots, 0])
        generated = (lengths[rows] > prefill[slots]) & (
            is_draft | (first_pos >= prefill[slots])
        )
        ids[indices] = torch.where(generated, values.to(ids.dtype), ids[indices])
        return indices
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
    ids = tokens.gather(1, (count.long() - 1).clamp_min(0)[:, None]).flatten()
    # Each request occupies one output row.
    last[slots, 0] = torch.where(active, ids, last[slots, 0])
    for i in range(tokens.shape[1]):
        write = valid & (count > i)
        # Only unused columns get a safe index; real writes must fit the row.
        write_pos = torch.where(write, old_total + i, 0)
        all_ids[slots, write_pos] = torch.where(
            write,
            tokens[:, i].to(all_ids.dtype),
            all_ids[slots, write_pos],
        )
        if bins is not None:
            bins[slots, tokens[:, i].clamp_min(0)] += write.to(bins.dtype)
    total[slots] = (old_total + torch.where(active, count, 0)).to(total.dtype)
    delta = -rejected if cu is None else cu[1:] - cu[:-1] - rejected
    computed[slots] += torch.where(valid, delta, 0).to(computed.dtype)


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


def apply_block_table_writes(self):
    for table in self.block_tables:
        table.apply_write()
    self.num_blocks.copy_to_uva()


def compute_slot_mappings(self, mapping, cu, positions, num_tokens_padded, out=None):
    padded = num_tokens_padded
    out = self.slot_mappings if out is None else out
    offsets, rows = token_rows(cu, positions.numel())
    slots = mapping[rows].long()
    active = (offsets < cu[-1]) & (slots >= 0)
    # Mask dummy and padded lanes before indexing, as the Triton load does.
    slots = torch.where(active, slots, 0)
    positions = torch.where(active, positions, 0)
    for i, table in enumerate(self.block_tables):
        if not self._slot_mapping_enabled[i]:
            out[i, :padded].fill_(-1)
            continue
        block_size = self.block_sizes[i]
        physical = table.gpu[slots, (positions // block_size).long()]
        values = physical.long() * block_size + positions % block_size
        out[i, : positions.numel()] = torch.where(active, values, -1)
        out[i, positions.numel() : padded].fill_(-1)
    return out[:, :padded]


def make_block_pointer_tensor(self, tensors):
    # Tensor operations use tensor references; the upstream descriptor remains
    # on the host because MPS does not support uint64 device tensors.
    return torch.tensor([tensor.data_ptr() for tensor in tensors], dtype=torch.uint64)


def preprocess_state(self, input_batch, block_tables, kv_cache_config, computed):
    """Copy scheduler-owned state on align boundaries; no speculative offsets."""
    if not self._align_mode or not input_batch.num_reqs:
        return
    group_ids, spec = self._get_mamba_group_info(kv_cache_config)
    slots = input_batch.idx_mapping.long()
    previous = self._mamba_state_idx_gpu[slots].long()
    end = computed[slots] + input_batch.query_start_loc.diff()
    current = ((end + spec.block_size - 1) // spec.block_size - 1).long()
    moved = (previous >= 0) & (previous != current)
    layers = self.vllm_config.compilation_config.static_forward_context
    for group_id in group_ids:
        table = block_tables[group_id]
        source = table.gather(1, previous.clamp_min(0)[:, None]).flatten().long()
        destination = table.gather(1, current[:, None]).flatten().long()
        for name in kv_cache_config.kv_cache_groups[group_id].layer_names:
            for state in layers[name].kv_cache:
                mask = moved.view(-1, *([1] * (state.ndim - 1)))
                state[destination] = torch.where(
                    mask, state[source], state[destination]
                )
    self._mamba_state_idx_gpu[slots] = current.to(torch.int32)


def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens=None):
    # Hybrid speculation is rejected at configuration time. Upstream starts
    # with the neutral acceptance count (1), including unsampled prefill steps.
    pass


def prepare_rope_positions(self, mapping, cu, prefill, computed):
    offsets, rows = token_rows(cu, self.positions.shape[1] - 1)
    slots = mapping[rows].long()
    original = computed[slots].long() + offsets - cu[rows]
    channels = (
        slots[None, :] * self.num_dims
        + torch.arange(self.num_dims, device=mapping.device)[:, None]
    )
    cached = self.prefill_positions.gpu[
        channels, original.clamp(0, self.max_model_len - 1)[None, :]
    ]
    positions = torch.where(
        (computed[slots] < prefill[slots])[None, :],
        cached,
        (original + self.prefill_delta.gpu[slots])[None, :],
    )
    self.positions[:, :-1].copy_(torch.where(offsets[None, :] < cu[-1], positions, 0))
