# SPDX-License-Identifier: Apache-2.0
"""MPS repetition penalties for greedy Omni generation."""

import torch

from vllm_metal.pytorch_backend.runtime import cpu_mirror


def bincount(mapping, all_ids, prompt_len, prefill_len, prompt_mask, counts, max_len):
    slots = cpu_mirror(mapping).tolist()
    prompts = cpu_mirror(prompt_len)
    prefills = cpu_mirror(prefill_len)
    ids = all_ids[slots, :max_len].cpu().long()
    bits = 1 << torch.arange(32, dtype=torch.int64)
    for row, slot in enumerate(slots):
        prompt_end, end = int(prompts[slot]), int(prefills[slot])
        mask = torch.zeros(prompt_mask.shape[1] * 32, dtype=torch.int64)
        mask[ids[row, :prompt_end]] = 1
        prompt_mask[slot].copy_((mask.view(-1, 32) * bits).sum(-1).to(torch.int32))
        counts[slot].copy_(
            torch.bincount(ids[row, prompt_end:end], minlength=counts.shape[1])
        )


def apply_penalties(
    logits,
    mapping,
    token_ids,
    local_pos,
    repetition,
    frequency,
    presence,
    prompt_mask,
    counts,
):
    slots = mapping.long()
    vocab = torch.arange(logits.shape[-1], device=logits.device)
    seen_prompt = (
        (prompt_mask[slots[:, None], vocab[None, :] // 32] >> (vocab % 32)) & 1
    ).bool()
    output_counts = counts[slots]
    seen_output = output_counts > 0
    penalty = repetition[slots, None]
    penalized = torch.where(logits > 0, logits / penalty, logits * penalty)
    logits.copy_(torch.where(seen_prompt | seen_output, penalized, logits))


def install():
    from vllm.v1.worker.gpu.sample import penalties

    penalties.bincount = bincount
    penalties.apply_penalties = apply_penalties
