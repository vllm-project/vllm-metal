# SPDX-License-Identifier: Apache-2.0
"""MPS implementations of MRv2 sampling kernels; policy stays in vLLM."""

# vLLM v0.34.0 sampler policy:
# https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/worker/gpu/sample/sampler.py#L223-L357
# Kernel launch keywords retain the upstream Triton signatures.
# Temperature ABI: https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/worker/gpu/sample/gumbel.py
# Min-p ABI: https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/worker/gpu/sample/min_p.py
# Gumbel wrapper, rebound because its block combine needs Triton: https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/worker/gpu/sample/gumbel.py#L393-L489
# ruff: noqa: N803

from functools import cache

import torch
from vllm.utils.math_utils import cdiv

# Upstream's vocabulary block; the shader reduces one block per threadgroup.
_BLOCK_SIZE = 1024


@cache
def _gumbel_kernel():
    from vllm_metal import envs
    from vllm_metal.metal.build import prepare_metallib

    path = prepare_metallib(
        "sampling_kern", build_from_source=envs.VLLM_METAL_BUILD_FROM_SOURCE
    )
    library = torch.mps.load_metallib(path)
    # The shared shader accepts an optional candidate-to-vocabulary map.
    unused_ids = torch.empty(1, dtype=torch.int32, device="mps")
    return library, library.gumbel_sample, unused_ids


def temperature(logits, stride, mapping, values, vocab_size, BLOCK_SIZE):
    temp = values[mapping.long(), None]
    logits.div_(torch.where(temp == 0, 1, temp))


def min_p(logits, stride, mapping, values, vocab_size, BLOCK_SIZE):
    threshold = logits.amax(-1, keepdim=True) + values[mapping.long(), None].log()
    logits.masked_fill_(logits < threshold, -float("inf"))


def gumbel_sample(
    logits,
    expanded_idx_mapping,
    temperature,
    seed,
    pos,
    apply_temperature,
    is_drafting,
    use_fp64=False,
):
    if use_fp64:
        raise NotImplementedError("MPS does not support FP64 sampling")
    num_tokens, vocab_size = logits.shape
    num_blocks = cdiv(vocab_size, _BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float32)
    # Cast a broadcast source row once, even when there are many draws.
    if logits.stride(0) == 0:
        logits = logits[:1].float().expand_as(logits)
    else:
        logits = logits.float()
    _, kernel, unused_ids = _gumbel_kernel()
    kernel(
        logits,
        expanded_idx_mapping.long().contiguous(),
        seed,
        pos.long().contiguous(),
        temperature,
        local_argmax,
        local_max,
        (vocab_size, *logits.stride()),
        _BLOCK_SIZE,
        num_blocks,
        int(apply_temperature),
        unused_ids,
        0,
        threads=(256 * num_blocks, num_tokens, 1),
        group_size=(256, 1, 1),
    )
    # The first maximal block wins, so ties keep the lowest token ID as upstream.
    return local_argmax.gather(1, local_max.argmax(1, keepdim=True)).view(-1)
