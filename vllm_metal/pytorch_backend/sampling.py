# SPDX-License-Identifier: Apache-2.0
"""MPS implementations of MRv2 sampling kernels; policy stays in vLLM."""

# vLLM v0.32.0 sampler policy:
# https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/sample/sampler.py#L237-L356
# Kernel launch keywords retain the upstream Triton signatures.
# Temperature and Gumbel ABIs: https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/sample/gumbel.py
# Min-p ABI: https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/sample/min_p.py
# ruff: noqa: N803

from functools import cache

import torch


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
    argmax,
    argmax_stride,
    maximum,
    maximum_stride,
    logits_cache,
    cache_stride,
    cache_col_stride,
    cache_col,
    cache_source,
    cache_source_stride,
    logits,
    logits_stride,
    mapping,
    seeds,
    positions,
    temperatures,
    vocab_size,
    BLOCK_SIZE,
    IS_DRAFTING,
    APPLY_TEMPERATURE,
    USE_FP64,
    PER_TOKEN_COL,
):
    if USE_FP64:
        raise NotImplementedError("MPS does not support FP64 sampling")
    if IS_DRAFTING or logits_cache is not None:
        raise NotImplementedError("MPS does not support probabilistic draft sampling")
    rows = logits.shape[0]
    # Cast a broadcast source row once, even when there are many draws.
    if logits_stride == 0:
        logits = logits[:1].float().expand_as(logits)
    else:
        logits = logits.float()
    _, kernel, unused_ids = _gumbel_kernel()
    kernel(
        logits,
        mapping.long().contiguous(),
        seeds,
        positions.long().contiguous(),
        temperatures,
        argmax,
        maximum,
        (vocab_size, *logits.stride()),
        BLOCK_SIZE,
        argmax.shape[1],
        int(APPLY_TEMPERATURE),
        unused_ids,
        0,
        threads=(256 * argmax.shape[1], rows, 1),
        group_size=(256, 1, 1),
    )
