# SPDX-License-Identifier: Apache-2.0
"""Token sampling for paged attention decode and prefill.

Pure functions: logits in, token IDs out.  No model runner state accessed.
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import torch
from vllm.v1.sample.sampler import Sampler

from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
from vllm_metal.v1.sampling_batch import SamplingBatch


def _mlx_greedy_sample(logits: mx.array) -> mx.array:
    """Native MLX greedy sampling — avoids PyTorch round-trip."""
    return mx.argmax(logits, axis=-1)


def sample_decode_tokens(
    logits: mx.array,
    decode_reqs: list[tuple[str, object]],
    num_decode: int,
    sampler: Sampler,
    make_metadata: Callable,
    device: torch.device,
) -> list[int]:
    """Sample one token per decode request from evaluated logits.

    Args:
        logits: Full logits array, shape ``(1, total_tokens, vocab)``.
        decode_reqs: ``(req_id, RequestState)`` pairs for decode requests.
        num_decode: Number of decode requests (prefix of the token dimension).
        sampler: vLLM Sampler instance.
        make_metadata: Callable matching ``_make_sampling_metadata`` signature.
        device: PyTorch device for the torch bridge path.

    Returns:
        List of sampled token IDs, one per decode request.
    """
    if not decode_reqs:
        return []

    decode_logits = logits[0, :num_decode, :]  # (num_decode, vocab)

    sampling_params_list = [state.sampling_params for _, state in decode_reqs]
    if SamplingBatch.can_use_native_greedy(sampling_params_list):
        next_tokens_mlx = _mlx_greedy_sample(decode_logits)
        mx.eval(next_tokens_mlx)
        return list(next_tokens_mlx.tolist())

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
    logits_torch = mlx_to_torch(decode_logits.astype(mx.float32), device=device)
    metadata = make_metadata(
        sampling_params_list,
        prompt_token_ids_list,
        output_tokens_list,
        generators=generators,
    )
    output = sampler.forward(logits_torch, metadata)
    return [int(output.sampled_token_ids[i, 0].item()) for i in range(num_decode)]


def sample_prefill_tokens(
    logits: mx.array,
    prefill_reqs: list,
    cu_seqlens: list[int],
    num_decode: int,
    sampler: Sampler,
    make_metadata: Callable,
    device: torch.device,
) -> list[int]:
    """Sample one token per prefill request from the last logit position.

    Args:
        logits: Full logits array, shape ``(1, total_tokens, vocab)``.
        prefill_reqs: List of ``PrefillRequest`` objects.
        cu_seqlens: Cumulative sequence lengths for logit position lookup.
        num_decode: Number of decode requests (offset into cu_seqlens).
        sampler: vLLM Sampler instance.
        make_metadata: Callable matching ``_make_sampling_metadata`` signature.
        device: PyTorch device for the torch bridge path.

    Returns:
        List of sampled token IDs, one per prefill request.
    """
    prefill_next_tokens: list[int] = []
    for j, pr in enumerate(prefill_reqs):
        last_idx = cu_seqlens[num_decode + j + 1] - 1
        last_logits = logits[:, last_idx : last_idx + 1, :]

        if pr.full_prompt_token_ids is not None:
            prompt_len = len(pr.full_prompt_token_ids)
        elif pr.prompt_len is not None:
            prompt_len = pr.prompt_len
        else:
            prompt_len = len(pr.token_ids)

        if SamplingBatch.can_use_native_greedy([pr.sampling_params]):
            next_token_mlx = _mlx_greedy_sample(last_logits[0])
            mx.eval(next_token_mlx)
            next_token = int(next_token_mlx.item())
        else:
            mx.eval(last_logits)
            logits_torch = mlx_to_torch(
                last_logits[0].astype(mx.float32), device=device
            )
            generators = {} if pr.generator is None else {0: pr.generator}
            prompt_for_meta = (
                pr.full_prompt_token_ids
                if pr.full_prompt_token_ids is not None
                else pr.token_ids
            )
            metadata = make_metadata(
                [pr.sampling_params],
                [prompt_for_meta[:prompt_len]],
                [prompt_for_meta[prompt_len:]],
                generators=generators,
            )
            output = sampler.forward(logits_torch, metadata)
            next_token = int(output.sampled_token_ids[0, 0].item())

        prefill_next_tokens.append(next_token)

    return prefill_next_tokens
