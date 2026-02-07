# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal kernel paged attention — verifies output matches non-paged path.

Requires ``kernels`` package with ``kernels-community/paged-attention`` support.

Run with:
    python -m pytest tests/test_metal_kernel_paged.py -v -s
"""

from __future__ import annotations

import mlx.core as mx
import pytest
import torch
from mlx_lm import load as mlx_lm_load
from mlx_lm.models.cache import make_prompt_cache

from vllm_metal.metal_kernel_backend.cache import MPSPagedKVCache
from vllm_metal.metal_kernel_backend.paged_attention import (
    MetalKernelPagedAttentionWrapper,
    patch_model_attention_metal_kernel,
)
from vllm_metal.paged_attention_common import (
    OffsetCache,
    clear_context,
    prepare_decode,
    prepare_prefill,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
BLOCK_SIZE = 16


# ---------------------------------------------------------------------------
# Skip if kernels package not available
# ---------------------------------------------------------------------------

try:
    from vllm_metal.metal_kernel_backend.kernel_loader import get_paged_attention_ops

    _ops = get_paged_attention_ops()
    _KERNEL_AVAILABLE = True
except Exception:
    _KERNEL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _KERNEL_AVAILABLE,
    reason="kernels-community/paged-attention not available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _greedy_generate_standard(model, token_ids: list[int], max_new: int) -> list[int]:
    """Generate tokens using the standard mlx_lm KVCache path."""
    cache = make_prompt_cache(model)

    # Prefill
    input_ids = mx.array([token_ids], dtype=mx.int32)
    logits = model(input_ids, cache=cache)
    next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    mx.eval(mx.array(next_tok), *[c.state for c in cache])
    generated = [next_tok]

    # Decode
    for _ in range(max_new - 1):
        input_ids = mx.array([[generated[-1]]], dtype=mx.int32)
        logits = model(input_ids, cache=cache)
        next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        mx.eval(mx.array(next_tok), *[c.state for c in cache])
        generated.append(next_tok)

    return generated


def _greedy_generate_metal_kernel(
    model, token_ids: list[int], max_new: int
) -> list[int]:
    """Generate tokens using the Metal kernel paged attention path."""
    args = model.args
    num_layers = args.num_hidden_layers
    num_kv_heads = args.num_key_value_heads
    head_dim = args.head_dim

    # Allocate generous block pool
    total_tokens = len(token_ids) + max_new + BLOCK_SIZE
    num_blocks = (total_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 4

    mps_cache = MPSPagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        dtype=torch.float16,
    )

    n_patched = patch_model_attention_metal_kernel(model, mps_cache, BLOCK_SIZE)
    assert n_patched == num_layers

    # Allocate blocks for this sequence
    seq_blocks_needed = (len(token_ids) + max_new + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_ids = mps_cache.allocate_blocks(0, seq_blocks_needed)

    # --- Prefill ---
    prepare_prefill(block_ids, len(token_ids), BLOCK_SIZE)
    offset_caches = [OffsetCache(0) for _ in range(num_layers)]

    input_ids = mx.array([token_ids], dtype=mx.int32)
    logits = model(input_ids, cache=offset_caches)
    next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    mx.eval(mx.array(next_tok))
    clear_context()
    generated = [next_tok]

    seq_len = len(token_ids)  # tokens stored in cache so far

    # --- Decode ---
    for _ in range(max_new - 1):
        prepare_decode([(block_ids, seq_len)], BLOCK_SIZE)
        offset_caches = [OffsetCache(seq_len) for _ in range(num_layers)]

        input_ids = mx.array([[generated[-1]]], dtype=mx.int32)
        logits = model(input_ids, cache=offset_caches)
        next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        mx.eval(mx.array(next_tok))
        clear_context()
        generated.append(next_tok)
        seq_len += 1

    return generated


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qwen3_model():
    """Load Qwen3-0.6B once for all tests in this module."""
    model, tokenizer = mlx_lm_load(MODEL_NAME)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMetalKernelPagedVsStandard:
    @pytest.mark.slow
    def test_greedy_output_matches(self, qwen3_model):
        """Metal kernel paged attention greedy decode must match standard path."""
        model, tokenizer = qwen3_model
        prompt = "The capital of France is"
        token_ids = tokenizer.encode(prompt)
        max_new = 20

        # Standard path
        ref_tokens = _greedy_generate_standard(model, token_ids, max_new)

        # Metal kernel path
        mk_tokens = _greedy_generate_metal_kernel(model, token_ids, max_new)

        assert ref_tokens == mk_tokens, (
            f"Token mismatch!\n"
            f"  Standard:     {ref_tokens}\n"
            f"  Metal kernel: {mk_tokens}"
        )

    @pytest.mark.slow
    def test_batched_decode_matches(self, qwen3_model):
        """Batched Metal kernel paged decode must match per-request sequential."""
        model, tokenizer = qwen3_model
        prompts = [
            "The capital of France is",
            "Machine learning is",
        ]
        max_new = 10

        # Generate reference tokens independently
        ref_all = []
        for prompt in prompts:
            token_ids = tokenizer.encode(prompt)
            ref_all.append(_greedy_generate_standard(model, token_ids, max_new))

        # Metal kernel path: prefill each, then batched decode
        args = model.args
        num_layers = args.num_hidden_layers
        num_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim

        total_max = (
            max(len(tokenizer.encode(p)) for p in prompts) + max_new + BLOCK_SIZE
        )
        num_blocks = ((total_max + BLOCK_SIZE - 1) // BLOCK_SIZE) * len(prompts) + 8

        mps_cache = MPSPagedKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_blocks=num_blocks,
            block_size=BLOCK_SIZE,
            dtype=torch.float16,
        )
        patch_model_attention_metal_kernel(model, mps_cache, BLOCK_SIZE)

        # Prefill each prompt
        all_token_ids = []
        all_block_ids = []
        all_seq_lens = []
        all_generated: list[list[int]] = []

        for i, prompt in enumerate(prompts):
            tids = tokenizer.encode(prompt)
            all_token_ids.append(tids)
            needed = (len(tids) + max_new + BLOCK_SIZE - 1) // BLOCK_SIZE
            bids = mps_cache.allocate_blocks(i, needed)
            all_block_ids.append(bids)

            prepare_prefill(bids, len(tids), BLOCK_SIZE)
            offset_caches = [OffsetCache(0) for _ in range(num_layers)]
            input_ids = mx.array([tids], dtype=mx.int32)
            logits = model(input_ids, cache=offset_caches)
            next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            mx.eval(mx.array(next_tok))
            clear_context()

            all_generated.append([next_tok])
            all_seq_lens.append(len(tids))

        # Batched decode steps
        for _step in range(max_new - 1):
            requests_info = []
            for i in range(len(prompts)):
                requests_info.append((all_block_ids[i], all_seq_lens[i]))

            prepare_decode(requests_info, BLOCK_SIZE)

            max_offset = max(all_seq_lens)
            offset_caches = [OffsetCache(max_offset) for _ in range(num_layers)]

            last_tokens = [gen[-1] for gen in all_generated]
            batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]
            logits = model(batched_input, cache=offset_caches)
            next_toks = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_toks)
            clear_context()

            for i in range(len(prompts)):
                tok = int(next_toks[i].item())
                all_generated[i].append(tok)
                all_seq_lens[i] += 1

        # Compare
        for i, prompt in enumerate(prompts):
            assert all_generated[i] == ref_all[i], (
                f"Mismatch for prompt {i} ({prompt!r}):\n"
                f"  Standard:     {ref_all[i]}\n"
                f"  Metal kernel: {all_generated[i]}"
            )


class TestMetalKernelPatchRouting:
    """Verify that the wrapper routes to metal kernel vs fallback."""

    @pytest.mark.slow
    def test_patch_replaces_self_attn(self, qwen3_model):
        """After patching, each layer's self_attn should be a wrapper."""
        model, _ = qwen3_model
        args = model.args

        mps_cache = MPSPagedKVCache(
            num_layers=args.num_hidden_layers,
            num_kv_heads=args.num_key_value_heads,
            head_dim=args.head_dim,
            num_blocks=32,
            block_size=BLOCK_SIZE,
            dtype=torch.float16,
        )
        patch_model_attention_metal_kernel(model, mps_cache, BLOCK_SIZE)

        layers = model.model.layers
        for i, layer in enumerate(layers):
            assert isinstance(layer.self_attn, MetalKernelPagedAttentionWrapper), (
                f"Layer {i} self_attn is {type(layer.self_attn).__name__}, "
                f"expected MetalKernelPagedAttentionWrapper"
            )

    @pytest.mark.slow
    def test_fallback_when_no_context(self, qwen3_model):
        """Without paged context, calls must fall back to original attention."""
        model, _ = qwen3_model
        args = model.args

        mps_cache = MPSPagedKVCache(
            num_layers=args.num_hidden_layers,
            num_kv_heads=args.num_key_value_heads,
            head_dim=args.head_dim,
            num_blocks=32,
            block_size=BLOCK_SIZE,
            dtype=torch.float16,
        )
        patch_model_attention_metal_kernel(model, mps_cache, BLOCK_SIZE)

        # Run forward without setting context → should use fallback
        cache = make_prompt_cache(model)
        input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
        logits = model(input_ids, cache=cache)
        mx.eval(logits)
        # If we got here without error, fallback worked
