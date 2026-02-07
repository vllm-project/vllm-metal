# SPDX-License-Identifier: Apache-2.0
"""Tests for paged attention v3 — verifies output matches non-paged path.

Run with:
    python -m pytest tests/test_paged_attention.py -v -s
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm import load as mlx_lm_load
from mlx_lm.models.cache import KVCache, make_prompt_cache

from vllm_metal.mlx_backend.cache import PagedKVCache
from vllm_metal.mlx_backend.paged_attention import (
    OffsetCache,
    PagedAttentionWrapper,
    clear_context,
    gather_kv_batched,
    get_paged_call_counts,
    patch_model_attention,
    prepare_decode,
    prepare_prefill,
    reset_paged_call_counts,
    write_kv_to_pool,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
BLOCK_SIZE = 16


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


def _greedy_generate_paged(model, token_ids: list[int], max_new: int) -> list[int]:
    """Generate tokens using the paged attention path."""
    args = model.args
    num_layers = args.num_hidden_layers
    num_kv_heads = args.num_key_value_heads
    head_dim = args.head_dim

    # Allocate generous block pool
    total_tokens = len(token_ids) + max_new + BLOCK_SIZE
    num_blocks = (total_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 4

    pool = PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
    )

    n_patched = patch_model_attention(model, pool, BLOCK_SIZE)
    assert n_patched == num_layers

    # Allocate blocks for this sequence
    seq_blocks_needed = (len(token_ids) + max_new + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_ids = pool.allocate_blocks(0, seq_blocks_needed)

    # --- Prefill ---
    prepare_prefill(block_ids, len(token_ids), BLOCK_SIZE)
    offset_caches = [OffsetCache(0) for _ in range(num_layers)]

    input_ids = mx.array([token_ids], dtype=mx.int32)
    logits = model(input_ids, cache=offset_caches)
    next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    mx.eval(mx.array(next_tok), pool.block_pool)
    clear_context()
    generated = [next_tok]

    seq_len = len(token_ids)  # tokens stored in pool so far

    # --- Decode ---
    for _ in range(max_new - 1):
        prepare_decode([(block_ids, seq_len)], BLOCK_SIZE)
        offset_caches = [OffsetCache(seq_len) for _ in range(num_layers)]

        input_ids = mx.array([[generated[-1]]], dtype=mx.int32)
        logits = model(input_ids, cache=offset_caches)
        next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        mx.eval(mx.array(next_tok), pool.block_pool)
        clear_context()
        generated.append(next_tok)
        seq_len += 1

    return generated


# ---------------------------------------------------------------------------
# Unit tests for components
# ---------------------------------------------------------------------------


class TestWriteAndGatherKV:
    def test_roundtrip_single_block(self):
        """Write KV to a block and gather it back — values should match."""
        num_blocks, num_layers, kv_heads, head_dim = 4, 2, 2, 8
        bs = 4  # block_size

        pool = mx.zeros((num_blocks, num_layers, 2, bs, kv_heads, head_dim))
        keys = mx.ones((3, kv_heads, head_dim))  # 3 tokens
        values = mx.full((3, kv_heads, head_dim), 2.0)
        slot_mapping = [0, 1, 2]  # block 0, slots 0-2

        write_kv_to_pool(pool, 0, keys, values, slot_mapping, bs)
        mx.eval(pool)

        # Gather back
        gathered_k, gathered_v = gather_kv_batched(pool, 0, [[0]], [3], bs)
        # (1, kv_heads, 3, head_dim)
        assert gathered_k.shape == (1, kv_heads, 3, head_dim)
        assert mx.allclose(gathered_k[0, 0, :, 0], mx.ones(3)).item()
        assert mx.allclose(gathered_v[0, 0, :, 0], mx.full((3,), 2.0)).item()

    def test_gather_with_left_padding(self):
        """Two sequences of different length should be left-padded."""
        num_blocks, num_layers, kv_heads, head_dim = 8, 1, 1, 4
        bs = 4

        pool = mx.zeros((num_blocks, num_layers, 2, bs, kv_heads, head_dim))

        # Seq 0: 2 tokens in block 0
        write_kv_to_pool(
            pool,
            0,
            mx.ones((2, kv_heads, head_dim)),
            mx.ones((2, kv_heads, head_dim)),
            [0, 1],
            bs,
        )
        # Seq 1: 4 tokens in block 1
        write_kv_to_pool(
            pool,
            0,
            mx.full((4, kv_heads, head_dim), 3.0),
            mx.full((4, kv_heads, head_dim), 3.0),
            [4, 5, 6, 7],
            bs,
        )
        mx.eval(pool)

        gathered_k, gathered_v = gather_kv_batched(pool, 0, [[0], [1]], [2, 4], bs)
        # max_len = 4, B = 2
        assert gathered_k.shape == (2, kv_heads, 4, head_dim)
        # Seq 0: 2 pad zeros + 2 ones
        assert mx.allclose(gathered_k[0, 0, :2, 0], mx.zeros(2)).item()
        assert mx.allclose(gathered_k[0, 0, 2:, 0], mx.ones(2)).item()
        # Seq 1: no padding, all 3s
        assert mx.allclose(gathered_k[1, 0, :, 0], mx.full((4,), 3.0)).item()


class TestOffsetCache:
    def test_offset_property(self):
        c = OffsetCache(42)
        assert c.offset == 42

    def test_make_mask_single_token(self):
        c = OffsetCache(10)
        assert c.make_mask(1) is None

    def test_make_mask_multi_token(self):
        c = OffsetCache(0)
        assert c.make_mask(5) == "causal"


class TestPrepare:
    def test_prepare_prefill_slot_mapping(self):
        prepare_prefill([10, 11], num_tokens=5, block_size=4)
        from vllm_metal.mlx_backend.paged_attention import get_context

        ctx = get_context()
        assert ctx is not None
        assert ctx.is_prefill
        # block 10: slots 40,41,42,43; block 11: slot 44
        assert ctx.slot_mapping == [40, 41, 42, 43, 44]
        clear_context()

    def test_prepare_decode(self):
        prepare_decode([([5, 6], 7)], block_size=4)
        from vllm_metal.mlx_backend.paged_attention import get_context

        ctx = get_context()
        assert ctx is not None
        assert not ctx.is_prefill
        # new_pos=7, block_ids[7//4]=block_ids[1]=6, slot=6*4+(7%4)=24+3=27
        assert ctx.slot_mapping == [27]
        assert ctx.context_lens == [8]
        assert ctx.offsets == [7]
        clear_context()


# ---------------------------------------------------------------------------
# Patch routing verification
# ---------------------------------------------------------------------------


class TestPatchRouting:
    """Verify that the wrapper actually routes to paged path vs fallback."""

    @pytest.mark.slow
    def test_patch_replaces_self_attn_with_wrapper(self, qwen3_model):
        """After patching, each layer's self_attn should be a PagedAttentionWrapper."""
        model, _ = qwen3_model
        layers = model.model.layers
        for i, layer in enumerate(layers):
            assert isinstance(layer.self_attn, PagedAttentionWrapper), (
                f"Layer {i} self_attn is {type(layer.self_attn).__name__}, "
                f"expected PagedAttentionWrapper"
            )

    @pytest.mark.slow
    def test_paged_path_is_taken_when_context_set(self, qwen3_model):
        """With paged context set, calls must go through paged path (not fallback)."""
        model, tokenizer = qwen3_model
        num_layers = model.args.num_hidden_layers

        reset_paged_call_counts(model)

        # Run one prefill with context set → should hit paged path
        token_ids = tokenizer.encode("Hello")
        pool = model.model.layers[0].self_attn._paged_kv_pool
        block_size = model.model.layers[0].self_attn._paged_block_size

        needed = (len(token_ids) + block_size - 1) // block_size
        block_ids = pool.allocate_blocks(99, needed)

        prepare_prefill(block_ids, len(token_ids), block_size)
        offset_caches = [OffsetCache(0) for _ in range(num_layers)]
        input_ids = mx.array([token_ids], dtype=mx.int32)
        model(input_ids, cache=offset_caches)
        mx.eval(pool.block_pool)
        clear_context()
        pool.free_sequence(99)

        counts = get_paged_call_counts(model)
        for i, (paged, fallback) in enumerate(counts):
            assert paged == 1, f"Layer {i}: expected 1 paged call, got {paged}"
            assert fallback == 0, (
                f"Layer {i}: expected 0 fallback calls, got {fallback}"
            )

    @pytest.mark.slow
    def test_fallback_path_when_no_context(self, qwen3_model):
        """Without paged context, calls must fall back to original attention."""
        model, _ = qwen3_model
        num_layers = model.args.num_hidden_layers

        reset_paged_call_counts(model)

        # Run one forward pass WITHOUT setting context → should hit fallback
        cache = make_prompt_cache(model)
        input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
        model(input_ids, cache=cache)
        mx.eval(mx.array(0))

        counts = get_paged_call_counts(model)
        for i, (paged, fallback) in enumerate(counts):
            assert paged == 0, f"Layer {i}: expected 0 paged calls, got {paged}"
            assert fallback == 1, f"Layer {i}: expected 1 fallback call, got {fallback}"


# ---------------------------------------------------------------------------
# End-to-end test: paged vs. standard output match
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qwen3_model():
    """Load Qwen3-0.6B once for all tests in this module.

    Also patches model attention so routing tests can verify call counts.
    The patch is idempotent — re-patching just updates pool references.
    """
    model, tokenizer = mlx_lm_load(MODEL_NAME)

    args = model.args
    total_tokens = 512
    num_blocks = (total_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 8

    pool = PagedKVCache(
        num_layers=args.num_hidden_layers,
        num_kv_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
    )
    patch_model_attention(model, pool, BLOCK_SIZE)

    return model, tokenizer


class TestPagedVsStandard:
    @pytest.mark.slow
    def test_greedy_output_matches(self, qwen3_model):
        """Paged attention greedy decode must match standard path token-for-token."""
        model, tokenizer = qwen3_model
        prompt = "The capital of France is"
        token_ids = tokenizer.encode(prompt)
        max_new = 20

        # Standard path
        ref_tokens = _greedy_generate_standard(model, token_ids, max_new)

        # Paged path
        paged_tokens = _greedy_generate_paged(model, token_ids, max_new)

        assert ref_tokens == paged_tokens, (
            f"Token mismatch!\n  Standard: {ref_tokens}\n  Paged:    {paged_tokens}"
        )

    @pytest.mark.slow
    def test_batched_decode_matches(self, qwen3_model):
        """Batched paged decode must match per-request sequential generation."""
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

        # Paged path: prefill each, then batched decode
        args = model.args
        num_layers = args.num_hidden_layers
        num_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim

        total_max = (
            max(len(tokenizer.encode(p)) for p in prompts) + max_new + BLOCK_SIZE
        )
        num_blocks = ((total_max + BLOCK_SIZE - 1) // BLOCK_SIZE) * len(prompts) + 8

        pool = PagedKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_blocks=num_blocks,
            block_size=BLOCK_SIZE,
        )
        patch_model_attention(model, pool, BLOCK_SIZE)

        # Prefill each prompt
        all_token_ids = []
        all_block_ids = []
        all_seq_lens = []
        all_generated: list[list[int]] = []

        for i, prompt in enumerate(prompts):
            tids = tokenizer.encode(prompt)
            all_token_ids.append(tids)
            needed = (len(tids) + max_new + BLOCK_SIZE - 1) // BLOCK_SIZE
            bids = pool.allocate_blocks(i, needed)
            all_block_ids.append(bids)

            prepare_prefill(bids, len(tids), BLOCK_SIZE)
            offset_caches = [OffsetCache(0) for _ in range(num_layers)]
            input_ids = mx.array([tids], dtype=mx.int32)
            logits = model(input_ids, cache=offset_caches)
            next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            mx.eval(mx.array(next_tok), pool.block_pool)
            clear_context()

            all_generated.append([next_tok])
            all_seq_lens.append(len(tids))

        # Batched decode steps
        for step in range(max_new - 1):
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
            mx.eval(next_toks, pool.block_pool)
            clear_context()

            for i in range(len(prompts)):
                tok = int(next_toks[i].item())
                all_generated[i].append(tok)
                all_seq_lens[i] += 1

        # Compare
        for i, prompt in enumerate(prompts):
            assert all_generated[i] == ref_all[i], (
                f"Mismatch for prompt {i} ({prompt!r}):\n"
                f"  Standard: {ref_all[i]}\n"
                f"  Paged:    {all_generated[i]}"
            )
