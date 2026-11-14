# SPDX-License-Identifier: Apache-2.0
"""Compare real paged draft KV with an independent native MLX-LM forward."""

import gc

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from mlx_lm.models import llama, qwen3
from mlx_lm.models.cache import make_prompt_cache
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

from tests.test_draft_model_proposer import _context, _request_state
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.v1.draft_model_proposer import DraftDims, DraftModelProposer
from vllm_metal.v1.spec_decode import SpeculativeDecodeController


def _draft_kv_cache_config(num_blocks: int) -> KVCacheConfig:
    """The scheduler-visible config ``_draft_layer_specs`` registers."""
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float32
    )
    names = ["draft_layers.0.self_attn", "draft_layers.1.self_attn"]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_groups=[KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec)],
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * num_blocks * spec.page_size_bytes,
                layers=names,
                layer_stride=num_blocks * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_layout="LBNHC",
    )


@pytest.mark.parametrize("family", [llama, qwen3], ids=["llama", "qwen3"])
@pytest.mark.parametrize("prompt_len", [15, 31])
@pytest.mark.parametrize("windowed", [False, True])
def test_draft_kv_matches_native_after_rejection_and_recompute(
    family, prompt_len, windowed, monkeypatch
):
    monkeypatch.setenv("VLLM_METAL_SPEC_VERIFY_WINDOW", "1" if windowed else "0")
    mx.random.seed(7)
    args = family.ModelArgs.from_dict(
        {
            "model_type": family.__name__.rsplit(".", 1)[-1],
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "rms_norm_eps": 1e-6,
            "vocab_size": 128,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        }
    )
    native, paged = family.Model(args), family.Model(args)
    mx.eval(native.parameters())
    paged.load_weights(tree_flatten(native.parameters()))
    proposer = DraftModelProposer(
        model=paged,
        block_size=16,
        max_model_len=128,
        num_layers=2,
        controller=SpeculativeDecodeController(),
        extract_logits=lambda logits: logits,
        merge_ingest_windows=windowed,
        allow_deferred_zero_k_ingest=True,
    )
    proposer._dims = DraftDims(num_layers=2, num_kv_heads=1, head_dim=64)
    # Bind the draft views into the shared storage, as serving does, and NaN
    # the whole backing first: every slot the asserts read must be written by
    # the proposer, never inherited from allocation.
    config = _draft_kv_cache_config(num_blocks=9)
    storage = KVCacheStorage(config)
    for tensor in storage.tensors.values():
        tensor.fill_(torch.nan)
    torch.mps.synchronize()
    proposer.bind_paged_cache(
        storage=storage,
        kv_cache_config=config,
        dtype=mx.float32,
        target_max_model_len=128,
    )
    assert proposer._scheduler_group_index == 0
    # An independent reader binding over the same storage: the cache asserts
    # below must see the proposer's writes through the shared backing.
    from vllm_metal.attention.runtime.sdpa import SDPAPagedAttentionRuntime

    runtime = SDPAPagedAttentionRuntime(block_size=16, dtype=mx.float32)
    runtime.initialize_from_config(
        config,
        ("draft_layers.0.self_attn", "draft_layers.1.self_attn"),
        storage=storage,
    )
    blocks = [5, 1, 7, 3, 6, 2, 8, 4]
    # Include the target's just-sampled token, making the first lookahead
    # write land at the start of a new physical page.
    history = [*range(1, prompt_len + 1), 42]

    def check_round():
        state = _request_state(scheduler_block_ids=blocks, token_ids=history)
        result = proposer.propose(
            _context("r", state, {"r": state}, num_speculative_tokens=3)
        )
        assert result is not None
        drafts = result.draft_token_ids[0]
        expected = []
        for _ in range(3):
            logits = native(mx.array([[*history, *expected]]))
            expected.append(int(mx.argmax(logits[0, -1])))
        assert drafts == expected

        # Verification can hide a broken drafter by rejecting its tokens.
        # Inspect the actual Metal cache too, against a fresh dense forward
        # that has neither block tables nor speculative validity tracking.
        written = [*history, *drafts[:-1]]
        cache = make_prompt_cache(native)
        mx.eval(native(mx.array([written]), cache=cache))
        slots = mx.array(
            [blocks[pos // 16] * 16 + pos % 16 for pos in range(len(written))]
        )
        for layer, reference in enumerate(cache):
            for name, expected_kv in (
                ("key_caches", reference.keys),
                ("value_caches", reference.values),
            ):
                actual = getattr(runtime.kv_cache, name)[layer].reshape(-1, 1, 64)
                actual = mx.take(actual, slots, axis=0).transpose(1, 0, 2)[None]
                np.testing.assert_allclose(
                    np.array(actual),
                    np.array(expected_kv[:, :, : len(written)]),
                    rtol=1e-4,
                    atol=1e-5,
                )
        return drafts

    for accepted in (0, 2, 3):
        drafts = check_round()
        # Replace the first unaccepted token; full acceptance adds a bonus.
        correction = (drafts[accepted] + 1) % 128 if accepted < 3 else 17
        history.extend([*drafts[:accepted], correction])

    # K=0 defers ingest, then K=3 must catch up before reading old lookahead.
    state = _request_state(scheduler_block_ids=blocks, token_ids=history)
    assert (
        proposer.propose(_context("r", state, {"r": state}, num_speculative_tokens=0))
        is None
    )
    history.extend([91, 23])
    check_round()

    # Recompute on reassigned, already-used pages after preemption.
    proposer.release_requests({"r"})
    blocks[0], blocks[1] = blocks[1], blocks[0]
    check_round()

    # Sleep frees the one backing the draft shares; the engine recomputes the
    # request after wake. Draft reads and writes must follow the new backing.
    gc.collect()
    before = torch.mps.driver_allocated_memory()
    storage.release()
    gc.collect()
    assert before - torch.mps.driver_allocated_memory() >= storage.nbytes
    storage.reallocate()
    for tensor in storage.tensors.values():
        tensor.fill_(torch.nan)
    torch.mps.synchronize()
    proposer.release_requests({"r"})
    check_round()
