"""Tests for draft-KV prefix caching in DraftModelProposer (#482 direction 1)."""

from types import SimpleNamespace

from vllm_metal.v1.draft_model_proposer import DraftModelProposer


def _proposer(block_size: int = 16, num_blocks: int = 64) -> DraftModelProposer:
    # _make_plan / _hash_prefix_blocks / allocator logic never touch the model,
    # controller, or extract_logits, so stubs are fine for these tests.
    return DraftModelProposer(
        model=None,
        block_size=block_size,
        num_blocks=num_blocks,
        num_layers=1,
        controller=None,
        extract_logits=None,
    )


def test_hash_prefix_blocks_full_only_deterministic_and_chained():
    p = _proposer(block_size=16)
    toks = list(range(40))  # 2 full blocks (32) + 8 trailing partial

    hashes = p._hash_prefix_blocks("r1", toks)
    assert len(hashes) == 2  # only full blocks are hashed
    assert p._hash_prefix_blocks("r1", toks) == hashes  # cache hit, idempotent

    # Chaining: same first block but a different second block -> same h[0],
    # different h[1] (h[1] depends on h[0] as parent, plus its own tokens).
    toks_b = list(range(16)) + list(range(100, 124))
    hb = p._hash_prefix_blocks("r2", toks_b)
    assert hb[0] == hashes[0]
    assert hb[1] != hashes[1]

    # A different first block changes every downstream hash.
    toks_c = list(range(50, 90))
    hc = p._hash_prefix_blocks("r3", toks_c)
    assert hc[0] != hashes[0]


def _state(token_ids):
    return SimpleNamespace(token_ids=list(token_ids))


def _warm(p, req_id, tokens, k=4):
    plan = p._make_plan(req_id, _state(tokens), k)
    assert plan is not None
    p._register_prefix(plan)
    return plan


def test_reuses_cached_prefix_after_finish():
    p = _proposer(block_size=16, num_blocks=64)
    tokens = list(range(40))  # 2 full blocks + 8-token partial
    p1 = _warm(p, "r1", tokens)
    assert p1.draft_seq_len == 0  # cold cache: nothing to reuse
    p._prune_finished({})  # r1 done; its 2 full blocks stay cached (idle)
    p2 = p._make_plan("r2", _state(tokens), 4)
    assert p2 is not None
    assert p2.draft_seq_len == 32  # 2 full blocks reused
    assert p2.ingest_tokens == tokens[32:40]  # only the partial suffix


def test_reuse_leaves_drafting_position_for_exact_multiple():
    p = _proposer(block_size=16, num_blocks=64)
    tokens = list(range(32))  # exactly 2 full blocks, no partial
    _warm(p, "r1", tokens)
    p._prune_finished({})
    p2 = p._make_plan("r2", _state(tokens), 4)
    assert p2 is not None
    assert p2.draft_seq_len == 16  # last block kept so a drafting row exists
    assert p2.ingest_tokens == tokens[16:32]


def test_shared_prefix_block_not_freed_while_referenced():
    p = _proposer(block_size=16, num_blocks=64)
    tokens = list(range(40))
    _warm(p, "r1", tokens)  # r1 caches 2 blocks, still live
    p2 = p._make_plan("r2", _state(tokens), 4)  # r2 reuses them
    assert p2.draft_seq_len == 32
    reused = p2.block_ids[:2]
    p._prune_finished({"r2": object()})  # r1 finishes, r2 still live
    for blk in reused:
        # r1's release must only drop the refcount, not free the block: r2
        # still holds a reference. (A block registered in the prefix cache
        # is never appended to a raw free list regardless of refcount, so
        # asserting free-list membership here would pass even if the
        # refcount bookkeeping were broken -- assert on the count itself.)
        assert p._pool.blocks[blk].ref_cnt > 0


def test_eviction_reuses_idle_cached_block_under_pressure():
    # 5 physical blocks -> 4 usable (BlockPool reserves id 0 as a null
    # placeholder), a tiny pool that exactly fits one requester at a time.
    p = _proposer(block_size=16, num_blocks=5)
    _warm(p, "a", list(range(48)))  # 3 full blocks cached + 1 partial
    p._prune_finished({})  # a done; 3 idle cached blocks
    assert len(p._pool.cached_block_hash_to_block) == 3
    p2 = p._make_plan("b", _state(list(range(100, 148))), 4)  # distinct content
    assert p2 is not None  # allocation succeeded by evicting idle cached blocks
    assert len(p._pool.cached_block_hash_to_block) < 3


# -- Adversarial cases from abcgco's PR #500 review (independently verified on
# an M4 Pro: pass on the prefix-cache branch, fail on its pre-cache base). --


def test_partial_prefix_match_reuses_only_leading_run():
    """A prompt diverging mid-way must reuse exactly the matching leading
    run of blocks and ingest everything after the divergence."""
    p = _proposer(block_size=16, num_blocks=64)
    _warm(p, "r1", list(range(40)))  # blocks 0,1 cached
    p._prune_finished({})
    tokens_b = list(range(16)) + list(range(100, 140))  # shares block 0 only
    p2 = p._make_plan("r2", _state(tokens_b), 4)
    assert p2 is not None
    assert p2.draft_seq_len == 16  # block 0 reused, block 1 diverged
    assert p2.ingest_tokens == tokens_b[16:]


def test_multi_turn_history_reuses_generated_blocks():
    """Turn 2 resubmits turn 1's prompt + generated tokens as its prefix.
    Full blocks containing generated content are registered too, so the
    whole turn-1 history is reused, not just the original prompt."""
    p = _proposer(block_size=16, num_blocks=64)
    history = list(range(40)) + list(range(200, 224))  # 64 = 4 full blocks
    _warm(p, "r1", history)
    p._prune_finished({})
    turn2 = history + list(range(300, 310))
    p2 = p._make_plan("r2", _state(turn2), 4)
    assert p2 is not None
    assert p2.draft_seq_len == 64
    assert p2.ingest_tokens == turn2[64:]


def test_reused_blocks_never_come_from_free_pool():
    """A reused block must not simultaneously sit in the pool's free queue
    (would be handed out again and overwritten under another request)."""
    p = _proposer(block_size=16, num_blocks=64)
    _warm(p, "r1", list(range(40)))
    p._prune_finished({})
    p2 = p._make_plan("r2", _state(list(range(40))), 4)
    assert p2 is not None
    free_ids = {b.block_id for b in p._pool.free_block_queue.get_all_free_blocks()}
    for blk in p2.block_ids[:2]:
        assert blk not in free_ids


def test_same_content_different_history_not_reused():
    """A block whose tokens reappear under a different preceding block must
    not be reused: its KV was computed attending to a different history, and
    the chained hash (parent-dependent) is what prevents the false hit."""
    p = _proposer(block_size=16, num_blocks=64)
    _warm(p, "r1", list(range(40)))  # r1 blocks: [0..16), [16..32)
    p._prune_finished({})
    # Same second-block tokens (16..32) behind a different first block.
    shifted = list(range(100, 116)) + list(range(16, 32)) + list(range(200, 208))
    p2 = p._make_plan("r2", _state(shifted), 4)
    assert p2 is not None
    assert p2.draft_seq_len == 0
    assert p2.ingest_tokens == shifted
