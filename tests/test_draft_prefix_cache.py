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

    hashes = p._hash_prefix_blocks(toks)
    assert len(hashes) == 2  # only full blocks are hashed
    assert p._hash_prefix_blocks(toks) == hashes  # deterministic

    # Chaining: same first block but a different second block -> same h[0],
    # different h[1] (h[1] depends on h[0] as parent, plus its own tokens).
    toks_b = list(range(16)) + list(range(100, 124))
    hb = p._hash_prefix_blocks(toks_b)
    assert hb[0] == hashes[0]
    assert hb[1] != hashes[1]

    # A different first block changes every downstream hash.
    toks_c = list(range(50, 90))
    hc = p._hash_prefix_blocks(toks_c)
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
        assert blk not in p._free_blocks  # shared blocks survive


def test_eviction_reuses_idle_cached_block_under_pressure():
    p = _proposer(block_size=16, num_blocks=4)  # tiny pool
    _warm(p, "a", list(range(48)))  # 3 full blocks cached + 1 partial
    p._prune_finished({})  # a done; 3 idle cached blocks
    assert len(p._cached) == 3
    p2 = p._make_plan("b", _state(list(range(100, 148))), 4)  # distinct content
    assert p2 is not None  # allocation succeeded by evicting idle cached blocks
    assert len(p._cached) < 3
