# SPDX-License-Identifier: Apache-2.0
"""Block drafting consumes actual scheduler prefix hits and shared Metal pages."""

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.request import Request

from tests.test_block_draft_proposer import (
    _assert_committed,
    _dense_tokens,
    _features,
    _prefill,
)
from tests.test_dflash_paged import make_cache
from tests.test_dspark_paged import make_cache as make_dspark_cache
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.v1.dflash_proposer import DFlashProposer
from vllm_metal.v1.dspark_proposer import DSparkProposer
from vllm_metal.v1.model_runner import RequestState
from vllm_metal.v1.spec_decode import SpeculativeDecodeController


@pytest.fixture(params=["dflash", "dspark"])
def runtime(request):
    common = {
        "num_draft_tokens": 3,
        "controller": SpeculativeDecodeController(),
        "enable_prefix_caching": True,
    }
    if request.param == "dspark":
        model, cache = make_dspark_cache()
        proposer = DSparkProposer(model, **common)
    else:
        model, embed, cache = make_cache()
        proposer = DFlashProposer(model, embed=embed, project=embed.as_linear, **common)
    # Distinct target/draft groups make a partial group miss observable.
    spec = cache.storage.specs["target"]
    names = ["target", *proposer.layer_names]
    size = 32 * spec.page_size_bytes
    config = KVCacheConfig(
        num_blocks=32,
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["target"], kv_cache_spec=spec),
            KVCacheGroupSpec(
                layer_names=list(proposer.layer_names),
                kv_cache_spec=spec,
                is_eagle_group=True,
            ),
        ],
        kv_cache_tensors=[
            KVCacheTensor(
                size=len(names) * size,
                layers=names,
                layer_stride=size,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_layout="LBNHC",
    )
    storage = KVCacheStorage(config)
    for tensor in storage.tensors.values():
        tensor.fill_(float("nan"))
    proposer.bind_cache(storage, group_index=1, max_model_len=64)
    init_none_hash(sha256)
    return proposer, config


def _manager(config, *, drop=True):
    config = replace(
        config,
        kv_cache_groups=[
            replace(group, is_eagle_group=drop and i == 1)
            for i, group in enumerate(config.kv_cache_groups)
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=64,
        scheduler_block_size=16,
        hash_block_size=16,
        enable_caching=True,
        use_eagle=drop,
    )


def _request(req_id, tokens):
    return Request(
        req_id,
        tokens,
        SamplingParams(temperature=0, max_tokens=8),
        None,
        block_hasher=get_request_block_hasher(16, sha256),
    )


def _allocate(manager, request, count=None):
    cached, start, _ = manager.get_computed_blocks(request)
    if count is None:
        count = len(request.prompt_token_ids) - start
    assert (
        manager.allocate_slots(
            request,
            count,
            num_new_computed_tokens=start,
            new_computed_blocks=cached,
            num_lookahead_tokens=4,
        )
        is not None
    )
    return start


def _context(manager, request, features, start, *, final=True):
    state = RequestState(
        token_ids=[*request.prompt_token_ids, 4] if final else request.prompt_token_ids,
        prompt_len=len(request.prompt_token_ids),
        sampling_params=request.sampling_params,
        block_ids=list(manager.get_blocks(request.request_id).get_block_ids()),
    )
    ctx = _prefill(state, features, start, final)
    return replace(
        ctx,
        prefill_reqs=[ctx.prefill_reqs[0]._replace(req_id=request.request_id)],
        request_states={request.request_id: state},
    )


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("drop", [False, True])
def test_scheduler_hits_share_only_committed_prefix(runtime, live, drop):
    proposer, config = runtime
    manager = _manager(config, drop=drop)
    first = _request("first", [1] * 47)
    features = _features(47)
    assert _allocate(manager, first) == 0
    assert proposer.propose(_context(manager, first, features, 0))[0] is not None
    if not live:
        manager.free(first)
        proposer.release_requests({"first"})

    second = _request("second", [1] * 32 + [3] * 7)
    start = _allocate(manager, second)
    assert start == (16 if drop else 32)
    blocks = manager.get_blocks("second").get_block_ids()[1]
    shared = [
        np.array(stored[mx.array(blocks[: start // 16])])
        for caches in (
            proposer.cache.cache.key_caches,
            proposer.cache.cache.value_caches,
        )
        for stored in caches
    ]
    full = [
        mx.concatenate([f[:32], tail])
        for f, tail in zip(features, _features(7), strict=True)
    ]
    result = proposer.propose(
        _context(manager, second, [f[start:] for f in full], start)
    )[0]
    assert (
        result.draft_token_ids
        == _dense_tokens(proposer, mx.array([4]), [f[None] for f in full], 3).tolist()
    )
    _assert_committed(
        proposer, manager.get_blocks(second.request_id).get_block_ids()[1], full
    )
    for saved, stored in zip(
        shared,
        [
            stored
            for caches in (
                proposer.cache.cache.key_caches,
                proposer.cache.cache.value_caches,
            )
            for stored in caches
        ],
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.array(stored[mx.array(blocks[: start // 16])]), saved
        )


@pytest.mark.parametrize("missing_group", [0, 1])
def test_missing_group_recomputes_common_suffix(runtime, missing_group):
    proposer, config = runtime
    manager = _manager(config, drop=False)
    first = _request("first", [1] * 47)
    features = _features(47)
    _allocate(manager, first)
    proposer.propose(_context(manager, first, features, 0))
    groups = manager.get_blocks("first").get_block_ids()
    manager.free(first)
    proposer.release_requests({"first"})
    # Remove one group's second block; the other group's longer hit must not
    # allow the request to skip the missing committed projections.
    manager.block_pool.evict_blocks({groups[missing_group][1]})
    second = _request("second", [1] * 39)
    start = _allocate(manager, second)
    assert start == 16
    full = [f[:39] for f in features]
    assert (
        proposer.propose(_context(manager, second, [f[start:] for f in full], start))[0]
        is not None
    )
    _assert_committed(
        proposer, manager.get_blocks(second.request_id).get_block_ids()[1], full
    )


def test_chunked_cached_prefill_commits_only_suffix(runtime):
    proposer, config = runtime
    manager = _manager(config)
    first = _request("first", [1] * 47)
    features = _features(47)
    _allocate(manager, first)
    proposer.propose(_context(manager, first, features, 0))
    manager.free(first)
    proposer.release_requests({"first"})
    second = _request("second", [1] * 39)
    start = _allocate(manager, second, count=16)
    assert start == 16
    assert (
        proposer.propose(
            _context(manager, second, [f[16:32] for f in features], 16, final=False)
        )[0]
        is None
    )
    second.num_computed_tokens = 32
    assert manager.allocate_slots(second, 7, num_lookahead_tokens=4) is not None
    result = proposer.propose(
        _context(manager, second, [f[32:39] for f in features], 32)
    )[0]
    _assert_committed(
        proposer,
        manager.get_blocks(second.request_id).get_block_ids()[1],
        [f[:39] for f in features],
    )
    assert (
        result.draft_token_ids
        == _dense_tokens(
            proposer, mx.array([4]), [f[None, :39] for f in features], 3
        ).tolist()
    )


def test_lookahead_and_partial_pages_are_not_reusable_prefixes(runtime):
    proposer, config = runtime
    manager = _manager(config, drop=False)
    free = manager.block_pool.get_num_free_blocks()
    first = _request("first", [1] * 47)
    _allocate(manager, first)
    result = proposer.propose(_context(manager, first, _features(47), 0))[0]
    # Drafting wrote temporary positions through the next page boundary.
    # A new prompt matching these tokens still only hits the full prompt pages.
    second = _request("second", [1] * 47 + [4] + result.draft_token_ids[0])
    _, start, _ = manager.get_computed_blocks(second)
    assert start == 32
    # Cache hits do not bypass capacity accounting or retain references when
    # allocation fails. The waiting request can retry after pages are released.
    held = manager.block_pool.get_new_blocks(manager.block_pool.get_num_free_blocks())
    cached, start, _ = manager.get_computed_blocks(second)
    assert (
        manager.allocate_slots(
            second,
            len(second.prompt_token_ids) - start,
            num_new_computed_tokens=start,
            new_computed_blocks=cached,
            num_lookahead_tokens=4,
        )
        is None
    )
    manager.block_pool.free_blocks(held)
    assert _allocate(manager, second) == 32
    manager.free(first)
    manager.free(second)
    proposer.release_requests({"first", "second"})
    assert manager.block_pool.get_num_free_blocks() == free


def test_same_step_prefix_consumer_waits_for_all_context_writes(runtime):
    proposer, config = runtime
    manager = _manager(config)
    first = _request("producer", [1] * 47)
    second = _request("consumer", [1] * 39)
    features = _features(47)
    _allocate(manager, first)
    # Upstream publishes scheduled prompt blocks before the worker executes.
    start = _allocate(manager, second)
    assert start == 16
    a = _context(manager, first, features, 0)
    b = _context(manager, second, [f[16:39] for f in features], start)
    # Consumer first catches a proposer that drafts per request before the
    # producer's committed prefix is projected in the same packed batch.
    ctx = replace(
        a,
        target_aux_hidden_states=tuple(mx.concatenate([f[16:39], f]) for f in features),
        prefill_reqs=[*b.prefill_reqs, *a.prefill_reqs],
        prefill_token_ids=[4, 4],
        prefill_result_modes=["final", "final"],
        request_states={**b.request_states, **a.request_states},
        cu_seqlens=[0, 23, 70],
    )
    result = proposer.propose(ctx)[0]
    assert result.req_ids == ["consumer", "producer"]
    _assert_committed(
        proposer,
        manager.get_blocks(second.request_id).get_block_ids()[1],
        [f[:39] for f in features],
    )
    assert (
        result.draft_token_ids[0]
        == _dense_tokens(
            proposer, mx.array([4]), [f[None, :39] for f in features], 3
        ).tolist()[0]
    )


@pytest.mark.parametrize("reuse_id", [False, True])
def test_resume_or_id_reuse_adopts_new_scheduler_hit(runtime, reuse_id):
    proposer, config = runtime
    manager = _manager(config)
    first = _request("r", [1] * 47)
    features = _features(47)
    _allocate(manager, first)
    proposer.propose(_context(manager, first, features, 0))
    manager.free(first)
    proposer.release_requests({"r"})
    resumed = _request("r" if reuse_id else "resumed", [1] * 39)
    start = _allocate(manager, resumed)
    assert start == 16
    assert (
        proposer.propose(
            _context(manager, resumed, [f[start:39] for f in features], start)
        )[0]
        is not None
    )
    _assert_committed(
        proposer,
        manager.get_blocks(resumed.request_id).get_block_ids()[1],
        [f[:39] for f in features],
    )


def test_evicted_page_ids_do_not_retain_request_coverage(runtime):
    proposer, config = runtime
    manager = _manager(config)
    first = _request("r", [1] * 47)
    _allocate(manager, first)
    proposer.propose(_context(manager, first, _features(47), 0))
    old_groups = manager.get_blocks("r").get_block_ids()
    old = old_groups[1]
    manager.free(first)
    proposer.release_requests({"r"})
    assert manager.reset_prefix_cache()
    # Occupy the pool, then release it so the next allocations reuse old IDs.
    held = manager.block_pool.get_new_blocks(manager.block_pool.get_num_free_blocks())
    by_id = {block.block_id: block for block in held}
    manager.block_pool.free_blocks([by_id[i] for group in old_groups for i in group])
    new = _request("r", [2] * 47)
    assert _allocate(manager, new) == 0
    features = _features(47)
    proposer.propose(_context(manager, new, features, 0))
    assert set(old) & set(manager.get_blocks("r").get_block_ids()[1])
    _assert_committed(
        proposer, manager.get_blocks(new.request_id).get_block_ids()[1], features
    )
