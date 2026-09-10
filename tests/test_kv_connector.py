# SPDX-License-Identifier: Apache-2.0
"""Tests for the file-based PD disaggregation KV connector."""

from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("vllm", reason="vllm not installed")

from vllm.config.kv_transfer import KVTransferConfig  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole  # noqa: E402

from vllm_metal.kv_connector import (  # noqa: E402
    MetalFileConnector,
    MetalFileConnectorMetadata,
    MetalKVBlockRegistry,
    MetalReqMeta,
)

BLOCK_SIZE = 2
NUM_BLOCKS = 6
NUM_LAYERS = 2
HEADS = 2
HEAD_DIM = 4


def _make_connector(tmp_path, role=KVConnectorRole.WORKER) -> MetalFileConnector:
    kv_config = KVTransferConfig(
        kv_connector="MetalFileConnector",
        kv_connector_module_path="vllm_metal.kv_connector",
        kv_role="kv_producer" if role == KVConnectorRole.WORKER else "kv_consumer",
        kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
    )
    vllm_config = SimpleNamespace(
        kv_transfer_config=kv_config,
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
    )
    return MetalFileConnector(vllm_config, role, SimpleNamespace())


def _make_registry(seed: float | None = None) -> MetalKVBlockRegistry:
    def layer() -> mx.array:
        data = mx.zeros((NUM_BLOCKS, BLOCK_SIZE, HEADS, HEAD_DIM), dtype=mx.float16)
        if seed is not None:
            data = data + seed
        return data

    return MetalKVBlockRegistry(
        key_caches=[layer() for _ in range(NUM_LAYERS)],
        value_caches=[layer() for _ in range(NUM_LAYERS)],
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
    )


def _bind_metadata(connector: MetalFileConnector, requests: list[MetalReqMeta]) -> None:
    connector.bind_connector_metadata(MetalFileConnectorMetadata(requests=requests))


class TestStoreLoadRoundtrip:
    def test_store_then_load_moves_blocks(self, tmp_path) -> None:
        producer = _make_connector(tmp_path)
        consumer = _make_connector(tmp_path)
        assert producer._folder_for([1, 2, 3], []) == consumer._folder_for(
            [1, 2, 3], []
        )

        producer_registry = _make_registry(seed=1.0)
        producer.set_block_registry(producer_registry)
        token_ids = list(range(1, 11))  # 10 tokens, aligned store = 8 tokens = 4 blocks
        store_blocks = [1, 3, 4, 0]
        _bind_metadata(
            producer,
            [
                MetalReqMeta(
                    token_ids=token_ids,
                    block_ids=store_blocks,
                    is_store=True,
                    mm_hashes=[],
                )
            ],
        )
        producer.save_finished_requests()
        producer.clear_connector_metadata()

        consumer_registry = _make_registry(seed=None)
        consumer.set_block_registry(consumer_registry)
        load_blocks = [5, 2, 0, 4]
        _bind_metadata(
            consumer,
            [
                MetalReqMeta(
                    token_ids=token_ids,
                    block_ids=load_blocks,
                    is_store=False,
                    mm_hashes=[],
                )
            ],
        )
        consumer.start_load_kv(None)

        for layer in range(NUM_LAYERS):
            for src_pos, dst_pos in enumerate(load_blocks):
                src_block = producer_registry.key_caches[layer][store_blocks[src_pos]]
                dst_block = consumer_registry.key_caches[layer][dst_pos]
                assert mx.all(src_block == dst_block).item() is True
                src_v = producer_registry.value_caches[layer][store_blocks[src_pos]]
                dst_v = consumer_registry.value_caches[layer][dst_pos]
                assert mx.all(src_v == dst_v).item() is True
        # Untouched consumer blocks stay zero.
        untouched = set(range(NUM_BLOCKS)) - set(load_blocks)
        for block in untouched:
            assert mx.all(consumer_registry.key_caches[0][block] == 0).item() is True

    def test_load_without_done_marker_raises(self, tmp_path) -> None:
        consumer = _make_connector(tmp_path)
        consumer.set_block_registry(_make_registry())
        # Build a folder with safetensors but no done marker.
        consumer._folder_for([7, 8, 9], [], create=True)
        _bind_metadata(
            consumer,
            [
                MetalReqMeta(
                    token_ids=[7, 8, 9],
                    block_ids=[0],
                    is_store=False,
                    mm_hashes=[],
                )
            ],
        )
        with pytest.raises(FileNotFoundError):
            consumer.start_load_kv(None)

    def test_geometry_mismatch_refuses_load(self, tmp_path) -> None:
        producer = _make_connector(tmp_path)
        producer.set_block_registry(_make_registry(seed=1.0))
        # Metadata token_ids are the aligned prefix (as build_connector_meta
        # produces) — both sides must key the identical list.
        tokens = list(range(12))
        _bind_metadata(
            producer,
            [
                MetalReqMeta(
                    token_ids=tokens,
                    block_ids=[1, 3],
                    is_store=True,
                    mm_hashes=[],
                )
            ],
        )
        producer.save_finished_requests()
        producer.clear_connector_metadata()

        # Consumer with a different head_dim must refuse the load.
        def wide_registry() -> MetalKVBlockRegistry:
            arr = mx.zeros(
                (NUM_BLOCKS, BLOCK_SIZE, HEADS, HEAD_DIM * 2), dtype=mx.float16
            )
            return MetalKVBlockRegistry(
                key_caches=[arr for _ in range(NUM_LAYERS)],
                value_caches=[arr for _ in range(NUM_LAYERS)],
                block_size=BLOCK_SIZE,
                num_blocks=NUM_BLOCKS,
            )

        consumer = _make_connector(tmp_path)
        consumer.set_block_registry(wide_registry())
        _bind_metadata(
            consumer,
            [
                MetalReqMeta(
                    token_ids=tokens,
                    block_ids=[5, 2],
                    is_store=False,
                    mm_hashes=[],
                )
            ],
        )
        with pytest.raises(ValueError, match="geometry mismatch"):
            consumer.start_load_kv(None)

    def test_store_prunes_oldest_beyond_limit(self, tmp_path) -> None:
        connector = _make_connector(tmp_path)
        connector._max_stored_prefixes = 1
        connector.set_block_registry(_make_registry(seed=1.0))
        for seed_tokens in ([1, 2, 3, 4], [5, 6, 7, 8]):
            _bind_metadata(
                connector,
                [
                    MetalReqMeta(
                        token_ids=seed_tokens,
                        block_ids=[0, 1],
                        is_store=True,
                        mm_hashes=[],
                    )
                ],
            )
            connector.save_finished_requests()
            connector.clear_connector_metadata()
        stored = list(os.listdir(str(tmp_path)))
        assert len(stored) == 1

    def test_matched_tokens_never_negative(self, tmp_path) -> None:
        connector = _make_connector(tmp_path, role=KVConnectorRole.SCHEDULER)
        aligned = _align(20)
        folder = connector._folder_for(list(range(aligned)), [], create=True)
        with open(os.path.join(folder, "done"), "wb") as fh:
            fh.write(b"ok")
        request = SimpleNamespace(
            request_id="r0",
            prompt_token_ids=list(range(20)),
            mm_features=[],
        )
        # Engine already computed more than the aligned prefix holds.
        matched, _ = connector.get_num_new_matched_tokens(request, aligned + 5)
        assert matched == 0


def _align(n: int) -> int:
    return (n - 1) // BLOCK_SIZE * BLOCK_SIZE


class TestSchedulerRole:
    def test_miss_then_hit_after_store(self, tmp_path) -> None:
        connector = _make_connector(tmp_path, role=KVConnectorRole.SCHEDULER)
        request = SimpleNamespace(
            request_id="r0",
            prompt_token_ids=list(range(20)),
            mm_features=[],
        )
        matched, async_load = connector.get_num_new_matched_tokens(request, 0)
        assert (matched, async_load) == (0, False)

        # Simulate a completed store: folder with done marker.
        aligned = (20 - 1) // BLOCK_SIZE * BLOCK_SIZE  # 18
        folder = connector._folder_for(list(range(aligned)), [], create=True)
        with open(os.path.join(folder, "done"), "wb") as fh:
            fh.write(b"ok")

        matched, async_load = connector.get_num_new_matched_tokens(request, 0)
        assert matched == aligned
        assert async_load is False

    def test_update_state_and_build_meta(self, tmp_path) -> None:
        connector = _make_connector(tmp_path, role=KVConnectorRole.SCHEDULER)
        request = SimpleNamespace(
            request_id="r1",
            prompt_token_ids=list(range(10)),
            mm_features=[],
        )
        connector.update_state_after_alloc(request, SimpleNamespace(), 8)

        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="r1",
                    prompt_token_ids=list(range(10)),
                    mm_features=[],
                    block_ids=[[0, 1, 2, 3, 4]],
                )
            ],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[],
                resumed_req_ids=[],
                num_computed_tokens=[],
                new_block_ids=[],
            ),
            num_scheduled_tokens={},
        )
        meta = connector.build_connector_meta(scheduler_output)
        assert len(meta.requests) == 1
        entry = meta.requests[0]
        assert entry.is_store is False
        assert entry.block_ids == [0, 1, 2, 3, 4]
        # State must be cleared after building metadata.
        assert connector._requests_need_load == {}
