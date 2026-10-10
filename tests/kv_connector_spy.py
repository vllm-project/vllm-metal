# SPDX-License-Identifier: Apache-2.0
"""A spy KV transfer group, for driving MetalKVConnector off-device."""

from __future__ import annotations

from types import SimpleNamespace


class SpyTransferGroup:
    """Records the connector step lifecycle under ``MetalKVConnector``.

    Keeps upstream's invariant that a step's results are collected while its
    metadata is still bound; a step closed after the metadata was cleared
    would trip the scheduler-side ``get_finished`` assertion.
    """

    def __init__(self) -> None:
        self.events: list[str] = []
        self.step = "?"
        self.metadata_bound = False
        # The step whose metadata is bound: a close is tagged with it, so a
        # leaked step closed later still shows as its own step.
        self.bound_step = "?"
        self.finish_forward_calls = 0
        self.finished_sending: set[str] | None = None
        self.finished_recving: set[str] | None = None
        self.invalid_block_ids: set[int] = set()

    def handle_preemptions(self, metadata) -> None:
        del metadata
        self.events.append(f"preempt:{self.step}")

    def bind_connector_metadata(self, metadata) -> None:
        del metadata
        self.metadata_bound = True
        self.bound_step = self.step
        self.events.append(f"open:{self.step}")

    def start_load_kv(self, forward_context, **kwargs) -> None:
        del forward_context, kwargs

    def finish_forward(self) -> None:
        self.finish_forward_calls += 1

    def wait_for_save(self) -> None:
        pass

    def get_transfer_results(self, finished_req_ids):
        del finished_req_ids
        if not self.metadata_bound:
            raise AssertionError("connector step closed after its metadata was cleared")
        self.events.append(f"close:{self.bound_step}")
        return SimpleNamespace(
            finished_sending=self.finished_sending,
            finished_recving=self.finished_recving or {f"r-{self.bound_step}"},
            failed_recving=set(),
        )

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set(self.invalid_block_ids)

    def get_kv_connector_stats(self):
        return None

    def get_kv_connector_kv_cache_events(self):
        return None

    def build_connector_worker_meta(self):
        return None

    def clear_connector_metadata(self) -> None:
        self.metadata_bound = False
