# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_metal.v1.cache_policy import WorkerCachePlanner


class TestQwenMTPWorkerBudget:
    def test_missing_optional_reporter_defaults_to_zero(self) -> None:
        planner = WorkerCachePlanner(SimpleNamespace(model_runner=SimpleNamespace()))

        assert planner._qwen_mtp_aux_bytes_per_block() == 0

    def test_public_reporter_is_used(self) -> None:
        planner = WorkerCachePlanner(
            SimpleNamespace(
                model_runner=SimpleNamespace(qwen_mtp_aux_bytes_per_block=lambda: 4096)
            )
        )

        assert planner._qwen_mtp_aux_bytes_per_block() == 4096

    def test_negative_reporter_fails_closed(self) -> None:
        planner = WorkerCachePlanner(
            SimpleNamespace(
                model_runner=SimpleNamespace(qwen_mtp_aux_bytes_per_block=lambda: -1)
            )
        )

        with pytest.raises(ValueError, match="cannot be negative"):
            planner._qwen_mtp_aux_bytes_per_block()
