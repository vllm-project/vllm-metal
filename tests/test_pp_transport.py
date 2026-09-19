# SPDX-License-Identifier: Apache-2.0
"""Validate JACCL configuration and bootstrap contracts without RDMA devices."""

import copy
import json
import logging
import os
import stat
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vllm_metal.distributed.transport import PipelineTransportConfig

TWO_RAILS = [
    [None, ["rdma_en1", "rdma_en2"]],
    [["rdma_en2", "rdma_en1"], None],
]
PEERS = ["169.254.10.1", "169.254.10.2"]
ENV_NAMES = (
    "JACCL_RANK",
    "MLX_RANK",
    "JACCL_IBV_DEVICES",
    "MLX_IBV_DEVICES",
    "JACCL_COORDINATOR",
    "MLX_JACCL_COORDINATOR",
    "JACCL_RING",
    "MLX_JACCL_RING",
    "MLX_HOSTFILE",
)


def config(matrix=None, **options):
    value = {"backend": "jaccl", "device_matrix": copy.deepcopy(TWO_RAILS)}
    if matrix is not None:
        value["device_matrix"] = matrix
    value.update(options)
    return PipelineTransportConfig.from_additional_config(
        {"pipeline_transport": value}, len(value["device_matrix"])
    )


def fake_mlx(monkeypatch, init):
    parent = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.distributed = SimpleNamespace(init=init)
    parent.core = core
    monkeypatch.setitem(sys.modules, "mlx", parent)
    monkeypatch.setitem(sys.modules, "mlx.core", core)


def group(rank=0, size=2):
    return SimpleNamespace(rank=lambda: rank, size=lambda: size)


class TestTransportConfig:
    def test_missing_configuration_keeps_ring_default(self):
        for additional in (None, {}, {"unrelated": True}, {"pipeline_transport": {}}):
            value = PipelineTransportConfig.from_additional_config(additional, 2)
            assert value.backend == "ring"
            assert value.device_matrix is None
        assert PipelineTransportConfig.from_additional_config({}, 1).backend == "ring"

    def test_ordered_lane_mapping_is_copied_and_frozen(self):
        matrix = copy.deepcopy(TWO_RAILS)
        value = config(matrix)
        matrix[0][1].reverse()
        assert json.loads(json.dumps(value.device_matrix)) == TWO_RAILS
        assert value.coordinator_port == 59451
        with pytest.raises(FrozenInstanceError):
            value.backend = "ring"
        with pytest.raises(TypeError):
            value.device_matrix[0][1][0] = "changed"

    def test_native_string_entries_and_two_rank_shared_neighbor_are_valid(self):
        value = config([[None, "rdma_en1"], ["rdma_en2", None]])
        assert json.loads(json.dumps(value.device_matrix)) == [
            [None, "rdma_en1"],
            ["rdma_en2", None],
        ]

    @pytest.mark.parametrize(
        "additional",
        [
            [],
            {"pipeline_transport": None},
            {"pipeline_transport": []},
            {"pipeline_transport": {"backend": "tcp"}},
            {"pipeline_transport": {"backend": True}},
            {"pipeline_transport": {"backend": "jaccl"}},
            {"pipeline_transport": {"backend": "ring", "device_matrix": TWO_RAILS}},
            {"pipeline_transport": {"backend": "ring", "coordinator_port": 1234}},
            {"pipeline_transport": {"backned": "jaccl"}},
        ],
    )
    def test_bad_options_fail_before_bootstrap(self, additional):
        with pytest.raises(ValueError, match="pipeline_transport"):
            PipelineTransportConfig.from_additional_config(additional, 2)

    @pytest.mark.parametrize("world_size", [0, -1, True, "2", 2.0])
    def test_world_size_is_a_positive_integer(self, world_size):
        with pytest.raises(ValueError, match="world_size"):
            PipelineTransportConfig.from_additional_config({}, world_size)

    @pytest.mark.parametrize(
        "matrix",
        [
            [[None]],
            [[None, "a"], ["b"]],
            [["self", "a"], ["b", None]],
            [[None, None], ["b", None]],
            [[None, []], ["b", None]],
            [[None, ""], ["b", None]],
            [[None, "   "], ["b", None]],
            [[None, "rdma\nen1"], ["b", None]],
            [[None, 17], ["b", None]],
            [[None, ["a", 3]], [["b", "c"], None]],
            [[None, ["a", "a"]], [["b", "c"], None]],
            [[None, ["a", "c"]], ["b", None]],
            [[None, "a", "a"], ["b", None, "c"], ["d", "e", None]],
        ],
    )
    def test_invalid_device_matrix_is_rejected(self, matrix):
        with pytest.raises(ValueError, match="pipeline_transport"):
            config(matrix)

    def test_matrix_size_must_match_pipeline_world_size(self):
        with pytest.raises(ValueError, match="world_size"):
            PipelineTransportConfig.from_additional_config(
                {
                    "pipeline_transport": {
                        "backend": "jaccl",
                        "device_matrix": TWO_RAILS,
                    }
                },
                3,
            )

    def test_all_ring_edges_including_closure_need_equal_width(self):
        # Every directed pair has equal width, but one edge has two lanes.
        matrix = [
            [None, ["a", "b"], "c"],
            [["d", "e"], None, "f"],
            ["g", "h", None],
        ]
        with pytest.raises(ValueError, match="lane"):
            config(matrix)
        missing_closure = [
            [None, "a", None, None],
            ["b", None, "c", None],
            [None, "d", None, "e"],
            [None, None, "f", None],
        ]
        with pytest.raises(ValueError, match="ring"):
            config(missing_closure)

    def test_four_rank_closed_ring_is_supported(self):
        matrix = [
            [None, "a", None, "b"],
            ["a", None, "b", None],
            [None, "a", None, "b"],
            ["b", None, "a", None],
        ]
        assert config(matrix).world_size == 4

    def test_more_than_four_rails_is_rejected_before_native_initialization(self):
        with pytest.raises(ValueError, match="4"):
            config(
                [
                    [None, [f"a{i}" for i in range(5)]],
                    [[f"b{i}" for i in range(5)], None],
                ]
            )

    @pytest.mark.parametrize("port", [0, 1023, 65536, True, "59451", 59451.0])
    def test_invalid_coordinator_port_is_rejected(self, port):
        with pytest.raises(ValueError, match="coordinator_port"):
            config(coordinator_port=port)


class TestJacclBootstrap:
    @pytest.fixture
    def previous_environment(self, monkeypatch):
        for index, name in enumerate(ENV_NAMES):
            if index % 2:
                monkeypatch.setenv(name, f"previous-{name}")
            else:
                monkeypatch.delenv(name, raising=False)
        return {name: os.environ.get(name) for name in ENV_NAMES}

    def test_strict_bootstrap_uses_rank_zero_ip_and_preserves_lane_order(
        self, monkeypatch, previous_environment, caplog
    ):
        caplog.set_level(logging.INFO, logger="vllm_metal.distributed.transport")
        paths = []
        expected_group = group(rank=1)

        def init(**kwargs):
            assert "MLX JACCL bootstrap: rank=1/2" in caplog.text
            assert "coordinator=169.254.10.1:59999" in caplog.text
            assert "rdma_en2" in caplog.text and "rdma_en1" in caplog.text
            assert kwargs == {"backend": "jaccl", "strict": True}
            assert os.environ["JACCL_RANK"] == os.environ["MLX_RANK"] == "1"
            assert os.environ["JACCL_RING"] == os.environ["MLX_JACCL_RING"] == "1"
            assert os.environ["JACCL_COORDINATOR"] == "169.254.10.1:59999"
            assert (
                os.environ["MLX_JACCL_COORDINATOR"] == os.environ["JACCL_COORDINATOR"]
            )
            assert os.environ["JACCL_IBV_DEVICES"] == os.environ["MLX_IBV_DEVICES"]
            assert "MLX_HOSTFILE" not in os.environ
            path = Path(os.environ["JACCL_IBV_DEVICES"])
            assert json.loads(path.read_text()) == TWO_RAILS
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            paths.append(path)
            return expected_group

        fake_mlx(monkeypatch, init)
        actual = config(coordinator_port=59999).bootstrap_jaccl(1, PEERS)
        assert actual is expected_group
        assert {
            name: os.environ.get(name) for name in ENV_NAMES
        } == previous_environment
        assert paths and all(not path.exists() for path in paths)

    @pytest.mark.parametrize("failure", ["init", "rank", "size"])
    def test_failed_bootstrap_never_falls_back_and_restores_environment(
        self, monkeypatch, previous_environment, failure
    ):
        paths, calls = [], []

        def init(**kwargs):
            calls.append(kwargs)
            paths.append(Path(os.environ["JACCL_IBV_DEVICES"]))
            if failure == "init":
                raise RuntimeError("RDMA device unavailable")
            return group(
                rank=1 if failure == "rank" else 0, size=1 if failure == "size" else 2
            )

        fake_mlx(monkeypatch, init)
        with pytest.raises(RuntimeError):
            config().bootstrap_jaccl(0, PEERS)
        assert calls == [{"backend": "jaccl", "strict": True}]
        assert {
            name: os.environ.get(name) for name in ENV_NAMES
        } == previous_environment
        assert paths and all(not path.exists() for path in paths)

    @pytest.mark.parametrize(
        "rank,peers",
        [
            (-1, PEERS),
            (2, PEERS),
            (True, PEERS),
            ("0", PEERS),
            (0, [PEERS[0]]),
            (0, [PEERS[0], PEERS[0]]),
            (0, ["mac-a.local", PEERS[1]]),
            (0, ["127.0.0.1", PEERS[1]]),
            (0, ["0.0.0.0", PEERS[1]]),
            (0, ["224.0.0.1", PEERS[1]]),
            (0, ["255.255.255.255", PEERS[1]]),
            (0, ["::1", PEERS[1]]),
            (0, ["169.254.10.1:59451", PEERS[1]]),
            (0, [1, PEERS[1]]),
        ],
    )
    def test_invalid_peers_or_rank_fail_before_network_or_environment_changes(
        self, monkeypatch, previous_environment, rank, peers
    ):
        calls = []
        fake_mlx(monkeypatch, lambda **kwargs: calls.append(kwargs))
        with pytest.raises(ValueError):
            config().bootstrap_jaccl(rank, peers)
        assert calls == []
        assert {
            name: os.environ.get(name) for name in ENV_NAMES
        } == previous_environment

    def test_ring_configuration_cannot_accidentally_bootstrap_jaccl(self):
        value = PipelineTransportConfig.from_additional_config({}, 2)
        with pytest.raises(ValueError, match="jaccl"):
            value.bootstrap_jaccl(0, PEERS)
