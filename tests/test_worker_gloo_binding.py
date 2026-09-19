# SPDX-License-Identifier: Apache-2.0
"""The explicit control IP applies to world and pipeline CPU groups."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vllm_metal.v1 import worker


@pytest.mark.parametrize("fail_subgroup", [False, True])
def test_multi_worker_binds_all_gloo_initialization_to_explicit_ip(
    monkeypatch, fail_subgroup
):
    monkeypatch.setenv("VLLM_HOST_IP", "192.168.1.145")
    active = []
    events = []

    @contextmanager
    def binding(ip):
        active.append(ip)
        events.append("enter")
        try:
            yield
        finally:
            active.clear()
            events.append("exit")

    def world(*args, **kwargs):
        assert active == ["192.168.1.145"]
        events.append("world")

    def subgroups(*args):
        assert active == ["192.168.1.145"]
        events.append("subgroups")
        if fail_subgroup:
            raise RuntimeError("subgroup failed")

    monkeypatch.setattr(worker, "bind_gloo_to_ipv4", binding, raising=False)
    monkeypatch.setattr(worker, "init_distributed_environment", world)
    monkeypatch.setattr(worker, "ensure_model_parallel_initialized", subgroups)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=2, tensor_parallel_size=1, pipeline_parallel_size=2
        )
    )
    if fail_subgroup:
        with pytest.raises(RuntimeError, match="subgroup failed"):
            worker.init_worker_distributed_environment(config, 0, "unused", 0)
    else:
        worker.init_worker_distributed_environment(config, 0, "unused", 0)
    assert not active
    assert events == ["enter", "world", "subgroups", "exit"]


@pytest.mark.parametrize("world_size,explicit_ip", [(1, True), (2, False)])
def test_single_worker_or_implicit_address_keeps_default_gloo(
    monkeypatch, world_size, explicit_ip
):
    if explicit_ip:
        monkeypatch.setenv("VLLM_HOST_IP", "192.168.1.145")
    else:
        monkeypatch.delenv("VLLM_HOST_IP", raising=False)

    def refuse(*args):
        raise AssertionError("binding should not be installed")

    monkeypatch.setattr(worker, "bind_gloo_to_ipv4", refuse, raising=False)
    monkeypatch.setattr(worker, "init_distributed_environment", lambda *a, **kw: None)
    monkeypatch.setattr(worker, "ensure_model_parallel_initialized", lambda *a: None)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=world_size,
            tensor_parallel_size=1,
            pipeline_parallel_size=world_size,
        )
    )
    worker.init_worker_distributed_environment(config, 0, "unused", 0)
