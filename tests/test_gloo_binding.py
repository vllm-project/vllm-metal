# SPDX-License-Identifier: Apache-2.0
"""Explicit per-worker Gloo addresses must cover world and subgroup creation."""

from datetime import timedelta

import pytest
import torch.distributed.distributed_c10d as c10d


@pytest.fixture
def native_gloo(monkeypatch):
    """Replace only native socket/group construction, retaining constructor state."""

    class Options:
        def __init__(self):
            self._timeout = timedelta(minutes=30)
            self._threads = 2
            self._devices = []
            self.global_ranks_in_group = []
            self.group_name = ""

    class NativeGloo:
        _Options = Options

        @staticmethod
        def create_device(*, hostname):
            return ("bound-device", hostname)

        def __init__(self, store, rank, size, options):
            self.arguments = (store, rank, size)
            self.options = options

    monkeypatch.setattr(c10d, "ProcessGroupGloo", NativeGloo)
    return NativeGloo


def test_world_and_subgroups_bind_the_worker_address_and_restore(native_gloo):
    from vllm_metal.distributed.gloo import bind_gloo_to_ipv4

    store = object()
    with bind_gloo_to_ipv4("192.168.1.137"):
        assert issubclass(c10d.ProcessGroupGloo, native_gloo)
        for rank, size in [(1, 2), (0, 1), (1, 2)]:
            group = c10d.ProcessGroupGloo(store, rank, size)
            assert isinstance(group, c10d.ProcessGroupGloo)
            assert group.arguments == (store, rank, size)
            assert group.options._devices == [("bound-device", "192.168.1.137")]
    assert c10d.ProcessGroupGloo is native_gloo


@pytest.mark.parametrize("form", ["default", "positional", "keyword"])
def test_constructor_timeout_forms_are_preserved(native_gloo, form):
    from vllm_metal.distributed.gloo import bind_gloo_to_ipv4

    timeout = timedelta(seconds=17)
    with bind_gloo_to_ipv4("192.168.1.145"):
        if form == "default":
            group = c10d.ProcessGroupGloo(object(), 0, 2)
            timeout = timedelta(minutes=30)
        elif form == "positional":
            group = c10d.ProcessGroupGloo(object(), 0, 2, timeout)
        else:
            group = c10d.ProcessGroupGloo(object(), 0, 2, timeout=timeout)
    assert group.options._timeout == timeout


@pytest.mark.parametrize("keyword", [False, True])
def test_options_overload_preserves_metadata_without_mutating_caller(
    native_gloo, keyword
):
    from vllm_metal.distributed.gloo import bind_gloo_to_ipv4

    options = native_gloo._Options()
    options._timeout = timedelta(seconds=23)
    options._threads = 7
    options._devices = [("original-device", "other-address")]
    options.global_ranks_in_group = [2, 3]
    options.group_name = "existing-group"
    with bind_gloo_to_ipv4("192.168.1.137"):
        if keyword:
            group = c10d.ProcessGroupGloo(object(), 0, 2, options=options)
        else:
            group = c10d.ProcessGroupGloo(object(), 0, 2, options)
    assert group.options is not options
    assert group.options._devices == [("bound-device", "192.168.1.137")]
    assert group.options._timeout == timedelta(seconds=23)
    assert group.options._threads == 7
    assert group.options.global_ranks_in_group == [2, 3]
    assert group.options.group_name == "existing-group"
    assert options._devices == [("original-device", "other-address")]


def test_binding_failure_restores_original_constructor(native_gloo, monkeypatch):
    from vllm_metal.distributed.gloo import bind_gloo_to_ipv4

    def fail(*, hostname):
        raise RuntimeError("address is not local")

    monkeypatch.setattr(native_gloo, "create_device", staticmethod(fail))
    with pytest.raises(RuntimeError, match="address is not local"):
        with bind_gloo_to_ipv4("192.168.1.137"):
            c10d.ProcessGroupGloo(object(), 0, 2)
    assert c10d.ProcessGroupGloo is native_gloo


@pytest.mark.parametrize(
    "address", [None, 123, "mac2.local", "::1", "0.0.0.0", "224.0.0.1"]
)
def test_invalid_binding_never_installs_a_constructor(native_gloo, address):
    from vllm_metal.distributed.gloo import bind_gloo_to_ipv4

    with pytest.raises(ValueError, match="IPv4"):
        with bind_gloo_to_ipv4(address):
            raise AssertionError("invalid address reached group initialization")
    assert c10d.ProcessGroupGloo is native_gloo


def test_bound_class_is_a_real_native_gloo_subclass():
    from vllm_metal.distributed.gloo import bind_gloo_to_ipv4

    original = c10d.ProcessGroupGloo
    with bind_gloo_to_ipv4("127.0.0.1"):
        assert c10d.ProcessGroupGloo is not original
        assert issubclass(c10d.ProcessGroupGloo, original)
    assert c10d.ProcessGroupGloo is original
