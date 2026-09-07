# SPDX-License-Identifier: Apache-2.0
"""Whole-model bridge and cache-allocation contracts."""

from types import SimpleNamespace

import psutil
import pytest
import torch
from torch import nn
from vllm.config import get_current_vllm_config, set_current_vllm_config
from vllm.v1.worker.cpu_model_runner import CPUModelRunner

from vllm_metal.pytorch_backend.model_runner import TorchModelRunner, _HostInputModel
from vllm_metal.pytorch_backend.worker import TorchWorker


class _Model(nn.Module):
    supports_pp = True

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8)

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        return self.embedding(input_ids) + positions[:, None]

    def compute_logits(self, hidden_states):
        return self.head(hidden_states)

    def embed_input_ids(self, input_ids):
        return self.embedding(input_ids)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_model_bridge_preserves_hidden_states_logits_and_capabilities(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    model = _Model().to(device)
    bridge = _HostInputModel(model, torch.device(device))
    inputs = {"input_ids": torch.tensor([1, 2]), "positions": torch.tensor([0, 1])}

    hidden = bridge(**inputs)

    torch.testing.assert_close(
        hidden, model(**{name: value.to(device) for name, value in inputs.items()})
    )
    # The CPU runner indexes the MPS hidden states using CPU logits indices.
    indices = torch.tensor([1])
    logits = bridge.compute_logits(hidden[indices])
    torch.testing.assert_close(logits, model.compute_logits(hidden[indices]).cpu())
    torch.testing.assert_close(
        bridge.embed_input_ids(inputs["input_ids"]),
        model.embed_input_ids(inputs["input_ids"].to(device)).cpu(),
    )
    assert bridge.supports_pp is model.supports_pp
    assert hidden.device.type == device
    assert logits.device.type == "cpu"


@pytest.mark.parametrize("fails", [False, True])
def test_kv_allocation_restores_control_device(monkeypatch, fails):
    runner = object.__new__(TorchModelRunner)
    runner.device = torch.device("cpu")
    runner.compute_device = torch.device("meta")
    config = SimpleNamespace()
    cache = {"layer": torch.empty(1)}

    def allocate(self, kv_cache_config, kernel_block_sizes, **kwargs):
        assert self.device == torch.device("meta")
        assert kv_cache_config is config
        assert kernel_block_sizes == [16]
        if fails:
            raise RuntimeError("allocation failed")
        return cache

    monkeypatch.setattr(CPUModelRunner, "initialize_kv_cache_tensors", allocate)

    if fails:
        with pytest.raises(RuntimeError, match="allocation failed"):
            runner.initialize_kv_cache_tensors(config, [16])
    else:
        assert runner.initialize_kv_cache_tensors(config, [16]) is cache
    assert runner.device == torch.device("cpu")


def test_worker_load_makes_config_available_to_upstream_weight_loader():
    worker = object.__new__(TorchWorker)
    worker.vllm_config = SimpleNamespace()

    def load_model(load_dummy_weights):
        assert get_current_vllm_config() is worker.vllm_config
        assert not load_dummy_weights

    worker.model_runner = SimpleNamespace(load_model=load_model)
    outer_config = SimpleNamespace()
    with set_current_vllm_config(outer_config):
        worker.load_model()
        assert get_current_vllm_config() is outer_config


@pytest.mark.parametrize(
    "available_mb,recommended_mb,allocated_mb,requested_mb,expected_mb",
    [
        pytest.param(4096, 2048, 0, None, 512, id="default-cap"),
        pytest.param(125, 2048, 0, None, 100, id="system-memory-limit"),
        pytest.param(4096, 250, 25, None, 100, id="mps-memory-limit"),
        pytest.param(4096, 2048, 0, 700, 700, id="explicit-overrides-default-cap"),
        pytest.param(4096, 250, 25, 101, None, id="explicit-exceeds-budget"),
        pytest.param(4096, 100, 51, None, None, id="mps-budget-exhausted"),
    ],
)
def test_worker_respects_kv_memory_limits(
    monkeypatch, available_mb, recommended_mb, allocated_mb, requested_mb, expected_mb
):
    """Cache sizing preserves RAM headroom and honors safe explicit requests."""
    mib = 1024**2
    worker = object.__new__(TorchWorker)
    worker.cache_config = SimpleNamespace(
        gpu_memory_utilization=0.5,
        kv_cache_memory_bytes=None if requested_mb is None else requested_mb * mib,
    )
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=available_mb * mib),
    )
    monkeypatch.setattr(torch.mps, "synchronize", lambda: None)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.mps, "recommended_max_memory", lambda: recommended_mb * mib
    )
    monkeypatch.setattr(
        torch.mps, "driver_allocated_memory", lambda: allocated_mb * mib
    )

    if expected_mb is None:
        with pytest.raises(ValueError, match="memory|budget"):
            worker.determine_available_memory()
    else:
        assert worker.determine_available_memory() == expected_mb * mib
