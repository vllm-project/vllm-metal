# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel sharding: partition parsing, weight slicing, masked routing."""

import pytest

from vllm_metal.distributed.experts import expert_partition


def test_partition_defaults_to_even_and_validates_overrides(monkeypatch):
    assert expert_partition(2, 128) == [64, 64]
    monkeypatch.setenv("VLLM_METAL_EXPERT_PARTITION", "56,72")
    assert expert_partition(2, 128) == [56, 72]
    for bad in ("64", "63,64", "0,128", "64,64,0", "a,b", ""):
        monkeypatch.setenv("VLLM_METAL_EXPERT_PARTITION", bad)
        with pytest.raises(ValueError, match="VLLM_METAL_EXPERT_PARTITION"):
            expert_partition(2, 128)


def test_partition_rejects_uneven_default(monkeypatch):
    monkeypatch.delenv("VLLM_METAL_EXPERT_PARTITION", raising=False)
    with pytest.raises(ValueError, match="divide evenly"):
        expert_partition(2, 127)


def _config(**overrides):
    from types import SimpleNamespace

    values = {
        "parallel_config": SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend="ray",
            enable_expert_parallel=False,
        ),
        "model_config": SimpleNamespace(
            hf_config=SimpleNamespace(model_type="gpt_oss"),
            quantization=None,
            multimodal_config=None,
            runner_type="generate",
        ),
        "scheduler_config": SimpleNamespace(async_scheduling=False),
        "speculative_config": None,
        "lora_config": None,
        "additional_config": {
            "tensor_transport": {
                "backend": "jaccl",
                "device_matrix": [
                    [None, ["rdma_en1", "rdma_en2"]],
                    [["rdma_en2", "rdma_en1"], None],
                ],
            }
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_expert_parallel_is_admitted_on_the_tp2_whitelist():
    from vllm_metal.distributed.tensor import validate_tensor_config

    cfg = _config()
    cfg.parallel_config.enable_expert_parallel = True
    validate_tensor_config(cfg)  # admitted
    cfg = _config()
    cfg.parallel_config.enable_expert_parallel = True
    cfg.scheduler_config.async_scheduling = True
    with pytest.raises(NotImplementedError):
        validate_tensor_config(cfg)
