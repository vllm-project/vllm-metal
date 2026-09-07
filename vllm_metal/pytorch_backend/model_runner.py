# SPDX-License-Identifier: Apache-2.0
"""Upstream vLLM execution with an MPS model and CPU scheduling tensors."""

from typing import Any, cast

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.cpu_model_runner import CPUModelRunner


class _HostInputModel(nn.Module):
    """Keep device transfers at model boundaries while sampling on the CPU."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        super().__init__()
        self.wrapped_model = model
        self.compute_device = device

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("wrapped_model"), name)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise NotImplementedError("The PyTorch Metal backend requires one stage.")
        inputs = {
            "input_ids": input_ids,
            "positions": positions,
            "inputs_embeds": inputs_embeds,
            **kwargs,
        }
        inputs = {
            key: value.to(self.compute_device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in inputs.items()
        }
        hidden_states = self.wrapped_model(intermediate_tensors=None, **inputs)
        if not isinstance(hidden_states, torch.Tensor):
            raise NotImplementedError(
                "The PyTorch Metal backend requires tensor hidden states."
            )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        model = cast(VllmModelForTextGeneration[torch.Tensor], self.wrapped_model)
        logits = model.compute_logits(hidden_states.to(self.compute_device))
        return None if logits is None else logits.cpu()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        model = cast(VllmModelForTextGeneration[torch.Tensor], self.wrapped_model)
        return model.embed_input_ids(input_ids.to(self.compute_device)).cpu()


class TorchModelRunner(CPUModelRunner):
    """Reuse upstream batching, cache management, model loading, and sampling."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.compute_device = torch.device("mps")
        # CPUModelRunner installs process-wide no-ops for accelerator APIs. MPS
        # is a real accelerator, so retain only its CPU buffer initialization.
        synchronize = torch.accelerator.synchronize
        empty_cache = torch.accelerator.empty_cache
        try:
            super().__init__(vllm_config, torch.device("cpu"))
        finally:
            torch.accelerator.synchronize = synchronize
            torch.accelerator.empty_cache = empty_cache

    def load_model(self, load_dummy_weights: bool = False) -> None:
        if load_dummy_weights:
            raise NotImplementedError("Elastic model loading is not supported on MPS.")
        self.vllm_config.load_config.device = str(self.compute_device)
        model = get_model(vllm_config=self.vllm_config)
        self.model = _HostInputModel(model, self.compute_device)
        self._setup_eagle3_aux_hidden_state_outputs()

    def get_model(self) -> nn.Module:
        return self.model.wrapped_model

    def initialize_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # vLLM 0.28 allocates then reshapes; main allocates structured views.
        # Both use self.device only for allocation in this method, so retain
        # upstream layout and binding logic without moving scheduler buffers.
        control_device = self.device
        self.device = self.compute_device
        try:
            return super().initialize_kv_cache_tensors(
                kv_cache_config, kernel_block_sizes, **kwargs
            )
        finally:
            self.device = control_device

    def _sync_device(self) -> None:
        torch.mps.synchronize()
