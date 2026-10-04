# SPDX-License-Identifier: Apache-2.0
"""Upstream vLLM Model Runner V2 with MPS device operations."""

import torch
from vllm.config import set_current_vllm_config
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_metal.pytorch_backend.runtime import install


class MPSModelRunner(GPUModelRunner):
    def __init__(self, vllm_config):
        install()
        super().__init__(vllm_config, torch.device("mps:0"))

    def load_model(self, *args, **kwargs):
        self.vllm_config.load_config.device = "mps"
        with set_current_vllm_config(self.vllm_config):
            super().load_model(*args, **kwargs)

    def warm_up(self):
        with set_current_vllm_config(self.vllm_config):
            self._dummy_run(min(16, self.max_num_tokens))
        torch.mps.synchronize()

    def get_cache_block_size_bytes(self):
        return sum(spec.page_size_bytes for spec in self.get_kv_cache_spec().values())
