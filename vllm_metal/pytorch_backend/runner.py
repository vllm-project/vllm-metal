# SPDX-License-Identifier: Apache-2.0
"""Upstream vLLM Model Runner V2 with MPS device operations."""

import torch
from vllm.config import set_current_vllm_config
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_metal.pytorch_backend.runtime import install


class MPSKVBlockZeroer:
    """Clear scheduler blocks through the allocator's existing tensor views."""

    def __init__(self, caches, num_blocks):
        self.pages = [
            cache.unflatten(0, (num_blocks, cache.shape[0] // num_blocks))
            for cache in caches
        ]

    def zero_block_ids(self, block_ids):
        if not block_ids or not self.pages:
            return
        indices = torch.tensor(block_ids, device="mps", dtype=torch.int64)
        for pages in self.pages:
            pages.index_fill_(0, indices, 0)


class MPSModelRunner(GPUModelRunner):
    def __init__(self, vllm_config):
        install()
        super().__init__(vllm_config, torch.device("mps:0"))

    def load_model(self, *args, **kwargs):
        self.vllm_config.load_config.device = "mps"
        with set_current_vllm_config(self.vllm_config):
            super().load_model(*args, **kwargs)

    def initialize_kv_cache(self, kv_cache_config, *args, **kwargs):
        super().initialize_kv_cache(kv_cache_config, *args, **kwargs)
        self.kv_block_zeroer = MPSKVBlockZeroer(
            self.kv_caches, kv_cache_config.num_blocks
        )

    def warm_up(self):
        with set_current_vllm_config(self.vllm_config):
            self._dummy_run(min(16, self.max_num_tokens))
        torch.mps.synchronize()

    def get_cache_block_size_bytes(self):
        return sum(spec.page_size_bytes for spec in self.get_kv_cache_spec().values())
