# SPDX-License-Identifier: Apache-2.0
"""Upstream vLLM Model Runner V2 with MPS device operations."""

import torch
from vllm.config import set_current_vllm_config
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
from vllm.v1.core.kv_cache_planning import get_kv_cache_groups_from_builder
from vllm.v1.worker.gpu.cudagraph_utils import _teardown_profiling_state
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_metal.pytorch_backend.bindings import install


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

    @torch.inference_mode()
    def profile_largest_step(self):
        """Return the MPS pool growth of the largest step the scheduler can run.

        Like the MLX profile, the forward runs that step with attention, as the
        fewest requests it can hold, on a throwaway KV cache, and logits cover
        every row a decode step can sample.
        """
        torch.accelerator.reset_peak_memory_stats()
        reserved = torch.accelerator.memory_reserved()
        max_num_reqs = self.max_num_reqs
        num_tokens = min(self.max_num_tokens, max_num_reqs * self.max_model_len)
        builder = get_kv_cache_config_builder(self.vllm_config)
        groups = get_kv_cache_groups_from_builder(
            builder, self.vllm_config, self.get_kv_cache_spec()
        )
        # Dummy block tables point every request at block 0.
        config = builder.get_kv_cache_config_from_groups(
            self.vllm_config, groups, num_blocks=1
        )
        self.initialize_kv_cache(config, is_profiling=True)
        try:
            # TODO: drop once upstream _dummy_run takes a request count; it
            # spreads its tokens over max_num_reqs requests.
            self.max_num_reqs = cdiv(num_tokens, self.max_model_len)
            with set_current_vllm_config(self.vllm_config):
                hidden_states, _ = self._dummy_run(num_tokens)
            rows = min(num_tokens, max_num_reqs * self.decode_query_len)
            self.model.compute_logits(hidden_states[-rows:])
            del hidden_states
            torch.mps.synchronize()
            return torch.accelerator.max_memory_reserved() - reserved
        finally:
            self.max_num_reqs = max_num_reqs
            del self.kv_block_zeroer
            _teardown_profiling_state(self)

    def warm_up(self):
        with set_current_vllm_config(self.vllm_config):
            self._dummy_run(min(16, self.max_num_tokens))
        torch.mps.synchronize()

    def get_cache_block_size_bytes(self):
        return sum(spec.page_size_bytes for spec in self.get_kv_cache_spec().values())
