# SPDX-License-Identifier: Apache-2.0
"""Synthetic Metal attention backend.

Kept in its own module so :mod:`vllm_metal.platform` (loaded during plugin
discovery) does not transitively import :mod:`vllm.v1.attention.backend`
and re-enter a partially initialized :mod:`vllm.distributed` in the
EngineCore subprocess.
"""

from vllm.v1.attention.backend import AttentionBackend, MultipleOf


class MetalBackend(AttentionBackend):
    """Synthetic backend that advertises Metal kernel block alignment.

    This class exists solely so the framework's block-size selection
    (Platform.update_block_size_for_backend → backend_cls.get_preferred_
    block_size, and the hybrid-block-size math via
    Platform._align_hybrid_block_size) can read Metal's MultipleOf(16)
    alignment constraint. The Metal paged-attention kernels are tuned for
    block_size=16; advertising MultipleOf(16) makes vLLM's selector default
    to 16 and lets hybrid models align to multiples of 16. It is never
    dispatched to as a real attention backend — the actual Metal paged
    attention lives in metal_kernel_backend/paged_attention.py. The
    unimplemented methods below intentionally raise: a loud failure is the
    correct behavior if upstream ever tries to use this as a real backend.
    """

    @staticmethod
    def get_name() -> str:
        return "METAL_ATTN"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError

    @staticmethod
    def get_kv_cache_shape(*args, **kwargs):
        raise NotImplementedError
