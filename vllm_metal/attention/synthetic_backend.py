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
    block_size=16, so default-sized runs keep 16 instead of vLLM's larger
    default, and hybrid models align to multiples of 16. It is never
    dispatched to as a real attention backend — the actual Metal paged
    attention lives in attention/impls/sdpa_wrapper.py. The
    unimplemented methods below intentionally raise: a loud failure is the
    correct behavior if upstream ever tries to use this as a real backend.
    """

    # The paged kernels' tuned block size; an explicit --block-size bypasses it.
    PREFERRED_BLOCK_SIZE = 16

    @staticmethod
    def get_name() -> str:
        return "METAL_ATTN"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return cls.PREFERRED_BLOCK_SIZE

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError
