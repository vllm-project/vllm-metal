# SPDX-License-Identifier: Apache-2.0
"""MPS fast paths for Omni's predictor, which defaults to forward_native."""

from functools import cache

from vllm_metal.pytorch_backend.normalization import rms_norm


@cache
def install():
    from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
        _RMSNorm,
        _RotaryEmbedding,
    )

    @_RMSNorm.register_oot
    class MPSPredictorRMSNorm(_RMSNorm):
        def forward_native(self, x):
            # Replace Omni's _RMSNorm.forward_native FP32 cast/reduction/rsqrt
            # chain with our F.rms_norm wrapper to reduce MPS kernel dispatches.
            # Fused arithmetic can round differently from the reference.
            # Keep upstream type promotion for mixed input/weight dtypes.
            if x.device.type == "mps" and x.dtype == self.weight.dtype:
                return rms_norm(x, self.weight, self.variance_epsilon)
            return super().forward_native(x)

    @_RotaryEmbedding.register_oot
    class MPSPredictorRotaryEmbedding(_RotaryEmbedding):
        def forward_native(self, x, position_ids):
            # Replace _RotaryEmbedding.forward_native's per-call frequency
            # matmul and cos/sin computation with Omni's cached tables, as its
            # CUDA/NPU/XPU paths do. Q/K rotation still runs separately.
            if x.device.type == "mps":
                return self._lookup(x, position_ids)
            return super().forward_native(x, position_ids)
