# SPDX-License-Identifier: Apache-2.0
"""Quantization helpers for vllm-metal load paths.

The AWQ load flow (entry-point preflight, ``mlx_lm.load`` invocation,
dtype-scoped cache key, post-load alignment of non-quantized floating
params) is owned end-to-end by ``awq_loader.AWQQuantLoader``;
``model_lifecycle`` dispatches the quantized branch to it. The actual
quantized GEMM kernel is ``mx.quantized_matmul`` in MLX core; mlx_lm
0.31.3+ provides the AWQ -> MLX-affine repack via
``_transform_awq_weights``.
"""
