# SPDX-License-Identifier: Apache-2.0
"""Shared constants for Metal paged-attention partitioning."""

PARTITION_SIZE = 512
PARTITION_THRESHOLD = 4096

# Query rows per threadgroup in the decode kernel's spec-verify window mode.
# 2 is the measured register/occupancy sweet spot on Apple GPUs: 2 rows run at
# 0.5-0.75x a one-row threadgroup per row, 4+ rows collapse to ~2x.  Injected
# into both the C++ dispatch (-DVLLM_METAL_PA_WINDOW_ROWS) and the shader
# source (#define) so host and kernel can never drift.
PA_WINDOW_ROWS = 2

# Largest head size window mode serves.  Per-thread register state in window
# mode is PA_WINDOW_ROWS * ceil(head_size / 32) accumulator floats, so 512
# would collapse occupancy the same way 4-row sub-windows did.  Models past
# this bound keep the expanded per-token verify layout (prepare_unified), and
# the C++ binding rejects wider window hints (-DVLLM_METAL_PA_WINDOW_MAX_HEAD).
PA_WINDOW_MAX_HEAD_SIZE = 256
