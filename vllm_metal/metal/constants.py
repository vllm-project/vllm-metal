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
