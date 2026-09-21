# SPDX-License-Identifier: Apache-2.0
"""The upstream KV cache layout advertised by the Metal worker."""

from __future__ import annotations

from vllm.v1.kv_cache_interface import KVCacheLayout

# vLLM's name for the page order Metal stores: [block, token, head, dim].
KV_CACHE_LAYOUT = KVCacheLayout.LBNHC.name
