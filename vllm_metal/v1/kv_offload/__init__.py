# SPDX-License-Identifier: Apache-2.0
"""KV offloading support for the Metal backend.

Implements the worker-side pieces of vLLM's OffloadingConnector for the MLX
paged KV cache: an OffloadingSpec that sizes and manages the host pool, an
OffloadingWorker that moves blocks between the wired MLX cache and pageable
host memory, and a connector subclass that registers them.

See docs/offload-design.md for the architecture.
"""
