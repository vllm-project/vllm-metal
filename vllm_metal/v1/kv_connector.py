# SPDX-License-Identifier: Apache-2.0
"""KV connector driver for the Metal model runner, in MRv2's shape.

vLLM's model runner v2 drives a KV connector through one object:
``pre_forward``, ``finish_forward``, ``post_forward`` and ``no_forward``. The
Metal runner uses the same object, so moving to MRv2 changes only the runner.
"""

from __future__ import annotations

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


class MetalKVConnector(ActiveKVConnector):
    """``ActiveKVConnector`` without its cache registration.

    Upstream registers a dict of torch tensors. The Metal runner registers
    ``KVCacheStorage`` itself (``register_kv_connector_caches``), so this keeps
    the step methods and sets only the state they use.
    """

    def __init__(self, vllm_config: VllmConfig):
        # Skips super().__init__, so this must set every field upstream's
        # step methods read. Recheck on each vLLM bump.
        self.vllm_config = vllm_config
        self.kv_connector = get_kv_transfer_group()
        self._pending_load_kwargs = None
        self._disabled = False
