# SPDX-License-Identifier: Apache-2.0
"""macOS storage operations for Omni's shared-memory connector."""

import hashlib
import tempfile
from multiprocessing import shared_memory

from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
    SharedMemoryConnector,
)
from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes


class MetalSharedMemoryConnector(SharedMemoryConnector):
    shm_directory = tempfile.gettempdir()

    def _shm_name(self, key):
        # Darwin allows 31 bytes, including the leading slash.
        return "omni_" + hashlib.sha256(key.encode()).hexdigest()[:24]

    def _shm_exists(self, key):
        try:
            segment = shared_memory.SharedMemory(name=self._shm_name(key))
        except FileNotFoundError:
            return False
        segment.close()
        return True

    def _write_bytes(self, payload, name):
        return shm_write_bytes(len(payload).to_bytes(8, "little") + payload, name=name)

    def _read_bytes(self, metadata):
        data = shm_read_bytes(metadata)
        size = int.from_bytes(data[:8], "little")
        if len(data) < 8 or size > len(data) - 8:
            raise ValueError("Invalid shared-memory payload length")
        # SharedMemory.size includes page padding on macOS.
        return data[8 : 8 + size]
