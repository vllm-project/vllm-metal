# SPDX-License-Identifier: Apache-2.0
"""Platform registration for the optional Omni engine."""

import psutil
import torch
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

from vllm_metal.platform import MetalPlatform


class MPSOmniPlatform(OmniPlatform, MetalPlatform):
    _omni_enum = OmniPlatformEnum.OOT
    device_type = "mps"
    device_name = "mps"
    dist_backend = "gloo"

    @classmethod
    def get_omni_ar_worker_cls(cls):
        return "vllm_metal.pytorch_backend.omni.MPSOmniARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls):
        return "vllm_metal.pytorch_backend.omni.MPSOmniGenerationWorker"

    @classmethod
    def get_default_stage_config_path(cls):
        return "vllm_omni/deploy"

    @classmethod
    def get_shared_memory_connector_cls(cls):
        from vllm_metal.pytorch_backend.omni_connector import MetalSharedMemoryConnector

        return MetalSharedMemoryConnector

    @classmethod
    def get_torch_device(cls, local_rank=None):
        return torch.device("mps:0")

    @classmethod
    def get_device_count(cls):
        return int(torch.backends.mps.is_available())

    @classmethod
    def supports_torch_inductor(cls):
        return False

    @classmethod
    def supports_talker_mtp_graph_capture(cls):
        return False

    @classmethod
    def supports_float64(cls):
        return False

    @classmethod
    def synchronize(cls):
        torch.mps.synchronize()

    @classmethod
    def get_free_memory(cls, device=None):
        return psutil.virtual_memory().available

    @classmethod
    def get_device_memory(cls, device=None):
        memory = psutil.virtual_memory()
        return memory.available, memory.total
