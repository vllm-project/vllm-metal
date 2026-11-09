# SPDX-License-Identifier: Apache-2.0
"""Platform registration for the optional Omni engine."""

import ctypes
import os

import psutil
import torch
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

from vllm_metal.platform import MetalPlatform

_libc = ctypes.CDLL(None, use_errno=True)
_RUSAGE_INFO_V4 = 4


class _RusageInfoV4(ctypes.Structure):
    """``struct rusage_info_v4`` from ``<sys/resource.h>``."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time",
            "ri_system_time",
            "ri_pkg_idle_wkups",
            "ri_interrupt_wkups",
            "ri_pageins",
            "ri_wired_size",
            "ri_resident_size",
            "ri_phys_footprint",
            "ri_proc_start_abstime",
            "ri_proc_exit_abstime",
            "ri_child_user_time",
            "ri_child_system_time",
            "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups",
            "ri_child_pageins",
            "ri_child_elapsed_abstime",
            "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
            "ri_cpu_time_qos_default",
            "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background",
            "ri_cpu_time_qos_utility",
            "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated",
            "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time",
            "ri_serviced_system_time",
            "ri_logical_writes",
            "ri_lifetime_max_phys_footprint",
            "ri_instructions",
            "ri_cycles",
            "ri_billed_energy",
            "ri_serviced_energy",
            "ri_interval_max_phys_footprint",
            "ri_runnable_time",
        )
    ]


def _interval_max_phys_footprint():
    info = _RusageInfoV4()
    if _libc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)):
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    return info.ri_interval_max_phys_footprint


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
    def get_diffusion_attn_backend_cls(
        cls, selected_backend, head_size, allow_trtllm_default=False
    ):
        from vllm_omni.diffusion.attention.backends.registry import (
            DiffusionAttentionBackendEnum,
        )

        if selected_backend is None:
            return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
        cls.validate_diffusion_attn_backend(selected_backend)
        return DiffusionAttentionBackendEnum[selected_backend.upper()].get_path()

    @classmethod
    def is_initialized(cls):
        return torch.backends.mps.is_available()

    # torch.mps keeps no peak statistics. Unified memory makes the process's
    # physical footprint the closest measure; the kernel tracks its peak.
    @classmethod
    def max_memory_reserved(cls, device=None):
        return _interval_max_phys_footprint()

    @classmethod
    def max_memory_allocated(cls, device=None):
        return torch.mps.current_allocated_memory()

    @classmethod
    def reset_peak_memory_stats(cls, device=None):
        # libproc exports this since macOS 10.14 without a public prototype;
        # without it the peak covers the process lifetime.
        reset = getattr(_libc, "proc_reset_footprint_interval", None)
        if reset is not None:
            reset(os.getpid())

    @classmethod
    def get_free_memory(cls, device=None):
        return psutil.virtual_memory().available

    @classmethod
    def get_device_memory(cls, device=None):
        memory = psutil.virtual_memory()
        return memory.available, memory.total
