# SPDX-License-Identifier: Apache-2.0
"""MPS execution adapters for the upstream Omni engine and stage runners."""

import torch
from vllm_omni.worker.mixins import OmniWorkerMixin
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

from vllm_metal.pytorch_backend.runner import MPSModelRunner
from vllm_metal.pytorch_backend.worker import MPSWorker


class MPSOmniARModelRunner(MPSModelRunner, OmniARModelRunner):
    def __init__(self, config):
        super().__init__(config)
        from vllm_metal.pytorch_backend.models.qwen3_code_predictor import install
        from vllm_metal.pytorch_backend.omni_logits import install as install_logits

        install()
        install_logits()


class MPSOmniGenerationModelRunner(MPSModelRunner, OmniGenerationModelRunner):
    def warm_up(self):
        self.profile_run()
        torch.mps.synchronize()


class MPSOmniWorker(OmniWorkerMixin, MPSWorker):
    is_omni = True

    def shutdown(self):
        try:
            if getattr(self, "model_runner", None) is not None:
                self.model_runner.shutdown()
        finally:
            super().shutdown()


class MPSOmniARWorker(MPSOmniWorker):
    model_runner_cls = MPSOmniARModelRunner


class MPSOmniGenerationWorker(MPSOmniWorker):
    model_runner_cls = MPSOmniGenerationModelRunner

    def determine_available_memory(self):
        return 0
