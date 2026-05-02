# SPDX-License-Identifier: Apache-2.0
"""vLLM v1 compatibility module for Metal platform."""

__all__ = ["MetalWorker"]


def __getattr__(name: str):
    if name == "MetalWorker":
        from vllm_metal.v1.worker import MetalWorker

        return MetalWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
