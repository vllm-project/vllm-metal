# SPDX-License-Identifier: Apache-2.0
"""Experimental GGUF loading support for vllm-metal."""

from vllm_metal.gguf.refs import GGUFReference, is_local_gguf, resolve_gguf_reference

__all__ = [
    "GGUFLoadError",
    "GGUFMLXLoader",
    "GGUFReference",
    "is_local_gguf",
    "resolve_gguf_reference",
]


def __getattr__(name: str):
    if name in {"GGUFLoadError", "GGUFMLXLoader"}:
        from vllm_metal.gguf.loader import GGUFLoadError, GGUFMLXLoader

        return {
            "GGUFLoadError": GGUFLoadError,
            "GGUFMLXLoader": GGUFMLXLoader,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
