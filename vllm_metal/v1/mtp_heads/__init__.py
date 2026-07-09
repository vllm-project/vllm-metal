# SPDX-License-Identifier: Apache-2.0
"""Native MTP heads package.

The contract and registry live in :mod:`vllm_metal.v1.mtp_heads.registry`; this
package module only re-exports them. Head implementations live in their own
modules and keep heavy (mlx) imports off the package import path.
"""

from __future__ import annotations

from vllm_metal.v1.mtp_heads.registry import (
    MTP_HEAD_REGISTRY,
    NativeMTPBuildContext,
    NativeMTPHead,
    draft_model_type,
    find_native_mtp_head,
    registered_mtp_head_types,
    unsupported_mtp_message,
)

__all__ = [
    "MTP_HEAD_REGISTRY",
    "NativeMTPBuildContext",
    "NativeMTPHead",
    "draft_model_type",
    "find_native_mtp_head",
    "registered_mtp_head_types",
    "unsupported_mtp_message",
]
