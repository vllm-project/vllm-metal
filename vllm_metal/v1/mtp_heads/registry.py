# SPDX-License-Identifier: Apache-2.0
"""Native MTP head contract and registry, keyed by draft ``model_type``.

A native MTP head runs a draft checkpoint's own multi-token-prediction layer as
a speculative proposer. Each head is a class implementing :class:`NativeMTPHead`:
it owns its runtime validation and builds the :class:`MetalProposer` that runs
it, so the model runner only looks the head up by draft ``model_type`` and
delegates. Gemma4 MTP stays on its dedicated in-model assistant path and is not
registered here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vllm_metal.v1.proposer import MetalProposer


@dataclass(frozen=True, slots=True)
class NativeMTPBuildContext:
    """Immutable inputs a native MTP head consults while building its proposer.

    Future head requirements extend this context (e.g. a target-modules
    accessor for heads that borrow target embeddings) rather than the
    ``build_proposer`` signature.
    """

    speculative_config: Any
    controller: Any
    vllm_config: Any
    target_config: Any
    dtype: Any


@runtime_checkable
class NativeMTPHead(Protocol):
    """Contract for a native MTP head.

    A head is keyed by its draft checkpoint's ``model_type`` and owns the two
    responsibilities the runner must not: runtime validation of the speculative
    config, and construction of the :class:`MetalProposer` that runs the head.
    ``build_proposer`` receives a :class:`NativeMTPBuildContext` — including
    ``vllm_config``, so config guards live in the head rather than the runner —
    and returns the proposer the runner installs. Heads keep their heavy (mlx)
    implementation modules off the package import path by importing them lazily
    inside ``build_proposer``.
    """

    model_type: str

    def build_proposer(self, context: NativeMTPBuildContext) -> MetalProposer:
        """Validate the speculative config and build this head's proposer."""
        ...


# Draft ``model_type`` -> head implementation. Head modules register their head
# here; the runner dispatches through ``find_native_mtp_head``.
MTP_HEAD_REGISTRY: Mapping[str, NativeMTPHead] = {}


def draft_model_type(speculative_config: Any | None) -> str | None:
    """Return the draft checkpoint ``model_type`` for an ``mtp`` config.

    Walks ``speculative_config.draft_model_config.hf_config.model_type``,
    tolerating ``SimpleNamespace``-style configs: a missing attribute anywhere in
    the chain, a non-``mtp`` method, or a non-string ``model_type`` all yield
    ``None`` rather than raising. Shared by ``find_native_mtp_head`` and the
    unsupported-head error message so the config walk has one implementation.
    """
    if (
        speculative_config is None
        or getattr(speculative_config, "method", None) != "mtp"
    ):
        return None
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    model_type = getattr(hf_config, "model_type", None)
    return model_type if isinstance(model_type, str) else None


def find_native_mtp_head(speculative_config: Any | None) -> NativeMTPHead | None:
    """Return the registered native MTP head for ``speculative_config``, if any.

    Checked *after* ``Gemma4MTPAssistantSource.is_gemma4_mtp`` in the runner's
    ``install_drafter`` dispatch, so Gemma4 keeps its dedicated path and its
    assistant configs never resolve here. Returns ``None`` unless the config is
    an ``mtp`` method whose draft checkpoint ``model_type`` is a string in the
    registry (read at call time, so a rebound registry stays in sync).
    """
    model_type = draft_model_type(speculative_config)
    if model_type is None:
        return None
    return MTP_HEAD_REGISTRY.get(model_type)


def registered_mtp_head_types() -> list[str]:
    """Return the sorted draft ``model_type``s that have a registered native head.

    Reads ``MTP_HEAD_REGISTRY`` through the module attribute at call time so a
    rebound registry is reflected; a ``from`` import of the dict would freeze a
    stale reference.
    """
    return sorted(MTP_HEAD_REGISTRY)


def unsupported_mtp_message(speculative_config: Any | None) -> str:
    """Build the fail-loud message for an ``mtp`` draft with no runnable head.

    Names the draft ``model_type`` and the registered native heads so the gap is
    obvious.
    """
    model_type = draft_model_type(speculative_config)
    return (
        f"No MTP head on Metal for draft model_type {model_type!r} "
        f"(registered native heads: {registered_mtp_head_types()}; plus "
        "Gemma4 MTP via the in-model assistant)."
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
