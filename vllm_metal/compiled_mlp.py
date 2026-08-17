# SPDX-License-Identifier: Apache-2.0
"""Compiled dispatch for stateless MLP/MoE blocks.

A decode step issues hundreds of small lazy ops, and the per-layer MLP
chain (router, expert gathers, gating elementwise, shared expert) is the
largest contributor. The blocks are stateless — activations in,
activations out, no cache — so ``mx.compile`` can trace each one whole
and fuse its elementwise glue, cutting the per-step op count. On the
quantized serving path the compiled output is bitwise identical to the
eager dispatch (pinned by tests and the parity harness); unquantized
fp16 fusion may reorder accumulation at the ulp level. Per-PR A/B
numbers live in the PR body.

Only decode-shaped calls route through the compiled trace; prefill-sized
calls keep the eager path so the compile cache holds a bounded set of
small shapes. The trace captures the inner module's weights on first
call, so install only runs when nothing rebinds them afterwards — LoRA
serves skip the install entirely (the adapter layers replace projection
modules inside the blocks at setup time and swap state per batch). The
wrapper state is single-threaded, matching the in-process uniproc
executor.
"""

from __future__ import annotations

from typing import Any, ClassVar

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from vllm.logger import init_logger

from vllm_metal import envs as metal_envs

logger = init_logger(__name__)


class CompiledMLPBlocks:
    """Owner for the compiled stateless-MLP decode dispatch."""

    # Route only dispatch-bound decode calls through the compiled trace:
    # at small token counts the per-op scheduling overhead dominates and
    # fusing the block wins; at larger decode batches the matmuls dominate
    # (mlx_lm also switches its expert dispatch at 64 routed indices) and
    # the eager path measures faster, so those keep it. Together with the
    # layout/dtype checks in routes_compiled this bounds each module's
    # compile cache to at most this many shapes per serving dtype.
    MAX_COMPILED_TOKENS: ClassVar[int] = 4

    @classmethod
    def install(cls, model: Any) -> int:
        """Wrap every stateless MLP/MoE block in *model*.

        Collects targets via ``named_modules()`` and replaces them through
        ``update_modules(tree_unflatten(...))`` — the same replacement
        idiom the LoRA wrapper uses. Gated by ``VLLM_METAL_COMPILED_MLP``;
        idempotent per module. Returns the number of wrapped modules.
        """
        if not metal_envs.VLLM_METAL_COMPILED_MLP:
            return 0
        from vllm_metal.platform import MetalPlatform

        if not MetalPlatform.is_available():
            return 0
        if not isinstance(model, nn.Module):
            raise TypeError(
                "CompiledMLPBlocks.install expects an mlx nn.Module, got "
                f"{type(model).__name__}"
            )
        policies = cls._target_policies()
        # Everything under an already-installed wrapper (its inner and any
        # target nested inside it, like the MoE block's shared expert) is a
        # bare target instance in the module tree; exclude the whole
        # subtree so a re-install is a no-op.
        wrapper_prefixes = [
            name + "."
            for name, module in model.named_modules()
            if isinstance(module, CompiledMLPBlock)
        ]
        candidates = {
            name: module
            for name, module in model.named_modules()
            if type(module) in policies
            and not any(name.startswith(prefix) for prefix in wrapper_prefixes)
        }
        # Wrap outermost targets only: a target nested inside another (the
        # MoE block's shared-expert MLP) compiles as part of the outer trace.
        outermost = [
            name
            for name in candidates
            if not any(
                name.startswith(other + ".") for other in candidates if other != name
            )
        ]
        replacements = [
            (name, policies[type(candidates[name])](candidates[name]))
            for name in sorted(outermost)
        ]
        if replacements:
            model.update_modules(tree_unflatten(replacements))
            logger.info(
                "Metal: compiled MLP dispatch wrapped %d blocks "
                "(mx.compile on decode-shaped calls).",
                len(replacements),
            )
        return len(replacements)

    @classmethod
    def _target_policies(cls) -> dict[type, type]:
        """Target block type -> wrapper class (one per calling convention).

        The Qwen3-Next family blocks (Qwen3.5/3.6/3.8 share them via the
        qwen3_5 arch) take a plain activations-only call; mlx_vlm's dense
        and MoE variants add a default-off ``target_verify`` flag, whose
        signatures are validated here so an mlx-vlm bump that changes them
        fails fast instead of silently mis-routing. All targets are
        stateless by construction — extending this table requires the same
        property.
        """
        import inspect

        from mlx_lm.models.qwen3_next import Qwen3NextMLP, Qwen3NextSparseMoeBlock
        from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP
        from mlx_vlm.models.qwen3_5_moe.language import (
            Qwen3_5MoeMLP,
            Qwen3_5MoeSparseMoeBlock,
        )

        for target in (Qwen3_5MLP, Qwen3_5MoeMLP, Qwen3_5MoeSparseMoeBlock):
            params = list(inspect.signature(target.__call__).parameters)
            if params != ["self", "x", "target_verify"]:
                raise RuntimeError(
                    f"mlx_vlm {target.__name__}.__call__ signature changed "
                    f"({params}); update CompiledMLPBlocks' wrapper policy "
                    "before wrapping it."
                )
        return {
            Qwen3NextSparseMoeBlock: CompiledMLPBlock,
            Qwen3NextMLP: CompiledMLPBlock,
            Qwen3_5MLP: CompiledTargetVerifyMLPBlock,
            Qwen3_5MoeMLP: CompiledTargetVerifyMLPBlock,
            Qwen3_5MoeSparseMoeBlock: CompiledTargetVerifyMLPBlock,
        }


class CompiledMLPBlock(nn.Module):
    """Wrapper for unary stateless blocks: ``__call__(x)``.

    The inner block stays a normally registered child, so its weights
    remain visible to ``model.parameters()``/``tree_flatten`` and
    ``train()``/``eval()`` propagate as usual. The compiled trace captures
    the inner module's (load-final) weights; training-mode and
    prefill-sized calls delegate to the eager inner block unchanged.
    """

    def __init__(self, inner: Any) -> None:
        super().__init__()
        self.inner = inner
        # Inherit the inner module's mode: install runs on an already
        # eval'd model, and a fresh nn.Module defaults to training=True.
        self.train(inner.training)
        self._compiled = mx.compile(inner.__call__)

    def routes_compiled(self, x: mx.array) -> bool:
        """Whether a decode-shaped call goes through the compiled trace.

        Shared with the parity harness's reach spy. The layout/dtype
        checks also bound each module's compile cache to at most
        MAX_COMPILED_TOKENS shapes per serving dtype.
        """
        return (
            not self.training
            and x.ndim == 3
            and x.shape[0] == 1
            and x.shape[1] <= CompiledMLPBlocks.MAX_COMPILED_TOKENS
            and x.dtype in (mx.float16, mx.bfloat16)
        )

    def dispatch_compiled(self, x: mx.array) -> mx.array:
        """Run the compiled trace (the harness spy's single choke point)."""
        return self._compiled(x)

    def __call__(self, x: mx.array) -> mx.array:
        if self.routes_compiled(x):
            return self.dispatch_compiled(x)
        return self.inner(x)


class CompiledTargetVerifyMLPBlock(CompiledMLPBlock):
    """Wrapper for blocks called as ``__call__(x, target_verify=False)``.

    mlx_vlm's decoder layers pass the flag on every call; its default-off
    form is the plain call, while a truthy flag (spec-decode verify)
    delegates to the eager inner block unchanged.
    """

    def __call__(self, x: mx.array, target_verify: bool = False) -> mx.array:
        if target_verify is False and self.routes_compiled(x):
            return self.dispatch_compiled(x)
        return self.inner(x, target_verify)
