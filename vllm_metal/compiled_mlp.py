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

        Walks the module tree and replaces each exact-type target with a
        :class:`CompiledMLPBlock` wrapper. Gated by
        ``VLLM_METAL_COMPILED_MLP``; idempotent per module. Returns the
        number of wrapped modules.
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
        targets = tuple(policies)
        wrapped = 0

        def walk_container(owner: nn.Module, sub: Any) -> None:
            # children() rebuilds dict/list containers, so writes to them
            # never reach the model — a bare target here would be an
            # un-wrappable module and must not be skipped silently.
            if type(sub) in targets:
                raise RuntimeError(
                    "CompiledMLPBlocks.install found a "
                    f"{type(sub).__name__} inside a raw container of "
                    f"{type(owner).__name__}; it can only wrap module "
                    "attributes."
                )
            walk(sub)

        def walk(mod: Any) -> None:
            nonlocal wrapped
            if not isinstance(mod, nn.Module):
                return
            for name, child in mod.children().items():
                if type(child) in targets:
                    setattr(mod, name, CompiledMLPBlock(child, policies[type(child)]))
                    wrapped += 1
                elif isinstance(child, CompiledMLPBlock):
                    continue  # already wrapped (idempotency)
                elif isinstance(child, nn.Module):
                    walk(child)
                elif isinstance(child, dict):
                    for sub in child.values():
                        walk_container(mod, sub)
                elif isinstance(child, list):
                    for sub in child:
                        walk_container(mod, sub)

        walk(model)
        if wrapped:
            logger.info(
                "Metal: compiled MLP dispatch wrapped %d blocks "
                "(mx.compile on decode-shaped calls).",
                wrapped,
            )
        return wrapped

    # The plain activations-only call per target type: every entry is an
    # exact (args, kwargs) pair the wrapper may route through the trace.
    _PLAIN_CALL: ClassVar[tuple[tuple[tuple[Any, ...], dict[str, Any]], ...]] = (
        ((), {}),
    )
    _PLAIN_CALL_TARGET_VERIFY: ClassVar[
        tuple[tuple[tuple[Any, ...], dict[str, Any]], ...]
    ] = (
        ((), {}),
        ((False,), {}),
        ((), {"target_verify": False}),
    )

    @classmethod
    def _target_policies(cls) -> dict[type, tuple]:
        # The Qwen3-Next family blocks (Qwen3.5/3.6/3.8 share them via the
        # qwen3_5 arch): the sparse MoE block (router + expert gathers +
        # shared expert) and the dense MLP, plus mlx_vlm's dense MLP for
        # conditional-generation checkpoints served through the VLM loader.
        # All are stateless by construction — extending this table requires
        # the same property. mlx_vlm's MLP takes a default-off target_verify
        # flag; its signature is validated at install so an mlx-vlm bump
        # that changes it fails fast instead of silently mis-routing.
        import inspect

        from mlx_lm.models.qwen3_next import Qwen3NextMLP, Qwen3NextSparseMoeBlock
        from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

        params = list(inspect.signature(Qwen3_5MLP.__call__).parameters)
        if params != ["self", "x", "target_verify"]:
            raise RuntimeError(
                "mlx_vlm Qwen3_5MLP.__call__ signature changed "
                f"({params}); update CompiledMLPBlocks' plain-call policy "
                "before wrapping it."
            )
        return {
            Qwen3NextSparseMoeBlock: cls._PLAIN_CALL,
            Qwen3NextMLP: cls._PLAIN_CALL,
            Qwen3_5MLP: cls._PLAIN_CALL_TARGET_VERIFY,
        }


class CompiledMLPBlock(nn.Module):
    """Per-module wrapper routing decode-shaped calls to a compiled trace.

    The inner block stays a normally registered child, so its weights
    remain visible to ``model.parameters()``/``tree_flatten`` and
    ``train()``/``eval()`` propagate as usual. The compiled trace captures
    the inner module's (load-final) weights; training-mode calls and
    prefill-sized calls delegate to the eager inner block unchanged.
    """

    def __init__(self, inner: Any, plain_calls: tuple) -> None:
        super().__init__()
        self.inner = inner
        # Inherit the inner module's mode: install runs on an already
        # eval'd model, and a fresh nn.Module defaults to training=True.
        self.train(inner.training)
        self._plain_calls = plain_calls
        self._compiled = mx.compile(inner.__call__)

    def routes_compiled(self, x: mx.array, args: tuple, kwargs: dict) -> bool:
        """Whether this exact call goes through the compiled trace.

        The single routing decision, shared with the parity harness's
        reach spy: the call must match one of the target type's exact
        plain-call forms, the wrapper must be in eval mode, and the input
        must be a decode-shaped single-row batch in a serving dtype (the
        layout/dtype checks also keep the compile cache bounded to at
        most MAX_COMPILED_TOKENS x two dtypes specializations).
        """
        return (
            (args, kwargs) in self._plain_calls
            and not self.training
            and x.ndim == 3
            and x.shape[0] == 1
            and x.shape[1] <= CompiledMLPBlocks.MAX_COMPILED_TOKENS
            and x.dtype in (mx.float16, mx.bfloat16)
        )

    def __call__(self, x: mx.array, *args: Any, **kwargs: Any) -> mx.array:
        if self.routes_compiled(x, args, kwargs):
            return self._compiled(x)
        return self.inner(x, *args, **kwargs)
