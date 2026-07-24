# SPDX-License-Identifier: Apache-2.0
"""GLM-4.7-Flash native MTP head."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_metal.v1.mtp_heads.registry import (
    NativeMTPBuildContext,
    NativeMTPHeadRegistry,
)

if TYPE_CHECKING:
    from vllm_metal.v1.proposer import MetalProposer


class Glm4MoeLiteMTPHead:
    """Native MTP head for GLM-4.7-Flash (``glm4_moe_lite_mtp``)."""

    model_type = "glm4_moe_lite_mtp"
    max_num_speculative_tokens = 1

    def build_proposer(self, context: NativeMTPBuildContext) -> MetalProposer:
        speculative_config = context.speculative_config
        if context.vllm_config.cache_config.enable_prefix_caching:
            raise NotImplementedError(
                "Native MTP speculative decoding on Metal requires prefix "
                "caching to be OFF: cached prompt tokens skip the target "
                "forward that produces the hidden states the MTP head "
                "consumes, so their slots could never be built. Restart with "
                "--no-enable-prefix-caching."
            )
        if speculative_config.num_speculative_tokens > self.max_num_speculative_tokens:
            raise NotImplementedError(
                "Native MTP head "
                f"{self.model_type!r} supports at most "
                f"{self.max_num_speculative_tokens} speculative "
                f"token(s), got num_speculative_tokens="
                f"{speculative_config.num_speculative_tokens}."
            )
        if context.vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Native MTP speculative decoding on Metal does not support "
                "pipeline parallelism: a non-last pipeline stage does not produce "
                "the target final hidden states the MTP head consumes. Run with "
                "pipeline_parallel_size=1."
            )

        from vllm_metal.v1.mtp_heads.glm4_moe_lite_mtp import Glm4MoeLiteMTPHeadLoader
        from vllm_metal.v1.mtp_proposer import NativeMTPProposer

        return NativeMTPProposer.build(
            speculative_config=speculative_config,
            controller=context.controller,
            loader=Glm4MoeLiteMTPHeadLoader(),
            model_type=self.model_type,
            target_config=context.target_config,
        )


NativeMTPHeadRegistry.register(Glm4MoeLiteMTPHead())
