# SPDX-License-Identifier: Apache-2.0
"""The supported Metal DSpark contract, checked before loading model weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import DSparkConfig

if TYPE_CHECKING:
    from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig


def is_dspark_drafter(draft_model_config: ModelConfig | None) -> bool:
    if draft_model_config is None:
        return False
    hf = draft_model_config.hf_config
    return hasattr(hf, "block_size") and hasattr(hf, "target_layer_ids")


def is_dspark_config(spec: SpeculativeConfig | None) -> bool:
    return spec is not None and (
        spec.method == "dspark"
        or (spec.method == "draft_model" and is_dspark_drafter(spec.draft_model_config))
    )


def validate_dspark_config(config: VllmConfig) -> None:
    """Validate the resolved upstream DTO; ordinary spec methods are unchanged.

    Only the standalone Qwen3 path is enabled; it serves greedy requests and plain
    sampled ones (temperature/top-k/top-p without penalties or token constraints). Model-only Gemma4
    reference tests remain available without advertising a target adapter.
    """
    spec = config.speculative_config
    if not is_dspark_config(spec):
        return
    assert spec is not None
    if config.lora_config is not None:
        raise NotImplementedError("Metal DSpark does not yet support LoRA")
    if config.parallel_config.tensor_parallel_size != 1:
        raise NotImplementedError("Metal DSpark requires tensor_parallel_size=1")

    target = config.model_config
    draft = spec.draft_model_config
    if target is None or draft is None:
        raise ValueError(
            "Metal DSpark requires resolved target and draft model configs"
        )
    if (
        target.hf_text_config.model_type != "qwen3"
        or target.architectures != ["Qwen3ForCausalLM"]
        or draft.architectures != ["Qwen3DSparkModel"]
    ):
        raise NotImplementedError(
            "Metal DSpark currently supports standalone Qwen3DSparkModel drafts "
            "paired with Qwen3ForCausalLM targets; Gemma4, V4 and other "
            "architectures require separate qualification"
        )

    dspark = DSparkConfig.from_dict(draft.hf_config.to_dict())
    target_hf = target.hf_text_config
    for field, expected in (
        ("hidden_size", dspark.hidden_size),
        ("vocab_size", dspark.vocab_size),
        ("num_hidden_layers", dspark.num_target_layers),
    ):
        if getattr(target_hf, field) != expected:
            raise ValueError(f"Metal DSpark target/draft {field} mismatch")
    if not 1 <= spec.num_speculative_tokens <= dspark.block_size:
        raise ValueError(
            "Metal DSpark requires 1 <= num_speculative_tokens <= "
            f"the checkpoint block_size ({dspark.block_size})"
        )

    # vLLM options that select upstream implementations this proposer does not use.
    unsupported = {
        "enable_adaptive_verification": spec.enable_adaptive_verification,
        "draft_sample_method": spec.draft_sample_method != "greedy",
        "rejection_sample_method": spec.rejection_sample_method != "standard",
        "dspark_draft_topk": spec.dspark_draft_topk is not None
        or getattr(draft.hf_config, "dspark_draft_topk", None) is not None,
        "quantization": spec.quantization is not None,
        "kv_cache_dtype": spec.kv_cache_dtype is not None,
        "max_model_len": spec.max_model_len is not None,
        "attention_backend": spec.attention_backend is not None,
        "draft_load_config": spec.draft_load_config is not None,
        "disable_padded_drafter_batch": spec.disable_padded_drafter_batch,
        "use_local_argmax_reduction": spec.use_local_argmax_reduction,
        "draft_tensor_parallel_size": spec.draft_tensor_parallel_size not in (None, 1),
    }
    for option, enabled in unsupported.items():
        if enabled:
            raise NotImplementedError(f"Metal DSpark does not yet implement {option}")


def present_dspark_as_draft_model(config: VllmConfig) -> None:
    """Re-label a validated DSpark pair so upstream admits it on the V1 runner.

    vLLM implements DSpark only in its GPU V2 model runner and, after the
    platform hook, rejects ``method="dspark"`` for the V1 runner that Metal
    uses (``VllmConfig._get_v1_model_runner_unsupported_features``). Metal
    serves DSpark with its own proposer, so the hook presents the pair to
    upstream as ``draft_model``, the method vLLM infers for a drafter it does
    not recognise. The runner and cache planner key on
    :func:`is_dspark_config`, which recognises the drafter under either name,
    and check it before the autoregressive draft-model path. ``parallel_drafting``
    stays set, so the scheduler keeps booking ``num_speculative_tokens`` draft
    slots per request.
    """
    spec = config.speculative_config
    if spec is not None and spec.method == "dspark":
        spec.method = "draft_model"
