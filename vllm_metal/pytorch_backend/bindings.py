# SPDX-License-Identifier: Apache-2.0

from functools import partial

import torch


def register_kernel_overrides():
    from vllm.triton_utils.dispatcher import register_kernels

    from vllm_metal.pytorch_backend import speculative_ops

    module = "vllm.v1.worker.gpu.spec_decode.rejection_sampler"
    register_kernels(
        {
            f"{module}._gather_draft_sampled_kernel": speculative_ops.gather_draft_sampled,
            f"{module}._flatten_sampled_kernel": speculative_ops.flatten_sampled,
        }
    )


def _install_input_ops():
    from vllm.v1.worker.gpu import block_table, input_batch, model_runner
    from vllm.v1.worker.gpu.mm.rope import RopeState
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    from vllm.v1.worker.gpu.sample import sampler

    from vllm_metal.pytorch_backend import input_ops

    RopeState.prepare_positions = input_ops.prepare_rope_positions

    for name in (
        "prepare_prefill_inputs",
        "prepare_pos_seq_lens",
        "combine_sampled_and_draft_tokens",
        "expand_idx_mapping",
        "post_update",
        "post_update_num_computed_tokens",
    ):
        setattr(input_batch, name, getattr(input_ops, name))
        setattr(model_runner, name, getattr(input_ops, name))
    input_batch.get_num_sampled_and_rejected = input_ops.get_num_sampled_and_rejected
    sampler.get_num_sampled_and_rejected = input_ops.get_num_sampled_and_rejected
    block_table.BlockTables.gather_block_tables = input_ops.gather_block_tables
    block_table.BlockTables.apply_staged_writes = input_ops.apply_block_table_writes
    block_table.BlockTables.compute_slot_mappings = input_ops.compute_slot_mappings
    block_table.BlockTables._make_ptr_tensor = input_ops.make_block_pointer_tensor
    MambaHybridModelState.preprocess_state = input_ops.preprocess_state
    MambaHybridModelState.postprocess_state = input_ops.postprocess_state


def _install_speculative_ops():
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar import speculator

    from vllm_metal.pytorch_backend import input_ops, runtime, speculative_ops

    rejection_sampler.compute_topk_scores = runtime.compute_topk_scores
    speculator.prepare_prefill_inputs = speculative_ops.prepare_prefill_inputs
    speculator.prepare_decode_inputs = speculative_ops.prepare_decode_inputs
    speculator.update_draft_inputs = speculative_ops.update_draft_inputs
    rejection_sampler.rejection_sample = speculative_ops.rejection_sample
    rejection_sampler.get_num_sampled_and_rejected = (
        input_ops.get_num_sampled_and_rejected
    )


def install_omni():
    from vllm.v1.worker.gpu.sample import penalties

    from vllm_metal.pytorch_backend import omni_logits
    from vllm_metal.pytorch_backend.models.qwen3_code_predictor import (
        install as install_predictor,
    )

    install_predictor()
    penalties.bincount = omni_logits.bincount
    penalties.apply_penalties = omni_logits.apply_penalties


def install():
    from vllm.model_executor.layers.mamba.short_conv import ShortConv
    from vllm.v1.worker.gpu import buffer_utils, model_runner
    from vllm.v1.worker.gpu.sample import sampler

    from vllm_metal.pytorch_backend import runtime

    if getattr(buffer_utils, "_metal_installed", False):
        return

    @ShortConv.register_oot
    class MPSShortConv(ShortConv):
        # Reuse upstream Torch ops without its CPU/CUDA custom-op wrapper.
        forward = ShortConv.forward_native

    # MRV2 0.32 uses these CUDA names even with an MPS device. Use native
    # PyTorch objects; partial avoids duplicate Dynamo handler registration.
    torch.cuda.Stream = torch.Stream
    torch.cuda.Event = partial(torch.Event, device="mps")
    torch.cuda.current_stream = partial(torch.accelerator.current_stream)
    torch.cuda.set_stream = partial(torch.accelerator.set_stream)
    # Explicit vLLM 0.32 bindings for the supported single-device path.
    buffer_utils.NonUvaBuffer = runtime.MPSStagingBuffer
    model_runner.async_tensor_h2d = runtime.copy_to_device
    buffer_utils.StagedWriteTensor.apply_write = runtime.staged_write
    buffer_utils.UvaBackedTensor.copy_to_uva = runtime.copy_changed_state

    _install_input_ops()
    _install_speculative_ops()
    # validate_request limits sampling to greedy; scoring uses upstream Torch ops.
    sampler.gumbel_sample = lambda logits, *args, **kwargs: logits.argmax(dim=-1)
    sampler.compute_topk_scores = runtime.compute_topk_scores
    buffer_utils._metal_installed = True
