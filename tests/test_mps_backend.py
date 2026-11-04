# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MPS request mapping, cache sizing and normalization parity."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.states import RequestState

from vllm_metal.pytorch_backend.bindings import install
from vllm_metal.pytorch_backend.input_ops import post_update

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


@pytest.fixture
def state():
    install()
    return RequestState(4, 64, 16, 0, 64, torch.device("mps"))


@pytest.mark.parametrize("num_groups", [1, 2])
def test_request_slots_and_dummy_padding(num_groups):
    install()
    tables = BlockTables(
        [16] * num_groups,
        4,
        16,
        [4] * num_groups,
        torch.device("mps"),
    )
    tables.append_block_ids(3, ([8, 9], [12, 13])[:num_groups], True)
    tables.append_block_ids(0, ([2, 4], [6, 7])[:num_groups], True)
    tables.apply_staged_writes()
    assert tables.num_blocks.gpu.tolist() == [[2, 0, 0, 2]] * num_groups
    mapping = torch.tensor([0, 3], dtype=torch.int32, device="mps")
    gathered = tables.gather_block_tables(mapping, num_reqs_padded=3)
    assert [table[:, :2].tolist() for table in gathered] == [
        [[2, 4], [8, 9], [0, 0]],
        [[6, 7], [12, 13], [0, 0]],
    ][:num_groups]
    # Real/dummy control from vLLM v0.32.0, tests/v1/worker/test_gpu_block_table.py
    # (test_dummy_request_slot_mapping_is_pad). Keep our mixed rows and padding.
    query_start_loc = torch.tensor([0, 1, 4], dtype=torch.int32, device="mps")
    positions = torch.tensor([17, 14, 15, 16, 99999, 99999], device="mps")
    slots = tables.compute_slot_mappings(
        mapping, query_start_loc, positions, num_tokens_padded=8
    )
    expected_slots = [
        [65, 142, 143, 144] + [PAD_SLOT_ID] * 4,
        [113, 206, 207, 208] + [PAD_SLOT_ID] * 4,
    ]
    assert slots.tolist() == expected_slots[:num_groups]
    # The new dummy row must clear a stale real slot, not preserve existing PAD.
    slots[:, 4] = slots[:, 0].clone()
    dummy = tables.compute_slot_mappings(
        torch.tensor([0, 3, -1], dtype=torch.int32, device="mps"),
        torch.tensor([0, 1, 4, 5], dtype=torch.int32, device="mps"),
        positions,
        num_tokens_padded=8,
    )
    assert dummy.tolist() == expected_slots[:num_groups]


@pytest.mark.parametrize("num_sampled", [0, 1, 3])
def test_removed_request_writeback_at_capacity(state, num_sampled):
    # The second request disappears before writeback. Its -1 slot must not
    # alias slot zero and overwrite the surviving request's sampled token.
    start = state.all_token_ids.gpu.shape[1] - num_sampled
    state.num_computed_tokens.gpu[0] = start - 1
    state.total_len.gpu[0] = start
    state.last_sampled_tokens[0, 0] = 40
    state.all_token_ids.gpu[0].fill_(7)
    # Unused output columns may extend beyond the end of a full history row.
    sampled_ids = [41, 42, 43]
    post_update(
        torch.tensor([0, -1], dtype=torch.int32, device="mps"),
        state.num_computed_tokens.gpu,
        state.last_sampled_tokens,
        None,
        torch.tensor([sampled_ids, [999] * len(sampled_ids)], device="mps"),
        torch.full((2,), num_sampled, dtype=torch.int32, device="mps"),
        torch.zeros(2, dtype=torch.int32, device="mps"),
        torch.tensor(
            [0, num_sampled, 2 * num_sampled], dtype=torch.int32, device="mps"
        ),
        state.all_token_ids.gpu,
        state.total_len.gpu,
    )
    expected_last = sampled_ids[num_sampled - 1] if num_sampled else 40
    assert state.last_sampled_tokens[0, 0].item() == expected_last
    assert (
        state.all_token_ids.gpu[0].tolist() == [7] * start + sampled_ids[:num_sampled]
    )
    assert state.total_len.gpu[0].item() == start + num_sampled
    assert state.num_computed_tokens.gpu[0].item() == start - 1 + num_sampled


def test_writeback_does_not_clamp_out_of_bounds_tokens():
    """A broken capacity contract must fail, not overwrite the final token.

    Use CPU tensors so a failed bound does not poison the MPS command buffer.
    """
    with pytest.raises(IndexError, match="out of bounds"):
        post_update(
            mapping=torch.tensor([0]),
            computed=torch.tensor([6]),
            last=torch.tensor([[40]]),
            bins=None,
            tokens=torch.tensor([[41, 42, 43]]),
            count=torch.tensor([3]),
            rejected=torch.tensor([0]),
            cu=None,
            all_ids=torch.zeros((1, 8), dtype=torch.int64),
            total=torch.tensor([7]),
        )


@pytest.fixture
def mps_memory(monkeypatch):
    """Fake a 4 GiB GPU working set with 128 MiB already allocated.

    Yields the native override that sets maxBufferLength for the test.
    """
    from vllm_metal.pytorch_backend.mps_ops import _load_mps_module

    monkeypatch.setattr(torch.mps, "synchronize", lambda: None)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 128 << 20)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 4 << 30)
    override = _load_mps_module()._override_max_buffer_length_for_test
    override(4 << 30)
    try:
        yield override
    finally:
        override(-1)


BUFFER_LIMITS = pytest.mark.parametrize(
    ("limit", "largest"),
    [
        # PyTorch rounds the backing heap up to 2 MiB, so an unaligned limit
        # still leaves only 1 GiB usable.
        ((1 << 30) + (1 << 20), 1 << 30),
        # PyTorch rejects size == maxBufferLength.
        (1 << 30, (1 << 30) - (2 << 20)),
    ],
    ids=["unaligned", "aligned"],
)


@pytest.mark.parametrize(
    ("config_name", "fields", "width"),
    [
        ("Qwen3Config", {"hidden_size": 64, "intermediate_size": 192}, 192),
        ("OPTConfig", {"hidden_size": 64, "ffn_dim": 192}, 192),
        ("GPT2Config", {"n_embd": 64, "n_inner": 192}, 192),
        ("GPT2Config", {"n_embd": 64}, 256),
        ("PretrainedConfig", {"hidden_size": 64}, None),
    ],
)
def test_cache_budget_handles_model_dimensions(mps_memory, config_name, fields, width):
    import transformers

    from vllm_metal.pytorch_backend.worker import MPSWorker

    hf = getattr(transformers, config_name)(**fields)
    worker = SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None, gpu_memory_utilization=0.5
        ),
        model_config=SimpleNamespace(
            hf_text_config=hf, get_hidden_size=lambda: hf.hidden_size
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )
    if width is None:
        with pytest.raises(ValueError, match="--kv-cache-memory-bytes"):
            MPSWorker.determine_available_memory(worker)
    else:
        expected = (2 << 30) - (128 << 20) - (512 << 20) - 8 * (width * 2 + 64 * 8) * 4
        assert MPSWorker.determine_available_memory(worker) == expected
    # Explicit budgets bypass dimension introspection, including unknown layouts.
    worker.cache_config.kv_cache_memory_bytes = 1234
    worker.model_config = None
    assert MPSWorker.determine_available_memory(worker) == 1234


@BUFFER_LIMITS
def test_cache_budget_fits_one_metal_buffer(mps_memory, limit, largest):
    """Upstream puts all KV layers in one buffer, so the budget must fit in it."""
    import transformers

    from vllm_metal.pytorch_backend.worker import MPSWorker

    hf = transformers.Qwen3Config(hidden_size=64, intermediate_size=192)
    worker = SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None, gpu_memory_utilization=0.9
        ),
        model_config=SimpleNamespace(hf_text_config=hf, get_hidden_size=lambda: 64),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )
    mps_memory(limit)
    assert MPSWorker.determine_available_memory(worker) == largest


@BUFFER_LIMITS
def test_explicit_cache_budget_must_fit_one_metal_buffer(mps_memory, limit, largest):
    from vllm_metal.pytorch_backend.worker import MPSWorker

    worker = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_memory_bytes=largest),
    )
    mps_memory(limit)
    assert MPSWorker.determine_available_memory(worker) == largest
    worker.cache_config.kv_cache_memory_bytes = largest + 1
    with pytest.raises(ValueError, match=f"at most {largest} bytes"):
        MPSWorker.determine_available_memory(worker)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    ("fused", "num_tokens", "hidden_size"),
    [
        (False, 4, 128),
        (True, 4, 128),
        (True, 1, 1024),
        (True, 4, 1024),
        (True, 3, 1028),
    ],
    ids=["plain", "residual", "decode", "batched-decode", "tail"],
)
def test_rms_norm_parity(dtype, fused, num_tokens, hidden_size):
    from vllm import ir

    import vllm_metal.pytorch_backend.normalization  # noqa: F401

    op = ir.ops.fused_add_rms_norm if fused else ir.ops.rms_norm
    args = op.generate_inputs(
        num_tokens=num_tokens, hidden_size=hidden_size, dtype=dtype, device="cpu"
    )
    args[-2].copy_(
        torch.tensor([1.25, 1.75, 0.75, 1.5], dtype=dtype).repeat(hidden_size // 4)
    )
    if fused:
        # This row distinguishes FP32 accumulation and cast-before-weight from
        # rounding the sum early or applying the learned weight before casting.
        args[0][0].fill_(1)
        args[1][0] = (
            torch.tensor([0.125, 0.375, 0.625, 0.875], dtype=dtype).repeat(
                hidden_size // 4
            )
            * torch.finfo(dtype).eps
        )
    with op.set_priority(["native"]):
        expected = op(*args)
    gpu_args = tuple(x.to("mps") if isinstance(x, torch.Tensor) else x for x in args)
    with op.set_priority(["torch_mps", "native"]):
        assert op.dispatch(*gpu_args).provider == "torch_mps"
        actual = op(*gpu_args)
    if fused:
        actual, residual = actual
        expected, expected_residual = expected
        torch.testing.assert_close(residual.cpu(), expected_residual, atol=0, rtol=0)
        torch.testing.assert_close(actual[0].cpu(), expected[0], atol=0, rtol=0)
        for original, gpu_arg in zip(args[:-1], gpu_args[:-1], strict=True):
            torch.testing.assert_close(gpu_arg.cpu(), original, atol=0, rtol=0)
    torch.testing.assert_close(actual.cpu(), expected, **op.get_tolerance(dtype))


@pytest.mark.parametrize("strided_arg", [0, 1, 2], ids=["input", "residual", "weight"])
def test_residual_rms_norm_falls_back_for_noncontiguous_inputs(strided_arg):
    from vllm import ir

    from vllm_metal.pytorch_backend import normalization

    op = ir.ops.fused_add_rms_norm
    args = list(
        op.generate_inputs(
            num_tokens=4,
            hidden_size=1024,
            dtype=torch.bfloat16,
            epsilon=1e-3,
            device="mps",
        )
    )
    source = args[strided_arg]
    storage = torch.empty(
        (*source.shape[:-1], source.shape[-1] * 2), device="mps", dtype=source.dtype
    )
    args[strided_arg] = storage[..., ::2]
    args[strided_arg].copy_(source)
    assert not args[strided_arg].is_contiguous()
    before = [arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args]
    with op.set_priority(["native"]):
        expected = op(*before)
    with (
        op.set_priority(["torch_mps", "native"]),
        patch.object(
            normalization,
            "_add_rms_norm_kernel",
            side_effect=AssertionError(
                "Noncontiguous inputs must bypass the fused kernel"
            ),
        ) as kernel_loader,
    ):
        assert op.dispatch(*args).provider == "torch_mps"
        actual = op(*args)
        kernel_loader.assert_not_called()
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            result.cpu(), reference, **op.get_tolerance(source.dtype)
        )
    for arg, original in zip(args[:-1], before[:-1], strict=True):
        torch.testing.assert_close(arg.cpu(), original, atol=0, rtol=0)


@pytest.mark.parametrize("case", ["empty", "unweighted", "partial-variance"])
def test_residual_rms_norm_retains_native_fallback(case):
    from vllm import ir

    import vllm_metal.pytorch_backend.normalization  # noqa: F401

    op = ir.ops.fused_add_rms_norm
    args = list(
        op.generate_inputs(
            num_tokens=0 if case == "empty" else 1,
            hidden_size=1024,
            dtype=torch.bfloat16,
            device="cpu",
        )
    )
    if case == "unweighted":
        args[2] = None
    elif case == "partial-variance":
        args.append(512)
    with op.set_priority(["native"]):
        expected = op(*args)
    gpu_args = tuple(
        arg.to("mps") if isinstance(arg, torch.Tensor) else arg for arg in args
    )
    with op.set_priority(["torch_mps", "native"]):
        assert op.dispatch(*gpu_args).provider == "native"
        actual = op(*gpu_args)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            result.cpu(), reference, **op.get_tolerance(torch.bfloat16)
        )


def test_shortconv_checkpoint_survives_resume():
    from vllm_metal.pytorch_backend.input_ops import preprocess_state
    from vllm_metal.pytorch_backend.runner import MPSKVBlockZeroer

    device = torch.device("mps")
    cache = torch.arange(12, dtype=torch.float16, device=device).reshape(3, 2, 2)
    checkpoint = cache[2].clone()
    zeroer = MPSKVBlockZeroer([cache.flatten()], num_blocks=3)
    spec = SimpleNamespace(block_size=4)
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["conv"])])
    context = SimpleNamespace(
        static_forward_context={"conv": SimpleNamespace(kv_cache=[cache])}
    )
    state = SimpleNamespace(
        _align_mode=True,
        _mamba_state_idx_gpu=torch.tensor([-1, 0], device=device, dtype=torch.int32),
        _get_mamba_group_info=lambda _: ([0], spec),
        vllm_config=SimpleNamespace(compilation_config=context),
    )
    batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([1], device=device, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], device=device, dtype=torch.int32),
    )
    tables = (torch.tensor([[2, 0]], device=device, dtype=torch.int32),)
    computed = torch.tensor([0, 4], device=device, dtype=torch.int32)

    # Resume the four-token prefix in block 2, continuing in recycled block 0.
    expected = cache.clone()
    expected[0].zero_()
    zeroer.zero_block_ids([0])
    assert torch.equal(cache, expected)
    preprocess_state(state, batch, tables, config, computed)
    assert torch.equal(cache[0], checkpoint)

    # Advancing within the same block must not reload or mutate the checkpoint.
    cache[0].add_(1)
    computed[1] = 5
    preprocess_state(state, batch, tables, config, computed)
    assert torch.equal(cache[0], checkpoint + 1)
    assert torch.equal(cache[2], checkpoint)

    # Re-admission at the cached prefix must restore the original state again.
    state._mamba_state_idx_gpu[1] = 0
    computed[1] = 4
    preprocess_state(state, batch, tables, config, computed)
    assert torch.equal(cache[0], checkpoint)
    assert torch.equal(cache[2], checkpoint)


@pytest.mark.parametrize("num_logprobs", [0, 2])
def test_mps_logprobs(num_logprobs):
    from vllm_metal.pytorch_backend.runtime import compute_topk_scores

    logits = torch.tensor([[3.0, 3.0, 1.0, -2.0], [1.0, 5.0, 2.0, 0.0]], device="mps")
    # Include the selected token even outside top-k, with correct tie ranks.
    actual = compute_topk_scores(
        logits, num_logprobs, torch.tensor([2, 1], device="mps")
    )
    expected = logits.cpu().log_softmax(-1)
    assert actual.logprobs.device.type == "mps"
    assert actual.logprob_token_ids[:, 0].tolist() == [2, 1]
    assert actual.selected_token_ranks.tolist() == [3, 1]
    torch.testing.assert_close(
        actual.logprobs.cpu(),
        expected.gather(-1, actual.logprob_token_ids.cpu().long()),
    )
    torch.testing.assert_close(
        actual.logprobs[:, 1:].cpu(), expected.topk(num_logprobs, -1).values
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,heads,kv_heads,head_size,gapped_heads",
    [(1, 16, 8, 128, False), (37, 3, 1, 64, False), (37, 4, 2, 64, True)],
)
def test_rope_matches_upstream(dtype, rows, heads, kv_heads, head_size, gapped_heads):
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    from vllm_metal.pytorch_backend.rotary import apply_rope

    # Packed projections exercise row strides and nonzero storage offsets.
    widths = [heads * head_size, kv_heads * head_size, kv_heads * head_size]
    packed = torch.randn(rows + 1, sum(widths), device="mps", dtype=dtype)
    q, k, _ = packed[1:].split(widths, dim=-1)
    if gapped_heads:
        q = q.view(rows, heads, head_size)[:, ::2]
        k = k.view(rows, kv_heads, head_size)[:, ::2]
    positions = torch.randperm(64, device="mps")[:rows]
    cache = torch.randn(64, head_size, device="mps", dtype=dtype)
    expected = RotaryEmbedding.forward_static(
        positions, q, k, head_size, head_size, cache, True
    )
    actual = apply_rope(positions, q, k, cache, head_size)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_vision_encoder_attention_runs_on_mps():
    # Upstream registers this op for the platform's CPU key only.
    from vllm.v1.attention.ops.vit_attn_wrappers import vit_torch_sdpa_wrapper

    install()
    q, k, v = torch.randn(3, 1, 9, 4, 64).unbind(0)
    cu_seqlens = torch.tensor([0, 2, 9], dtype=torch.int32)
    expected = vit_torch_sdpa_wrapper(q, k, v, cu_seqlens=cu_seqlens)
    actual = vit_torch_sdpa_wrapper(
        q.to("mps"), k.to("mps"), v.to("mps"), cu_seqlens=cu_seqlens.to("mps")
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("logprobs", [None, 5])
def test_chunked_verification_matches_full_batch(
    run_in_spawn_process, monkeypatch, logprobs
):
    """Verify fixed logits in a fresh worker, before other kernels are loaded."""
    monkeypatch.setenv("VLLM_METAL_BACKEND", "mps")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    run_in_spawn_process(_check_chunked_verification, logprobs)


def _check_chunked_verification(logprobs):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler

    from vllm_metal.platform import MetalPlatform

    MetalPlatform.register_triton_kernel_overrides()
    install()
    state = RequestState(4, 64, 16, 0, 64, torch.device("mps"))
    sampler = Sampler(
        SimpleNamespace(reasoning_config=None), 4, 64, torch.device("mps"), state
    )
    state.prefill_len.np[:] = 10
    state.prefill_len.copy_to_uva()
    for slot in range(4):
        sampler.add_request(slot, SamplingParams(temperature=0, logprobs=logprobs))
    sampler.apply_staged_writes()
    verifier = rejection_sampler.RejectionSampler(
        sampler,
        SimpleNamespace(
            num_speculative_tokens=3,
            enable_adaptive_verification=False,
            rejection_sample_method="standard",
        ),
        torch.device("mps"),
    )
    widths = np.array([4, 4, 3, 1], dtype=np.int32)
    cu_np = np.array([0, 4, 8, 11, 12], dtype=np.int32)
    slots_np = np.array([3, 0, 2, 1])

    def to_device(values):
        return torch.tensor(values, device="mps", dtype=torch.int32)

    batch = SimpleNamespace(
        num_reqs=4,
        input_ids=to_device([0, 1, 2, 3, 0, 5, 9, 7, 0, 2, 3, 0]),
        positions=to_device([10, 11, 12, 13, 10, 11, 12, 13, 8, 9, 10, 8]),
        logits_indices=torch.arange(12, device="mps"),
        idx_mapping=to_device(slots_np),
        idx_mapping_np=slots_np,
        expanded_idx_mapping=to_device(np.repeat(slots_np, widths)),
        expanded_local_pos=to_device([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0]),
        cu_num_logits=to_device(cu_np),
        cu_num_logits_np=cu_np,
        seq_lens=to_device([14, 14, 11, 9]),
        seq_lens_cpu_upper_bound=torch.tensor([14, 14, 11, 9]),
    )
    winners = to_device([1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 6]).long()
    logits = torch.arange(64, device="mps", dtype=torch.float32).repeat(12, 1) / 100
    logits.scatter_(1, winners[:, None], 10)
    # Chunk/full oracle from vLLM v0.32.0,
    # tests/v1/worker/test_gpu_rejection_sampler_chunking.py. Keep real verification
    # and our fixed acceptance/prefill expectations instead of its fake verifier.
    rejection_sampler.get_max_chunk_logits = lambda _: 1024
    full = verifier(logits, batch)
    rejection_sampler.get_max_chunk_logits = lambda _: 4
    chunked = verifier(logits, batch)
    assert full.num_sampled.tolist() == [4, 2, 1, 0]
    assert full.num_rejected.tolist() == [0, 2, 2, 0]
    assert full.sampled_token_ids.tolist() == [
        [1, 2, 3, 4],
        [5, 6, -1, -1],
        [2, -1, -1, -1],
        [6, -1, -1, -1],
    ]
    assert torch.equal(chunked.num_sampled, full.num_sampled)
    assert torch.equal(chunked.num_rejected, full.num_rejected)
    assert torch.equal(chunked.sampled_token_ids, full.sampled_token_ids)
    if logprobs is None:
        assert full.logprobs_tensors is None and chunked.logprobs_tensors is None
    else:
        scores = full.logprobs_tensors
        chunked_scores = chunked.logprobs_tensors
        assert scores is not None and chunked_scores is not None
        kept = torch.tensor([0, 1, 2, 3, 4, 5, 8], device="mps")
        torch.testing.assert_close(
            scores.logprob_token_ids[kept, 0], winners[kept].int()
        )
        expected = logits.log_softmax(-1).gather(1, winners[:, None])[:, 0]
        torch.testing.assert_close(scores.logprobs[kept, 0], expected[kept])
        assert scores.cu_num_generated_tokens == cu_np.tolist()
        assert torch.equal(chunked_scores.logprob_token_ids, scores.logprob_token_ids)
        assert torch.equal(chunked_scores.logprobs, scores.logprobs)
        assert torch.equal(
            chunked_scores.selected_token_ranks, scores.selected_token_ranks
        )
        assert chunked_scores.cu_num_generated_tokens == scores.cu_num_generated_tokens


def test_mrope_positions_preserve_ragged_prefill_and_decode():
    from vllm_metal.pytorch_backend.input_ops import prepare_rope_positions

    state = SimpleNamespace(
        num_dims=3,
        max_model_len=8,
        positions=torch.full((3, 7), -1, device="mps", dtype=torch.int64),
        prefill_positions=SimpleNamespace(
            gpu=torch.arange(4 * 3 * 8, device="mps").reshape(12, 8)
        ),
        prefill_delta=SimpleNamespace(gpu=torch.tensor([0, 7, 0, 0], device="mps")),
    )
    prepare_rope_positions(
        state,
        torch.tensor([3, 1], device="mps"),
        torch.tensor([0, 2, 5], device="mps"),
        torch.tensor([0, 4, 0, 3], device="mps"),
        torch.tensor([0, 4, 0, 1], device="mps"),
    )
    assert state.positions.cpu().tolist() == [
        [73, 74, 11, 12, 13, 0, -1],
        [81, 82, 11, 12, 13, 0, -1],
        [89, 90, 11, 12, 13, 0, -1],
    ]


def test_omni_repetition_penalty_matches_upstream_torch():
    """Sampler row order may differ from request-slot order."""
    from vllm.model_executor.layers.utils import apply_penalties as reference

    from vllm_metal.pytorch_backend.omni_logits import apply_penalties, bincount

    logits = torch.tensor([[2.0, -3.0, 4.0, -5.0], [-1.0, 2.0, -3.0, 4.0]])
    prompts = torch.tensor([[0, 1], [2, 3]])
    outputs = torch.tensor([[1, 1], [0, 3]])
    rep, freq, pres = (
        torch.tensor([1.2, 1.5]),
        torch.zeros(2),
        torch.zeros(2),
    )
    with torch.compiler.set_stance("force_eager"):
        expected = reference(
            logits.clone(), prompts.flip(0), outputs.flip(0), pres, freq, rep.flip(0)
        )
    ids = torch.cat([prompts, outputs], dim=1).to("mps")
    mask = torch.zeros(2, 1, dtype=torch.int32, device="mps")
    counts = torch.zeros(2, 4, dtype=torch.int32, device="mps")
    mapping = torch.tensor([1, 0], device="mps")
    bincount(mapping, ids, torch.tensor([2, 2]), torch.tensor([4, 4]), mask, counts, 4)
    actual = logits.to("mps")
    apply_penalties(
        actual,
        mapping,
        None,
        None,
        rep.to("mps"),
        freq.to("mps"),
        pres.to("mps"),
        mask,
        counts,
    )
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("with_metadata", [False, True])
def test_omni_macos_transport_handles_long_names_and_page_padding(with_metadata):
    pytest.importorskip("vllm_omni")
    from vllm_metal.pytorch_backend.omni_connector import MetalSharedMemoryConnector

    connector = MetalSharedMemoryConnector({})
    key = "speech-" + "long-request-id-" * 8
    try:
        ok, size, metadata = connector.put("0", "1", key, {"frame": 7})
        assert ok
        connector.reap_consumed()
        result = connector.get("0", "1", key, metadata if with_metadata else None)
        assert result == ({"frame": 7}, size)
        assert connector.get("0", "1", key) is None
    finally:
        connector.close()


@pytest.mark.parametrize(
    "length", [1, 5, 64, 65, 200], ids=["one", "short", "chunk", "chunk+1", "multi"]
)
def test_gdn_chunked_prefill_matches_transformers_recurrence(length):
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_recurrent_gated_delta_rule,
    )

    from vllm_metal.pytorch_backend.models.qwen3_5 import delta_rule_chunked

    heads, k_dim, v_dim = 3, 32, 16
    q, k = (
        torch.nn.functional.normalize(torch.randn(length, heads, k_dim), dim=-1)
        for _ in range(2)
    )
    v = torch.randn(length, heads, v_dim)
    g = -torch.rand(length, heads)
    beta = torch.rand(length, heads)
    state = torch.randn(heads, k_dim, v_dim) * 0.1
    expected, expected_state = torch_recurrent_gated_delta_rule(
        q[None], k[None], v[None], g[None], beta[None], state[None], True
    )
    out, final = delta_rule_chunked(
        *(x.to("mps") for x in (q * k_dim**-0.5, k, v, g, beta, state))
    )
    # The chunked form reorders the recurrence's fp32 sums.
    torch.testing.assert_close(out.cpu(), expected[0], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(final.cpu(), expected_state[0], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "pooled",
    [
        pytest.param(False, id="dense"),
        pytest.param(True, id="pooled", marks=pytest.mark.slow),
    ],
)
def test_gdn_state_carries_across_chunked_prefill_and_decode(pooled):
    """``pooled`` puts the caches at the start of one buffer holding more than
    INT_MAX fp32 elements, like the shared KV buffer on a large Mac. MPSGraph
    matmuls reject a view into it at a non-zero offset, so resumed prefill must
    copy its slot. Metal wires that whole buffer on first use, hence slow."""
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    from vllm_metal.pytorch_backend.models.qwen3_5 import _gdn_core
    from vllm_metal.pytorch_backend.mps_ops import _load_mps_module

    pool_bytes = (2**31 + (1 << 20)) * 4
    if pooled and (
        _load_mps_module().max_buffer_length() <= pool_bytes
        or torch.mps.recommended_max_memory() < 2 * pool_bytes
    ):
        pytest.skip("Needs a Metal buffer past INT_MAX fp32 elements, with headroom")
    k_heads, v_heads, k_dim, v_dim, width = 2, 4, 16, 8, 4
    channels = 2 * k_heads * k_dim + v_heads * v_dim
    layer = SimpleNamespace(
        key_dim=k_heads * k_dim,
        value_dim=v_heads * v_dim,
        tp_size=1,
        head_k_dim=k_dim,
        head_v_dim=v_dim,
        activation="silu",
        conv1d=SimpleNamespace(
            weight=torch.randn(channels, 1, width, device="mps"), bias=None
        ),
        A_log=torch.randn(v_heads, device="mps"),
        dt_bias=torch.randn(v_heads, device="mps"),
        kv_cache=[
            torch.randn(4, width - 1, channels, device="mps"),
            torch.randn(4, v_heads, v_dim, k_dim, device="mps"),
        ],
    )
    if pooled:
        pool = torch.empty(pool_bytes // 4, device="mps")
        start = 0
        for i, t in enumerate(layer.kv_cache):
            layer.kv_cache[i] = pool[start : start + t.numel()].view_as(t).copy_(t)
            start += t.numel()
    untouched = [t[2].clone() for t in layer.kv_cache]
    qkv = torch.randn(76, channels, device="mps")
    b, a = torch.randn(2, 76, v_heads, device="mps")

    def step(lo, hi, decode, resumed):
        # A padded decode row rides along with every prefill; it must stay zero.
        x = torch.cat([qkv[:1] * 0, qkv[lo:hi]])
        md = SimpleNamespace(
            num_decodes=2 if decode else 1,
            num_decode_tokens=2 if decode else 1,
            num_prefills=0 if decode else 1,
            num_prefill_tokens=0 if decode else hi - lo,
            non_spec_state_indices_tensor=torch.tensor([PAD_SLOT_ID, 1], device="mps"),
            non_spec_query_start_loc=torch.tensor([0, 1, 1 + hi - lo], device="mps"),
            has_initial_state=torch.tensor([False, resumed], device="mps"),
        )
        out = torch.zeros(len(x), v_heads, v_dim, device="mps")
        _gdn_core(
            layer,
            x,
            torch.cat([b[:1], b[lo:hi]]),
            torch.cat([a[:1], a[lo:hi]]),
            out,
            md,
        )
        assert torch.equal(out[0], torch.zeros_like(out[0]))
        return out[1:].cpu()

    got = torch.cat(
        [step(0, 70, False, False), step(70, 75, False, True), step(75, 76, True, True)]
    )

    weight = layer.conv1d.weight[:, 0].cpu()
    tail, state = torch.zeros(width - 1, channels), torch.zeros(v_heads, k_dim, v_dim)
    for t in range(76):
        hist = torch.cat([tail, qkv[t : t + 1].cpu()])
        x, tail = torch.nn.functional.silu((hist.T * weight).sum(-1)), hist[1:]
        q, k, v = x.split([k_heads * k_dim, k_heads * k_dim, v_heads * v_dim])
        q, k = (
            torch.nn.functional.normalize(
                y.view(k_heads, k_dim), dim=-1
            ).repeat_interleave(2, 0)
            for y in (q, k)
        )
        g = -torch.exp(layer.A_log.cpu()) * torch.nn.functional.softplus(
            a[t].cpu() + layer.dt_bias.cpu()
        )
        state = state * torch.exp(g)[:, None, None]
        delta = (
            v.view(v_heads, v_dim) - torch.einsum("hk,hkv->hv", k, state)
        ) * torch.sigmoid(b[t].cpu())[:, None]
        state = state + k[:, :, None] * delta[:, None, :]
        torch.testing.assert_close(
            got[t],
            torch.einsum("hk,hkv->hv", q * k_dim**-0.5, state),
            atol=1e-4,
            rtol=1e-4,
        )
    torch.testing.assert_close(
        layer.kv_cache[1][1].cpu().view(v_heads, k_dim, v_dim),
        state,
        atol=1e-4,
        rtol=1e-4,
    )
    assert all(
        torch.equal(t[2], u) for t, u in zip(layer.kv_cache, untouched, strict=True)
    )


def test_seeded_sampling_preserves_distribution_and_batch_isolation(
    run_in_spawn_process,
):
    # vLLM registers overrides at worker startup, before optional libraries load.
    run_in_spawn_process(_check_seeded_sampling)


def _check_seeded_sampling():
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    from vllm_metal.pytorch_backend.bindings import register_kernel_overrides

    register_kernel_overrides()
    rows = 8192
    # Cross a reduction-block boundary, including a partial final block.
    token_ids = torch.tensor([0, 1023, 1024])
    logits = torch.full((1, 1025), -float("inf"))
    logits[:, token_ids] = torch.tensor([0.2, 0.3, 0.5]).log()
    logits = logits.to("mps").expand(rows, -1)
    mapping = torch.zeros(rows, dtype=torch.int32, device="mps")
    positions = torch.arange(rows, device="mps")
    seed = torch.tensor([42], device="mps")
    temperature = torch.ones(1, device="mps")
    actual = gumbel_sample(logits, mapping, temperature, seed, positions, False, False)
    frequencies = torch.bincount(actual.cpu(), minlength=1025)[token_ids] / rows
    torch.testing.assert_close(
        frequencies, torch.tensor([0.2, 0.3, 0.5]), atol=0.02, rtol=0
    )
    # Request scheduling must not change the random stream for a given position.
    order = torch.arange(rows - 1, -1, -1, device="mps")
    reordered = gumbel_sample(
        logits, mapping, temperature, seed, positions[order], False, False
    )
    assert torch.equal(actual[order], reordered)
    greedy = gumbel_sample(
        logits, mapping, temperature * 0, seed, positions, False, False
    )
    assert greedy.eq(1024).all()
    with pytest.raises(NotImplementedError, match="probabilistic draft sampling"):
        gumbel_sample(
            logits[:1],
            mapping[:1],
            temperature,
            seed,
            positions[:1],
            apply_temperature=True,
            is_drafting=True,
            logits_cache=logits[:1, None, :],
            logits_cache_col=positions[:1],
        )


def test_mrv2_sampling_filters_preserve_request_mapping(run_in_spawn_process):
    run_in_spawn_process(_check_mrv2_sampling_filters)


def _check_mrv2_sampling_filters():
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    from vllm_metal.pytorch_backend.bindings import register_kernel_overrides

    register_kernel_overrides()
    install()
    state = RequestState(4, 64, 16, 0, 64, torch.device("mps"))
    state.prompt_len.np[:] = 10
    state.prompt_len.copy_to_uva()
    sampler = Sampler(
        SimpleNamespace(reasoning_config=None), 4, 5, torch.device("mps"), state
    )
    params = [
        SamplingParams(temperature=0),
        SamplingParams(temperature=0.5, top_k=2, top_p=0.95, seed=17),
        SamplingParams(
            temperature=1,
            top_k=3,
            top_p=0.7,
            min_p=0.2,
            seed=42,
        ),
    ]
    for slot, param in enumerate(params):
        sampler.add_request(slot, param)
    sampler.apply_staged_writes()
    slots = np.array([2, 0, 1])
    mapping = torch.tensor(slots, device="mps", dtype=torch.int32)
    logits = torch.tensor(
        [
            [2.0, 3.0, 4.0, 5.0, 1.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 4.0, 4.0, 0.0, 0.0],
        ],
        device="mps",
    )
    zeros = torch.zeros(3, device="mps", dtype=torch.int32)

    def sample(position):
        return sampler.sample(
            logits,
            mapping,
            mapping,
            slots,
            torch.full((3,), position, device="mps"),
            zeros,
            zeros,
            np.full(3, position + 1),
        )

    first, processed = sample(10)
    assert first[:2].tolist() == [3, 4]
    assert first[2].item() in (1, 2)
    assert processed[0].isfinite().tolist() == [False, False, False, True, False]
    assert processed[2, 1:3].tolist() == [8.0, 8.0]
    assert processed[2].isfinite().tolist() == [False, True, True, False, False]
    assert torch.equal(sample(10)[0], first)
