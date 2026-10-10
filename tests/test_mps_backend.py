# SPDX-License-Identifier: Apache-2.0
"""MPS request mapping, cache sizing and normalization parity."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.states import RequestState

from vllm_metal.pytorch_backend.input_ops import post_update
from vllm_metal.pytorch_backend.runtime import install

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


@pytest.fixture
def state():
    install()
    return RequestState(4, 64, 16, 0, 64, torch.device("mps"))


@pytest.mark.parametrize("num_groups", [1, 2])
def test_request_slots_and_removed_request_writeback(state, num_groups):
    tables = BlockTables(
        [16] * num_groups,
        4,
        16,
        [4] * num_groups,
        torch.device("mps"),
        [16] * num_groups,
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
    slots = tables.compute_slot_mappings(
        mapping,
        torch.tensor([0, 1, 4], dtype=torch.int32, device="mps"),
        torch.tensor([17, 14, 15, 16], dtype=torch.int64, device="mps"),
        num_tokens_padded=8,
    )
    expected_slots = [
        [65, 142, 143, 144, -1, -1, -1, -1],
        [113, 206, 207, 208, -1, -1, -1, -1],
    ]
    assert slots.tolist() == expected_slots[:num_groups]
    # The second request disappears before writeback. Its -1 slot must not
    # alias slot zero and overwrite the surviving request's sampled token.
    state.num_computed_tokens.gpu[0] = 17
    state.total_len.gpu[0] = 18
    state.last_sampled_tokens[0, 0] = 40
    post_update(
        torch.tensor([0, -1], dtype=torch.int32, device="mps"),
        state.num_computed_tokens.gpu,
        state.last_sampled_tokens,
        None,
        torch.tensor([[41], [999]], device="mps"),
        torch.ones(2, dtype=torch.int32, device="mps"),
        torch.zeros(2, dtype=torch.int32, device="mps"),
        torch.tensor([0, 1, 2], dtype=torch.int32, device="mps"),
        state.all_token_ids.gpu,
        state.total_len.gpu,
    )
    assert state.last_sampled_tokens[0, 0].item() == 41
    assert state.all_token_ids.gpu[0, 18].item() == 41
    assert state.total_len.gpu[0].item() == 19
    assert state.num_computed_tokens.gpu[0].item() == 18


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
def test_cache_budget_handles_model_dimensions(monkeypatch, config_name, fields, width):
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
    monkeypatch.setattr(torch.mps, "synchronize", lambda: None)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 128 << 20)
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(total=4 << 30))
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("fused", [False, True], ids=["plain", "residual"])
def test_rms_norm_parity(dtype, fused):
    from vllm import ir

    import vllm_metal.pytorch_backend.normalization  # noqa: F401

    op = ir.ops.fused_add_rms_norm if fused else ir.ops.rms_norm
    args = op.generate_inputs(num_tokens=4, hidden_size=128, dtype=dtype, device="cpu")
    args[-2].copy_(torch.tensor([1.25, 1.75, 0.75, 1.5], dtype=dtype).repeat(32))
    if fused:
        # This row distinguishes FP32 accumulation and cast-before-weight from
        # rounding the sum early or applying the learned weight before casting.
        args[0][0].fill_(1)
        args[1][0] = (
            torch.tensor([0.125, 0.375, 0.625, 0.875], dtype=dtype).repeat(32)
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
    torch.testing.assert_close(actual.cpu(), expected, **op.get_tolerance(dtype))


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
