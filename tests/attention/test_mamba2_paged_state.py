# SPDX-License-Identifier: Apache-2.0
"""Mamba-2 wrapper contract: the unmodified mlx_lm mixer over paged state rows."""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.nemotron_h import ModelArgs, NemotronHMamba2Mixer

from tests.stub_runner import NEMOTRON_H_TINY_ARGS
from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.attention.context import (
    PagedAttentionContext,
    clear_context,
    set_context,
)
from vllm_metal.attention.impls.mamba2 import Mamba2PagedStateWrapper

_HIDDEN = 32


@pytest.fixture(autouse=True)
def _no_context() -> Iterator[None]:
    clear_context()
    yield
    clear_context()


def _make_mixer(**overrides: object) -> NemotronHMamba2Mixer:
    mx.random.seed(0)
    mixer = NemotronHMamba2Mixer(ModelArgs(**{**NEMOTRON_H_TINY_ARGS, **overrides}))
    mx.eval(mixer.parameters())
    return mixer


def _make_cache(
    mixer: NemotronHMamba2Mixer, dtype: mx.Dtype, num_layers: int = 1
) -> GDNPagedStateCache:
    return GDNPagedStateCache(
        num_layers=num_layers,
        max_seqs=3,
        conv_kernel_dim=mixer.conv_kernel_size,
        conv_dim=mixer.conv_dim,
        num_v_heads=mixer.num_heads,
        value_head_dim=mixer.head_dim,
        key_head_dim=mixer.ssm_state_size,
        initial_seqs=3,
        dtype=dtype,
    )


def _reference(
    mixer: NemotronHMamba2Mixer, x: mx.array, cache: ArraysCache | None = None
) -> tuple[mx.array, ArraysCache]:
    cache = ArraysCache(size=2) if cache is None else cache
    out = mixer(x, mask=None, cache=cache)
    mx.eval(out, cache[0], cache[1])
    return out, cache


def _stacked(caches: list[ArraysCache]) -> ArraysCache:
    """Batch per-request caches the way mlx_lm would run them together."""
    batched = ArraysCache(size=2)
    batched[0] = mx.concatenate([c[0] for c in caches], axis=0)
    batched[1] = mx.concatenate([c[1] for c in caches], axis=0)
    return batched


def _tokens(num_tokens: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    return mx.random.normal((1, num_tokens, _HIDDEN)).astype(dtype)


def _set_step(cu_seqlens: list[int], slots: list[int], num_decode: int) -> None:
    set_context(
        PagedAttentionContext(
            slot_mapping=[0] * cu_seqlens[-1],
            cu_seqlens=cu_seqlens,
            state_slot_mapping=slots,
            num_decode_requests=num_decode,
        )
    )


def _pool_rows(
    cache: GDNPagedStateCache, slot: int, cache_idx: int = 0
) -> tuple[mx.array, mx.array]:
    return cache.conv_states[cache_idx][slot], cache.recurrent_states[cache_idx][slot]


class TestMamba2PagedStateWrapper:
    def test_prefill_runs_each_request_on_its_own_slot(self) -> None:
        mixer = _make_mixer()
        cache = _make_cache(mixer, mx.float32)
        wrapper = Mamba2PagedStateWrapper(mixer, 0, 0, cache)
        first, second = _tokens(5), _tokens(7)
        ref_first, ref_first_cache = _reference(mixer, first)
        ref_second, ref_second_cache = _reference(mixer, second)
        _set_step([0, 5, 12], [2, 0], num_decode=0)

        out = wrapper(mx.concatenate([first, second], axis=1))
        mx.eval(out)

        assert mx.array_equal(out[:, :5], ref_first)
        assert mx.array_equal(out[:, 5:], ref_second)
        conv, ssm = _pool_rows(cache, 2)
        assert mx.array_equal(conv, ref_first_cache[0][0])
        assert mx.array_equal(ssm, ref_first_cache[1][0])
        conv, ssm = _pool_rows(cache, 0)
        assert mx.array_equal(conv, ref_second_cache[0][0])
        assert mx.array_equal(ssm, ref_second_cache[1][0])
        untouched_conv, untouched_ssm = _pool_rows(cache, 1)
        assert not np.any(np.array(untouched_conv))
        assert not np.any(np.array(untouched_ssm))

    def test_decode_runs_all_requests_as_one_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mixer = _make_mixer()
        mixer.set_dtype(mx.bfloat16)
        cache = _make_cache(mixer, mx.bfloat16)
        wrapper = Mamba2PagedStateWrapper(mixer, 0, 0, cache)
        first, second = _tokens(5, mx.bfloat16), _tokens(7, mx.bfloat16)
        _, ref_first_cache = _reference(mixer, first)
        _, ref_second_cache = _reference(mixer, second)
        _set_step([0, 5, 12], [2, 0], num_decode=0)
        wrapper(mx.concatenate([first, second], axis=1))
        step_first, step_second = _tokens(1, mx.bfloat16), _tokens(1, mx.bfloat16)
        # mlx_lm batches decode rows through one call, and MLX's batched GEMM
        # rounds differently from single rows, so the reference is that batch.
        ref_cache = _stacked([ref_first_cache, ref_second_cache])
        ref_out, ref_cache = _reference(
            mixer, mx.concatenate([step_first, step_second], axis=0), ref_cache
        )
        _set_step([0, 1, 2], [2, 0], num_decode=2)
        calls: list[tuple[int, ...]] = []
        real_call = NemotronHMamba2Mixer.__call__

        def spy(self, x, mask=None, cache=None):
            calls.append(tuple(x.shape))
            return real_call(self, x, mask=mask, cache=cache)

        monkeypatch.setattr(NemotronHMamba2Mixer, "__call__", spy)

        out = wrapper(mx.concatenate([step_first, step_second], axis=1))
        mx.eval(out)

        assert calls == [(2, 1, _HIDDEN)]
        assert mx.array_equal(out[:, :1], ref_out[:1])
        assert mx.array_equal(out[:, 1:], ref_out[1:])
        conv, ssm = _pool_rows(cache, 2)
        assert mx.array_equal(conv, ref_cache[0][0])
        assert mx.array_equal(ssm, ref_cache[1][0])
        conv, ssm = _pool_rows(cache, 0)
        assert mx.array_equal(conv, ref_cache[0][1])
        assert mx.array_equal(ssm, ref_cache[1][1])

    def test_mixed_step_keeps_packed_output_order(self) -> None:
        mixer = _make_mixer()
        cache = _make_cache(mixer, mx.float32)
        wrapper = Mamba2PagedStateWrapper(mixer, 0, 0, cache)
        resident = _tokens(5)
        _, ref_resident_cache = _reference(mixer, resident)
        _set_step([0, 5], [2], num_decode=0)
        wrapper(resident)
        step, fresh = _tokens(1), _tokens(4)
        ref_step, _ = _reference(mixer, step, ref_resident_cache)
        ref_fresh, _ = _reference(mixer, fresh)
        _set_step([0, 1, 5], [2, 1], num_decode=1)

        out = wrapper(mx.concatenate([step, fresh], axis=1))
        mx.eval(out)

        assert mx.array_equal(out[:, :1], ref_step)
        assert mx.array_equal(out[:, 1:], ref_fresh)

    def test_chunked_prefill_resumes_from_pool_state(self) -> None:
        mixer = _make_mixer()
        cache = _make_cache(mixer, mx.float32)
        wrapper = Mamba2PagedStateWrapper(mixer, 0, 0, cache)
        prompt = _tokens(12)
        ref_out, ref_cache = _reference(mixer, prompt)

        _set_step([0, 7], [1], num_decode=0)
        head = wrapper(prompt[:, :7])
        _set_step([0, 5], [1], num_decode=0)
        tail = wrapper(prompt[:, 7:])
        out = mx.concatenate([head, tail], axis=1)
        mx.eval(out)

        conv, ssm = _pool_rows(cache, 1)
        assert mx.array_equal(conv, ref_cache[0][0])
        assert np.allclose(np.array(ssm), np.array(ref_cache[1][0]), atol=1e-6)
        assert np.allclose(np.array(out), np.array(ref_out), atol=1e-5)

    def test_no_context_delegates_to_the_mixer_unchanged(self) -> None:
        mixer = _make_mixer()
        cache = _make_cache(mixer, mx.float32)
        wrapper = Mamba2PagedStateWrapper(mixer, 0, 0, cache)
        prompt = _tokens(6)
        ref_out, _ = _reference(mixer, prompt)

        out = wrapper(prompt, cache=ArraysCache(size=2))
        mx.eval(out)

        assert mx.array_equal(out, ref_out)
        assert not np.any(np.array(cache.conv_states[0]))
        assert not np.any(np.array(cache.recurrent_states[0]))

    def test_paged_path_never_forwards_the_model_mask(self) -> None:
        mixer = _make_mixer()
        cache = _make_cache(mixer, mx.float32)
        wrapper = Mamba2PagedStateWrapper(mixer, 0, 0, cache)
        prompt = _tokens(6)
        ref_out, _ = _reference(mixer, prompt)
        _set_step([0, 6], [0], num_decode=0)

        out = wrapper(prompt, mask="causal")
        mx.eval(out)

        assert mx.array_equal(out, ref_out)

    def test_rebind_state_cache_repoints_the_pool_and_ordinal(self) -> None:
        mixer = _make_mixer()
        old_cache = _make_cache(mixer, mx.float32)
        new_cache = _make_cache(mixer, mx.float32, num_layers=2)
        wrapper = Mamba2PagedStateWrapper(mixer, 0, 0, old_cache)
        prompt = _tokens(3)
        _, ref_cache = _reference(mixer, prompt)
        _set_step([0, 3], [1], num_decode=0)

        wrapper.rebind_state_cache(new_cache, cache_idx=1)
        mx.eval(wrapper(prompt))

        conv, ssm = _pool_rows(new_cache, 1, cache_idx=1)
        assert mx.array_equal(conv, ref_cache[0][0])
        assert mx.array_equal(ssm, ref_cache[1][0])
        assert not np.any(np.array(new_cache.recurrent_states[0]))
        assert not np.any(np.array(old_cache.recurrent_states[0]))

    def test_rejects_ssm_state_size_off_the_kernel_grid(self) -> None:
        mixer = _make_mixer(ssm_state_size=48)
        cache = _make_cache(mixer, mx.float32)
        expected = (
            "Mamba-2 paged decode at layer 3 requires ssm_state_size to be a "
            "multiple of 32, got ssm_state_size=48."
        )

        with pytest.raises(NotImplementedError) as excinfo:
            Mamba2PagedStateWrapper(mixer, 3, 0, cache)
        assert str(excinfo.value) == expected

    def test_rejects_state_pool_dtype_that_differs_from_the_mixer(self) -> None:
        mixer = _make_mixer()
        cache = _make_cache(mixer, mx.bfloat16)

        with pytest.raises(ValueError, match="state pool dtype"):
            Mamba2PagedStateWrapper(mixer, 0, 0, cache)
