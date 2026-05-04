# SPDX-License-Identifier: Apache-2.0
"""Tests for MetalStructuredOutputApplier.apply_paged and grammar integration in
execute_model / sample_tokens."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.sampler import Sampler

import vllm_metal.v1.model_runner as mr
import vllm_metal.v1.structured_output as so
from tests.stub_runner import make_stub_runner
from vllm_metal.v1.structured_output import MetalStructuredOutputApplier

# Shared instance used by unit tests that call apply_paged directly.
_applier = MetalStructuredOutputApplier()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 64
NUM_BITMASK_WORDS = math.ceil(VOCAB_SIZE / 32)  # 2 int32 words for vocab=64


def _make_full_bitmask(num_rows: int = 1, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """All tokens allowed (all bits = 1, i.e. fill_value=-1 == 0xFFFFFFFF)."""
    words = math.ceil(vocab_size / 32)
    return np.full((num_rows, words), fill_value=-1, dtype=np.int32)


def _make_single_token_bitmask(
    token_id: int, vocab_size: int = VOCAB_SIZE
) -> np.ndarray:
    """Bitmask that allows only `token_id`, forbids everything else."""
    words = math.ceil(vocab_size / 32)
    bitmask = np.zeros((1, words), dtype=np.int32)
    word_idx = token_id // 32
    bit_idx = token_id % 32
    bitmask[0, word_idx] = 1 << bit_idx
    return bitmask


def _make_scheduler_output(
    req_ids: list[str],
    *,
    has_structured_output_requests: bool = False,
) -> SimpleNamespace:
    """Minimal SchedulerOutput stub — no spec-decode tokens."""
    return SimpleNamespace(
        scheduled_spec_decode_tokens={},
        num_scheduled_tokens=dict.fromkeys(req_ids, 1),
        total_num_scheduled_tokens=len(req_ids),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        ),
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        preempted_req_ids=set(),
        has_structured_output_requests=has_structured_output_requests,
    )


def _make_grammar_output(
    req_ids: list[str],
    bitmask: np.ndarray,
) -> SimpleNamespace:
    """Minimal GrammarOutput stub."""
    return SimpleNamespace(
        structured_output_request_ids=req_ids,
        grammar_bitmask=bitmask,
    )


def _uniform_logits_2d(batch: int, vocab: int = VOCAB_SIZE) -> mx.array:
    return mx.zeros((batch, vocab))


def _uniform_logits_3d(total_tokens: int, vocab: int = VOCAB_SIZE) -> mx.array:
    """Paged-path logits: shape (1, total_tokens, vocab)."""
    return mx.zeros((1, total_tokens, vocab))


def _to_numpy(arr: mx.array) -> np.ndarray:
    """Materialise an MLX array to numpy (triggers lazy evaluation)."""
    return np.array(arr)


def _make_decode_req(req_id: str) -> tuple[str, SimpleNamespace]:
    """Minimal (req_id, RequestState) pair for a decode request."""
    state = SimpleNamespace(
        token_ids=[1, 2, 3],
        prompt_len=2,
        cache=[],
        sampling_params=None,
        generator=None,
        generated_tokens=1,
    )
    return (req_id, state)


def _make_prefill_req(req_id: str, num_tokens: int) -> SimpleNamespace:
    """Minimal PrefillRequest for a prefill request."""
    return SimpleNamespace(
        req_id=req_id,
        token_ids=list(range(num_tokens)),
        block_ids=[],
        start_pos=0,
        prompt_len=num_tokens,
        full_prompt_token_ids=None,
        generator=None,
        sampling_params=None,
    )


def _build_cu_seqlens(num_decode: int, prefill_lens: list[int]) -> list[int]:
    """Build cumulative sequence lengths matching _start_paged_forward logic."""
    cu = [0]
    for _ in range(num_decode):
        cu.append(cu[-1] + 1)
    for length in prefill_lens:
        cu.append(cu[-1] + length)
    return cu


# ---------------------------------------------------------------------------
# MetalStructuredOutputApplier.apply_paged — 3D paged-path helper
# ---------------------------------------------------------------------------


class TestApplyGrammarBitmaskPaged:
    def test_decode_only_forbidden_tokens_neg_inf(self) -> None:
        """Decode request: forbidden tokens become -inf in the correct row."""
        allowed_token = 5
        # 2 decode requests → logits shape (1, 2, vocab)
        logits = _uniform_logits_3d(2)
        decode_reqs = [_make_decode_req("r0"), _make_decode_req("r1")]
        cu = _build_cu_seqlens(num_decode=2, prefill_lens=[])
        sched = _make_scheduler_output(["r0", "r1"])
        grammar = _make_grammar_output(
            ["r0"], _make_single_token_bitmask(allowed_token)
        )

        result = _to_numpy(
            _applier.apply_paged(sched, grammar, decode_reqs, [], cu, 2, logits)
        )

        # Row 0 (r0): only allowed_token survives
        assert np.isfinite(result[0, 0, allowed_token])
        assert result[0, 0, (allowed_token + 1) % VOCAB_SIZE] == float("-inf")
        # Row 1 (r1): unconstrained — all finite
        assert np.all(np.isfinite(result[0, 1]))

    def test_decode_row_mapping_correct(self) -> None:
        """Grammar is applied to the correct decode row (not always row 0)."""
        allowed_in_r1 = 11
        logits = _uniform_logits_3d(2)
        decode_reqs = [_make_decode_req("r0"), _make_decode_req("r1")]
        cu = _build_cu_seqlens(num_decode=2, prefill_lens=[])
        sched = _make_scheduler_output(["r0", "r1"])
        grammar = _make_grammar_output(
            ["r1"], _make_single_token_bitmask(allowed_in_r1)
        )

        result = _to_numpy(
            _applier.apply_paged(sched, grammar, decode_reqs, [], cu, 2, logits)
        )

        # Row 0 (r0): unconstrained
        assert np.all(np.isfinite(result[0, 0]))
        # Row 1 (r1): constrained
        assert np.isfinite(result[0, 1, allowed_in_r1])
        assert result[0, 1, (allowed_in_r1 + 1) % VOCAB_SIZE] == float("-inf")

    def test_prefill_last_token_row_constrained(self) -> None:
        """Grammar bitmask is applied to the last token row of a prefill request."""
        allowed_token = 9
        prefill_len = 4  # tokens at rows 0,1,2,3 in logits[0]
        # Only prefill, no decode. total_tokens = 4.
        logits = _uniform_logits_3d(prefill_len)
        prefill_reqs = [_make_prefill_req("p0", prefill_len)]
        cu = _build_cu_seqlens(num_decode=0, prefill_lens=[prefill_len])
        sched = _make_scheduler_output(["p0"])
        grammar = _make_grammar_output(
            ["p0"], _make_single_token_bitmask(allowed_token)
        )

        result = _to_numpy(
            _applier.apply_paged(sched, grammar, [], prefill_reqs, cu, 0, logits)
        )

        # Only the LAST row (row 3) should be constrained
        last_row = prefill_len - 1
        assert np.isfinite(result[0, last_row, allowed_token])
        assert result[0, last_row, (allowed_token + 1) % VOCAB_SIZE] == float("-inf")
        # Earlier rows untouched
        for row in range(last_row):
            assert np.all(np.isfinite(result[0, row]))

    def test_mixed_decode_and_prefill(self) -> None:
        """Grammar constraints work correctly across decode + prefill in one batch."""
        allowed_decode = 7
        allowed_prefill = 15
        # 1 decode (row 0) + 1 prefill of 3 tokens (rows 1,2,3)
        logits = _uniform_logits_3d(4)
        decode_reqs = [_make_decode_req("d0")]
        prefill_reqs = [_make_prefill_req("p0", 3)]
        cu = _build_cu_seqlens(num_decode=1, prefill_lens=[3])
        sched = _make_scheduler_output(["d0", "p0"])
        bitmask = np.vstack(
            [
                _make_single_token_bitmask(allowed_decode),
                _make_single_token_bitmask(allowed_prefill),
            ]
        )
        grammar = _make_grammar_output(["d0", "p0"], bitmask)

        result = _to_numpy(
            _applier.apply_paged(
                sched, grammar, decode_reqs, prefill_reqs, cu, 1, logits
            )
        )

        # Decode row 0 (d0)
        assert np.isfinite(result[0, 0, allowed_decode])
        assert result[0, 0, (allowed_decode + 1) % VOCAB_SIZE] == float("-inf")
        # Prefill last token row 3 (p0)
        assert np.isfinite(result[0, 3, allowed_prefill])
        assert result[0, 3, (allowed_prefill + 1) % VOCAB_SIZE] == float("-inf")
        # Prefill intermediate rows 1,2 untouched
        for row in [1, 2]:
            assert np.all(np.isfinite(result[0, row]))

    def test_no_constrained_requests_returns_unchanged_logits(self) -> None:
        """If no scheduled request has grammar constraints, logits are returned as-is."""
        data = np.random.randn(1, 2, VOCAB_SIZE).astype(np.float32)
        logits = mx.array(data)
        decode_reqs = [_make_decode_req("r0"), _make_decode_req("r1")]
        cu = _build_cu_seqlens(num_decode=2, prefill_lens=[])
        sched = _make_scheduler_output(["r0", "r1"])
        # Grammar output references a request not in this batch
        grammar = _make_grammar_output(["absent"], _make_single_token_bitmask(0))

        result = _to_numpy(
            _applier.apply_paged(sched, grammar, decode_reqs, [], cu, 2, logits)
        )

        np.testing.assert_allclose(result, data, rtol=1e-5)

    def test_empty_structured_output_request_ids_returns_unchanged(self) -> None:
        """GrammarOutput with empty request list must not touch logits."""
        data = np.random.randn(1, 1, VOCAB_SIZE).astype(np.float32)
        logits = mx.array(data)
        decode_reqs = [_make_decode_req("r0")]
        cu = _build_cu_seqlens(num_decode=1, prefill_lens=[])
        sched = _make_scheduler_output(["r0"])
        # empty structured_output_request_ids — grammar engine sent a no-op
        grammar = _make_grammar_output(
            [], np.zeros((0, NUM_BITMASK_WORDS), dtype=np.int32)
        )

        result = _to_numpy(
            _applier.apply_paged(sched, grammar, decode_reqs, [], cu, 1, logits)
        )

        np.testing.assert_allclose(result, data, rtol=1e-5)

    def test_dtype_preserved(self) -> None:
        """Output dtype must match the input dtype."""
        logits = mx.zeros((1, 1, VOCAB_SIZE), dtype=mx.float16)
        decode_reqs = [_make_decode_req("r0")]
        cu = _build_cu_seqlens(num_decode=1, prefill_lens=[])
        sched = _make_scheduler_output(["r0"])
        grammar = _make_grammar_output(["r0"], _make_single_token_bitmask(0))

        result = _applier.apply_paged(sched, grammar, decode_reqs, [], cu, 1, logits)

        assert result.dtype == mx.float16

    def test_raises_if_xgrammar_missing(self) -> None:
        logits = _uniform_logits_3d(1)
        decode_reqs = [_make_decode_req("r0")]
        cu = _build_cu_seqlens(num_decode=1, prefill_lens=[])
        sched = _make_scheduler_output(["r0"])
        grammar = _make_grammar_output(["r0"], _make_single_token_bitmask(0))

        with patch.object(so, "xgr", None):
            with pytest.raises(RuntimeError, match="xgrammar is required"):
                _applier.apply_paged(sched, grammar, decode_reqs, [], cu, 1, logits)

    def test_rejects_2d_logits(self) -> None:
        """The paged helper must reject 2D logits with a clear assertion."""
        logits_2d = _uniform_logits_2d(2)
        decode_reqs = [_make_decode_req("r0")]
        cu = _build_cu_seqlens(num_decode=1, prefill_lens=[])
        sched = _make_scheduler_output(["r0"])
        grammar = _make_grammar_output(["r0"], _make_single_token_bitmask(0))

        with pytest.raises(AssertionError):
            _applier.apply_paged(sched, grammar, decode_reqs, [], cu, 1, logits_2d)

    def test_batch_order_independent_of_grammar_output_order(self) -> None:
        """Bitmask must reach the correct row even when grammar_output order differs."""
        allowed_r0 = 3
        allowed_r1 = 19
        # bitmask is ordered [r1, r0] but batch is [r0, r1]
        bitmask = np.vstack(
            [
                _make_single_token_bitmask(allowed_r1),
                _make_single_token_bitmask(allowed_r0),
            ]
        )
        logits = _uniform_logits_3d(2)
        decode_reqs = [_make_decode_req("r0"), _make_decode_req("r1")]
        cu = _build_cu_seqlens(num_decode=2, prefill_lens=[])
        sched = _make_scheduler_output(["r0", "r1"])
        grammar = _make_grammar_output(["r1", "r0"], bitmask)

        result = _to_numpy(
            _applier.apply_paged(sched, grammar, decode_reqs, [], cu, 2, logits)
        )

        assert np.isfinite(result[0, 0, allowed_r0])
        assert result[0, 0, (allowed_r0 + 1) % VOCAB_SIZE] == float("-inf")
        assert np.isfinite(result[0, 1, allowed_r1])
        assert result[0, 1, (allowed_r1 + 1) % VOCAB_SIZE] == float("-inf")

    def test_dtype_preserved_for_bfloat16(self) -> None:
        logits = mx.zeros((1, 1, VOCAB_SIZE), dtype=mx.bfloat16)
        decode_reqs = [_make_decode_req("r0")]
        cu = _build_cu_seqlens(num_decode=1, prefill_lens=[])
        sched = _make_scheduler_output(["r0"])
        grammar = _make_grammar_output(["r0"], _make_single_token_bitmask(0))

        result = _applier.apply_paged(sched, grammar, decode_reqs, [], cu, 1, logits)

        assert result.dtype == mx.bfloat16

    def test_input_array_not_mutated(self) -> None:
        """Caller-held input mx.array must be unchanged after the call."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((1, 2, VOCAB_SIZE)).astype(np.float32)
        logits = mx.array(data)
        decode_reqs = [_make_decode_req("r0"), _make_decode_req("r1")]
        cu = _build_cu_seqlens(num_decode=2, prefill_lens=[])
        sched = _make_scheduler_output(["r0", "r1"])
        grammar = _make_grammar_output(["r0"], _make_single_token_bitmask(5))

        _applier.apply_paged(sched, grammar, decode_reqs, [], cu, 2, logits)

        np.testing.assert_array_equal(_to_numpy(logits), data)

    def test_non_constrained_rows_unchanged_at_float16(self) -> None:
        """Non-constrained rows must round-trip bit-identically at fp16."""
        rng = np.random.default_rng(7)
        data = rng.standard_normal((1, 2, VOCAB_SIZE)).astype(np.float16)
        logits = mx.array(data)
        decode_reqs = [_make_decode_req("r0"), _make_decode_req("r1")]
        cu = _build_cu_seqlens(num_decode=2, prefill_lens=[])
        sched = _make_scheduler_output(["r0", "r1"])
        # Only r0 is constrained; r1 (row 1) must come back unchanged.
        grammar = _make_grammar_output(["r0"], _make_single_token_bitmask(5))

        result = _to_numpy(
            _applier.apply_paged(sched, grammar, decode_reqs, [], cu, 2, logits)
        )

        np.testing.assert_array_equal(result[0, 1], data[0, 1])

    def test_non_constrained_rows_unchanged_at_bfloat16(self) -> None:
        """Non-constrained rows must round-trip bit-identically at bfloat16."""
        rng = np.random.default_rng(13)
        data_fp32 = rng.standard_normal((1, 2, VOCAB_SIZE)).astype(np.float32)
        logits = mx.array(data_fp32).astype(mx.bfloat16)
        decode_reqs = [_make_decode_req("r0"), _make_decode_req("r1")]
        cu = _build_cu_seqlens(num_decode=2, prefill_lens=[])
        sched = _make_scheduler_output(["r0", "r1"])
        grammar = _make_grammar_output(["r0"], _make_single_token_bitmask(5))

        result = _applier.apply_paged(sched, grammar, decode_reqs, [], cu, 2, logits)

        # Row 1 (r1) is unconstrained — values must match the input row exactly.
        np.testing.assert_array_equal(
            np.array(result[0, 1].astype(mx.float32)),
            np.array(logits[0, 1].astype(mx.float32)),
        )

    def test_multiple_prefill_requests_correct_last_row(self) -> None:
        """cu_seqlens index must identify the last row for each prefill correctly."""
        # 1 decode (row 0) + prefill-0 of 3 tokens (rows 1,2,3)
        #                   + prefill-1 of 5 tokens (rows 4,5,6,7,8)
        allowed_p0 = 7
        allowed_p1 = 13
        total_tokens = 1 + 3 + 5
        logits = _uniform_logits_3d(total_tokens)
        decode_reqs = [_make_decode_req("d0")]
        prefill_reqs = [_make_prefill_req("p0", 3), _make_prefill_req("p1", 5)]
        cu = _build_cu_seqlens(num_decode=1, prefill_lens=[3, 5])
        sched = _make_scheduler_output(["d0", "p0", "p1"])
        bitmask = np.vstack(
            [
                _make_single_token_bitmask(allowed_p0),
                _make_single_token_bitmask(allowed_p1),
            ]
        )
        grammar = _make_grammar_output(["p0", "p1"], bitmask)

        result = _to_numpy(
            _applier.apply_paged(
                sched, grammar, decode_reqs, prefill_reqs, cu, 1, logits
            )
        )

        # d0 row 0: unconstrained
        assert np.all(np.isfinite(result[0, 0]))
        # p0 last row = 1 + 3 - 1 = 3
        assert np.isfinite(result[0, 3, allowed_p0])
        assert result[0, 3, (allowed_p0 + 1) % VOCAB_SIZE] == float("-inf")
        # p1 last row = 1 + 3 + 5 - 1 = 8
        assert np.isfinite(result[0, 8, allowed_p1])
        assert result[0, 8, (allowed_p1 + 1) % VOCAB_SIZE] == float("-inf")
        # intermediate rows untouched
        for row in [1, 2, 4, 5, 6, 7]:
            assert np.all(np.isfinite(result[0, row]))

    def test_spec_decode_raises_when_constrained_req_has_spec_tokens(self) -> None:
        """Paged helper must raise when a structured-output request has spec-decode."""
        logits = _uniform_logits_3d(2)
        decode_reqs = [_make_decode_req("r0")]
        cu = _build_cu_seqlens(num_decode=1, prefill_lens=[])
        sched = SimpleNamespace(
            scheduled_spec_decode_tokens={"r0": [9]},
            num_scheduled_tokens={"r0": 2},
            total_num_scheduled_tokens=2,
            finished_req_ids=set(),
        )
        grammar = _make_grammar_output(["r0"], _make_single_token_bitmask(0))

        with pytest.raises(NotImplementedError, match="speculative decoding"):
            _applier.apply_paged(sched, grammar, decode_reqs, [], cu, 1, logits)

    def test_spec_decode_plain_req_does_not_block_structured_req(self) -> None:
        """Plain req with spec tokens must not block a structured-output req.

        r1 (plain) has spec tokens; r0 (structured) does not.
        The guard must allow this batch — only raise when the structured-output
        request itself overlaps with spec-decode.
        """
        allowed_token = 5
        logits = _uniform_logits_3d(3)
        decode_reqs = [_make_decode_req("r0"), _make_decode_req("r1")]
        cu = _build_cu_seqlens(num_decode=2, prefill_lens=[])
        sched = SimpleNamespace(
            scheduled_spec_decode_tokens={"r1": [99]},
            num_scheduled_tokens={"r0": 1, "r1": 2},
            total_num_scheduled_tokens=3,
            finished_req_ids=set(),
        )
        grammar = _make_grammar_output(
            ["r0"], _make_single_token_bitmask(allowed_token)
        )

        # Must not raise — r0 (structured) has no spec tokens.
        result = _applier.apply_paged(sched, grammar, decode_reqs, [], cu, 2, logits)

        result_np = _to_numpy(result)
        # r0 (row 0) is constrained to allowed_token
        assert np.isfinite(result_np[0, 0, allowed_token])
        assert result_np[0, 0, (allowed_token + 1) % VOCAB_SIZE] == float("-inf")
        # r1 (row 1) is unconstrained — all logits finite
        assert np.all(np.isfinite(result_np[0, 1]))
        # row 2 is r1's spec-token slot; it is outside grammar_output.structured_output_request_ids
        # so it receives no masking — intentionally not asserted here.


# ---------------------------------------------------------------------------
# execute_model — non-paged path raises early for structured output
# ---------------------------------------------------------------------------


class TestSampleTokensGrammarNonPagedPath:
    def _make_runner(self) -> mr.MetalModelRunner:
        return make_stub_runner()

    def test_non_paged_path_raises_early_in_execute_model(self) -> None:
        """Non-paged path must raise NotImplementedError in execute_model before
        any forward pass runs, when structured output is requested."""
        runner = self._make_runner()
        scheduler_output = _make_scheduler_output(
            [], has_structured_output_requests=True
        )

        with pytest.raises(NotImplementedError, match="non-paged"):
            runner.execute_model(scheduler_output)

    def test_non_paged_path_no_raise_without_grammar(self) -> None:
        """Non-paged path with grammar_output=None must return output normally."""
        runner = self._make_runner()
        pending = ModelRunnerOutput(
            req_ids=["r0"],
            req_id_to_index={"r0": 0},
            sampled_token_ids=[[5]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None],
        )
        runner._pending_output = pending

        out = runner.sample_tokens(grammar_output=None)
        assert out is pending


# ---------------------------------------------------------------------------
# sample_tokens — paged path applies grammar bitmask before sampling
# ---------------------------------------------------------------------------


class TestSampleTokensGrammarPagedPath:
    """Integration tests for the paged decode path: execute_model → sample_tokens."""

    def test_grammar_constraint_applied_on_paged_decode(self) -> None:
        """Grammar bitmask must constrain greedy sampling on the paged path.

        Injects _execute_model_state directly (no live model available in unit
        tests) to isolate the sample_tokens bitmask-then-sample ordering.
        Covers the state that execute_model leaves behind after a paged forward.
        """
        vocab = 64
        allowed_token = 5

        runner = make_stub_runner(
            model_args={"vocab_size": vocab},
            _paged_request_seq_lens={},
        )
        runner._sampler = Sampler()

        req_state = mr.RequestState(
            token_ids=[1],
            prompt_len=1,
            cache=[],
            sampling_params=SamplingParams(temperature=0.0),
            generator=None,
            generated_tokens=0,
        )
        decode_reqs = [("r0", req_state)]

        # Uniform logits: without bitmask, greedy argmax = token 0.
        logits = mx.zeros((1, 1, vocab))

        scheduler_output = _make_scheduler_output(["r0"])
        grammar_output = _make_grammar_output(
            ["r0"], _make_single_token_bitmask(allowed_token)
        )

        # Simulate the paged forward state that execute_model leaves behind.
        batch = mr._ExecutionBatch()
        batch.paged_decode_reqs = decode_reqs

        runner._execute_model_state = mr._PagedForwardState(
            batch=batch,
            prefill_reqs=[],
            decode_reqs=decode_reqs,
            scheduler_output=scheduler_output,
            logits=logits,
            cu_seqlens=[0, 1],
            num_decode=1,
            spec_metadata=None,
        )

        output = runner.sample_tokens(grammar_output=grammar_output)

        assert output is not None
        assert output.sampled_token_ids[0][0] == allowed_token
