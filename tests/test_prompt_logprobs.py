# SPDX-License-Identifier: Apache-2.0
"""Prompt logprobs accounting and tensor construction."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import torch

from vllm_metal.v1.prompt_logprobs import (
    PromptLogprobsAccumulator,
    PromptLogprobsWindow,
    gather_prompt_logprobs,
    prompt_logprobs_window,
)


class TestPromptLogprobsWindow:
    def test_single_chunk_scores_every_prompt_token_but_the_first(self) -> None:
        window = prompt_logprobs_window(start_pos=0, num_tokens=5, prompt_len=5)

        assert window == PromptLogprobsWindow(0, 4, completes=True)
        assert window.first_target == 1

    def test_intermediate_chunk_scores_all_its_rows(self) -> None:
        window = prompt_logprobs_window(start_pos=0, num_tokens=3, prompt_len=8)

        assert window == PromptLogprobsWindow(0, 3, completes=False)

    def test_final_chunk_drops_the_row_that_predicts_the_sampled_token(self) -> None:
        window = prompt_logprobs_window(start_pos=3, num_tokens=5, prompt_len=8)

        # rows for positions 3..7 -> targets 4..7 (four prompt tokens)
        assert window == PromptLogprobsWindow(3, 4, completes=True)

    def test_chunk_ending_one_before_the_prompt_end_is_not_yet_complete(self) -> None:
        # Positions 0..6 forwarded for an 8-token prompt: the == case in vLLM.
        window = prompt_logprobs_window(start_pos=0, num_tokens=7, prompt_len=8)

        assert window == PromptLogprobsWindow(0, 7, completes=False)

        final = prompt_logprobs_window(start_pos=7, num_tokens=1, prompt_len=8)
        assert final == PromptLogprobsWindow(7, 0, completes=True)

    def test_one_token_prompt_has_nothing_to_score(self) -> None:
        window = prompt_logprobs_window(start_pos=0, num_tokens=1, prompt_len=1)

        assert window == PromptLogprobsWindow(0, 0, completes=True)


class TestGatherPromptLogprobs:
    def test_column_zero_is_the_target_and_rank_is_one_based(self) -> None:
        logits = mx.array(
            [
                [0.0, 1.0, 3.0, 2.0],
                [5.0, 0.0, 0.0, 0.0],
            ],
            dtype=mx.float16,
        )

        tensors = gather_prompt_logprobs(logits, [1, 0], num_logprobs=2)

        expected = torch.log_softmax(
            torch.tensor(np.array(logits.astype(mx.float32))), dim=-1
        )
        assert tensors.logprob_token_ids.shape == (2, 3)
        assert tensors.logprobs.shape == (2, 3)
        assert tensors.logprob_token_ids[:, 0].tolist() == [1, 0]
        torch.testing.assert_close(tensors.logprobs[0, 0], expected[0, 1])
        torch.testing.assert_close(tensors.logprobs[1, 0], expected[1, 0])
        # Top-2 alternatives in descending order.
        assert tensors.logprob_token_ids[0, 1:].tolist() == [2, 3]
        assert tensors.logprob_token_ids[1, 1:].tolist() == [0, 1]
        # Token 1 in row 0 is the third highest logit; token 0 in row 1 the highest.
        assert tensors.selected_token_ranks.tolist() == [3, 1]

    def test_zero_logprobs_keeps_only_the_target_column(self) -> None:
        logits = mx.array([[0.0, 2.0, 1.0]], dtype=mx.float32)

        tensors = gather_prompt_logprobs(logits, [2], num_logprobs=0)

        assert tensors.logprob_token_ids.shape == (1, 1)
        assert tensors.logprob_token_ids.tolist() == [[2]]
        assert tensors.selected_token_ranks.tolist() == [2]

    def test_rejects_row_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="num_targets, vocab"):
            gather_prompt_logprobs(mx.zeros((2, 4)), [1], num_logprobs=1)


class TestPromptLogprobsAccumulator:
    def test_chunks_fill_disjoint_positions_of_one_tensor_set(self) -> None:
        acc = PromptLogprobsAccumulator(prompt_len=6, num_logprobs=1)
        first = gather_prompt_logprobs(
            mx.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.0]]), [1, 0, 1], num_logprobs=1
        )
        second = gather_prompt_logprobs(mx.array([[2.0, 0.0], [0.0, 2.0]]), [0, 0], 1)

        acc.fill(prompt_logprobs_window(start_pos=0, num_tokens=3, prompt_len=6), first)
        acc.fill(
            prompt_logprobs_window(start_pos=3, num_tokens=3, prompt_len=6), second
        )

        assert acc.tensors.logprob_token_ids.shape == (5, 2)
        assert acc.tensors.logprob_token_ids[:, 0].tolist() == [1, 0, 1, 0, 0]
        torch.testing.assert_close(acc.tensors.logprobs[:3], first.logprobs)
        torch.testing.assert_close(acc.tensors.logprobs[3:], second.logprobs)
        assert acc.tensors.selected_token_ranks.tolist() == (
            first.selected_token_ranks.tolist() + second.selected_token_ranks.tolist()
        )

    def test_rejects_chunk_that_does_not_match_its_window(self) -> None:
        acc = PromptLogprobsAccumulator(prompt_len=4, num_logprobs=1)
        chunk = gather_prompt_logprobs(mx.array([[0.0, 1.0]]), [1], num_logprobs=1)

        with pytest.raises(ValueError, match="window expects 2"):
            acc.fill(
                prompt_logprobs_window(start_pos=0, num_tokens=2, prompt_len=4), chunk
            )

    def test_rejects_window_past_the_prompt(self) -> None:
        acc = PromptLogprobsAccumulator(prompt_len=3, num_logprobs=0)
        chunk = gather_prompt_logprobs(mx.array([[0.0, 1.0], [1.0, 0.0]]), [1, 0], 0)

        with pytest.raises(ValueError, match="exceeds"):
            acc.fill(
                PromptLogprobsWindow(start_pos=1, num_logits=2, completes=True), chunk
            )


class TestPromptLogprobsTracker:
    def test_delivers_only_on_the_completing_chunk(self) -> None:
        from vllm_metal.v1.prompt_logprobs import PromptLogprobsTracker

        tracker = PromptLogprobsTracker()
        prompt = [3, 1, 0, 2, 1, 0]
        vocab = 4
        full_rows = mx.arange(len(prompt) * vocab, dtype=mx.float32).reshape(
            len(prompt), vocab
        ) * mx.array([0.1])

        first = tracker.observe_chunk(
            "req-a",
            prompt_token_ids=prompt,
            start_pos=0,
            num_tokens=3,
            chunk_logits=full_rows[0:3],
            num_logprobs=1,
        )
        assert first is None

        final = tracker.observe_chunk(
            "req-a",
            prompt_token_ids=prompt,
            start_pos=3,
            num_tokens=3,
            chunk_logits=full_rows[3:6],
            num_logprobs=1,
        )
        assert final is not None
        # One row per scored prompt position, column 0 is the prompt token.
        assert final.logprob_token_ids.shape == (len(prompt) - 1, 2)
        assert final.logprob_token_ids[:, 0].tolist() == prompt[1:]
        # Values match a one-shot gather over the same rows.
        expected = gather_prompt_logprobs(full_rows[:5], prompt[1:], 1)
        torch.testing.assert_close(final.logprobs, expected.logprobs)
        assert final.selected_token_ranks.tolist() == (
            expected.selected_token_ranks.tolist()
        )
        # Delivery clears the in-progress slot.
        assert tracker._in_progress == {}

    def test_single_chunk_prompt_completes_immediately(self) -> None:
        from vllm_metal.v1.prompt_logprobs import PromptLogprobsTracker

        tracker = PromptLogprobsTracker()
        prompt = [2, 0, 1]
        rows = mx.array(
            [[0.0, 1.0, 2.0], [2.0, 0.0, 1.0], [1.0, 2.0, 0.0]], dtype=mx.float32
        )

        tensors = tracker.observe_chunk(
            "req-b",
            prompt_token_ids=prompt,
            start_pos=0,
            num_tokens=3,
            chunk_logits=rows,
            num_logprobs=0,
        )

        assert tensors is not None
        # The last row predicts the first sampled token and is not scored.
        assert tensors.logprob_token_ids.shape == (2, 1)
        assert tensors.logprob_token_ids[:, 0].tolist() == prompt[1:]

    def test_one_token_prompt_delivers_empty_tensors(self) -> None:
        from vllm_metal.v1.prompt_logprobs import PromptLogprobsTracker

        tracker = PromptLogprobsTracker()
        tensors = tracker.observe_chunk(
            "req-c",
            prompt_token_ids=[5],
            start_pos=0,
            num_tokens=1,
            chunk_logits=mx.zeros((1, 8)),
            num_logprobs=2,
        )

        assert tensors is not None
        assert tensors.logprob_token_ids.shape == (0, 3)

    def test_rejects_row_count_mismatch(self) -> None:
        from vllm_metal.v1.prompt_logprobs import PromptLogprobsTracker

        tracker = PromptLogprobsTracker()
        with pytest.raises(ValueError, match=r"\(3, vocab\)"):
            tracker.observe_chunk(
                "req-d",
                prompt_token_ids=[1, 2, 3, 4],
                start_pos=0,
                num_tokens=3,
                chunk_logits=mx.zeros((2, 8)),
                num_logprobs=1,
            )

    def test_discard_drops_in_progress_state(self) -> None:
        from vllm_metal.v1.prompt_logprobs import PromptLogprobsTracker

        tracker = PromptLogprobsTracker()
        tracker.observe_chunk(
            "req-e",
            prompt_token_ids=[1, 2, 3, 4],
            start_pos=0,
            num_tokens=2,
            chunk_logits=mx.zeros((2, 8)),
            num_logprobs=1,
        )
        assert "req-e" in tracker._in_progress

        tracker.discard({"req-e", "never-seen"})

        assert tracker._in_progress == {}

    def test_wants(self) -> None:
        from vllm.sampling_params import SamplingParams

        from vllm_metal.v1.prompt_logprobs import PromptLogprobsTracker

        assert PromptLogprobsTracker.wants(SamplingParams(prompt_logprobs=1))
        assert not PromptLogprobsTracker.wants(SamplingParams())
        assert not PromptLogprobsTracker.wants(None)


class TestFullPromptLogprobs:
    def test_matches_chunked_gather(self) -> None:
        from vllm_metal.v1.prompt_logprobs import full_prompt_logprobs

        prompt = [0, 2, 1, 3]
        rows = mx.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 0.0],
                [1.0, 3.0, 0.0, 2.0],
                [9.0, 9.0, 9.0, 9.0],  # predicts the sampled token; ignored
            ],
            dtype=mx.float32,
        )

        tensors = full_prompt_logprobs(rows, prompt, num_logprobs=2)

        assert tensors.logprob_token_ids.shape == (3, 3)
        assert tensors.logprob_token_ids[:, 0].tolist() == prompt[1:]
        expected = gather_prompt_logprobs(rows[:3], prompt[1:], 2)
        torch.testing.assert_close(tensors.logprobs, expected.logprobs)

    def test_one_token_prompt(self) -> None:
        from vllm_metal.v1.prompt_logprobs import full_prompt_logprobs

        tensors = full_prompt_logprobs(mx.zeros((1, 4)), [7], num_logprobs=1)

        assert tensors.logprob_token_ids.shape == (0, 2)


class TestRunnerWiring:
    """The two #680 symptoms, pinned at the runner seams.

    ``prompt_logprobs_dict`` was hardcoded ``{}`` in ``_build_output`` and
    nothing ever produced tensors, so ``echo+logprobs`` 500'd (the API layer
    indexed a generated-token dict with a prompt token id) and bare
    ``prompt_logprobs`` returned ``[None]`` silently.
    """

    def _runner(self):
        from tests.stub_runner import make_stub_runner

        return make_stub_runner(num_kv_heads=2)

    def test_build_output_carries_prompt_logprobs_dict(self) -> None:
        import vllm_metal.v1.model_runner as mr

        batch = mr._ExecutionBatch()
        batch.add_output("req-0", [1])
        tensors = gather_prompt_logprobs(mx.zeros((1, 8)), [3], num_logprobs=1)
        batch.prompt_logprobs_dict["req-0"] = tensors

        output = mr.MetalModelRunner._build_output(batch)

        assert output.prompt_logprobs_dict == {"req-0": tensors}

    def test_prefill_single_populates_prompt_logprobs(self) -> None:
        from types import SimpleNamespace

        from vllm.sampling_params import SamplingParams

        vocab = 32
        prompt = [4, 7, 1, 9]
        rows = mx.arange(len(prompt) * vocab, dtype=mx.float32).reshape(
            1, len(prompt), vocab
        ) * mx.array([0.01])

        class _TinyModel(SimpleNamespace):
            def make_cache(self):
                return []

            def __call__(self, input_ids, cache=None):
                return rows

        runner = self._runner()
        runner.model = _TinyModel()

        next_token, _cache, _logprobs, prompt_logprobs = runner._prefill_single(
            prompt,
            SamplingParams(temperature=0, prompt_logprobs=1),
        )

        assert isinstance(next_token, int)
        assert prompt_logprobs is not None
        assert prompt_logprobs.logprob_token_ids.shape == (len(prompt) - 1, 2)
        assert prompt_logprobs.logprob_token_ids[:, 0].tolist() == prompt[1:]

        # Without the flag the extra work is skipped entirely.
        _, _, _, none_logprobs = runner._prefill_single(
            prompt, SamplingParams(temperature=0)
        )
        assert none_logprobs is None

    def test_paged_gather_delivers_on_completing_chunk(self) -> None:
        from vllm.sampling_params import SamplingParams

        import vllm_metal.v1.model_runner as mr

        runner = self._runner()
        prompt = [3, 1, 4, 1, 5, 9]
        vocab = 16
        params = SamplingParams(temperature=0, prompt_logprobs=1)

        def _prefill(start: int, end: int, prompt_len: int | None):
            return mr.PrefillRequest(
                req_id="req-p",
                token_ids=prompt[start:end],
                sampling_params=params,
                block_ids=[[0]],
                generator=None,
                prompt_len=prompt_len,
                start_pos=start,
                full_prompt_token_ids=prompt,
            )

        batch = mr._ExecutionBatch()
        chunk1 = mx.zeros((1, 4, vocab))
        runner._gather_prefill_prompt_logprobs(
            batch, [_prefill(0, 4, None)], chunk1, [0, 4], 0
        )
        assert batch.prompt_logprobs_dict == {}

        chunk2 = mx.zeros((1, 2, vocab))
        runner._gather_prefill_prompt_logprobs(
            batch, [_prefill(4, 6, len(prompt))], chunk2, [0, 2], 0
        )
        tensors = batch.prompt_logprobs_dict["req-p"]
        assert tensors.logprob_token_ids.shape == (len(prompt) - 1, 2)
        assert tensors.logprob_token_ids[:, 0].tolist() == prompt[1:]

    def test_paged_gather_rejects_pruned_segment_rows(self) -> None:
        from vllm.sampling_params import SamplingParams

        import vllm_metal.v1.model_runner as mr

        runner = self._runner()
        prompt = [3, 1, 4, 1]
        prefill = mr.PrefillRequest(
            req_id="req-q",
            token_ids=prompt,
            sampling_params=SamplingParams(temperature=0, prompt_logprobs=1),
            block_ids=[[0]],
            generator=None,
            prompt_len=len(prompt),
            start_pos=0,
            full_prompt_token_ids=prompt,
        )
        batch = mr._ExecutionBatch()

        # A selective-logits layout would leave one row per prefill segment.
        with pytest.raises(RuntimeError, match="selective-logits"):
            runner._gather_prefill_prompt_logprobs(
                batch, [prefill], mx.zeros((1, 1, 8)), [0, 1], 0
            )

    def test_paged_gather_requires_full_prompt(self) -> None:
        from vllm.sampling_params import SamplingParams

        import vllm_metal.v1.model_runner as mr

        runner = self._runner()
        prefill = mr.PrefillRequest(
            req_id="req-r",
            token_ids=[1, 2],
            sampling_params=SamplingParams(temperature=0, prompt_logprobs=1),
            block_ids=[[0]],
            generator=None,
            prompt_len=None,
            start_pos=0,
            full_prompt_token_ids=None,
        )
        batch = mr._ExecutionBatch()

        with pytest.raises(RuntimeError, match="full prompt"):
            runner._gather_prefill_prompt_logprobs(
                batch, [prefill], mx.zeros((1, 2, 8)), [0, 2], 0
            )


@pytest.mark.slow
def test_prompt_logprobs_end_to_end_paged():
    """Bare ``prompt_logprobs`` returns one real entry per prompt token (#680).

    Covers both reported symptoms at the engine level: the silent ``[None]``
    (the dict below used to stay empty) and the data the OpenAI ``echo`` +
    ``logprobs`` path indexes into (its 500 came from these tensors never
    existing).  ``max_num_batched_tokens`` forces chunked prefill so the
    accounting crosses chunk boundaries.
    """
    from vllm import LLM, SamplingParams

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        mp.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        mp.setenv("VLLM_METAL_MEMORY_FRACTION", "0.3")

        llm = LLM(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512,
            max_num_seqs=2,
            max_num_batched_tokens=16,
            enable_chunked_prefill=True,
        )
        prompt = (
            "The three most important properties of a distributed cache "
            "are consistency, availability, and partition tolerance."
        )
        sp = SamplingParams(temperature=0, max_tokens=4, prompt_logprobs=1, logprobs=1)
        [output] = llm.generate([prompt], sp)

        prompt_len = len(output.prompt_token_ids)
        assert prompt_len > 16, "prompt must span multiple prefill chunks"
        prompt_logprobs = output.prompt_logprobs
        assert prompt_logprobs is not None
        assert len(prompt_logprobs) == prompt_len
        assert prompt_logprobs[0] is None
        for position, (token_id, entry) in enumerate(
            zip(output.prompt_token_ids[1:], prompt_logprobs[1:], strict=True), 1
        ):
            assert entry is not None, f"position {position} silently empty"
            assert token_id in entry, f"prompt token missing at {position}"
            assert entry[token_id].logprob <= 0.0
