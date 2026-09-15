# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from tools.check_sd_lossless import drafted_tokens, main


class TestDraftedTokens:
    # The shape below is what `llm.get_metrics()` actually produces through this
    # tool, copied from a real DSpark run: the keys carry a `spec_decode_` prefix
    # and no `_total` suffix. Asserting an invented shape here would let the gate
    # report "no metrics" for every real run while the tests stayed green.
    REAL = {
        "gen_tokens": 768,
        "elapsed_s": 1.72,
        "tok_per_s": 447.45,
        "spec_decode_num_drafts": 343,
        "spec_decode_num_draft_tokens": 681,
        "spec_decode_num_accepted_tokens": 407,
        "spec_decode_num_accepted_tokens_per_pos": [249, 158],
    }

    def test_reads_the_real_metric_shape(self):
        assert drafted_tokens(self.REAL) == 681

    def test_prefers_draft_tokens_over_draft_count(self):
        assert (
            drafted_tokens(
                {"spec_decode_num_drafts": 343, "spec_decode_num_draft_tokens": 681}
            )
            == 681
        )

    def test_falls_back_to_the_draft_count(self):
        assert drafted_tokens({"spec_decode_num_drafts": 32}) == 32

    def test_tolerates_a_total_suffix(self):
        assert drafted_tokens({"spec_decode_num_draft_tokens_total": 128}) == 128

    def test_sums_a_per_engine_series(self):
        assert drafted_tokens({"spec_decode_num_draft_tokens": [12, 20]}) == 32

    def test_zero_is_reported_as_zero_not_missing(self):
        # The distinction matters: 0 means the drafter demonstrably never ran,
        # None means we cannot tell. Both are inconclusive, for different reasons.
        assert drafted_tokens({"spec_decode_num_draft_tokens": 0}) == 0

    def test_accepted_tokens_alone_do_not_count_as_drafting(self):
        assert drafted_tokens({"spec_decode_num_accepted_tokens": 407}) is None

    def test_a_target_only_run_reports_nothing(self):
        # The base engine emits no spec-decode counters at all.
        assert drafted_tokens({"gen_tokens": 768, "elapsed_s": 1.17}) is None

    def test_a_metrics_error_is_unknown(self):
        assert (
            drafted_tokens({"metrics_error": "RuntimeError()", "gen_tokens": 64})
            is None
        )


class TestVerdict:
    """A speculative run that never drafted must not be reported as lossless."""

    PROMPT = "The capital of France is"

    @staticmethod
    def _result(tokens, stats):
        return {
            "tokens": {TestVerdict.PROMPT: tokens},
            "topk": {TestVerdict.PROMPT: [[(t, -0.1)] for t in tokens]},
            "stats": stats,
        }

    @pytest.fixture
    def run(self, monkeypatch):
        def _run(sd_tokens, sd_stats, base_tokens=(1, 2, 3)):
            calls = []

            def fake_run_engine(args, spec_config, out_path, *, logprobs):
                calls.append(spec_config)
                if spec_config is None:
                    return self._result(list(base_tokens), {"gen_tokens": 3})
                return self._result(list(sd_tokens), sd_stats)

            monkeypatch.setattr("tools.check_sd_lossless.run_engine", fake_run_engine)
            monkeypatch.setattr(
                sys, "argv", ["check_sd_lossless.py", "--prompt", self.PROMPT]
            )
            return main()

        return _run

    def test_identical_tokens_without_drafting_are_inconclusive(self, run):
        # The regression: a target-only fallback emits the target's own tokens,
        # so every prompt matches exactly and nothing failed -- but verification
        # was never exercised, so this is not evidence of losslessness.
        assert run((1, 2, 3), {"spec_decode_num_draft_tokens": 0}) == 2

    def test_identical_tokens_without_metrics_are_inconclusive(self, run):
        assert run((1, 2, 3), {"gen_tokens": 3}) == 2

    def test_identical_tokens_with_drafting_pass(self, run):
        assert run((1, 2, 3), {"spec_decode_num_draft_tokens": 12}) == 0

    def test_a_divergence_still_fails(self, run):
        assert run((1, 9, 3), {"spec_decode_num_draft_tokens": 12}) == 1
