# SPDX-License-Identifier: Apache-2.0

from tools.check_parity import compare_results


class TestParityTool:
    @staticmethod
    def _result(tokens, candidates):
        return {
            "prompt": "test prompt",
            "tokens": tokens,
            "text": str(tokens),
            "top_logprobs": [
                [
                    {"id": token, "text": str(token), "rank": rank, "logprob": -rank}
                    for rank, token in enumerate(step, start=1)
                ]
                for step in candidates
            ],
        }

    def test_exact_mode_rejects_different_tokens(self):
        ref = self._result([1, 2], [])
        assert compare_results([ref], [ref], max_tokens=2)
        got = self._result([1, 3], [])
        assert not compare_results([ref], [got], max_tokens=2)

    def test_top_k_accepts_first_divergence_only(self, capsys):
        ref = self._result([1, 5, 8], [[1], [5, 6]])
        got = self._result([1, 6, 9], [[1], [6, 5]])
        assert compare_results([ref], [got], max_tokens=3, top_k=2)
        output = capsys.readouterr().out
        assert output.startswith("TOP_K_MATCH")
        assert "remaining continuation not compared" in output

    def test_top_k_requires_mutual_membership_within_requested_rank(self):
        ref = self._result([1, 5], [[1], [5, 6]])
        # The sampled token can be returned outside the requested top-K.
        got = self._result([1, 6], [[1], [6, 7, 5]])
        assert not compare_results([ref], [got], max_tokens=2, top_k=2)

    def test_incomplete_generation_cannot_pass_top_k(self):
        ref = self._result([1], [[1, 6]])
        got = self._result([6], [[6, 1]])
        assert not compare_results([ref], [got], max_tokens=2, top_k=2)
