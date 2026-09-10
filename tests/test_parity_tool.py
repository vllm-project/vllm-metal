# SPDX-License-Identifier: Apache-2.0

import io
import json
from threading import Barrier

import pytest

from tools.check_parity import compare_results, http_generate, http_result


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

    def test_http_logprobs_exclude_sampled_token_outside_top_k(self):
        choice = {
            "token_ids": [6],
            "text": "six",
            "logprobs": {
                "top_logprobs": [{"token_id:6": -3, "token_id:1": -1, "token_id:5": -2}]
            },
        }
        got = http_result(choice, top_k=2)
        assert {entry["id"] for entry in got["top_logprobs"][0]} == {1, 5}

    def test_http_requests_overlap_and_keep_reference_order(self, monkeypatch):
        barrier = Barrier(2, timeout=5)

        def respond(request, timeout):
            assert request.full_url == "http://localhost:8000/v1/completions"
            payload = json.loads(request.data)
            assert payload["ignore_eos"] and payload["return_token_ids"]
            assert payload["return_tokens_as_token_ids"]
            assert payload["temperature"] == 0
            barrier.wait()  # Both requests must be in flight together.
            [token] = payload["prompt"]
            return io.BytesIO(
                json.dumps(
                    {
                        "choices": [
                            {
                                "prompt_token_ids": [token],
                                "token_ids": [token + 1],
                                "text": str(token + 1),
                                "logprobs": {
                                    "top_logprobs": [{f"token_id:{token + 1}": -1}]
                                },
                            }
                        ]
                    }
                ).encode()
            )

        monkeypatch.setattr("urllib.request.urlopen", respond)
        results = http_generate(
            "http://localhost:8000/v1/",
            "model",
            [{"input_ids": [10]}, {"input_ids": [20]}],
            max_tokens=1,
            top_k=1,
            concurrency=2,
        )
        assert [result["tokens"] for result in results] == [[11], [21]]

    def test_http_rejects_changed_input_ids(self, monkeypatch):
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *args, **kwargs: io.BytesIO(
                b'{"choices": [{"prompt_token_ids": [99]}]}'
            ),
        )
        with pytest.raises(ValueError, match="input token IDs"):
            http_generate("http://localhost/v1", "model", [{"input_ids": [1]}], 1)
