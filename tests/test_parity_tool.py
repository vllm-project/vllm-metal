# SPDX-License-Identifier: Apache-2.0

import io
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from tools.check_parity import check_parity, compare_results, http_generate, http_result


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

    @pytest.mark.parametrize(
        ("batch_size", "expected_prompts"),
        [(1, [[10], [20], [30]]), (2, [[[10], [20]], [30]])],
    )
    def test_http_prompt_batches_keep_reference_order(
        self, monkeypatch, batch_size, expected_prompts
    ):
        sent = []

        def respond(request, timeout):
            assert request.full_url == "http://localhost:8000/v1/completions"
            payload = json.loads(request.data)
            assert payload["ignore_eos"] and payload["return_token_ids"]
            assert payload["return_tokens_as_token_ids"]
            assert payload["temperature"] == 0
            sent.append(payload["prompt"])
            prompts = payload["prompt"]
            if isinstance(prompts[0], int):
                prompts = [prompts]
            choices = [
                {
                    "index": i,
                    "prompt_token_ids": ids,
                    "token_ids": [ids[0] + 1],
                    "text": str(ids[0] + 1),
                    "logprobs": {"top_logprobs": [{f"token_id:{ids[0] + 1}": -1}]},
                }
                for i, ids in enumerate(prompts)
            ]
            return io.BytesIO(json.dumps({"choices": choices[::-1]}).encode())

        monkeypatch.setattr("urllib.request.urlopen", respond)
        results = http_generate(
            "http://localhost:8000/v1/",
            "model",
            [{"input_ids": [10]}, {"input_ids": [20]}, {"input_ids": [30]}],
            max_tokens=1,
            top_k=1,
            batch_size=batch_size,
        )
        assert sent == expected_prompts
        assert [result["tokens"] for result in results] == [[11], [21], [31]]

    @pytest.mark.parametrize("indices", [[0], [0, 0]])
    def test_http_rejects_missing_or_duplicate_choices(self, monkeypatch, indices):
        choices = [
            {"index": i, "prompt_token_ids": [1], "token_ids": [2], "text": "two"}
            for i in indices
        ]
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *args, **kwargs: io.BytesIO(
                json.dumps({"choices": choices}).encode()
            ),
        )
        with pytest.raises(ValueError):
            http_generate(
                "http://localhost/v1",
                "model",
                [{"input_ids": [1]}] * 2,
                1,
                batch_size=2,
            )

    def test_http_rejects_changed_input_ids(self, monkeypatch):
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *args, **kwargs: io.BytesIO(
                b'{"choices": [{"index": 0, "prompt_token_ids": [99]}]}'
            ),
        )
        with pytest.raises(ValueError, match="input token IDs"):
            http_generate("http://localhost/v1", "model", [{"input_ids": [1]}], 1)

    def test_canonical_flow_reuses_reference_and_server(self, monkeypatch, tmp_path):
        events = []
        row = {**self._result([2], []), "input_ids": [1]}

        def generate_reference(command, **kwargs):
            path = Path(command[command.index("--generate-reference") + 1])
            assert json.loads(path.read_text()) == ["test prompt"]
            path.write_text(json.dumps([row]))
            events.append("reference exited")

        @contextmanager
        def server(model, max_model_len, max_num_seqs, log_path, env):
            assert max_num_seqs == 2
            events.append("server started")
            yield "http://localhost/v1"
            events.append("server stopped")

        def generate(base_url, model, reference, max_tokens, top_k, batch_size):
            events.append(f"batch size {batch_size}")
            return reference

        monkeypatch.setattr("subprocess.run", generate_reference)
        monkeypatch.setattr("tools.check_parity.serving", server)
        monkeypatch.setattr("tools.check_parity.http_generate", generate)
        assert check_parity(str(tmp_path), ["test prompt"], 1, output_dir=tmp_path)
        assert events == [
            "reference exited",
            "server started",
            "batch size 1",
            "batch size 2",
            "server stopped",
        ]
