# SPDX-License-Identifier: Apache-2.0
"""Behavior matrix for the shared sampling admission policy.

The expected values are frozen literals captured from the pre-policy
implementations (``sampling_batch`` native gates and ``platform``
MPS request validation) before they were centralized, so the matrix is an
independent oracle for the documented semantics rather than a restatement
of the policy code:

- native greedy/random eligibility per feature, with ``allow_seed`` for the
  seeded-row (sync and deferred) native random path;
- MPS admission per control, under plain MPS, Omni (repetition penalty
  allowed), and speculation (non-zero temperature rejected exactly, not by
  the greedy epsilon);
- API ``bad_words`` (read pre-tokenization by request validation) vs
  tokenized ``bad_words_token_ids`` (what the native gates see);
- prompt logprobs (gathered outside sampling) vs sample logprobs;
- ``min_tokens`` pending-aware on the native paths vs rejected outright at
  MPS admission.
"""

from __future__ import annotations

import pytest
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from vllm_metal.v1.sampling_policy import (
    GREEDY_TEMPERATURE_EPS,
    params_allow_native_greedy,
    params_allow_native_random,
    unsupported_mlx_scoring_option,
    unsupported_request_controls,
)


def _sp(**kwargs) -> SamplingParams:
    return SamplingParams(**kwargs)


def _tokenized_bad_words() -> SamplingParams:
    # bad_words reaches the gates only after update_from_tokenizer; set the
    # tokenized form directly to model that state without a tokenizer.
    params = _sp(temperature=0.7)
    params._bad_words_token_ids = [[99]]
    return params


def _sub_eps_temperature() -> SamplingParams:
    # SamplingParams clamps 0 < t < 0.01 to 0.01 at construction; assign
    # after init to pin the gates' epsilon boundary.
    params = _sp(temperature=0.7)
    params.temperature = 1e-6
    return params


def _structured_outputs() -> SamplingParams:
    return _sp(
        temperature=0.7,
        structured_outputs=StructuredOutputsParams(choice=["y", "n"]),
    )


# (label, params builder, output_lengths)
_CASES = [
    ("plain-greedy", lambda: [_sp(temperature=0.0)], [0]),
    (
        "plain-random",
        lambda: [_sp(temperature=0.7, top_k=20, top_p=0.95)],
        [0],
    ),
    ("clamped-min-temperature", lambda: [_sp(temperature=0.01)], [0]),
    ("sub-eps-temperature", lambda: [_sub_eps_temperature()], [0]),
    ("seeded-greedy", lambda: [_sp(temperature=0.0, seed=7)], [0]),
    ("seeded-random", lambda: [_sp(temperature=0.7, seed=7)], [0]),
    ("top-k", lambda: [_sp(temperature=0.7, top_k=40)], [0]),
    ("top-p", lambda: [_sp(temperature=0.7, top_p=0.8)], [0]),
    ("min-p", lambda: [_sp(temperature=0.7, min_p=0.1)], [0]),
    (
        "mixed-greedy-random",
        lambda: [_sp(temperature=0.0), _sp(temperature=0.7)],
        [0, 0],
    ),
    (
        "frequency-penalty",
        lambda: [_sp(temperature=0.0, frequency_penalty=0.5)],
        [0],
    ),
    (
        "presence-penalty",
        lambda: [_sp(temperature=0.0, presence_penalty=0.5)],
        [0],
    ),
    (
        "repetition-penalty",
        lambda: [_sp(temperature=0.0, repetition_penalty=1.2)],
        [0],
    ),
    ("sample-logprobs", lambda: [_sp(temperature=0.0, logprobs=1)], [0]),
    (
        "specific-token-logprobs",
        lambda: [_sp(temperature=0.0, logprob_token_ids=[1, 2])],
        [0],
    ),
    (
        "prompt-logprobs",
        lambda: [_sp(temperature=0.7, prompt_logprobs=1)],
        [0],
    ),
    (
        "allowed-token-ids",
        lambda: [_sp(temperature=0.0, allowed_token_ids=[1, 2])],
        [0],
    ),
    ("api-bad-words", lambda: [_sp(temperature=0.7, bad_words=["foo"])], [0]),
    ("tokenized-bad-words", lambda: [_tokenized_bad_words()], [0]),
    ("logit-bias", lambda: [_sp(temperature=0.0, logit_bias={1: 5.0})], [0]),
    ("structured-outputs", lambda: [_structured_outputs()], [0]),
    (
        "min-tokens-pending",
        lambda: [_sp(temperature=0.0, min_tokens=2), _sp(temperature=0.0)],
        [1, 0],
    ),
    (
        "min-tokens-reached",
        lambda: [_sp(temperature=0.0, min_tokens=2), _sp(temperature=0.0)],
        [2, 0],
    ),
    (
        "min-tokens-default-lengths",
        lambda: [_sp(temperature=0.0, min_tokens=2)],
        None,
    ),
    ("empty-batch", lambda: [], None),
]

# Frozen from the pre-policy gates on 2026-11-11:
# (native_greedy, native_random, native_random_allow_seed,
#  mps_controls, mps_controls_omni, mps_controls_speculative)
_EXPECTED = {
    "plain-greedy": (True, True, True, [], [], []),
    "plain-random": (False, True, True, [], [], ["temperature"]),
    "clamped-min-temperature": (False, True, True, [], [], ["temperature"]),
    "sub-eps-temperature": (True, True, True, [], [], ["temperature"]),
    "seeded-greedy": (True, False, True, [], [], []),
    "seeded-random": (False, False, True, [], [], ["temperature"]),
    "top-k": (False, True, True, [], [], ["temperature"]),
    "top-p": (False, True, True, [], [], ["temperature"]),
    "min-p": (False, True, True, [], [], ["temperature"]),
    "mixed-greedy-random": (False, True, True, [], [], []),
    "frequency-penalty": (
        False,
        False,
        False,
        ["frequency_penalty"],
        ["frequency_penalty"],
        ["frequency_penalty"],
    ),
    "presence-penalty": (
        False,
        False,
        False,
        ["presence_penalty"],
        ["presence_penalty"],
        ["presence_penalty"],
    ),
    "repetition-penalty": (
        False,
        False,
        False,
        ["repetition_penalty"],
        [],
        ["repetition_penalty"],
    ),
    "sample-logprobs": (False, False, False, [], [], []),
    "specific-token-logprobs": (
        False,
        False,
        False,
        ["logprob_token_ids"],
        ["logprob_token_ids"],
        ["logprob_token_ids"],
    ),
    "prompt-logprobs": (False, True, True, [], [], ["temperature"]),
    "allowed-token-ids": (
        False,
        False,
        False,
        ["allowed_token_ids"],
        ["allowed_token_ids"],
        ["allowed_token_ids"],
    ),
    "api-bad-words": (
        False,
        True,
        True,
        ["bad_words"],
        ["bad_words"],
        ["temperature", "bad_words"],
    ),
    "tokenized-bad-words": (False, False, False, [], [], ["temperature"]),
    "logit-bias": (
        False,
        False,
        False,
        ["logit_bias"],
        ["logit_bias"],
        ["logit_bias"],
    ),
    "structured-outputs": (
        False,
        False,
        False,
        ["structured_outputs"],
        ["structured_outputs"],
        ["temperature", "structured_outputs"],
    ),
    "min-tokens-pending": (
        False,
        False,
        False,
        ["min_tokens"],
        ["min_tokens"],
        ["min_tokens"],
    ),
    "min-tokens-reached": (
        True,
        True,
        True,
        ["min_tokens"],
        ["min_tokens"],
        ["min_tokens"],
    ),
    "min-tokens-default-lengths": (
        False,
        False,
        False,
        ["min_tokens"],
        ["min_tokens"],
        ["min_tokens"],
    ),
    "empty-batch": (True, False, False, [], [], []),
}


class TestSamplingPolicyMatrix:
    @pytest.mark.parametrize(
        ("label", "make_params", "output_lengths"),
        _CASES,
        ids=[case[0] for case in _CASES],
    )
    def test_feature_matrix(self, label: str, make_params, output_lengths) -> None:
        params_list = make_params()
        (
            native_greedy,
            native_random,
            native_random_allow_seed,
            mps,
            mps_omni,
            mps_spec,
        ) = _EXPECTED[label]

        assert (
            params_allow_native_greedy(params_list, output_lengths) is native_greedy
        ), "native greedy eligibility changed"
        assert (
            params_allow_native_random(params_list, output_lengths) is native_random
        ), "native random eligibility changed"
        assert (
            params_allow_native_random(params_list, output_lengths, allow_seed=True)
            is native_random_allow_seed
        ), "seeded native random eligibility changed"

        if not params_list:
            return  # admission is per request; the empty batch has none
        params = params_list[0]
        assert unsupported_request_controls(params) == mps, (
            "plain MPS admission changed"
        )
        assert (
            unsupported_request_controls(params, allow_repetition_penalty=True)
            == mps_omni
        ), "Omni MPS admission changed"
        assert unsupported_request_controls(params, speculative=True) == mps_spec, (
            "speculative MPS admission changed"
        )

    def test_greedy_epsilon_threshold(self) -> None:
        """The native greedy boundary is the epsilon, not exact zero."""
        assert GREEDY_TEMPERATURE_EPS == 1e-5
        greedy = _sp(temperature=0.0)
        greedy.temperature = GREEDY_TEMPERATURE_EPS
        assert not params_allow_native_greedy([greedy])
        greedy.temperature = GREEDY_TEMPERATURE_EPS / 2
        assert params_allow_native_greedy([greedy])
        # MPS speculation reads the temperature exactly, unlike the gates.
        assert unsupported_request_controls(greedy, speculative=True) == ["temperature"]

    def test_length_mismatch_raises_only_without_a_pending_row(self) -> None:
        """The min-tokens reduction short-circuits before zip-strict can
        validate a trailing extra length — the pre-policy gate behavior."""
        # First row past min_tokens: the reduction walks on and zip-strict
        # rejects the over-long list.
        with pytest.raises(ValueError, match="zip"):
            params_allow_native_greedy([_sp(temperature=0.0, min_tokens=2)], [2, 99])
        with pytest.raises(ValueError, match="zip"):
            params_allow_native_random([_sp(temperature=0.7, min_tokens=2)], [2, 99])
        # First row still pending: any() stops before the extra length is
        # consumed, so the gate answers False instead of raising.
        assert not params_allow_native_greedy(
            [_sp(temperature=0.0, min_tokens=2)], [0, 99]
        )
        assert not params_allow_native_random(
            [_sp(temperature=0.7, min_tokens=2)], [0, 99]
        )

    def test_multiple_violations_keep_report_order(self) -> None:
        params = _sp(
            temperature=0.7,
            frequency_penalty=0.5,
            allowed_token_ids=[1],
            logit_bias={2: 1.0},
            min_tokens=3,
            logprob_token_ids=[4],
        )
        assert unsupported_request_controls(params, speculative=True) == [
            "temperature",
            "frequency_penalty",
            "allowed_token_ids",
            "logit_bias",
            "min_tokens",
            "logprob_token_ids",
        ]


class TestMlxScoringOption:
    """MLX rejects these at admission; MPS serves them through MRv2."""

    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            (_sp(temperature=0, logprobs=5), None),
            (_sp(temperature=0, prompt_logprobs=1, prompt_logprob_start=0), None),
            (_sp(temperature=0, logprobs=-1), ("logprobs", "logprobs=-1")),
            (
                _sp(temperature=0, prompt_logprobs=1, prompt_logprob_start=2),
                ("prompt_logprob_start", "prompt_logprob_start"),
            ),
            (
                _sp(
                    temperature=0,
                    logprobs=-1,
                    prompt_logprobs=1,
                    prompt_logprob_start=2,
                ),
                ("prompt_logprob_start", "prompt_logprob_start"),
            ),
        ],
        ids=["top-k", "prompt-start-0", "full-vocab", "prompt-start", "both"],
    )
    def test_first_rejected_option(self, params, expected) -> None:
        assert unsupported_mlx_scoring_option(params) == expected
