# SPDX-License-Identifier: Apache-2.0
"""The numerical qualification tool must fail on invalid or missing evidence."""

import hashlib
from pathlib import Path

import mlx.core as mx
import pytest

import tools.dflash_parity as dflash_parity
import tools.dspark_paged_parity as dspark_paged_parity
import tools.dspark_parity as dspark_parity
from tools.attention_bench_utils import compare, native_source_hashes
from vllm_metal.v1.draft_checkpoint import load_draft_weights


@pytest.mark.parametrize(
    "actual,expected,check",
    [
        # Non-finite, empty, mismatched or out-of-tolerance inputs are rejected.
        ([[float("nan")]], [[float("nan")]], "error"),
        ([[float("inf")]], [[float("inf")]], "error"),
        ([], [], "error"),
        ([[1.0, 2.0]], [[1.0]], "error"),
        ([[1.0]], [[1.1]], "error"),
        # Exact and tolerated matches report their difference.
        ([[1.0]], [[1.0]], {"exact": True, "max_abs_error": 0.0}),
        ([[1.0001]], [[1.0]], {"exact": False, "max_abs_error": 1e-4}),
    ],
)
def test_compare(actual, expected, check):
    if check == "error":
        with pytest.raises((AssertionError, ValueError)):
            compare(mx.array(actual), mx.array(expected))
        return
    result = compare(mx.array(actual), mx.array(expected))
    assert result["exact"] is check["exact"]
    assert result["max_abs_error"] == pytest.approx(check["max_abs_error"], rel=0.05)


@pytest.mark.parametrize(
    "sources",
    [
        dflash_parity.NATIVE_SOURCES,
        dspark_parity.NATIVE_SOURCES,
        dspark_paged_parity.NATIVE_SOURCES,
    ],
    ids=["dflash", "dspark", "dspark-paged"],
)
def test_reports_hash_the_module_that_admits_the_weights(sources) -> None:
    # The report offers these hashes as evidence for the numbers it records.
    # load_draft_weights holds the tensor, precision and finite-value rules,
    # so an edit there must change a hash. Find its file through the function
    # rather than by name, so a later move keeps this check honest.
    rules = Path(load_draft_weights.__code__.co_filename)

    hashes = native_source_hashes(*sources)

    assert hashes[rules.name] == hashlib.sha256(rules.read_bytes()).hexdigest()
