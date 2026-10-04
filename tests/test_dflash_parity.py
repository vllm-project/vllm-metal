# SPDX-License-Identifier: Apache-2.0
"""The numerical qualification tool must fail on invalid or missing evidence."""

import hashlib
from pathlib import Path

import mlx.core as mx
import pytest

import tools.dflash_parity as dflash_parity
import tools.dspark_paged_parity as dspark_paged_parity
import tools.dspark_parity as dspark_parity
from tools.attention_bench_utils import (
    compare,
    native_source_hashes,
    source_file_hashes,
)
from vllm_metal.v1.draft_checkpoint import load_draft_weights


@pytest.mark.parametrize(
    "actual,expected",
    [
        ([[float("nan")]], [[float("nan")]]),
        ([[float("inf")]], [[float("inf")]]),
        ([], []),
        ([[1.0, 2.0]], [[1.0]]),
        ([[1.0]], [[1.1]]),
    ],
)
def test_compare_rejects_invalid_evidence(actual, expected):
    with pytest.raises((AssertionError, ValueError)):
        compare(mx.array(actual), mx.array(expected))


def test_compare_distinguishes_exact_and_tolerated_matches():
    exact = compare(mx.array([[1.0]]), mx.array([[1.0]]))
    tolerated = compare(mx.array([[1.0001]]), mx.array([[1.0]]))

    assert exact == {"exact": True, "max_abs_error": 0.0}
    assert tolerated["exact"] is False
    assert tolerated["max_abs_error"] == pytest.approx(1e-4, rel=0.05)


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


def test_source_file_hashes_use_relative_paths(tmp_path: Path) -> None:
    source = tmp_path / "nested" / "source.py"
    source.parent.mkdir()
    source.write_text("source")

    assert source_file_hashes(tmp_path, [source]) == {
        "nested/source.py": hashlib.sha256(b"source").hexdigest()
    }
