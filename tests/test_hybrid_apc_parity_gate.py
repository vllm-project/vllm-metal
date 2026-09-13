# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the parity gate's mismatch classification.

The gate's engine-facing halves (children, arms) need local weights, but
the classification core is pure and is where review found two
false-PASS holes: a diagnostic rerun that converged used to drop the
strict failure outright, and a rerun telling a different story could
waive the strict failure with its own tie. These tests pin the contract:
only a *reproduced* flip on an exact tie between exactly the two chosen
tokens may be waived.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent / "tools" / "hybrid_apc_parity_matrix.py"
_spec = importlib.util.spec_from_file_location("hybrid_apc_parity_matrix", TOOL)
gate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gate
_spec.loader.exec_module(gate)


def _run(
    strict_ref,
    strict_cached,
    diag_ref=None,
    diag_cached=None,
    ref_gaps=None,
    cached_gaps=None,
):
    return gate.classify_mismatches(
        strict_ref, strict_cached, diag_ref, diag_cached, ref_gaps, cached_gaps
    )


class TestFirstFlip:
    def test_none_when_equal(self) -> None:
        assert gate.first_flip([1, 2], [1, 2]) is None

    def test_length_difference_is_a_flip(self) -> None:
        assert gate.first_flip([101], [101, 102]) == 1
        assert gate.first_flip([101, 102, 103], [101]) == 1


class TestClassifyMismatches:
    def test_reproduced_exact_tie_between_chosen_tokens_waives(self) -> None:
        # The observed 27B batch shape: cached arm ties exactly on the two
        # tokens the arms respectively chose.
        divergent, waived = _run(
            {"case": [5979]},
            {"case": [1528]},
            {"case": [5979]},
            {"case": [1528]},
            ref_gaps={"case": [[0.25, 5979, 1528]]},
            cached_gaps={"case": [[0.0, 1528, 5979]]},
        )
        assert divergent == [] and waived == ["case"]

    def test_converged_rerun_retains_failure(self) -> None:
        # Review P1: strict diverges, diagnostic rerun converges — the
        # strict failure must survive (previously it was dropped).
        divergent, waived = _run(
            {"case": [101]},
            {"case": [999]},
            {"case": [101]},
            {"case": [101]},
        )
        assert divergent == ["case"] and waived == []

    def test_rerun_flip_differs_from_strict_stays_failure(self) -> None:
        # Review P2: the rerun's own tie is about a different execution
        # (101/102); it may not excuse the strict 101/999 divergence.
        divergent, waived = _run(
            {"case": [101]},
            {"case": [999]},
            {"case": [101]},
            {"case": [102]},
            ref_gaps={"case": [[0.0, 101, 102]]},
            cached_gaps={"case": [[0.0, 101, 102]]},
        )
        assert divergent == ["case"] and waived == []

    def test_tie_between_unrelated_tokens_stays_failure(self) -> None:
        # Exact tie exists but not between the two tokens the arms chose.
        divergent, waived = _run(
            {"case": [101]},
            {"case": [999]},
            {"case": [101]},
            {"case": [999]},
            ref_gaps={"case": [[0.0, 101, 102]]},
            cached_gaps={"case": [[0.5, 999, 1]]},
        )
        assert divergent == ["case"] and waived == []

    def test_no_diagnostic_pass_stays_failure(self) -> None:
        divergent, waived = _run({"case": [1]}, {"case": [2]})
        assert divergent == ["case"] and waived == []

    def test_prefix_truncation_mismatch_stays_failure(self) -> None:
        # Pure length difference: recorded as a mismatch, no exception.
        divergent, waived = _run(
            {"case": [7, 7]},
            {"case": [7, 7, 9]},
            {"case": [7, 7]},
            {"case": [7, 7, 9]},
        )
        assert divergent == ["case"] and waived == []

    def test_reproduced_flip_but_common_prefix_differs_stays_failure(self) -> None:
        divergent, waived = _run(
            {"case": [5, 101]},
            {"case": [5, 999]},
            {"case": [6, 101]},
            {"case": [6, 999]},
            ref_gaps={"case": [[9.0, 0, 0], [0.0, 101, 999]]},
        )
        assert divergent == ["case"] and waived == []
