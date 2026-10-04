# SPDX-License-Identifier: Apache-2.0
"""Numerical evidence must fail on missing data, non-finite values, or wrong IDs."""

import mlx.core as mx
import pytest
import torch

from tools.dspark_parity import check_tokens, compare


@pytest.mark.parametrize(
    "actual,expected",
    [
        ([], []),
        ([[1]], [[1, 2]]),
        ([[float("nan")]], [[float("nan")]]),
        ([[float("inf")]], [[float("inf")]]),
        ([[1]], [[2]]),
    ],
)
def test_compare_rejects_invalid_evidence(actual, expected):
    with pytest.raises((ValueError, AssertionError)):
        compare(mx.array(actual), torch.tensor(expected), atol=1e-3, rtol=1e-3)


def test_compare_distinguishes_exact_from_tolerant_results():
    assert compare(mx.array([[1.0]]), torch.tensor([[1.0]]), atol=1e-3, rtol=1e-3)[
        "exact"
    ]
    result = compare(mx.array([[1.0001]]), torch.tensor([[1.0]]), atol=1e-3, rtol=1e-3)
    assert not result["exact"] and result["max_abs_error"] > 0


def test_proposal_mismatch_is_not_accepted_as_a_near_tie():
    native = mx.array([[[18.25, 18.5]]])
    reference = torch.tensor([[[18.25, 18.25]]])
    # Floating-point tolerance alone would accept this difference.
    compare(native, reference, atol=0.25, rtol=0.02)
    with pytest.raises(AssertionError, match="row 0, position 0"):
        check_tokens(mx.array([[1]]), torch.tensor([[0]]), native, reference)


def test_proposal_shapes_must_match():
    with pytest.raises(AssertionError, match="shapes"):
        check_tokens(mx.array([[1]]), torch.tensor([[1, 2]]), None, None)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_reference_device_tensors_compare_and_report_mismatches(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    native = mx.array([[[1.0, 2.0]]])
    reference = torch.tensor([[[1.0, 2.0]]], device=device, requires_grad=True)
    tokens = torch.tensor([[1]], device=device)
    assert compare(native, reference, atol=0, rtol=0)["exact"]
    check_tokens(mx.array([[1]]), tokens, native, reference)
    with pytest.raises(AssertionError, match="native token 0, reference token 1"):
        check_tokens(mx.array([[0]]), tokens, native, reference)


@pytest.mark.parametrize("value", [None, "1"])
@pytest.mark.parametrize(
    "module",
    [
        "tools.dflash_serving_parity",
        "tools.dspark_parity",
        "tools.dspark_paged_parity",
    ],
)
def test_import_does_not_override_caller_tf32_setting(monkeypatch, value, module):
    import importlib
    import os

    if value is None:
        monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    else:
        monkeypatch.setenv("MLX_ENABLE_TF32", value)
    importlib.reload(importlib.import_module(module))
    assert os.environ.get("MLX_ENABLE_TF32") == value


@pytest.mark.parametrize("problem", ["token", "logits", "confidence", "missing", "nan"])
def test_paged_report_keeps_failed_gates_explicit(problem):
    from tools.dspark_paged_parity import compare_outputs

    expected = (mx.array([[1]]), mx.array([[[1.0, 2.0]]]), mx.array([[0.5]]))
    actual = list(expected)
    if problem == "token":
        actual[0] = mx.array([[0]])
    elif problem == "logits":
        actual[1] = expected[1] + 1
    elif problem == "confidence":
        actual[2] = expected[2] + 1
    elif problem == "missing":
        actual[2] = None
    else:
        actual[1] = mx.full((1, 1, 2), float("nan"))
    checks = compare_outputs(actual, expected, atol=0.015, rtol=0.02)
    assert not all(check["passed"] for check in checks.values())
    assert checks["tokens"]["passed"] == (problem != "token")


def test_paged_report_accepts_optional_confidence_and_tolerant_tensors():
    from tools.dspark_paged_parity import compare_outputs

    expected = (mx.array([[1]]), mx.array([[[1.0, 2.0]]]), None)
    actual = (expected[0], expected[1] + 0.001, None)
    checks = compare_outputs(actual, expected, atol=0.015, rtol=0.02)
    assert all(check["passed"] for check in checks.values())
    assert checks["logits"]["max_abs_error"] > 0


def test_paged_report_rejects_missing_proposal_ids():
    from tools.dspark_paged_parity import compare_outputs

    outputs = (mx.array([[]]), mx.array([[[1.0, 2.0]]]), None)
    checks = compare_outputs(outputs, outputs, atol=0.015, rtol=0.02)
    assert not checks["tokens"]["passed"]


def test_failed_paged_report_is_saved_but_cli_exits_unsuccessfully(
    monkeypatch, tmp_path
):
    import json
    import sys

    import tools.dspark_paged_parity as parity

    output = tmp_path / "failed.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["parity", "--target", ".", "--draft", ".", "--output", str(output)],
    )
    monkeypatch.setattr(parity, "qualify", lambda _: {"passed": False})
    with pytest.raises(SystemExit, match="qualification failed"):
        parity.main()
    assert json.loads(output.read_text()) == {"passed": False}
    # A rerun cannot leave or overwrite an earlier report and look successful.
    with pytest.raises(SystemExit):
        parity.main()
    assert json.loads(output.read_text()) == {"passed": False}
