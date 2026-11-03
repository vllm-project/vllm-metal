# SPDX-License-Identifier: Apache-2.0
"""Shared shader parity against the unchanged PR #1241 PyTorch recurrence."""

import os
import subprocess
import sys

import pytest
import torch

from scripts.bench_mps_gdn_shared_kernel import DTYPES, make_case, reference_result

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Requires Apple Silicon MPS"
)


@pytest.fixture(autouse=True)
def _seed_random_generators():
    # Override conftest's MLX-seeding fixture for this explicitly MPS-only suite.
    torch.manual_seed(1241)


@pytest.fixture(scope="module", autouse=True)
def _error_table_header():
    print(
        "\n| Case | Dtype | Output max abs | Output max rel | State max abs | State max rel |"
    )
    print("|---|---|---:|---:|---:|---:|")


def _errors(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    return diff.max().item(), (
        diff / expected.float().abs().clamp_min(1e-6)
    ).max().item()


CASES = [
    ("decode-B1", (1,), (1,), True),
    ("decode-B8", (1,) * 8, tuple(reversed(range(8))), True),
    *[(f"prefill-{n}", (n,), (1,), False) for n in (1, 5, 63, 64, 65, 200, 528)],
    ("mixed-PAD", (1, 5, 65, 0), (3, -1, 0, 2), False),
]


@pytest.mark.parametrize("dtype_name", DTYPES)
@pytest.mark.parametrize("name,lengths,slots,decode", CASES, ids=[c[0] for c in CASES])
@torch.inference_mode()
def test_mps_gdn_shared_kernel(name, lengths, slots, decode, dtype_name):
    from vllm_metal.pytorch_backend.mps_ops import gdn_linear_attention

    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
    before_mlx = {n for n in sys.modules if n == "mlx" or n.startswith("mlx.")}
    mixed = name == "mixed-PAD"
    case = make_case(
        lengths,
        slots,
        DTYPES[dtype_name],
        nonzero_state=decode or mixed,
        padded_state=mixed,
    )
    saved = case.state_storage.clone()
    expected_y, expected_state = reference_result(case, decode=decode)
    gdn_linear_attention(*case.args())
    # Consume on the same PyTorch stream before any explicit host synchronize.
    actual_y = (case.y.float() + 0.0).cpu()
    actual_state = (case.state + 0.0).cpu()
    expected_y, expected_state = expected_y.cpu(), expected_state.cpu()
    out_abs, out_rel = _errors(actual_y, expected_y)
    state_abs, state_rel = _errors(actual_state, expected_state)
    print(
        f"\n| {name} | {dtype_name} | {out_abs:.6e} | {out_rel:.6e} | "
        f"{state_abs:.6e} | {state_rel:.6e} |",
        flush=True,
    )
    # Only Y is rounded to the input dtype. State and recurrence remain fp32.
    atol, rtol = {
        "fp32": (1e-5, 1e-4),
        "fp16": (2e-5, 1e-3),
        "bf16": (2e-4, 8e-3),
    }[dtype_name]
    torch.testing.assert_close(actual_y, expected_y, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_state, expected_state, atol=1e-5, rtol=1e-4)

    # -1 must not touch the prefix guard; row gaps and inactive slots are exact.
    changed = torch.zeros_like(saved, dtype=torch.bool)
    row_size = case.state[0].numel()
    for length, slot in zip(lengths, slots, strict=True):
        if slot >= 0 and length:
            changed[slot + 1, :row_size] = True
    torch.testing.assert_close(
        case.state_storage.cpu()[~changed.cpu()],
        saved.cpu()[~changed.cpu()],
        atol=0,
        rtol=0,
    )
    start = 0
    for length, slot in zip(lengths, slots, strict=True):
        if slot < 0:
            assert torch.all(actual_y[start : start + length] == 13.0)
        start += length
    assert {n for n in sys.modules if n == "mlx" or n.startswith("mlx.")} == before_mlx


def test_mps_gdn_does_not_import_mlx():
    # A fresh interpreter also proves independence when another suite imported MLX.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
from scripts.bench_mps_gdn_shared_kernel import make_case
from vllm_metal.pytorch_backend.mps_ops import gdn_linear_attention
assert not any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules)
case = make_case((1,), (0,), torch.float32)
gdn_linear_attention(*case.args())
torch.mps.synchronize()
assert not any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules)
""",
        ],
        check=True,
    )
