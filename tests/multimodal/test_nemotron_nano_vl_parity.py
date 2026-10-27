# SPDX-License-Identifier: Apache-2.0
"""Opt-in paged serving parity against a fresh direct MLX-VLM reference.

Set NEMOTRON_NANO_VL_PARITY_MODEL and NEMOTRON_NANO_VL_PROCESSOR_SOURCE to
local copies of the pinned checkpoint and NVIDIA processor. The reference
process exits before the paged engine starts. No weights are downloaded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.nano_vl_reference import make_images, sha256

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests/fixtures/nano_vl_v2_cases.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text())
CASES = {case["name"]: case for case in FIXTURE["cases"]}
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    model_path = os.environ.get("NEMOTRON_NANO_VL_PARITY_MODEL")
    if not model_path:
        pytest.skip("Set NEMOTRON_NANO_VL_PARITY_MODEL to the pinned local checkpoint")
    source = os.environ.get("NEMOTRON_NANO_VL_PROCESSOR_SOURCE")
    if not source:
        pytest.fail("Set NEMOTRON_NANO_VL_PROCESSOR_SOURCE to the pinned NVIDIA source")
    output_dir = tmp_path_factory.mktemp("nano-vl-reference")
    output = output_dir / "reference.json"
    env = {
        **os.environ,
        **FIXTURE["reference"]["environment"],
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    with (output_dir / "reference.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.nano_vl_reference",
                "--model",
                str(Path(model_path).resolve()),
                "--fixture",
                str(FIXTURE_PATH),
                "--processor-source",
                str(Path(source).resolve()),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=900,
        )
    report = json.loads(output.read_text())
    assert report["fixture_sha256"] == sha256(FIXTURE_PATH)
    assert set(report["cases"]) == set(CASES)
    return report["cases"]


@pytest.fixture(scope="module")
def engine(reference):
    from vllm import LLM
    from vllm.platforms import current_platform

    from vllm_metal.platform import MetalPlatform

    assert isinstance(current_platform, MetalPlatform)
    # Match the direct reference's math mode, independently of the unit suite's
    # strict FP32 setting. MLX reads this flag once: use a fresh spawned engine
    # so a prior test's Metal operations cannot freeze a different value.
    with pytest.MonkeyPatch.context() as patch:
        for name, value in FIXTURE["reference"]["environment"].items():
            patch.setenv(name, value)
        patch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
        patch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        llm = LLM(
            model=os.environ["NEMOTRON_NANO_VL_PARITY_MODEL"],
            max_model_len=2048,
            max_num_seqs=2,
            max_num_batched_tokens=512,
            gpu_memory_utilization=0.50,
            kv_cache_memory_bytes=1024**3,
            enable_prefix_caching=False,
            enforce_eager=True,
            seed=0,
            generation_config="vllm",
            limit_mm_per_prompt={"image": 2},
            mm_processor_kwargs={"max_num_tiles": FIXTURE["max_num_tiles"]},
        )
        try:
            yield llm
        finally:
            llm.llm_engine.engine_core.shutdown()


def _request(case: dict, expected: dict) -> dict:
    images = make_images(case, FIXTURE)
    request = {"prompt": expected["prompt"]}
    if images:
        request["multi_modal_data"] = {"image": images}
    return request


def _assert_parity(result, expected: dict) -> None:
    assert result.prompt_token_ids == expected["prompt_token_ids"]
    assert list(result.outputs[0].token_ids) == expected["output_token_ids"]
    assert result.outputs[0].finish_reason == expected["finish_reason"]


@pytest.mark.parametrize("name", CASES)
def test_nano_image_greedy_tokens_match_direct_reference(engine, reference, name):
    from vllm import SamplingParams

    case = CASES[name]
    result = engine.generate(
        [_request(case, reference[name])],
        SamplingParams(temperature=0, top_p=1, max_tokens=32, seed=0),
        use_tqdm=False,
    )[0]
    _assert_parity(result, reference[name])


@pytest.mark.parametrize(
    "names",
    [
        ("square", "resized"),
        ("two_images", "reversed_images"),
        ("wide_tiled", "text_only"),
    ],
)
def test_nano_concurrent_requests_keep_reference_tokens(engine, reference, names):
    from vllm import SamplingParams

    results = engine.generate(
        [_request(CASES[name], reference[name]) for name in names],
        SamplingParams(temperature=0, top_p=1, max_tokens=32, seed=0),
        use_tqdm=False,
    )
    for result, name in zip(results, names, strict=True):
        _assert_parity(result, reference[name])
