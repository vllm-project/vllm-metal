# SPDX-License-Identifier: Apache-2.0
"""Tests for YOCO fast-prefill indexing helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from vllm_metal.yoco_fast_prefill import (
    build_yoco_reduced_context_from_full_metadata,
    try_enable_gemma4_yoco_fast_prefill,
)

_REAL_GEMMA4_MODEL_ENV = "GEMMA4_FAST_PREFILL_MODEL_PATH"
_REAL_GEMMA4_FALLBACK_MODEL_ENV = "GEMMA4_MODEL_PATH"
_REAL_GEMMA4_PROMPTS = [
    "The capital of France is",
    "One plus one equals",
]
_REAL_GEMMA4_MAX_MODEL_LEN = 1024
_REAL_GEMMA4_MAX_TOKENS = 8
_FAST_PREFILL_ENABLED_ATTR = "_vllm_metal_yoco_fast_prefill_enabled"
_REAL_RESULT_MARKER = "VLLM_METAL_YOCO_FAST_PREFILL_RESULT="


def _logged_warning_text(warning: Mock) -> str:
    rendered = []
    for call in warning.call_args_list:
        if not call.args:
            continue
        fmt = str(call.args[0])
        rendered.append(fmt % call.args[1:] if len(call.args) > 1 else fmt)
    return "\n".join(rendered)


def _run_real_gemma4_paged_path(
    model_path: str,
    *,
    fast_prefill: bool,
) -> tuple[dict[str, list[int]], bool]:
    memory_fraction = os.environ.get(
        "GEMMA4_FAST_PREFILL_MEMORY_FRACTION",
        os.environ.get("VLLM_METAL_MEMORY_FRACTION", "0.5"),
    )
    env = os.environ.copy()
    env.update(
        {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_METAL_USE_PAGED_ATTENTION": "1",
            "VLLM_METAL_KV_SHARING_FAST_PREFILL": "1" if fast_prefill else "0",
            "VLLM_METAL_MEMORY_FRACTION": memory_fraction,
        }
    )

    script = f"""
import json
from contextlib import suppress

from vllm import LLM, SamplingParams

model_path = {model_path!r}
prompts = {json.dumps(_REAL_GEMMA4_PROMPTS)}
enabled_attr = {_FAST_PREFILL_ENABLED_ATTR!r}
llm = LLM(
    model=model_path,
    max_model_len={_REAL_GEMMA4_MAX_MODEL_LEN},
    max_num_seqs={len(_REAL_GEMMA4_PROMPTS)},
    enable_prefix_caching=False,
)
runner = llm.llm_engine.model_executor.driver_worker.model_runner
model = runner.model
language_model = getattr(model, "language_model", None)
candidates = [
    model,
    getattr(model, "model", None),
    language_model,
    getattr(language_model, "model", None),
]
fast_prefill_enabled = any(
    bool(getattr(candidate, enabled_attr, False))
    for candidate in candidates
    if candidate is not None
)
sp = SamplingParams(
    temperature=0,
    max_tokens={_REAL_GEMMA4_MAX_TOKENS},
    ignore_eos=True,
)
outputs = llm.generate(prompts, sp)
tokens = {{o.prompt: list(o.outputs[0].token_ids) for o in outputs}}
print(
    {_REAL_RESULT_MARKER!r}
    + json.dumps({{"tokens": tokens, "enabled": fast_prefill_enabled}}),
    flush=True,
)
engine = getattr(llm, "llm_engine", None)
engine_core = getattr(engine, "engine_core", None)
shutdown = getattr(engine_core, "shutdown", None)
if callable(shutdown):
    with suppress(Exception):
        shutdown()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        raise AssertionError(
            "real Gemma4 paged-path subprocess failed "
            f"(fast_prefill={fast_prefill}, code={result.returncode})\n"
            f"{output[-8000:]}"
        )
    for line in output.splitlines():
        if line.startswith(_REAL_RESULT_MARKER):
            payload = json.loads(line[len(_REAL_RESULT_MARKER) :])
            return payload["tokens"], bool(payload["enabled"])
    raise AssertionError(
        f"real Gemma4 paged-path subprocess did not report result\n{output[-8000:]}"
    )


@pytest.mark.parametrize(
    ("model_args", "expected_reason"),
    [
        (
            {"model_type": "qwen3", "num_hidden_layers": 32, "num_kv_shared_layers": 8},
            "model_type='qwen3' is not Gemma4",
        ),
        (
            {
                "model_type": "gemma4",
                "num_hidden_layers": 42,
                "num_kv_shared_layers": 0,
            },
            "num_kv_shared_layers must be positive",
        ),
        (
            {"model_type": "gemma4", "num_hidden_layers": 42},
            "num_kv_shared_layers is missing or not an int",
        ),
        (
            {
                "model_type": "gemma4",
                "num_hidden_layers": 18,
                "num_kv_shared_layers": 18,
            },
            "num_kv_shared_layers must be smaller than num_hidden_layers (18 >= 18)",
        ),
        (
            {
                "model_type": "gemma4",
                "num_hidden_layers": "bad",
                "num_kv_shared_layers": 1,
            },
            "num_hidden_layers is missing or not an int",
        ),
        (
            {"num_hidden_layers": 42, "num_kv_shared_layers": 18},
            "model_type=None is not Gemma4",
        ),
    ],
)
def test_try_enable_logs_ineligible_gemma4_yoco_shape(
    model_args,
    expected_reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning = Mock()
    monkeypatch.setattr("vllm_metal.yoco_fast_prefill.logger.warning", warning)

    assert not try_enable_gemma4_yoco_fast_prefill(
        object(),
        model_args,
        use_paged_attention=True,
    )
    assert expected_reason in _logged_warning_text(warning)


def test_try_enable_can_skip_ineligible_model_without_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning = Mock()
    debug = Mock()
    monkeypatch.setattr("vllm_metal.yoco_fast_prefill.logger.warning", warning)
    monkeypatch.setattr("vllm_metal.yoco_fast_prefill.logger.debug", debug)

    assert not try_enable_gemma4_yoco_fast_prefill(
        object(),
        {
            "model_type": "qwen3",
            "num_hidden_layers": 32,
            "num_kv_shared_layers": 8,
        },
        use_paged_attention=True,
        warn_on_skip=False,
    )
    warning.assert_not_called()
    debug.assert_called_once()


def test_try_enable_logs_without_paged_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning = Mock()
    monkeypatch.setattr("vllm_metal.yoco_fast_prefill.logger.warning", warning)

    assert not try_enable_gemma4_yoco_fast_prefill(
        object(),
        {
            "model_type": "gemma4",
            "num_hidden_layers": 42,
            "num_kv_shared_layers": 18,
        },
        use_paged_attention=False,
    )
    assert "paged attention is disabled" in _logged_warning_text(warning)


def test_try_enable_logs_when_not_all_layers_are_paged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning = Mock()
    monkeypatch.setattr("vllm_metal.yoco_fast_prefill.logger.warning", warning)

    assert not try_enable_gemma4_yoco_fast_prefill(
        object(),
        {
            "model_type": "gemma4",
            "num_hidden_layers": 4,
            "num_kv_shared_layers": 2,
        },
        use_paged_attention=True,
        num_paged_layers=3,
    )
    assert "only 3/4 layers use paged attention" in _logged_warning_text(warning)


def test_try_enable_logs_missing_num_hidden_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning = Mock()
    monkeypatch.setattr("vllm_metal.yoco_fast_prefill.logger.warning", warning)

    assert not try_enable_gemma4_yoco_fast_prefill(
        object(),
        {
            "model_type": "gemma4",
            "num_kv_shared_layers": 18,
        },
        use_paged_attention=True,
    )
    assert "num_hidden_layers is missing or not an int" in _logged_warning_text(warning)


def test_reduced_context_from_full_paged_metadata() -> None:
    meta = build_yoco_reduced_context_from_full_metadata(
        slot_mapping=[
            2 * 4 + 2,
            3 * 4 + 1,
            4 * 4 + 0,
            4 * 4 + 1,
            4 * 4 + 2,
            4 * 4 + 3,
            5 * 4 + 0,
            7 * 4 + 1,
            7 * 4 + 2,
            7 * 4 + 3,
        ],
        block_tables=[
            [1, 2],
            [3],
            [4, 5],
            [6, 7, 8],
        ],
        context_lens=[7, 2, 5, 8],
        offsets=[6, 1, 0, 5],
        cu_seqlens=[0, 1, 2, 7, 10],
    )

    assert meta.selected_query_indices == [0, 1, 6, 9]
    assert meta.slot_mapping == [2 * 4 + 2, 3 * 4 + 1, 5 * 4 + 0, 7 * 4 + 3]
    assert meta.block_tables == [[1, 2], [3], [4, 5], [6, 7, 8]]
    assert meta.context_lens == [7, 2, 5, 8]
    assert meta.offsets == [6, 1, 4, 7]
    assert meta.cu_seqlens == [0, 1, 2, 3, 4]


def test_mlx_slice_assignment_writes_in_place() -> None:
    import mlx.core as mx

    h = mx.zeros((1, 5, 2), dtype=mx.float32)
    selected = mx.array([1, 3], dtype=mx.int32)
    cross_h = mx.array([[[7.0, 8.0], [9.0, 10.0]]], dtype=mx.float32)

    h[:, selected, :] = cross_h
    mx.eval(h)

    assert h[0, 0, 0].item() == 0.0
    assert h[0, 1, 0].item() == 7.0
    assert h[0, 3, 0].item() == 9.0


def test_reduced_context_from_empty_full_paged_metadata() -> None:
    meta = build_yoco_reduced_context_from_full_metadata(
        slot_mapping=[],
        block_tables=[],
        context_lens=[],
        offsets=[],
        cu_seqlens=[0],
    )

    assert meta.selected_query_indices == []
    assert meta.slot_mapping == []
    assert meta.block_tables == []
    assert meta.context_lens == []
    assert meta.offsets == []
    assert meta.cu_seqlens == [0]


def test_try_enable_gemma4_yoco_fast_prefill_reduces_shared_layer_queries() -> None:
    import mlx.core as mx

    from vllm_metal.paged_attention_common import (
        PagedAttentionContext,
        clear_context,
        get_context,
        set_context,
    )

    class _Config:
        num_hidden_layers = 4
        num_kv_shared_layers = 2

    class _Layer:
        def __init__(self, layer_idx: int) -> None:
            self.layer_idx = layer_idx
            self.calls = []

        def __call__(
            self,
            h,
            mask,
            cache,
            *,
            per_layer_input=None,
            shared_kv=None,
            offset=None,
        ):
            ctx = get_context()
            self.calls.append(
                {
                    "tokens": h.shape[1],
                    "cu_seqlens": tuple(ctx.cu_seqlens),
                    "offsets": tuple(ctx.offsets),
                    "gdn_slot_mapping": ctx.gdn_slot_mapping,
                }
            )
            return h + (self.layer_idx + 1), (f"k{self.layer_idx}", "v"), offset

    class _TextModel:
        config = _Config()
        embed_scale = 1
        hidden_size_per_layer_input = 0

        def __init__(self) -> None:
            self.layers = [_Layer(i) for i in range(4)]
            self.previous_kvs = [0, 1, 0, 1]
            self.embed_tokens = lambda inputs: inputs
            self.norm = lambda h: h

        def _get_per_layer_inputs(self, input_ids, input_embeddings=None):
            raise AssertionError("per-layer input lookup should not run")

        def _project_per_layer_inputs(self, input_embeddings, per_layer_inputs=None):
            raise AssertionError("per-layer input projection should not run")

        def _make_masks(self, h, cache):
            return [None] * len(self.layers)

        def __call__(
            self,
            inputs=None,
            cache=None,
            input_embeddings=None,
            per_layer_inputs=None,
        ):
            return input_embeddings + 100

    text_model = _TextModel()
    top_model = type("_TopModel", (), {"model": text_model})()

    assert try_enable_gemma4_yoco_fast_prefill(
        top_model,
        {
            "model_type": "gemma4_text",
            "num_hidden_layers": 4,
            "num_kv_shared_layers": 2,
        },
        use_paged_attention=True,
        num_paged_layers=4,
    )

    inputs = mx.zeros((1, 5, 2), dtype=mx.float32)
    fallback = text_model(input_embeddings=inputs)
    mx.eval(fallback)
    assert fallback[0, 0, 0].item() == 100

    ctx = PagedAttentionContext(
        slot_mapping=[0, 1, 2, 3, 4],
        block_tables=[[0, 1]],
        context_lens=[5],
        offsets=[0],
        cu_seqlens=[0, 5],
        gdn_slot_mapping=[17],
    )
    set_context(ctx)
    try:
        out = text_model(input_embeddings=inputs, cache=[object()] * 4)
        assert get_context() is ctx
    finally:
        clear_context()

    mx.eval(out)
    assert [layer.calls[0]["tokens"] for layer in text_model.layers] == [5, 5, 1, 1]
    assert text_model.layers[2].calls[0]["cu_seqlens"] == (0, 1)
    assert text_model.layers[2].calls[0]["offsets"] == (4,)
    assert text_model.layers[0].calls[0]["gdn_slot_mapping"] == [17]
    assert text_model.layers[2].calls[0]["gdn_slot_mapping"] is None
    assert text_model.layers[3].calls[0]["gdn_slot_mapping"] is None
    assert out[0, 0, 0].item() == 3
    assert out[0, 4, 0].item() == 10


@pytest.mark.slow
def test_real_gemma4_paged_path_matches_with_fast_prefill_off_on() -> None:
    model_path = os.environ.get(_REAL_GEMMA4_MODEL_ENV) or os.environ.get(
        _REAL_GEMMA4_FALLBACK_MODEL_ENV
    )
    if not model_path:
        pytest.skip(
            f"Set {_REAL_GEMMA4_MODEL_ENV} or {_REAL_GEMMA4_FALLBACK_MODEL_ENV} "
            "to run the real Gemma4 fast-prefill parity test"
        )
    if not os.path.isdir(model_path):
        pytest.skip(f"{model_path} is not a directory")

    off_tokens, off_enabled = _run_real_gemma4_paged_path(
        model_path,
        fast_prefill=False,
    )
    assert not off_enabled

    on_tokens, on_enabled = _run_real_gemma4_paged_path(
        model_path,
        fast_prefill=True,
    )

    assert on_enabled
    assert on_tokens == off_tokens


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "slot_mapping": [],
                "block_tables": [],
                "context_lens": [],
                "offsets": [],
                "cu_seqlens": [],
            },
            "cu_seqlens must start with 0",
        ),
        (
            {
                "slot_mapping": [],
                "block_tables": [],
                "context_lens": [],
                "offsets": [],
                "cu_seqlens": [1],
            },
            "cu_seqlens must start with 0",
        ),
        (
            {
                "slot_mapping": [1],
                "block_tables": [[1]],
                "context_lens": [1],
                "offsets": [],
                "cu_seqlens": [0, 1],
            },
            "block_tables, context_lens, and offsets must match",
        ),
        (
            {
                "slot_mapping": [1, 2],
                "block_tables": [[1]],
                "context_lens": [1],
                "offsets": [0],
                "cu_seqlens": [0, 1],
            },
            "slot_mapping length must match final cu_seqlens",
        ),
        (
            {
                "slot_mapping": [],
                "block_tables": [[1]],
                "context_lens": [1],
                "offsets": [0],
                "cu_seqlens": [0, 1],
            },
            "slot_mapping shorter than cu_seqlens",
        ),
        (
            {
                "slot_mapping": [1],
                "block_tables": [[1]],
                "context_lens": [1],
                "offsets": [0],
                "cu_seqlens": [0, 0],
            },
            "cu_seqlens segments must be positive and increasing",
        ),
    ],
)
def test_reduced_context_from_full_metadata_validates_inputs(
    kwargs,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_yoco_reduced_context_from_full_metadata(**kwargs)
