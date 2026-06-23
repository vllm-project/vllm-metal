# SPDX-License-Identifier: Apache-2.0
"""Deterministic loader-contract tests for ``vllm_metal.gguf.loader``.

Synthetic GGUF fixtures built with ``gguf.GGUFWriter`` + a tiny config exercise
the loader contract offline: partition, install, tie-skip, bias side-map,
completeness, and fail-fast rejection of unsupported files.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from vllm_metal.gguf.loader import GGUFLoadError, GGUFModelLoader  # noqa: E402
from vllm_metal.gguf.wrappers import GGUFLinear  # noqa: E402

QT = gguf.GGMLQuantizationType


def _tiny_config(model_type: str, **overrides) -> dict:
    config = {
        "model_type": model_type,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
        "max_position_embeddings": 512,
    }
    config.update(overrides)
    return config


def _dims(config: dict) -> dict:
    heads = config["num_attention_heads"]
    head_dim = config.get("head_dim", config["hidden_size"] // heads)
    return {
        "h": config["hidden_size"],
        "layers": config["num_hidden_layers"],
        "inter": config["intermediate_size"],
        "vocab": config["vocab_size"],
        "qd": heads * head_dim,
        "kvd": config["num_key_value_heads"] * head_dim,
        "hd": head_dim,
    }


def _dense_tensor_specs(config: dict, *, has_qk_norm: bool, with_bias: bool) -> dict:
    """Return ``{gguf_name: (kind, shape)}`` for a dense decoder GGUF.

    ``kind`` is ``"q"`` (quantized weight) or ``"f"`` (F32 plain weight/bias).
    """
    d = _dims(config)
    specs: dict[str, tuple[str, tuple[int, ...]]] = {
        "token_embd.weight": ("q", (d["vocab"], d["h"])),
        "output_norm.weight": ("f", (d["h"],)),
    }
    if not config["tie_word_embeddings"]:
        specs["output.weight"] = ("q", (d["vocab"], d["h"]))
    for i in range(d["layers"]):
        p = f"blk.{i}."
        specs[p + "attn_q.weight"] = ("q", (d["qd"], d["h"]))
        specs[p + "attn_k.weight"] = ("q", (d["kvd"], d["h"]))
        specs[p + "attn_v.weight"] = ("q", (d["kvd"], d["h"]))
        specs[p + "attn_output.weight"] = ("q", (d["h"], d["qd"]))
        specs[p + "attn_norm.weight"] = ("f", (d["h"],))
        specs[p + "ffn_norm.weight"] = ("f", (d["h"],))
        specs[p + "ffn_gate.weight"] = ("q", (d["inter"], d["h"]))
        specs[p + "ffn_up.weight"] = ("q", (d["inter"], d["h"]))
        specs[p + "ffn_down.weight"] = ("q", (d["h"], d["inter"]))
        if has_qk_norm:
            specs[p + "attn_q_norm.weight"] = ("f", (d["hd"],))
            specs[p + "attn_k_norm.weight"] = ("f", (d["hd"],))
        if with_bias:
            specs[p + "attn_q.bias"] = ("f", (d["qd"],))
            specs[p + "attn_k.bias"] = ("f", (d["kvd"],))
            specs[p + "attn_v.bias"] = ("f", (d["kvd"],))
    return specs


def _write_gguf(
    path: Path,
    arch: str,
    specs: dict,
    *,
    quant_type=QT.Q8_0,
    quant_overrides: dict[str, QT] | None = None,
    inject: dict | None = None,
) -> None:
    rng = np.random.default_rng(0)
    writer = gguf.GGUFWriter(str(path), arch)
    for name, (kind, shape) in {**specs, **(inject or {})}.items():
        data = rng.standard_normal(shape).astype(np.float32)
        qtype = (quant_overrides or {}).get(name)
        if kind == "q" or qtype is not None:
            raw_dtype = qtype or quant_type
            quant_input = data.reshape(1, -1) if data.ndim == 1 else data
            raw = gguf.quants.quantize(quant_input, raw_dtype)
            writer.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=raw_dtype)
        else:
            writer.add_tensor(name, data, raw_dtype=QT.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _build_dense_fixture(
    tmp_path: Path,
    model_type: str,
    *,
    config_overrides: dict | None = None,
    has_qk_norm: bool,
    with_bias: bool = False,
    inject: dict | None = None,
    drop: set[str] | None = None,
    quant_type=QT.Q8_0,
    quant_overrides: dict[str, QT] | None = None,
    gguf_arch: str | None = None,
) -> tuple[str, str]:
    """Write a tiny ``config.json`` + a matching dense GGUF; return (gguf, dir).

    By default the GGUF's ``general.architecture`` matches the config's
    ``model_type``; tests can override it to exercise file/config mismatch paths.
    """
    config = {**_tiny_config(model_type), **(config_overrides or {})}
    (tmp_path / "config.json").write_text(json.dumps(config))
    specs = _dense_tensor_specs(config, has_qk_norm=has_qk_norm, with_bias=with_bias)
    for name in drop or set():
        specs.pop(name, None)
    gguf_path = tmp_path / f"{model_type}.gguf"
    _write_gguf(
        gguf_path,
        gguf_arch or config["model_type"],
        specs,
        quant_type=quant_type,
        quant_overrides=quant_overrides,
        inject=inject,
    )
    return str(gguf_path), str(tmp_path)


def _gguf_module_histogram(model: nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}

    def walk(module: nn.Module) -> None:
        for child in module.children().values():
            for leaf in child if isinstance(child, list) else [child]:
                if isinstance(leaf, nn.Module):
                    counts[type(leaf).__name__] = counts.get(type(leaf).__name__, 0) + 1
                    walk(leaf)

    walk(model)
    return counts


@pytest.mark.parametrize("quant_type", [QT.Q8_0, QT.Q4_0])
def test_loads_dense_qwen3_installs_wrappers(tmp_path, quant_type):
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path, "qwen3", has_qk_norm=True, quant_type=quant_type
    )
    model, _ = GGUFModelLoader(
        gguf_path,
        config_dir=cfg_dir,
        target_dtype=mx.float32,
    ).load()
    hist = _gguf_module_histogram(model)
    assert hist.get("GGUFEmbedding") == 1
    assert hist.get("GGUFLinear") == 2 * 7
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape == (1, 3, 256)


def test_skips_tie_redundant_output(tmp_path):
    # Tied config but the GGUF still carries a redundant Q8_0 output.weight.
    d = _dims(_tiny_config("qwen3"))
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen3",
        has_qk_norm=True,
        inject={"output.weight": ("q", (d["vocab"], d["h"]))},
    )
    # Must NOT raise (the redundant output is tie-skipped, not unmapped-failed).
    model, _ = GGUFModelLoader(
        gguf_path,
        config_dir=cfg_dir,
        target_dtype=mx.float32,
    ).load()
    assert not hasattr(model, "lm_head")
    # The tied head runs through the GGUFEmbedding's as_linear.
    assert _gguf_module_histogram(model).get("GGUFEmbedding") == 1


def test_skips_tie_redundant_output_when_config_omits_tie_flag(tmp_path):
    d = _dims(_tiny_config("qwen2"))
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen2",
        has_qk_norm=False,
        with_bias=True,
        inject={"output.weight": ("q", (d["vocab"], d["h"]))},
    )
    config_path = Path(cfg_dir) / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("tie_word_embeddings")
    config_path.write_text(json.dumps(config))

    model, _ = GGUFModelLoader(
        gguf_path,
        config_dir=cfg_dir,
        target_dtype=mx.float32,
    ).load()

    assert not hasattr(model, "lm_head")
    assert _gguf_module_histogram(model).get("GGUFEmbedding") == 1


def test_untied_qwen2_attaches_bias_and_installs_lm_head(tmp_path):
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen2",
        config_overrides={"tie_word_embeddings": False},
        has_qk_norm=False,
        with_bias=True,
    )
    model, _ = GGUFModelLoader(
        gguf_path,
        config_dir=cfg_dir,
        target_dtype=mx.float32,
    ).load()
    # Assert through the model's own public structure, not loader internals.
    q_proj = model.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, GGUFLinear)
    assert "bias" in q_proj  # F32 bias paired from the side-map
    assert isinstance(model.lm_head, GGUFLinear)  # untied output -> real lm_head
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == 256


def test_completeness_fails_on_dropped_norm(tmp_path):
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path, "qwen3", has_qk_norm=True, drop={"blk.1.ffn_norm.weight"}
    )
    with pytest.raises(GGUFLoadError, match="Incomplete GGUF load"):
        GGUFModelLoader(
            gguf_path,
            config_dir=cfg_dir,
            target_dtype=mx.float32,
        ).load()


def test_validate_fails_on_wrong_shape(tmp_path):
    d = _dims(_tiny_config("qwen3"))
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen3",
        has_qk_norm=True,
        inject={"blk.0.attn_q.weight": ("q", (d["qd"] + 16, d["h"]))},
    )
    with pytest.raises(GGUFLoadError, match="dims"):
        GGUFModelLoader(
            gguf_path,
            config_dir=cfg_dir,
            target_dtype=mx.float32,
        ).load()


def test_validate_fails_on_wrong_shape_bias(tmp_path):
    d = _dims(_tiny_config("qwen2"))
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen2",
        config_overrides={"tie_word_embeddings": False},
        has_qk_norm=False,
        with_bias=True,
        inject={"blk.0.attn_q.bias": ("f", (d["qd"] + 16,))},
    )
    with pytest.raises(GGUFLoadError, match="blk.0.attn_q.bias"):
        GGUFModelLoader(
            gguf_path,
            config_dir=cfg_dir,
            target_dtype=mx.float32,
        ).load()


def test_rejects_missing_required_bias(tmp_path):
    # qwen2 q/k/v are nn.Linear(bias=True); installing a wrapper drops the original
    # bias leaf, so a GGUF that omits a required bias must fail at install.
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen2",
        config_overrides={"tie_word_embeddings": False},
        has_qk_norm=False,
        with_bias=True,
        drop={"blk.0.attn_q.bias"},
    )
    with pytest.raises(
        GGUFLoadError,
        match="'model.layers.0.self_attn.q_proj' expects a bias",
    ):
        GGUFModelLoader(
            gguf_path,
            config_dir=cfg_dir,
            target_dtype=mx.float32,
        ).load()


def test_rejects_non_dense_arch(tmp_path):
    # A linear-attention/fused-QKV hybrid arch is rejected by the dense allowlist.
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path, "qwen3", config_overrides={"model_type": "qwen3_5"}, has_qk_norm=True
    )
    with pytest.raises(
        GGUFLoadError, match="'qwen3_5' is not a supported dense decoder"
    ):
        GGUFModelLoader(
            gguf_path,
            config_dir=cfg_dir,
            target_dtype=mx.float32,
        ).load()


def test_rejects_gguf_config_arch_mismatch(tmp_path):
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen2",
        has_qk_norm=False,
        gguf_arch="qwen3",
    )
    with pytest.raises(GGUFLoadError, match="does not match config model_type"):
        GGUFModelLoader(
            gguf_path,
            config_dir=cfg_dir,
            target_dtype=mx.float32,
        ).load()


def test_rejects_out_of_scope_tensor(tmp_path):
    d = _dims(_tiny_config("qwen3"))
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen3",
        has_qk_norm=True,
        inject={"blk.0.attn_qkv.weight": ("q", (d["qd"], d["h"]))},
    )
    with pytest.raises(
        GGUFLoadError, match="Out-of-scope GGUF tensor 'blk.0.attn_qkv.weight'"
    ):
        GGUFModelLoader(
            gguf_path,
            config_dir=cfg_dir,
            target_dtype=mx.float32,
        ).load()


def test_rejects_vision_tensor(tmp_path):
    d = _dims(_tiny_config("qwen3"))
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen3",
        has_qk_norm=True,
        inject={"v.blk.0.attn_q.weight": ("q", (d["qd"], d["h"]))},
    )
    with pytest.raises(
        GGUFLoadError, match="Out-of-scope GGUF tensor 'v.blk.0.attn_q.weight'"
    ):
        GGUFModelLoader(
            gguf_path,
            config_dir=cfg_dir,
            target_dtype=mx.float32,
        ).load()


def test_rejects_unsupported_qtype(tmp_path):
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen3",
        has_qk_norm=True,
        quant_overrides={"blk.0.ffn_up.weight": QT.Q4_1},
    )
    with pytest.raises(
        GGUFLoadError,
        match="Unsupported qtype Q4_1 on mapped weight 'blk.0.ffn_up.weight'",
    ):
        GGUFModelLoader(
            gguf_path,
            config_dir=cfg_dir,
            target_dtype=mx.float32,
        ).load()


def test_rejects_quantized_bias(tmp_path):
    # An additive bias must be plain F32/F16/BF16; a quantized bias is rejected in
    # preflight (it would otherwise slip past the weight-only qtype check).
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen2",
        config_overrides={"tie_word_embeddings": False},
        has_qk_norm=False,
        with_bias=True,
        quant_overrides={"blk.0.attn_q.bias": QT.Q8_0},
    )
    with pytest.raises(
        GGUFLoadError, match="Unsupported qtype Q8_0 on bias 'blk.0.attn_q.bias'"
    ):
        GGUFModelLoader(
            gguf_path,
            config_dir=cfg_dir,
            target_dtype=mx.float32,
        ).load()


def test_rejects_missing_config(tmp_path):
    gguf_path, _ = _build_dense_fixture(tmp_path, "qwen3", has_qk_norm=True)
    empty = tmp_path / "no_config"
    empty.mkdir()
    with pytest.raises(GGUFLoadError, match="No config.json"):
        GGUFModelLoader(
            gguf_path,
            config_dir=str(empty),
            target_dtype=mx.float32,
        ).load()


def test_for_model_builds_loader_that_stays_quantized(tmp_path):
    # The lifecycle-facing factory yields a loader whose model keeps the GGUF
    # quantized wrappers (Q8_0 in memory), not a dense dequantized fallback.
    gguf_path, cfg_dir = _build_dense_fixture(tmp_path, "qwen3", has_qk_norm=True)
    loader = GGUFModelLoader.for_model(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.float32
    )
    assert isinstance(loader, GGUFModelLoader)
    model, _ = loader.load()
    hist = _gguf_module_histogram(model)
    assert hist.get("GGUFEmbedding") == 1
    assert hist.get("GGUFLinear") == 2 * 7


def test_for_model_rejects_remote_reference(tmp_path):
    # A non-local model id (remote repo:quant) is out of scope; for_model must
    # fail fast rather than fall through to the generic loader.
    _, cfg_dir = _build_dense_fixture(tmp_path, "qwen3", has_qk_norm=True)
    with pytest.raises(GGUFLoadError, match="not a local .gguf file"):
        GGUFModelLoader.for_model(
            "org/qwen3-gguf:Q8_0", config_dir=cfg_dir, target_dtype=mx.float32
        )


def test_for_model_rejects_missing_tokenizer_dir(tmp_path):
    # --tokenizer omitted: config_dir resolves to the .gguf itself (no config.json).
    gguf_path, _ = _build_dense_fixture(tmp_path, "qwen3", has_qk_norm=True)
    with pytest.raises(GGUFLoadError, match="needs --tokenizer"):
        GGUFModelLoader.for_model(
            gguf_path, config_dir=gguf_path, target_dtype=mx.float32
        )


def test_cache_key_is_stable_dtype_and_config_dir_scoped(tmp_path):
    gguf_path, cfg_dir = _build_dense_fixture(tmp_path, "qwen3", has_qk_norm=True)
    key = GGUFModelLoader.cache_key(gguf_path, cfg_dir, target_dtype=mx.float16)
    assert key == GGUFModelLoader.cache_key(gguf_path, cfg_dir, target_dtype=mx.float16)
    # dtype-scoped: a different dtype must not be cross-served from cache.
    assert key != GGUFModelLoader.cache_key(
        gguf_path, cfg_dir, target_dtype=mx.bfloat16
    )
    # config_dir-scoped: the same .gguf with a different --tokenizer dir is a
    # different model (a .gguf carries weights only) and must not collide.
    assert key != GGUFModelLoader.cache_key(
        gguf_path, cfg_dir + "_other", target_dtype=mx.float16
    )
    # Full contract: model path + config-dir + dtype-scoped loader segment.
    assert key == (gguf_path, f"gguf:{cfg_dir}:{mx.float16}")
