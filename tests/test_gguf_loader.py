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


def _write_minimal_tokenizer(config_dir: str, vocab_size: int) -> None:
    """Write a tiny WordLevel fast tokenizer into a synthetic config dir.

    transformers builds a llama tokenizer through ``LlamaTokenizer``, which
    needs real tokenizer files (unlike qwen, which constructs from config
    alone), so the llama fixtures must ship a tokenizer to reach ``load()``.
    """

    def _special(i: int, content: str) -> dict:
        return {
            "id": i,
            "content": content,
            "special": True,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
        }

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    vocab.update({f"t{i}": i for i in range(3, vocab_size)})
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [_special(0, "<unk>"), _special(1, "<s>"), _special(2, "</s>")],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    }
    dir_path = Path(config_dir)
    (dir_path / "tokenizer.json").write_text(json.dumps(tokenizer))
    (dir_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
    )


@pytest.mark.parametrize("quant_type", [QT.Q8_0, QT.Q4_0])
def test_loads_dense_llama_installs_wrappers(tmp_path, quant_type):
    # llama: separate q/k/v, no q/k norm, no attention bias, tied embeddings.
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path, "llama", has_qk_norm=False, quant_type=quant_type
    )
    _write_minimal_tokenizer(cfg_dir, 256)
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


def test_untied_llama_installs_lm_head(tmp_path):
    # The 8B-shaped path: untied output, no biases (llama attention_bias=False).
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "llama",
        config_overrides={"tie_word_embeddings": False},
        has_qk_norm=False,
        with_bias=False,
    )
    _write_minimal_tokenizer(cfg_dir, 256)
    model, _ = GGUFModelLoader(
        gguf_path,
        config_dir=cfg_dir,
        target_dtype=mx.float32,
    ).load()
    q_proj = model.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, GGUFLinear)
    assert "bias" not in q_proj  # llama q/k/v are bias-free
    assert isinstance(model.lm_head, GGUFLinear)  # untied output -> real lm_head
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == 256


def test_llama_skips_tie_redundant_output_when_config_omits_tie_flag(tmp_path):
    # Omitted-tie branch for the new arch: the loader must read the resolved
    # model.args tie flag (llama default True), not a config.get default.
    d = _dims(_tiny_config("llama"))
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "llama",
        has_qk_norm=False,
        inject={"output.weight": ("q", (d["vocab"], d["h"]))},
    )
    config_path = Path(cfg_dir) / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("tie_word_embeddings")
    config_path.write_text(json.dumps(config))
    _write_minimal_tokenizer(cfg_dir, 256)

    model, _ = GGUFModelLoader(
        gguf_path,
        config_dir=cfg_dir,
        target_dtype=mx.float32,
    ).load()

    assert not hasattr(model, "lm_head")
    assert _gguf_module_histogram(model).get("GGUFEmbedding") == 1


@pytest.mark.parametrize(
    "moe_tensor",
    ["blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_exps.weight"],
)
def test_rejects_out_of_scope_moe_tensor_under_llama_arch(tmp_path, moe_tensor):
    # Guard-widening regression: admitting "llama" makes llama-family MoE files
    # (Mixtral et al., general.architecture="llama") reachable. The router
    # (ffn_gate_inp) and experts (ffn_gate_exps) must both fail fast at preflight,
    # not slip through to an "unmapped tensor" error deep in the load.
    d = _dims(_tiny_config("llama"))
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "llama",
        has_qk_norm=False,
        inject={moe_tensor: ("f", (2, d["h"]))},
    )
    with pytest.raises(GGUFLoadError, match="Out-of-scope GGUF tensor"):
        GGUFModelLoader(
            gguf_path,
            config_dir=cfg_dir,
            target_dtype=mx.float32,
        ).load()


def _rope_inv_index(out_features, n_head):
    # Independent reference for llama.cpp's inverse q/k RoPE row permutation.
    return (
        mx.arange(out_features)
        .reshape(n_head, out_features // n_head // 2, 2)
        .swapaxes(1, 2)
        .reshape(out_features)
    )


def _plain_qk_fixture(tmp_path, model_type, *, has_qk_norm=False):
    # A dense fixture whose attn_q/attn_k are written PLAIN (F32), to exercise
    # the plain-weight branch of _partition (llama.cpp permutes q/k regardless
    # of qtype, so the plain path must un-permute too).
    d = _dims(_tiny_config(model_type))
    drop = {f"blk.{i}.attn_{p}.weight" for i in range(d["layers"]) for p in ("q", "k")}
    inject = {}
    for i in range(d["layers"]):
        inject[f"blk.{i}.attn_q.weight"] = ("f", (d["qd"], d["h"]))
        inject[f"blk.{i}.attn_k.weight"] = ("f", (d["kvd"], d["h"]))
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path, model_type, has_qk_norm=has_qk_norm, drop=drop, inject=inject
    )
    _write_minimal_tokenizer(cfg_dir, 256)
    return gguf_path, cfg_dir, d


def test_plain_llama_qk_are_row_unpermuted(tmp_path):
    gguf_path, cfg_dir, d = _plain_qk_fixture(tmp_path, "llama")
    raw_q = mx.load(gguf_path)["blk.0.attn_q.weight"]

    model, _ = GGUFModelLoader(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.float32
    ).load()

    cfg = _tiny_config("llama")
    n_head, n_kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    raw_k = mx.load(gguf_path)["blk.0.attn_k.weight"]
    got_q = model.model.layers[0].self_attn.q_proj.weight
    got_k = model.model.layers[0].self_attn.k_proj.weight
    assert bool(mx.array_equal(got_q, raw_q[_rope_inv_index(d["qd"], n_head)]).item())
    assert bool(mx.array_equal(got_k, raw_k[_rope_inv_index(d["kvd"], n_kv)]).item())
    # A non-trivial reorder actually happened (not identity), for q and k (GQA).
    assert not bool(mx.array_equal(got_q, raw_q).item())
    assert not bool(mx.array_equal(got_k, raw_k).item())


def test_plain_qwen_qk_are_not_permuted(tmp_path):
    # qwen uses the HF RoPE layout: q/k must load byte-identical (no permute).
    gguf_path, cfg_dir, _ = _plain_qk_fixture(tmp_path, "qwen3", has_qk_norm=True)
    raw_q = mx.load(gguf_path)["blk.0.attn_q.weight"]

    model, _ = GGUFModelLoader(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.float32
    ).load()

    got = model.model.layers[0].self_attn.q_proj.weight
    assert bool(mx.array_equal(got, raw_q).item())


def test_llama_loads_when_config_omits_head_dim(tmp_path):
    # head_dim is resolved onto the built q/k projections, not model.args, so the
    # permute index must read out_features off the built object (#449 omitted-
    # default). Every other fixture sets head_dim; this pops it.
    gguf_path, cfg_dir = _build_dense_fixture(tmp_path, "llama", has_qk_norm=False)
    _write_minimal_tokenizer(cfg_dir, 256)
    config_path = Path(cfg_dir) / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("head_dim")  # derived head_dim == hidden_size // heads == 16
    config_path.write_text(json.dumps(config))

    model, _ = GGUFModelLoader(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.float32
    ).load()

    assert _gguf_module_histogram(model).get("GGUFLinear") == 2 * 7
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape == (1, 3, 256)


def test_llama_qk_bias_is_row_unpermuted(tmp_path):
    # llama.cpp permutes q/k bias with the same RoPE index as the weight; a
    # llama with attention_bias must un-permute the bias too, or attention is
    # silently wrong (permuted weight + un-permuted bias). v bias is untouched.
    d = _dims(_tiny_config("llama"))
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "llama",
        config_overrides={"attention_bias": True},
        has_qk_norm=False,
        with_bias=True,
        # llama attention_bias also biases o_proj; _dense_tensor_specs only writes
        # q/k/v biases, so supply the output bias the built model expects.
        inject={
            f"blk.{i}.attn_output.bias": ("f", (d["h"],)) for i in range(d["layers"])
        },
    )
    _write_minimal_tokenizer(cfg_dir, 256)
    arrays = mx.load(gguf_path)
    raw_q_bias, raw_v_bias = arrays["blk.0.attn_q.bias"], arrays["blk.0.attn_v.bias"]

    model, _ = GGUFModelLoader(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.float32
    ).load()

    attn = model.model.layers[0].self_attn
    n_head = _tiny_config("llama")["num_attention_heads"]
    assert bool(
        mx.array_equal(
            attn.q_proj.bias, raw_q_bias[_rope_inv_index(d["qd"], n_head)]
        ).item()
    )
    assert not bool(mx.array_equal(attn.q_proj.bias, raw_q_bias).item())
    # v bias is not RoPE'd: byte-identical.
    assert bool(mx.array_equal(attn.v_proj.bias, raw_v_bias).item())


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


def test_direct_loader_keeps_quantized_wrappers(tmp_path):
    gguf_path, cfg_dir = _build_dense_fixture(tmp_path, "qwen3", has_qk_norm=True)
    loader = GGUFModelLoader(gguf_path, config_dir=cfg_dir, target_dtype=mx.float32)
    assert isinstance(loader, GGUFModelLoader)
    model, _ = loader.load()
    hist = _gguf_module_histogram(model)
    assert hist.get("GGUFEmbedding") == 1
    assert hist.get("GGUFLinear") == 2 * 7


def test_direct_loader_rejects_remote_reference(tmp_path):
    _, cfg_dir = _build_dense_fixture(tmp_path, "qwen3", has_qk_norm=True)
    with pytest.raises(GGUFLoadError, match="Not a local .gguf file"):
        GGUFModelLoader(
            "org/qwen3-gguf:Q8_0", config_dir=cfg_dir, target_dtype=mx.float32
        ).load()


def test_direct_loader_rejects_missing_config_dir(tmp_path):
    gguf_path, _ = _build_dense_fixture(tmp_path, "qwen3", has_qk_norm=True)
    with pytest.raises(GGUFLoadError, match="No config.json"):
        GGUFModelLoader(gguf_path, config_dir=gguf_path, target_dtype=mx.float32).load()
