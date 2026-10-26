# SPDX-License-Identifier: Apache-2.0
"""Deterministic loader-contract tests for ``vllm_metal.gguf.loader``.

Synthetic GGUF fixtures built with ``gguf.GGUFWriter`` + a tiny config exercise
the loader contract offline: partition, install, tie-skip, bias side-map,
completeness, and fail-fast rejection of unsupported files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from huggingface_hub import constants as hf_constants
from mlx.utils import tree_flatten
from mlx_lm import utils as mlx_lm_utils

gguf = pytest.importorskip("gguf")

import vllm_metal.gguf.loader as gguf_loader  # noqa: E402
from tests.gguf_kquant_fixtures import build_kquant_blocks  # noqa: E402
from vllm_metal.gguf.adapter import GGUFModelAdapter  # noqa: E402
from vllm_metal.gguf.loader import GGUFLoadError, GGUFModelLoader  # noqa: E402
from vllm_metal.gguf.mlx_native import GGUFMLXQuantizedTensor  # noqa: E402
from vllm_metal.gguf.raw_block import (  # noqa: E402
    RAW_KERNEL_GGUF_TYPES,
    GGUFRawBlockTensor,
)
from vllm_metal.gguf.source import GGUFLoadSource  # noqa: E402
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


# K-quant superblocks are 256 wide, so K-quant fixtures need 256-wide rows.
_KQUANT_CONFIG = {"hidden_size": 256, "intermediate_size": 256, "head_dim": 64}


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
    Tests may inject ``"iq4_xs"``/``"q6_k"`` zero blocks for tables that are
    never read or must be rejected; K-quant ``"q"`` weights get field-built
    blocks because gguf-py can dequantize K-quants but not quantize them.
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
    values: dict[str, np.ndarray] | None = None,
) -> None:
    rng = np.random.default_rng(0)
    writer = gguf.GGUFWriter(str(path), arch)
    for name, (kind, shape) in {**specs, **(inject or {})}.items():
        data = rng.standard_normal(shape).astype(np.float32)
        data = (values or {}).get(name, data)
        qtype = (quant_overrides or {}).get(name)
        if kind in ("iq4_xs", "q6_k"):
            raw_qtype = {"iq4_xs": QT.IQ4_XS, "q6_k": QT.Q6_K}[kind]
            block_size, type_size = gguf.GGML_QUANT_SIZES[raw_qtype]
            assert shape[-1] % block_size == 0
            packed_shape = (
                *shape[:-1],
                shape[-1] // block_size * type_size,
            )
            raw = np.zeros(packed_shape, dtype=np.uint8)
            writer.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=raw_qtype)
        elif kind == "q" or qtype is not None:
            raw_dtype = qtype or quant_type
            if raw_dtype in (QT.Q2_K, QT.Q3_K, QT.Q4_K, QT.Q5_K, QT.Q6_K):
                raw = build_kquant_blocks(shape[0], shape[-1], raw_dtype)
            else:
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
    values: dict[str, np.ndarray] | None = None,
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
        values=values,
    )
    return str(gguf_path), str(tmp_path)


def _file_quant_tensor(
    gguf_path: str, name: str
) -> GGUFMLXQuantizedTensor | GGUFRawBlockTensor:
    """Build the file's untransformed quantized tensor straight from its bytes."""
    tensor = next(t for t in gguf.GGUFReader(gguf_path).tensors if t.name == name)
    logical = tuple(int(dim) for dim in reversed(tensor.shape))
    data = tensor.data.reshape(-1)
    if tensor.tensor_type in RAW_KERNEL_GGUF_TYPES:
        return GGUFRawBlockTensor.from_raw_blocks(data, logical, tensor.tensor_type)
    return GGUFMLXQuantizedTensor.from_raw_blocks(data, logical, tensor.tensor_type)


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


def _assert_dense_wrapper_histogram(model: nn.Module) -> None:
    hist = _gguf_module_histogram(model)
    assert hist.get("GGUFEmbedding") == 1
    assert hist.get("GGUFLinear") == 2 * 7


def _assert_forward_vocab_shape(model: nn.Module) -> None:
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape == (1, 3, 256)


@pytest.mark.parametrize("quant_type", [QT.Q8_0, QT.Q4_0, QT.Q4_1, QT.Q5_0, QT.Q5_1])
@pytest.mark.parametrize(
    ("model_type", "has_qk_norm", "needs_tokenizer"),
    [("qwen3", True, False), ("llama", False, True)],
)
def test_loads_dense_model_installs_wrappers(
    tmp_path, quant_type, model_type, has_qk_norm, needs_tokenizer
):
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path, model_type, has_qk_norm=has_qk_norm, quant_type=quant_type
    )
    if needs_tokenizer:
        _write_minimal_tokenizer(cfg_dir, 256)
    model, _ = GGUFModelLoader(
        gguf_path,
        config_dir=cfg_dir,
        target_dtype=mx.float32,
    ).load()
    _assert_dense_wrapper_histogram(model)
    _assert_forward_vocab_shape(model)


@pytest.mark.parametrize("target_dtype", [mx.bfloat16, mx.float16])
def test_plain_tensors_follow_target_dtype(tmp_path, target_dtype):
    """Norm weights are F32 in the file but must load at the compute dtype.

    Left as F32 they promote the activations on the first RMSNorm and the
    whole forward pass runs in float32 whatever ``--dtype`` asked for.
    """
    gguf_path, cfg_dir = _build_dense_fixture(tmp_path, "qwen3", has_qk_norm=True)
    model, _ = GGUFModelLoader(
        gguf_path,
        config_dir=cfg_dir,
        target_dtype=target_dtype,
    ).load()

    plain = {
        name: leaf.dtype
        for name, leaf in tree_flatten(model.parameters())
        if isinstance(leaf, mx.array) and mx.issubdtype(leaf.dtype, mx.floating)
    }
    assert plain, "fixture should expose plain floating parameters"
    assert any("norm" in name for name in plain)
    assert {str(dtype) for dtype in plain.values()} == {str(target_dtype)}

    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.dtype == target_dtype


def test_loads_cached_remote_model_offline(tmp_path, monkeypatch):
    cache_dir = tmp_path / "hub"
    repository = cache_dir / "models--org--tiny-GGUF"
    revision = "a" * 40
    snapshot = repository / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (repository / "refs").mkdir()
    (repository / "refs" / "weights-revision").write_text(revision)
    gguf_path, config_dir = _build_dense_fixture(snapshot, "qwen3", has_qk_norm=True)
    weights_path = snapshot / "tiny-Q8_0.gguf"
    Path(gguf_path).rename(weights_path)
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    source = GGUFLoadSource.from_model_config(
        SimpleNamespace(
            quantization="gguf",
            model_weights="org/tiny-GGUF:Q8_0",
            model=config_dir,
            tokenizer=config_dir,
            revision="weights-revision",
            tokenizer_revision=None,
            hf_token=None,
        ),
        SimpleNamespace(download_dir=str(cache_dir), ignore_patterns=None),
    )
    assert source is not None
    model, _ = GGUFModelLoader(
        source.weights_path,
        config_dir=source.config_dir,
        tokenizer_dir=source.tokenizer_dir,
        target_dtype=mx.float32,
    ).load()
    reference, _ = GGUFModelLoader(
        weights_path, config_dir=config_dir, target_dtype=mx.float32
    ).load()
    tokens = mx.array([[1, 2, 3]])
    np.testing.assert_array_equal(np.array(model(tokens)), np.array(reference(tokens)))


@pytest.mark.parametrize("output_kind", ["q6_k", "iq4_xs"])
def test_skips_tie_redundant_output(tmp_path, output_kind):
    # Tied config, but the GGUF still carries a redundant output.weight, as
    # llama.cpp exports of tied models do. It is never read, so even a qtype
    # the loader cannot execute (IQ4_XS) does not block the model.
    d = _dims(_tiny_config("qwen3", **_KQUANT_CONFIG))
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen3",
        config_overrides=_KQUANT_CONFIG,
        has_qk_norm=True,
        inject={"output.weight": (output_kind, (d["vocab"], d["h"]))},
    )

    model, _ = GGUFModelLoader(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.float32
    ).load()

    assert not hasattr(model, "lm_head")
    # The tied head runs through the GGUFEmbedding's as_linear.
    assert _gguf_module_histogram(model).get("GGUFEmbedding") == 1


@pytest.mark.parametrize(
    ("model_type", "has_qk_norm", "with_bias", "needs_tokenizer"),
    [("qwen2", False, True, False), ("llama", False, False, True)],
)
def test_skips_tie_redundant_output_when_config_omits_tie_flag(
    tmp_path, model_type, has_qk_norm, with_bias, needs_tokenizer
):
    d = _dims(_tiny_config(model_type))
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        model_type,
        has_qk_norm=has_qk_norm,
        with_bias=with_bias,
        inject={"output.weight": ("q", (d["vocab"], d["h"]))},
    )
    config_path = Path(cfg_dir) / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("tie_word_embeddings")
    config_path.write_text(json.dumps(config))
    if needs_tokenizer:
        _write_minimal_tokenizer(cfg_dir, 256)

    model, _ = GGUFModelLoader(
        gguf_path,
        config_dir=cfg_dir,
        target_dtype=mx.float32,
    ).load()

    assert not hasattr(model, "lm_head")
    assert _gguf_module_histogram(model).get("GGUFEmbedding") == 1


@pytest.mark.parametrize(
    ("model_type", "has_qk_norm", "with_bias", "needs_tokenizer"),
    [("qwen2", False, True, False), ("llama", False, False, True)],
)
def test_untied_model_installs_lm_head_and_bias_policy(
    tmp_path, model_type, has_qk_norm, with_bias, needs_tokenizer
):
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        model_type,
        config_overrides={"tie_word_embeddings": False},
        has_qk_norm=has_qk_norm,
        with_bias=with_bias,
    )
    if needs_tokenizer:
        _write_minimal_tokenizer(cfg_dir, 256)
    model, _ = GGUFModelLoader(
        gguf_path,
        config_dir=cfg_dir,
        target_dtype=mx.float32,
    ).load()
    # Assert through the model's own public structure, not loader internals.
    q_proj = model.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, GGUFLinear)
    assert ("bias" in q_proj) is with_bias
    assert isinstance(model.lm_head, GGUFLinear)  # untied output -> real lm_head
    _assert_forward_vocab_shape(model)


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


@pytest.mark.parametrize(
    "quant_type",
    [
        QT.Q8_0,
        QT.Q4_0,
        QT.Q4_1,
        QT.Q5_0,
        QT.Q5_1,
        QT.Q2_K,
        QT.Q3_K,
        QT.Q4_K,
        QT.Q5_K,
        QT.Q6_K,
    ],
)
def test_quantized_llama_qk_are_row_unpermuted(tmp_path, quant_type):
    # The main quantized-path behavior: installed q/k GGUFLinear tensors carry the
    # RoPE-un-permuted quantized weight, not the raw (llama.cpp-permuted) one.
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "llama",
        config_overrides=_KQUANT_CONFIG,
        has_qk_norm=False,
        quant_type=quant_type,
    )
    _write_minimal_tokenizer(cfg_dir, 256)
    cfg = _tiny_config("llama", **_KQUANT_CONFIG)
    d = _dims(cfg)
    raw_q = _file_quant_tensor(gguf_path, "blk.0.attn_q.weight")
    raw_k = _file_quant_tensor(gguf_path, "blk.0.attn_k.weight")
    exp_q = raw_q.permute_rows(_rope_inv_index(d["qd"], cfg["num_attention_heads"]))
    exp_k = raw_k.permute_rows(_rope_inv_index(d["kvd"], cfg["num_key_value_heads"]))

    model, _ = GGUFModelLoader(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.float32
    ).load()

    got_q = model.model.layers[0].self_attn.q_proj.tensor
    got_k = model.model.layers[0].self_attn.k_proj.tensor
    assert bool(mx.array_equal(got_q.qweight, exp_q.qweight).item())
    assert bool(mx.array_equal(got_k.qweight, exp_k.qweight).item())
    assert not bool(mx.array_equal(got_q.qweight, raw_q.qweight).item())
    assert not bool(mx.array_equal(got_k.qweight, raw_k.qweight).item())


def test_loads_untied_mistral_via_llama_arch_mapping(tmp_path):
    # Real Mistral GGUFs declare general.architecture="llama" while the config
    # says model_type="mistral"; the adapter mapping admits the pair and the
    # llama RoPE q/k un-permutation applies through it. Mirrors the real v0.3
    # shape: untied (lm_head present) and head_dim OMITTED (hidden // heads).
    # Arrange
    d = _dims(_tiny_config("mistral"))
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "mistral",
        config_overrides={"tie_word_embeddings": False},
        has_qk_norm=False,
        gguf_arch="llama",
    )
    _write_minimal_tokenizer(cfg_dir, 256)
    config_path = Path(cfg_dir) / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("head_dim")  # derived head_dim == hidden_size // heads == 16
    config_path.write_text(json.dumps(config))
    cfg = _tiny_config("mistral")
    raw_q = _file_quant_tensor(gguf_path, "blk.0.attn_q.weight")
    raw_k = _file_quant_tensor(gguf_path, "blk.0.attn_k.weight")
    exp_q = raw_q.permute_rows(_rope_inv_index(d["qd"], cfg["num_attention_heads"]))
    exp_k = raw_k.permute_rows(_rope_inv_index(d["kvd"], cfg["num_key_value_heads"]))

    # Act
    model, _ = GGUFModelLoader(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.float32
    ).load()
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)

    # Assert
    assert isinstance(model.lm_head, GGUFLinear)  # untied head installed
    got_q = model.model.layers[0].self_attn.q_proj.tensor
    got_k = model.model.layers[0].self_attn.k_proj.tensor
    for got, exp, raw in ((got_q, exp_q, raw_q), (got_k, exp_k, raw_k)):
        assert bool(mx.array_equal(got.qweight, exp.qweight).item())
        assert bool(mx.array_equal(got.scales, exp.scales).item())
        assert bool(mx.array_equal(got.biases, exp.biases).item())
        assert not bool(mx.array_equal(got.qweight, raw.qweight).item())
    assert out.shape == (1, 3, 256)


def test_resolve_arch_accepts_llama_gguf_with_mistral_config():
    # The one admitted pairing: a llama-arch .gguf with a mistral config.
    arch = GGUFModelAdapter.resolve_arch(gguf_arch="llama", config_model_type="mistral")

    assert arch == "llama"


@pytest.mark.parametrize(
    ("gguf_arch", "config_model_type"),
    [
        ("mistral", "mistral"),
        ("mistral3", "mistral"),
        ("mistral4", "mistral"),
        ("qwen35moe", "qwen3_5_moe"),
    ],
)
def test_resolve_arch_rejects_unsupported_gguf_archs(gguf_arch, config_model_type):
    # The config-side mapping never widens the .gguf side: a file declaring a
    # raw "mistral" arch is not a llama.cpp product, and the multimodal
    # mistral3/mistral4 and MoE qwen35moe archs are out of scope; all stay
    # allowlist-rejected.
    with pytest.raises(GGUFLoadError) as unsupported:
        GGUFModelAdapter.resolve_arch(
            gguf_arch=gguf_arch, config_model_type=config_model_type
        )

    assert str(unsupported.value) == (
        f"Architecture {gguf_arch!r} is not supported; the GGUF loader supports "
        "['llama', 'qwen2', 'qwen3', 'qwen35']."
    )


def test_resolve_arch_mismatch_names_raw_and_mapped_config_type():
    # The mismatch error shows the GGUF arch, the raw config model_type, and
    # the GGUF arch that model_type maps to.
    with pytest.raises(GGUFLoadError) as mismatch:
        GGUFModelAdapter.resolve_arch(gguf_arch="qwen2", config_model_type="mistral")

    assert str(mismatch.value) == (
        "GGUF architecture 'qwen2' does not match config model_type 'mistral' "
        "(maps to GGUF arch 'llama'); the .gguf and config_dir describe "
        "different models."
    )


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


def test_rejects_unsupported_arch(tmp_path):
    # llama.cpp converts Qwen3.5 as "qwen35", so a raw "qwen3_5" arch stays rejected.
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path, "qwen3", config_overrides={"model_type": "qwen3_5"}, has_qk_norm=True
    )
    with pytest.raises(GGUFLoadError, match="'qwen3_5' is not supported"):
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


@pytest.mark.parametrize(
    ("name", "spec"),
    [("blk.0.attn_qkv.weight", ("q", (64, 64))), ("blk.0.ssm_a", ("f", (4,)))],
    ids=["attn_qkv", "ssm"],
)
def test_rejects_out_of_scope_tensor(tmp_path, name, spec):
    # qwen35 admits these markers; every other arch still rejects them.
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path, "qwen3", has_qk_norm=True, inject={name: spec}
    )
    with pytest.raises(GGUFLoadError, match=f"Out-of-scope GGUF tensor '{name}'"):
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


def test_rejects_unsupported_qtype_before_model_allocation(tmp_path, monkeypatch):
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen3",
        config_overrides=_KQUANT_CONFIG,
        has_qk_norm=True,
        inject={"blk.0.ffn_up.weight": ("iq4_xs", (256, 256))},
    )

    def fail_load_model(*args, **kwargs):
        raise AssertionError("qtype preflight must run before load_model")

    monkeypatch.setattr(gguf_loader, "load_model", fail_load_model)
    with pytest.raises(
        GGUFLoadError,
        match="Unsupported qtype IQ4_XS on mapped weight 'blk.0.ffn_up.weight'",
    ):
        GGUFModelLoader(
            gguf_path,
            config_dir=cfg_dir,
            target_dtype=mx.float32,
        ).load()


def test_builds_the_skeleton_lazily(tmp_path, monkeypatch):
    # The GGUF replaces every skeleton parameter, so the skeleton's random-init
    # weights must never be evaluated.
    gguf_path, cfg_dir = _build_dense_fixture(tmp_path, "qwen3", has_qk_norm=True)
    real_load_model = gguf_loader.load_model
    lazy_flags = []

    def spy_load_model(*args, **kwargs):
        lazy_flags.append(kwargs.get("lazy"))
        return real_load_model(*args, **kwargs)

    monkeypatch.setattr(gguf_loader, "load_model", spy_load_model)

    GGUFModelLoader(gguf_path, config_dir=cfg_dir, target_dtype=mx.float32).load()

    assert lazy_flags == [True]


def test_lazy_skeleton_keeps_peak_memory_off_the_eager_path(tmp_path, monkeypatch):
    """Peak-memory regression check for the lazy skeleton.

    ``load_model(lazy=False)`` ends with ``mx.eval(model.parameters())``,
    materializing every skeleton parameter the GGUF then replaces — pure
    overhead.  Load the same fixture twice, once lazily and once with the
    regression forced (lazy=False monkeypatched in), and pin the lazy peak
    well under the eager one: if the lazy flag is lost the two peaks
    converge and the ratio collapses toward 1.
    """
    # Inflated dims so the fp32 skeleton is tens of MB — well above the
    # quantized GGUF payload and any per-process noise floor.
    config_overrides = {
        "hidden_size": 512,
        "num_hidden_layers": 4,
        "intermediate_size": 1024,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "vocab_size": 1024,
    }
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path, "qwen3", config_overrides=config_overrides, has_qk_norm=True
    )
    real_load_model = gguf_loader.load_model

    def load_eager(*args, **kwargs):
        kwargs["lazy"] = False
        return real_load_model(*args, **kwargs)

    def peak_of_load() -> int:
        # Peak above what the process already holds, so memory retained by
        # earlier tests (or the first load) cannot skew the ratio.
        mx.reset_peak_memory()
        baseline = mx.get_active_memory()
        GGUFModelLoader(gguf_path, config_dir=cfg_dir, target_dtype=mx.float32).load()
        return mx.get_peak_memory() - baseline

    lazy_peak = peak_of_load()
    monkeypatch.setattr(gguf_loader, "load_model", load_eager)
    eager_peak = peak_of_load()

    assert lazy_peak < eager_peak
    # The lazy peak is the GGUF payload alone (~1 byte/param Q8); the eager
    # peak adds the full fp32 skeleton (~4 bytes/param), so lazy should sit
    # far below — the bound keeps slack for CI variance while still failing
    # the moment the flag is dropped (ratio -> 1).
    assert lazy_peak <= 0.75 * eager_peak


def _mixed_precision_predicate(path: str, module: nn.Module) -> dict | bool:
    # mlx_lm's mixed recipes record per-layer entries like this in the config.
    return {"group_size": 64, "bits": 8} if path.endswith("down_proj") else True


def test_mlx_quantized_companion_loads_like_plain_companion(tmp_path):
    gguf_path, plain_dir = _build_dense_fixture(tmp_path, "qwen3", has_qk_norm=True)
    quantized_dir = tmp_path / "mlx-quantized"
    model, config = mlx_lm_utils.load_model(Path(plain_dir), strict=False)
    model, quantized_config = mlx_lm_utils.quantize_model(
        model,
        config,
        group_size=None,
        bits=None,
        quant_predicate=_mixed_precision_predicate,
    )
    mlx_lm_utils.save_model(quantized_dir, model)
    mlx_lm_utils.save_config(quantized_config, quantized_dir / "config.json")
    tokens = mx.array([[1, 2, 3]])
    reference, _ = GGUFModelLoader(
        gguf_path, config_dir=plain_dir, target_dtype=mx.float32
    ).load()
    expected = np.array(reference(tokens))

    loaded, _ = GGUFModelLoader(
        gguf_path, config_dir=quantized_dir, target_dtype=mx.float32
    ).load()
    actual = np.array(loaded(tokens))

    np.testing.assert_array_equal(actual, expected)


def test_rejects_unsupported_untied_output(tmp_path):
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen3",
        config_overrides={**_KQUANT_CONFIG, "tie_word_embeddings": False},
        has_qk_norm=True,
        inject={"output.weight": ("iq4_xs", (256, 256))},
    )

    with pytest.raises(
        GGUFLoadError,
        match="Unsupported qtype IQ4_XS on mapped weight 'output.weight'",
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
        GGUFLoadError, match="Unsupported qtype Q8_0 on 'blk.0.attn_q.bias'"
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
    _assert_dense_wrapper_histogram(model)


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


@pytest.mark.parametrize(
    "shard_name",
    [
        "qwen3-Q8_0-00001-of-00002.gguf",
        "qwen3-Q8_0-00002-of-00002.gguf",
        "qwen3-Q8_0-1-of-2.gguf",
    ],
)
def test_rejects_local_sharded_gguf_before_reading(tmp_path, shard_name):
    # A dummy payload proves the guard fires on the name, before GGUFReader.
    shard_path = tmp_path / shard_name
    shard_path.write_bytes(b"not a gguf payload")
    expected = (
        f"Sharded GGUF files are not supported yet: {str(shard_path)!r}; "
        "pass a single-file .gguf or merge the shards."
    )

    with pytest.raises(GGUFLoadError) as excinfo:
        GGUFModelLoader(
            str(shard_path), config_dir=str(tmp_path), target_dtype=mx.float32
        ).load()

    assert str(excinfo.value) == expected


def test_loads_single_file_set_named_like_one_shard(tmp_path):
    gguf_path, cfg_dir = _build_dense_fixture(tmp_path, "qwen3", has_qk_norm=True)
    single = tmp_path / "qwen3-Q8_0-00001-of-00001.gguf"
    Path(gguf_path).rename(single)

    model, _ = GGUFModelLoader(
        str(single), config_dir=cfg_dir, target_dtype=mx.float32
    ).load()

    _assert_dense_wrapper_histogram(model)


@pytest.mark.parametrize("plain_type", [QT.F16, QT.F32, QT.BF16])
def test_loads_plain_typed_checkpoint_exactly(tmp_path, plain_type):
    # llama.cpp plain exports keep norms F32; matrix weights carry the type.
    # Values below float16's normal range catch any detour through float16.
    tiny = np.random.default_rng(1).standard_normal((128, 64)).astype(np.float32)
    tiny[0] *= 1e-7
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen3",
        has_qk_norm=True,
        quant_type=plain_type,
        values={"blk.0.ffn_gate.weight": tiny},
    )
    source = next(
        t
        for t in gguf.GGUFReader(gguf_path).tensors
        if t.name == "blk.0.ffn_gate.weight"
    )
    expected = gguf.quants.dequantize(source.data, plain_type).astype(np.float32)

    model, _ = GGUFModelLoader(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.float32
    ).load()
    hist = _gguf_module_histogram(model)
    installed = np.array(model.model.layers[0].mlp.gate_proj.weight)

    assert "GGUFEmbedding" not in hist
    assert "GGUFLinear" not in hist
    assert np.array_equal(installed, expected)
    _assert_forward_vocab_shape(model)


def test_rejects_unsupported_kquant_weight_at_preflight(tmp_path):
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen3",
        has_qk_norm=True,
        inject={"blk.0.ffn_gate.weight": ("iq4_xs", (128, 256))},
    )

    with pytest.raises(GGUFLoadError) as excinfo:
        GGUFModelLoader(gguf_path, config_dir=cfg_dir, target_dtype=mx.float32).load()

    assert str(excinfo.value) == (
        "Unsupported qtype IQ4_XS on mapped weight 'blk.0.ffn_gate.weight'; "
        "only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K "
        "(and plain F32/F16/BF16) are supported."
    )


@pytest.mark.parametrize(
    ("body_type", "embd_type"),
    [
        pytest.param(QT.Q3_K, QT.Q6_K, id="q3_k_m"),
        pytest.param(QT.Q4_K, QT.Q6_K, id="q4_k_m"),
        pytest.param(QT.Q5_K, QT.Q6_K, id="q5_k_m"),
        pytest.param(QT.Q4_K, QT.Q8_0, id="q4_k_l"),
        pytest.param(QT.Q4_0, QT.Q6_K, id="q4_0-q6_k-embd"),
        pytest.param(QT.Q4_K, QT.Q5_0, id="q4_k-q5_0-embd"),
        pytest.param(QT.Q5_K, QT.Q5_1, id="q5_k-q5_1-embd"),
    ],
)
def test_loads_llama_cpp_qtype_mixes(tmp_path, body_type, embd_type):
    # llama.cpp mixes qtypes inside one file: Q6_K, Q8_0 or Q5_0/Q5_1 fallback
    # embeddings next to a K-quant or Q4_0 body.
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen3",
        config_overrides=_KQUANT_CONFIG,
        has_qk_norm=True,
        quant_type=body_type,
        quant_overrides={"token_embd.weight": embd_type},
    )

    model, _ = GGUFModelLoader(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.float32
    ).load()

    assert model.model.embed_tokens.tensor.qweight_type == embd_type
    assert model.model.layers[0].mlp.down_proj.tensor.qweight_type == body_type
    _assert_forward_vocab_shape(model)


def test_loads_untied_q6_k_output_head(tmp_path):
    # Untied llama.cpp K-quant exports store the output head in Q6_K.
    gguf_path, cfg_dir = _build_dense_fixture(
        tmp_path,
        "qwen3",
        config_overrides={**_KQUANT_CONFIG, "tie_word_embeddings": False},
        has_qk_norm=True,
        quant_type=QT.Q4_K,
        quant_overrides={"output.weight": QT.Q6_K},
    )

    model, _ = GGUFModelLoader(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.float32
    ).load()

    assert model.lm_head.tensor.qweight_type == QT.Q6_K
    _assert_forward_vocab_shape(model)


def test_loads_read_only_gguf_file(tmp_path):
    # Hugging Face's shared blob store keeps downloads read-only, so loading
    # must only need read access to the file.
    gguf_path, cfg_dir = _build_dense_fixture(tmp_path, "qwen3", has_qk_norm=True)
    Path(gguf_path).chmod(0o444)

    model, _ = GGUFModelLoader(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.float32
    ).load()

    _assert_dense_wrapper_histogram(model)


# Every projection row is 256 wide so Q6_K superblocks fit, and the gated-delta
# kernel splits the key head dim across 32 lanes.
_QWEN35_TEXT_CONFIG = {
    "model_type": "qwen3_5_text",
    "hidden_size": 256,
    "intermediate_size": 256,
    "num_hidden_layers": 4,
    "full_attention_interval": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "vocab_size": 256,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 512,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 32,
}
# full_attention_interval 2 makes layers 1 and 3 full attention.
_QWEN35_GDN_LAYERS = (0, 2)

# GGUF tensor (after ``blk.N.``) -> mlx-lm parameter (after ``layers.N.``).
_QWEN35_LAYER_TENSORS = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
}
_QWEN35_GLOBAL_TENSORS = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
_QWEN35_LIVE_TO_GGUF_LAYER = {
    live: kind for kind, live in _QWEN35_LAYER_TENSORS.items()
}
_QWEN35_LIVE_TO_GGUF_GLOBAL = {
    live: name for name, live in _QWEN35_GLOBAL_TENSORS.items()
}
# llama.cpp keeps 1-D tensors and the depthwise conv kernel in F32.
_QWEN35_PLAIN_TENSORS = frozenset(
    {"ssm_a", "ssm_dt.bias", "ssm_conv1d.weight"}
    | {kind for kind in _QWEN35_LAYER_TENSORS if kind.endswith("norm.weight")}
    | {"output_norm.weight"}
)


@dataclass(frozen=True)
class _GDNGeometry:
    num_k_heads: int
    num_v_heads: int
    v_head_dim: int


# nk != nv // nk in r2/r3, so the tiled V-head order is not its own inverse.
_GDN_R1 = _GDNGeometry(num_k_heads=4, num_v_heads=4, v_head_dim=64)
_GDN_R2 = _GDNGeometry(num_k_heads=4, num_v_heads=8, v_head_dim=32)
_GDN_R3 = _GDNGeometry(num_k_heads=2, num_v_heads=6, v_head_dim=128)


def _qwen35_config(geometry: _GDNGeometry, *, tie: bool | None) -> dict:
    """A Qwen3_5ForConditionalGeneration config; ``tie=None`` omits the text
    tie flag the way the real 9B/27B configs do. The top-level flag always
    disagrees with the text one, so only the built text model resolves it."""
    text = {
        **_QWEN35_TEXT_CONFIG,
        "linear_num_key_heads": geometry.num_k_heads,
        "linear_num_value_heads": geometry.num_v_heads,
        "linear_value_head_dim": geometry.v_head_dim,
    }
    if tie is not None:
        text["tie_word_embeddings"] = tie
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "tie_word_embeddings": not tie,
        "text_config": text,
    }


def _qwen35_gguf_name(param_path: str) -> str:
    """Live mlx-lm parameter path -> the GGUF tensor llama.cpp writes for it."""
    path = param_path.removeprefix("language_model.")
    if not path.startswith("model.layers."):
        return _QWEN35_LIVE_TO_GGUF_GLOBAL[path]
    _, _, layer, leaf = path.split(".", 3)
    return f"blk.{layer}.{_QWEN35_LIVE_TO_GGUF_LAYER[leaf]}"


def _qwen35_kind(gguf_name: str) -> str:
    return gguf_name.split(".", 2)[2] if gguf_name.startswith("blk.") else gguf_name


def _reorder_v_heads(
    tensor: np.ndarray, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int
) -> np.ndarray:
    """numpy port of llama.cpp's ``_LinearAttentionVReorderBase._reorder_v_heads``."""
    shape = list(tensor.shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1 :]
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.reshape(new_shape).transpose(perm).reshape(shape)


def _v_head_width(kind: str, geometry: _GDNGeometry) -> int:
    wide = (
        "attn_qkv.weight",
        "ssm_conv1d.weight",
        "attn_gate.weight",
        "ssm_out.weight",
    )
    return geometry.v_head_dim if kind in wide else 1


def _qwen35_llama_cpp_tensor(
    kind: str, hf: np.ndarray, geometry: _GDNGeometry
) -> np.ndarray:
    """What llama.cpp's Qwen3.5 converter writes for one HF-layout tensor."""
    nk, width = geometry.num_k_heads, _v_head_width(kind, geometry)
    ratio = geometry.num_v_heads // nk
    qk_rows = 2 * nk * _QWEN35_TEXT_CONFIG["linear_key_head_dim"]
    if kind == "ssm_conv1d.weight":
        hf = hf[..., 0]
    if kind in ("attn_qkv.weight", "ssm_conv1d.weight"):
        v_rows = _reorder_v_heads(hf[qk_rows:], 0, nk, ratio, width)
        return np.concatenate([hf[:qk_rows], v_rows])
    if kind in ("attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight"):
        return _reorder_v_heads(hf, 0, nk, ratio, width)
    if kind == "ssm_dt.bias":
        return _reorder_v_heads(hf, 0, nk, ratio, 1)
    if kind == "ssm_a":
        return -np.exp(_reorder_v_heads(hf, 0, nk, ratio, 1))
    if kind == "ssm_out.weight":
        return _reorder_v_heads(hf, 1, nk, ratio, width)
    return hf


def _hf_order_index(kind: str, geometry: _GDNGeometry) -> np.ndarray:
    """Position in the llama.cpp layout of each HF row (ssm_out: column)."""
    nk, width = geometry.num_k_heads, _v_head_width(kind, geometry)
    tiled = _reorder_v_heads(
        np.arange(geometry.num_v_heads * width),
        0,
        nk,
        geometry.num_v_heads // nk,
        width,
    )
    index = np.argsort(tiled)
    if kind not in ("attn_qkv.weight", "ssm_conv1d.weight"):
        return index
    qk_rows = 2 * nk * _QWEN35_TEXT_CONFIG["linear_key_head_dim"]
    return np.concatenate([np.arange(qk_rows), qk_rows + index])


def _qwen35_hf_value(rng: np.random.Generator, kind: str, shape: tuple) -> np.ndarray:
    # Scaled so every head's gate stays sensitive to its own A_log/dt_bias/a.
    if kind == "ssm_a":
        value = np.log(rng.uniform(1.0, 16.0, shape))
    elif kind.endswith("norm.weight"):
        value = 1.0 + 0.1 * rng.standard_normal(shape)
    elif len(shape) > 1 and kind != "token_embd.weight":
        value = rng.standard_normal(shape) / np.sqrt(shape[1])
    else:
        value = rng.standard_normal(shape)
    return value.astype(np.float32)


def _build_qwen35_fixture(
    tmp_path: Path,
    geometry: _GDNGeometry,
    *,
    tie: bool | None = True,
    quant_type=QT.Q8_0,
    quant_overrides: dict[str, QT] | None = None,
    inject: dict | None = None,
) -> tuple[str, str, dict[str, np.ndarray]]:
    """Write a tiny Qwen3.5 config and the GGUF llama.cpp converts it to.

    Returns ``(gguf_path, config_dir, hf)``; ``hf`` maps each GGUF name to the
    HF-layout source value it was converted from.
    """
    config = _qwen35_config(geometry, tie=tie)
    (tmp_path / "config.json").write_text(json.dumps(config))
    skeleton, _ = mlx_lm_utils.load_model(tmp_path, lazy=True, strict=False)
    rng = np.random.default_rng(0)
    hf = {}
    for path, param in tree_flatten(skeleton.parameters()):
        name = _qwen35_gguf_name(path)
        hf[name] = _qwen35_hf_value(rng, _qwen35_kind(name), tuple(param.shape))
    values = {
        name: _qwen35_llama_cpp_tensor(_qwen35_kind(name), value, geometry)
        for name, value in hf.items()
        if name not in (inject or {})
    }
    specs = {
        name: ("f" if _qwen35_kind(name) in _QWEN35_PLAIN_TENSORS else "q", v.shape)
        for name, v in values.items()
    }
    gguf_path = tmp_path / "qwen35.gguf"
    _write_gguf(
        gguf_path,
        "qwen35",
        specs,
        quant_type=quant_type,
        quant_overrides=quant_overrides,
        inject=inject,
        values=values,
    )
    return str(gguf_path), str(tmp_path), hf


def _qwen35_source_model(cfg_dir: str, hf: dict[str, np.ndarray]) -> nn.Module:
    """The dense mlx-lm model the fixture's GGUF was converted from."""
    model, _ = mlx_lm_utils.load_model(Path(cfg_dir), lazy=True, strict=False)
    params = [path for path, _ in tree_flatten(model.parameters())]
    model.load_weights(
        [(path, mx.array(hf[_qwen35_gguf_name(path)])) for path in params]
    )
    model.eval()
    return model


def _gguf_dense(gguf_path: str, name: str) -> np.ndarray:
    """A file tensor's values, dequantized by the gguf-py oracle."""
    tensor = next(t for t in gguf.GGUFReader(gguf_path).tensors if t.name == name)
    return gguf.quants.dequantize(tensor.data, tensor.tensor_type).astype(np.float32)


def _packed_arrays(qt: GGUFMLXQuantizedTensor | GGUFRawBlockTensor) -> list:
    if isinstance(qt, GGUFRawBlockTensor):
        return [qt.qweight]
    return [qt.qweight, qt.scales, qt.biases]


def _gdn(model: nn.Module, layer: int) -> nn.Module:
    return model.language_model.model.layers[layer].linear_attn


@pytest.mark.parametrize(
    "geometry",
    [
        pytest.param(_GDN_R1, id="r1"),
        pytest.param(_GDN_R2, id="r2"),
        pytest.param(_GDN_R3, id="r3"),
    ],
)
def test_qwen35_loads_the_model_it_was_converted_from(tmp_path, geometry):
    # A plain file loads back to the exact source model, whatever the V-head ratio.
    gguf_path, cfg_dir, hf = _build_qwen35_fixture(
        tmp_path, geometry, tie=False, quant_type=QT.F32
    )
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    expected = np.array(_qwen35_source_model(cfg_dir, hf)(tokens))

    model, _ = GGUFModelLoader(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.float32
    ).load()
    logits = np.array(model(tokens))

    np.testing.assert_allclose(
        logits, expected, rtol=0, atol=float(np.abs(expected).max()) * 1e-6
    )


@pytest.mark.parametrize(
    "geometry", [pytest.param(_GDN_R2, id="r2"), pytest.param(_GDN_R3, id="r3")]
)
@pytest.mark.parametrize("quant_type", [QT.Q8_0, QT.Q6_K], ids=["affine", "raw_block"])
def test_qwen35_quantized_v_head_rows_are_untiled(tmp_path, quant_type, geometry):
    gguf_path, cfg_dir, _ = _build_qwen35_fixture(
        tmp_path, geometry, quant_type=quant_type
    )
    projections = {
        "attn_qkv.weight": "in_proj_qkv",
        "attn_gate.weight": "in_proj_z",
        "ssm_alpha.weight": "in_proj_a",
        "ssm_beta.weight": "in_proj_b",
    }
    raw = {
        (i, kind): _file_quant_tensor(gguf_path, f"blk.{i}.{kind}")
        for i in _QWEN35_GDN_LAYERS
        for kind in projections
    }
    expected = {
        key: qt.permute_rows(mx.array(_hf_order_index(key[1], geometry)))
        for key, qt in raw.items()
    }

    model, _ = GGUFModelLoader(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.float32
    ).load()

    for (i, kind), raw_qt in raw.items():
        got = getattr(_gdn(model, i), projections[kind]).tensor
        pairs = zip(
            _packed_arrays(got), _packed_arrays(expected[(i, kind)]), strict=True
        )
        assert all(bool(mx.array_equal(a, b).item()) for a, b in pairs), (i, kind)
        assert not bool(mx.array_equal(got.qweight, raw_qt.qweight).item()), (i, kind)


@pytest.mark.parametrize(
    "geometry", [pytest.param(_GDN_R2, id="r2"), pytest.param(_GDN_R3, id="r3")]
)
@pytest.mark.parametrize(
    "ssm_out_type", [QT.Q8_0, QT.Q6_K], ids=["affine", "raw_block"]
)
def test_qwen35_ssm_out_reads_hf_order_columns(tmp_path, ssm_out_type, geometry):
    gguf_path, cfg_dir, _ = _build_qwen35_fixture(
        tmp_path,
        geometry,
        quant_overrides={
            f"blk.{i}.ssm_out.weight": ssm_out_type for i in _QWEN35_GDN_LAYERS
        },
    )
    columns = _hf_order_index("ssm_out.weight", geometry)
    width = geometry.num_v_heads * geometry.v_head_dim
    # The GDN runtime calls out_proj on (1, tokens, value_dim).
    x = np.random.default_rng(1).standard_normal((1, 3, width)).astype(np.float32)
    stored = {
        i: _gguf_dense(gguf_path, f"blk.{i}.ssm_out.weight") for i in _QWEN35_GDN_LAYERS
    }
    expected = {i: x @ w[:, columns].T for i, w in stored.items()}
    untransformed = {i: x @ w.T for i, w in stored.items()}

    model, _ = GGUFModelLoader(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.float32
    ).load()
    out = {i: np.array(_gdn(model, i).out_proj(mx.array(x))) for i in stored}

    for i in stored:
        atol = float(np.abs(expected[i]).max()) * 4e-5
        np.testing.assert_allclose(out[i], expected[i], rtol=0, atol=atol)
        assert not np.allclose(out[i], untransformed[i], rtol=0, atol=atol)


@pytest.mark.parametrize(
    ("tie", "has_lm_head"),
    [
        pytest.param(True, False, id="text-tie-true"),
        pytest.param(False, True, id="text-tie-false"),
        pytest.param(None, True, id="text-tie-omitted"),
    ],
)
def test_qwen35_tie_flag_resolves_on_text_model(tmp_path, tie, has_lm_head):
    # This fixture always carries output.weight; a tied model must skip it.
    text = _QWEN35_TEXT_CONFIG
    output = {"output.weight": ("q", (text["vocab_size"], text["hidden_size"]))}
    gguf_path, cfg_dir, _ = _build_qwen35_fixture(
        tmp_path, _GDN_R1, tie=tie, inject=output
    )

    model, _ = GGUFModelLoader(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.float32
    ).load()

    assert isinstance(model.language_model.get("lm_head"), GGUFLinear) is has_lm_head


@pytest.mark.parametrize(
    ("kind", "source_shape"),
    [("ssm_dt.bias", (16,)), ("ssm_conv1d.weight", (1024, 4))],
    ids=["ssm_dt", "ssm_conv1d"],
)
def test_qwen35_rejects_oversized_source_before_transform(tmp_path, kind, source_shape):
    # Twice the r2 rows: a reorder gather would mask the extra rows.
    gguf_path, cfg_dir, _ = _build_qwen35_fixture(
        tmp_path, _GDN_R2, inject={f"blk.0.{kind}": ("f", source_shape)}
    )

    with pytest.raises(
        GGUFLoadError, match=rf"'blk\.0\.{kind}' shape \({source_shape[0]},"
    ):
        GGUFModelLoader(gguf_path, config_dir=cfg_dir, target_dtype=mx.float32).load()


def test_qwen35_rejects_quantized_ssm_a(tmp_path):
    # ssm_a has no .weight suffix, so it must be plain like a bias.
    gguf_path, cfg_dir, _ = _build_qwen35_fixture(
        tmp_path, _GDN_R1, inject={"blk.0.ssm_a": ("q", (32,))}
    )

    with pytest.raises(GGUFLoadError, match="Unsupported qtype Q8_0 on 'blk.0.ssm_a'"):
        GGUFModelLoader(gguf_path, config_dir=cfg_dir, target_dtype=mx.float32).load()


@pytest.mark.parametrize(
    "moe_tensor", ["blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_exps.weight"]
)
def test_qwen35_rejects_out_of_scope_moe_tensor(tmp_path, moe_tensor):
    # qwen35 admits only its own fused-QKV and SSM substrings, not qwen35moe's.
    gguf_path, cfg_dir, _ = _build_qwen35_fixture(
        tmp_path, _GDN_R1, inject={moe_tensor: ("f", (2, 256))}
    )

    with pytest.raises(GGUFLoadError, match=f"Out-of-scope GGUF tensor '{moe_tensor}'"):
        GGUFModelLoader(gguf_path, config_dir=cfg_dir, target_dtype=mx.float32).load()


def test_qwen35_a_log_keeps_float32_under_bfloat16(tmp_path):
    gguf_path, cfg_dir, hf = _build_qwen35_fixture(tmp_path, _GDN_R1)

    model, _ = GGUFModelLoader(
        gguf_path, config_dir=cfg_dir, target_dtype=mx.bfloat16
    ).load()

    for i in _QWEN35_GDN_LAYERS:
        a_log = _gdn(model, i).A_log
        assert a_log.dtype == mx.float32
        np.testing.assert_allclose(
            np.array(a_log), hf[f"blk.{i}.ssm_a"], rtol=1e-6, atol=0
        )
        assert _gdn(model, i).dt_bias.dtype == mx.bfloat16
