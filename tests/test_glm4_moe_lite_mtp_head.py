# SPDX-License-Identifier: Apache-2.0
"""Tests for GLM-4.7-Flash native MTP head loading and validation contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vllm_metal.v1.mtp_heads.glm4_moe_lite_mtp import (
    GLM4_MOE_LITE_MTP_MODEL_TYPE,
    GLM4_MOE_LITE_TARGET_MODEL_TYPE,
    Glm4MoeLiteMTPArgs,
    Glm4MoeLiteMTPHeadLoader,
    Glm4MoeLiteMTPHeadMetadata,
    Glm4MoeLiteMTPModel,
)


@pytest.fixture(autouse=True)
def _reset_head_cache():
    Glm4MoeLiteMTPHeadLoader.clear_cache()
    yield
    Glm4MoeLiteMTPHeadLoader.clear_cache()


# Identity lives at the top level; the source model config is nested under
# ``text_config`` (the hosted drafter-split schema). Overrides route to the top
# level for these keys, else into ``text_config``.
_TOP_LEVEL_KEYS = frozenset(
    {"model_type", "block_size", "tie_word_embeddings", "model_file"}
)


def _head_config(**overrides: object) -> dict[str, object]:
    """A hosted GLM-4.7-Flash MTP head config (drafter-split ``text_config`` schema)."""
    text: dict[str, object] = {
        "architectures": ["Glm4MoeLiteForCausalLM"],
        "model_type": GLM4_MOE_LITE_TARGET_MODEL_TYPE,
        "num_hidden_layers": 47,
        "num_nextn_predict_layers": 1,
        "hidden_size": 2048,
        "vocab_size": 154880,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 192,
        "v_head_dim": 256,
        "num_attention_heads": 20,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-5,
    }
    config: dict[str, object] = {
        "model_type": GLM4_MOE_LITE_MTP_MODEL_TYPE,
        "block_size": 2,
        "tie_word_embeddings": False,
    }
    for key, value in overrides.items():
        target = config if key in _TOP_LEVEL_KEYS else text
        target[key] = value
    config["text_config"] = text
    return config


def _target_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "model_type": "glm4_moe_lite",
        "hidden_size": 2048,
        "vocab_size": 154880,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 192,
        "v_head_dim": 256,
        "num_attention_heads": 20,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-5,
    }
    config.update(overrides)
    return config


def _speculative_config(
    *,
    model_type: str = GLM4_MOE_LITE_MTP_MODEL_TYPE,
    model: str = "/head",
) -> SimpleNamespace:
    return SimpleNamespace(
        method="mtp",
        draft_model_config=SimpleNamespace(
            model=model,
            revision=None,
            hf_config=SimpleNamespace(model_type=model_type),
        ),
    )


def _fake_head_model(*, missing: tuple[str, ...] = ()) -> object:
    params: dict[str, object] = {
        "lm_head": {"weight": mx.zeros((1,))},
        "model": {"embed_tokens": {"weight": mx.zeros((1,))}},
    }
    if "lm_head.weight" in missing:
        params["lm_head"] = {}
    if "model.embed_tokens.weight" in missing:
        params["model"] = {}

    class _FakeModel:
        def parameters(self) -> dict[str, object]:
            return params

    return _FakeModel()


# --------------------------------------------------------------------------
# Metadata parsing
# --------------------------------------------------------------------------


def test_from_config_parses_valid_head() -> None:
    metadata = Glm4MoeLiteMTPHeadMetadata.from_config(_head_config())

    assert metadata.model_type == GLM4_MOE_LITE_MTP_MODEL_TYPE
    assert metadata.architectures == ("Glm4MoeLiteForCausalLM",)
    assert metadata.hidden_size == 2048
    assert metadata.vocab_size == 154880
    assert metadata.kv_lora_rank == 512
    assert metadata.num_nextn_predict_layers == 1


def test_from_config_rejects_missing_text_config() -> None:
    config = _head_config()
    del config["text_config"]

    with pytest.raises(ValueError, match="text_config"):
        Glm4MoeLiteMTPHeadMetadata.from_config(config)


def test_from_config_rejects_missing_num_nextn() -> None:
    config = _head_config()
    del config["text_config"]["num_nextn_predict_layers"]

    with pytest.raises(ValueError, match="num_nextn_predict_layers"):
        Glm4MoeLiteMTPHeadMetadata.from_config(config)


def test_from_config_rejects_missing_block_size() -> None:
    config = _head_config()
    del config["block_size"]

    with pytest.raises(ValueError, match="block_size"):
        Glm4MoeLiteMTPHeadMetadata.from_config(config)


def test_from_config_rejects_wrong_model_type() -> None:
    with pytest.raises(ValueError, match="got 'glm4_moe_lite'"):
        Glm4MoeLiteMTPHeadMetadata.from_config(_head_config(model_type="glm4_moe_lite"))


def test_from_config_rejects_bad_num_nextn() -> None:
    with pytest.raises(ValueError, match="num_nextn_predict_layers=1"):
        Glm4MoeLiteMTPHeadMetadata.from_config(_head_config(num_nextn_predict_layers=2))


def test_from_config_rejects_inconsistent_block_size() -> None:
    # block_size must equal num_nextn_predict_layers + 1 (the drafter-split
    # schema's MTP-depth field); an inconsistent value fails loud.
    with pytest.raises(ValueError, match="block_size must be"):
        Glm4MoeLiteMTPHeadMetadata.from_config(_head_config(block_size=3))


def test_from_config_rejects_non_integer_field() -> None:
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        Glm4MoeLiteMTPHeadMetadata.from_config(_head_config(hidden_size=0))


# --------------------------------------------------------------------------
# Target compatibility
# --------------------------------------------------------------------------


def test_validate_compatible_accepts_match() -> None:
    metadata = Glm4MoeLiteMTPHeadMetadata.from_config(_head_config())

    metadata.validate_compatible_with(_target_config())


def test_validate_compatible_reads_nested_text_config() -> None:
    metadata = Glm4MoeLiteMTPHeadMetadata.from_config(_head_config())

    metadata.validate_compatible_with({"text_config": _target_config()})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hidden_size", 4096),
        ("vocab_size", 32000),
        ("kv_lora_rank", 256),
        ("qk_rope_head_dim", 32),
        ("qk_nope_head_dim", 128),
        ("v_head_dim", 128),
        ("num_attention_heads", 32),
        ("rope_theta", 500000.0),
        ("rms_norm_eps", 1e-6),
    ],
)
def test_validate_compatible_rejects_each_field_mismatch(
    field: str,
    value: object,
) -> None:
    metadata = Glm4MoeLiteMTPHeadMetadata.from_config(_head_config())

    with pytest.raises(ValueError, match=f"head {field} must match target {field}"):
        metadata.validate_compatible_with(_target_config(**{field: value}))


def test_validate_compatible_reports_both_values() -> None:
    metadata = Glm4MoeLiteMTPHeadMetadata.from_config(_head_config())

    with pytest.raises(ValueError, match="head=2048, target=4096"):
        metadata.validate_compatible_with(_target_config(hidden_size=4096))


def test_validate_compatible_rejects_missing_target_field() -> None:
    metadata = Glm4MoeLiteMTPHeadMetadata.from_config(_head_config())
    target = _target_config()
    del target["kv_lora_rank"]

    with pytest.raises(ValueError, match="missing 'kv_lora_rank'"):
        metadata.validate_compatible_with(target)


def test_validate_compatible_rejects_wrong_target_model_type() -> None:
    metadata = Glm4MoeLiteMTPHeadMetadata.from_config(_head_config())

    with pytest.raises(
        ValueError, match="requires a 'glm4_moe_lite' target model"
    ) as excinfo:
        metadata.validate_compatible_with(_target_config(model_type="llama"))
    assert "model_type='llama'" in str(excinfo.value)


def test_validate_compatible_rejects_missing_target_model_type() -> None:
    metadata = Glm4MoeLiteMTPHeadMetadata.from_config(_head_config())
    target = _target_config()
    del target["model_type"]

    with pytest.raises(ValueError, match="model_type=None"):
        metadata.validate_compatible_with(target)


def test_validate_compatible_reads_target_model_type_in_text_config() -> None:
    # model_type lives only inside the nested text_config; the head still
    # accepts it (mirrors how vLLM nests the backbone config).
    metadata = Glm4MoeLiteMTPHeadMetadata.from_config(_head_config())

    metadata.validate_compatible_with({"text_config": _target_config()})


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------


def test_get_model_classes_returns_head_classes() -> None:
    model_cls, args_cls = Glm4MoeLiteMTPHeadLoader._get_model_classes(_head_config())

    assert model_cls is Glm4MoeLiteMTPModel
    assert args_cls is Glm4MoeLiteMTPArgs


def test_get_model_classes_rejects_wrong_model_type() -> None:
    with pytest.raises(ValueError, match="mlx_vlm") as excinfo:
        Glm4MoeLiteMTPHeadLoader._get_model_classes(
            _head_config(model_type="glm4_moe_lite")
        )
    # Points users at the alternative head sources, not a deleted local script.
    assert "GLM-4.7-Flash-MTP" in str(excinfo.value)


def test_loader_loads_and_validates_with_injected_fakes() -> None:
    load_calls: list[dict[str, object]] = []
    download_calls: list[tuple[str, str | None]] = []
    fake_model = _fake_head_model()

    def _load_model(model_path: object, *args: object, **kwargs: object):
        load_calls.append({"path": model_path, "kwargs": kwargs})
        get_model_classes = kwargs["get_model_classes"]
        assert callable(get_model_classes)
        assert get_model_classes(_head_config()) == (
            Glm4MoeLiteMTPModel,
            Glm4MoeLiteMTPArgs,
        )
        return fake_model, _head_config()

    loader = Glm4MoeLiteMTPHeadLoader(
        load_model_fn=_load_model,
        download_fn=lambda name, rev: download_calls.append((name, rev)) or Path(name),
    )

    runtime = loader.load_if_needed(
        speculative_config=_speculative_config(),
        target_config=_target_config(),
    )

    assert runtime.model is fake_model
    assert runtime.metadata.hidden_size == 2048
    assert download_calls == [("/head", None)]
    assert load_calls[0]["kwargs"]["strict"] is True


def test_loader_caches_runtime_and_revalidates_target() -> None:
    load_calls = 0

    def _load_model(*args: object, **kwargs: object):
        nonlocal load_calls
        load_calls += 1
        return _fake_head_model(), _head_config()

    loader = Glm4MoeLiteMTPHeadLoader(
        load_model_fn=_load_model,
        download_fn=lambda name, rev: Path(name),
        model_path_resolver=lambda name: name,
    )

    first = loader.load_if_needed(
        speculative_config=_speculative_config(),
        target_config=_target_config(),
    )
    second = loader.load_if_needed(
        speculative_config=_speculative_config(),
        target_config=_target_config(),
    )

    assert first is second
    assert load_calls == 1

    with pytest.raises(ValueError, match="head hidden_size must match"):
        loader.load_if_needed(
            speculative_config=_speculative_config(),
            target_config=_target_config(hidden_size=4096),
        )
    assert load_calls == 1


def test_loader_keeps_revisions_in_separate_cache_slots() -> None:
    load_calls = 0
    download_calls: list[tuple[str, str | None]] = []

    def _load_model(*args: object, **kwargs: object):
        nonlocal load_calls
        load_calls += 1
        return _fake_head_model(), _head_config()

    loader = Glm4MoeLiteMTPHeadLoader(
        load_model_fn=_load_model,
        download_fn=lambda name, rev: download_calls.append((name, rev)) or Path(name),
        model_path_resolver=lambda name: name,
    )
    spec_a = _speculative_config()
    spec_a.revision = "rev-a"
    spec_a.draft_model_config.revision = "rev-a"
    spec_b = _speculative_config()
    spec_b.revision = "rev-b"
    spec_b.draft_model_config.revision = "rev-b"

    first = loader.load_if_needed(
        speculative_config=spec_a, target_config=_target_config()
    )
    again = loader.load_if_needed(
        speculative_config=spec_a, target_config=_target_config()
    )
    other = loader.load_if_needed(
        speculative_config=spec_b, target_config=_target_config()
    )

    assert first is again
    assert first is not other
    assert load_calls == 2
    assert download_calls == [("/head", "rev-a"), ("/head", "rev-b")]


def test_loader_rejects_custom_model_file_before_load(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(_head_config(model_file="modeling_glm.py"))
    )
    loader = Glm4MoeLiteMTPHeadLoader(
        load_model_fn=lambda *a, **k: pytest.fail("load_model called"),
        download_fn=lambda name, rev: tmp_path,
        model_path_resolver=lambda name: name,
    )

    with pytest.raises(ValueError, match="custom model_file"):
        loader.load_if_needed(
            speculative_config=_speculative_config(),
            target_config=_target_config(),
        )


def test_loader_rejects_raw_target_repo_config(tmp_path: Path) -> None:
    # Pointing at the raw GLM-4.7-Flash target repo (top-level
    # model_type=glm4_moe_lite) is rejected before load: only the MTP head, whose
    # top-level model_type is glm4_moe_lite_mtp, is a valid source.
    (tmp_path / "config.json").write_text(json.dumps(_target_config()))
    loader = Glm4MoeLiteMTPHeadLoader(
        load_model_fn=lambda *a, **k: pytest.fail("load_model called"),
        download_fn=lambda name, rev: tmp_path,
        model_path_resolver=lambda name: name,
    )

    with pytest.raises(ValueError, match="glm4_moe_lite_mtp") as excinfo:
        loader.load_if_needed(
            speculative_config=_speculative_config(),
            target_config=_target_config(),
        )
    assert "mlx_vlm" in str(excinfo.value)


def test_loader_prevalidates_config_file_before_load(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_head_config(hidden_size=4096)))
    loader = Glm4MoeLiteMTPHeadLoader(
        load_model_fn=lambda *a, **k: pytest.fail("load_model called"),
        download_fn=lambda name, rev: tmp_path,
        model_path_resolver=lambda name: name,
    )

    with pytest.raises(ValueError, match="head hidden_size must match"):
        loader.load_if_needed(
            speculative_config=_speculative_config(),
            target_config=_target_config(),
        )


def test_default_loader_rejects_missing_config_before_load(tmp_path: Path) -> None:
    loader = Glm4MoeLiteMTPHeadLoader(
        download_fn=lambda name, rev: tmp_path,
        model_path_resolver=lambda name: name,
    )

    with pytest.raises(ValueError, match="must contain config.json"):
        loader.load_if_needed(
            speculative_config=_speculative_config(),
            target_config=_target_config(),
        )


def test_loader_rejects_invalid_json_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{", encoding="utf-8")
    loader = Glm4MoeLiteMTPHeadLoader(
        load_model_fn=lambda *a, **k: pytest.fail("load_model called"),
        download_fn=lambda name, rev: tmp_path,
        model_path_resolver=lambda name: name,
    )

    with pytest.raises(ValueError, match="not valid JSON"):
        loader.load_if_needed(
            speculative_config=_speculative_config(),
            target_config=_target_config(),
        )


@pytest.mark.parametrize("missing", ["lm_head.weight", "model.embed_tokens.weight"])
def test_loader_asserts_head_tensors_present(missing: str) -> None:
    loader = Glm4MoeLiteMTPHeadLoader(
        load_model_fn=lambda *a, **k: (
            _fake_head_model(missing=(missing,)),
            _head_config(),
        ),
        download_fn=lambda name, rev: Path(name),
        model_path_resolver=lambda name: name,
    )

    with pytest.raises(ValueError, match=f"missing {missing!r}"):
        loader.load_if_needed(
            speculative_config=_speculative_config(),
            target_config=_target_config(),
        )


def test_default_loader_snapshot_allow_patterns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    huggingface_hub = pytest.importorskip("huggingface_hub")
    (tmp_path / "config.json").write_text(json.dumps(_head_config()))
    (tmp_path / "model.safetensors").write_text("", encoding="utf-8")
    snapshot_calls: list[dict[str, object]] = []

    def _snapshot_download(*args: object, **kwargs: object) -> str:
        snapshot_calls.append({"args": args, "kwargs": kwargs})
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snapshot_download)
    spec = _speculative_config(model="zai-org/GLM-4.7-Flash-mtp-head")
    loader = Glm4MoeLiteMTPHeadLoader(
        load_model_fn=lambda *a, **k: (_fake_head_model(), _head_config()),
    )

    loader.load_if_needed(speculative_config=spec, target_config=_target_config())

    assert snapshot_calls == [
        {
            "args": ("zai-org/GLM-4.7-Flash-mtp-head",),
            "kwargs": {
                "revision": None,
                "allow_patterns": [
                    "config.json",
                    "*.safetensors",
                    "*.safetensors.index.json",
                ],
            },
        }
    ]


def test_default_loader_loads_real_head_from_disk(tmp_path: Path) -> None:
    # Load-bearing positive path: a real on-disk head checkpoint flows through the
    # default loader + mlx_lm.load_model, validates against a matching target, and
    # produces finite logits. Small dims are nested under text_config (the hosted
    # drafter-split schema); from_dict promotes them to build the tiny model.
    from mlx.utils import tree_flatten
    from mlx_lm.models.cache import KVCache

    dims: dict[str, object] = {
        "vocab_size": 32,
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "n_shared_experts": 1,
        "n_routed_experts": 4,
        "kv_lora_rank": 16,
        "q_lora_rank": 32,
        "qk_rope_head_dim": 8,
        "qk_nope_head_dim": 16,
        "v_head_dim": 16,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-5,
    }

    mx.random.seed(0)
    model = Glm4MoeLiteMTPModel(Glm4MoeLiteMTPArgs(**dims))
    mx.eval(model.parameters())
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": GLM4_MOE_LITE_MTP_MODEL_TYPE,
                "block_size": 2,
                "text_config": {
                    "model_type": GLM4_MOE_LITE_TARGET_MODEL_TYPE,
                    "architectures": ["Glm4MoeLiteForCausalLM"],
                    "num_hidden_layers": 47,
                    "num_nextn_predict_layers": 1,
                    **dims,
                },
            }
        )
    )
    target = _target_config(
        hidden_size=64,
        vocab_size=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        v_head_dim=16,
        num_attention_heads=4,
    )

    loader = Glm4MoeLiteMTPHeadLoader(model_path_resolver=lambda name: name)
    runtime = loader.load_if_needed(
        speculative_config=_speculative_config(model=str(tmp_path)),
        target_config=target,
    )

    assert isinstance(runtime.model, Glm4MoeLiteMTPModel)
    assert runtime.metadata.hidden_size == 64
    assert runtime.metadata.num_nextn_predict_layers == 1

    x = runtime.model.build_slot_inputs(mx.array([1, 2, 3]), mx.zeros((3, 64)), 0)
    logits = runtime.model.compute_logits(runtime.model.forward_slots(x, KVCache()))
    mx.eval(logits)
    assert tuple(logits.shape) == (3, 32)
    assert bool(mx.all(mx.isfinite(logits)))


# --------------------------------------------------------------------------
# Args
# --------------------------------------------------------------------------


def test_args_reject_wrong_model_type() -> None:
    with pytest.raises(ValueError, match="model_type"):
        Glm4MoeLiteMTPArgs(model_type="glm4_moe_lite")


def test_args_reject_bad_num_nextn() -> None:
    with pytest.raises(ValueError, match="num_nextn_predict_layers"):
        Glm4MoeLiteMTPArgs(num_nextn_predict_layers=2)


def test_args_reject_layer_idx_that_would_not_be_moe() -> None:
    # first_k_dense_replace=1 (default) is not divisible by moe_layer_freq=2, so
    # stock Glm4MoeLiteDecoderLayer would build a *dense* mtp_block, not the MoE
    # the trained nextn head requires. Fail fast rather than mis-build.
    with pytest.raises(ValueError, match="must be divisible by"):
        Glm4MoeLiteMTPArgs(moe_layer_freq=2)


def test_args_accept_layer_idx_divisible_by_moe_layer_freq() -> None:
    # first_k_dense_replace=2 is divisible by moe_layer_freq=2 -> MoE layer.
    args = Glm4MoeLiteMTPArgs(first_k_dense_replace=2, moe_layer_freq=2)

    assert args.first_k_dense_replace == 2
    assert args.moe_layer_freq == 2


def test_args_reject_non_positive_moe_layer_freq() -> None:
    with pytest.raises(ValueError, match="moe_layer_freq to be a positive"):
        Glm4MoeLiteMTPArgs(moe_layer_freq=0)


def test_args_from_dict_ignores_extra_config_keys() -> None:
    args = Glm4MoeLiteMTPArgs.from_dict({**_head_config(), "unrelated": 123})

    assert args.model_type == GLM4_MOE_LITE_MTP_MODEL_TYPE
    assert args.hidden_size == 2048
    backbone = args.backbone_args()
    assert backbone.model_type == "glm4_moe_lite"
    assert backbone.hidden_size == 2048
