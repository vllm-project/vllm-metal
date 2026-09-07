# SPDX-License-Identifier: Apache-2.0
"""Deterministic tests for the GGUF vLLM engine integration.

These exercise the gguf-package-free layer (detection, EngineArgs rewrite,
marker quantization config) through the public vLLM seams with real
``EngineArgs`` objects, so they run on a default install: fixtures write dummy
``.gguf`` files (detection reads the suffix / magic bytes, never the payload)
and a tiny ``config.json``. The regression test drives
``create_engine_config()`` end to end — the test that would have caught the
vLLM 0.24 in-tree GGUF removal (#463) silently breaking GGUF serve.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import vllm.engine.arg_utils as arg_utils_module
from huggingface_hub import constants as hf_constants
from vllm.engine.arg_utils import EngineArgs

from vllm_metal.gguf import source as gguf_source
from vllm_metal.gguf import vllm_integration
from vllm_metal.gguf.vllm_integration import (
    GGUFEngineIntegration,
    MetalGGUFConfig,
)

_TINY_CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
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


@pytest.fixture(autouse=True)
def _registered() -> None:
    # The entry point is exercised by a real install; unit tests apply the
    # registration explicitly (idempotent).
    vllm_integration.register()


@pytest.fixture()
def config_dir(tmp_path: Path) -> str:
    directory = tmp_path / "config"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(_TINY_CONFIG))
    return str(directory)


@pytest.fixture()
def gguf_file(tmp_path: Path) -> str:
    # Detection reads the suffix (or magic bytes), never the payload, so a
    # dummy file stands in for a real checkpoint.
    path = tmp_path / "tiny.gguf"
    path.write_bytes(b"GGUF-dummy")
    return str(path)


def _engine_args(**kwargs):
    return EngineArgs(**kwargs)


def _remote_gguf_model_config(**overrides: object) -> SimpleNamespace:
    values = {
        "quantization": "gguf",
        "model_weights": "Qwen/Qwen3-0.6B-GGUF:Q8_0",
        "model": "Qwen/Qwen3-0.6B",
        "tokenizer": None,
        "revision": None,
        "tokenizer_revision": None,
        "hf_token": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_create_engine_config_routes_local_gguf(gguf_file, config_dir) -> None:
    """The #463 regression test: a local .gguf drives the FULL engine-config
    build (speculator probe, model-config rewrite, and the marker quantization
    config exercised by VllmConfig) and lands with the fields the Metal
    lifecycle consumes.
    """
    vllm_config = _engine_args(
        model=gguf_file, tokenizer=config_dir
    ).create_engine_config()
    model_config = vllm_config.model_config

    assert model_config.quantization == "gguf"
    assert model_config.model == config_dir
    assert model_config.model_weights == gguf_file
    assert model_config.tokenizer == config_dir
    assert model_config.hf_config.model_type == "qwen3"
    assert model_config.served_model_name == gguf_file


def test_hf_config_path_beats_tokenizer(gguf_file, config_dir, tmp_path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(json.dumps(_TINY_CONFIG))

    model_config = _engine_args(
        model=gguf_file, tokenizer=str(other), hf_config_path=config_dir
    ).create_model_config()

    assert model_config.model == config_dir


def test_parent_dir_fallback_without_tokenizer(tmp_path) -> None:
    # config.json next to the .gguf: no --tokenizer needed.
    (tmp_path / "config.json").write_text(json.dumps(_TINY_CONFIG))
    gguf_path = tmp_path / "tiny.gguf"
    gguf_path.write_bytes(b"GGUF-dummy")

    model_config = _engine_args(model=str(gguf_path)).create_model_config()

    assert model_config.model == str(tmp_path)
    assert model_config.model_weights == str(gguf_path)


def test_gguf_tokenizer_is_skipped_in_precedence(tmp_path) -> None:
    # A tokenizer that is itself a GGUF file cannot be the config source; the
    # parent dir wins.
    (tmp_path / "config.json").write_text(json.dumps(_TINY_CONFIG))
    gguf_path = tmp_path / "tiny.gguf"
    gguf_path.write_bytes(b"GGUF-dummy")

    model_config = _engine_args(
        model=str(gguf_path), tokenizer=str(gguf_path)
    ).create_model_config()

    assert model_config.model == str(tmp_path)


def test_missing_config_json_fails_fast(gguf_file, tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError) as exc_info:
        _engine_args(model=gguf_file, tokenizer=str(empty)).create_model_config()
    assert str(exc_info.value) == (
        f"Serving GGUF model {gguf_file!r} needs a config source: "
        f"{str(empty)!r} has no config.json. A .gguf carries weights only; "
        "pass --tokenizer <dir> pointing at the model's "
        "config/tokenizer directory."
    )


def test_missing_config_json_via_hf_config_path_fails_fast(gguf_file, tmp_path) -> None:
    # The hf-config-path branch flows through the same guard as the tokenizer
    # and parent-dir branches.
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError) as exc_info:
        _engine_args(model=gguf_file, hf_config_path=str(empty)).create_model_config()
    assert str(exc_info.value) == (
        f"Serving GGUF model {gguf_file!r} needs a config source: "
        f"{str(empty)!r} has no config.json. A .gguf carries weights only; "
        "pass --tokenizer <dir> pointing at the model's "
        "config/tokenizer directory."
    )


def test_magic_bytes_without_suffix_fails_fast(tmp_path, config_dir) -> None:
    # Detected as GGUF by magic bytes, but MLX's loader dispatches on the file
    # extension, so a suffix-less file must be rejected up front with the
    # rename hint instead of dying later in the worker.
    suffixless = tmp_path / "model.bin"
    suffixless.write_bytes(b"GGUF" + b"\x00" * 8)

    with pytest.raises(ValueError) as exc_info:
        _engine_args(model=str(suffixless), tokenizer=config_dir).create_model_config()
    assert str(exc_info.value) == (
        f"{str(suffixless)!r} is a GGUF file (magic bytes) but "
        "vllm-metal requires the .gguf extension; add the "
        f"extension (e.g. {suffixless.name + '.gguf'!r})."
    )


def test_explicit_non_gguf_quantization_fails_fast(gguf_file, config_dir) -> None:
    with pytest.raises(ValueError, match="Cannot serve GGUF model"):
        _engine_args(
            model=gguf_file, tokenizer=config_dir, quantization="awq"
        ).create_model_config()


def test_non_gguf_model_is_untouched(config_dir) -> None:
    # A plain HF directory flows through the wrap unmodified.
    args = _engine_args(model=config_dir)
    model_config = args.create_model_config()

    assert args.model == config_dir
    assert model_config.quantization is None
    assert model_config.model_weights == ""


@pytest.mark.parametrize("offline", [False, True])
def test_create_model_config_routes_remote_gguf_reference(
    config_dir, monkeypatch, offline
) -> None:
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", offline)
    reference = "Qwen/Qwen3-0.6B-GGUF:Q8_0"

    model_config = _engine_args(
        model=reference, tokenizer=config_dir
    ).create_model_config()

    assert model_config.quantization == "gguf"
    assert model_config.model == config_dir
    assert model_config.model_weights == reference
    assert model_config.served_model_name == reference


def test_remote_gguf_reference_defaults_config_to_weights_repo() -> None:
    assert (
        GGUFEngineIntegration._resolve_config_source(
            "Qwen/Qwen3-0.6B-GGUF:Q8_0", tokenizer=None, hf_config_path=None
        )
        == "Qwen/Qwen3-0.6B-GGUF"
    )


def test_register_is_idempotent() -> None:
    before_config = EngineArgs.create_model_config
    before_probe = arg_utils_module.maybe_override_with_speculators
    before_model_path = arg_utils_module.get_model_path
    vllm_integration.register()
    vllm_integration.register()

    assert EngineArgs.create_model_config is before_config
    assert arg_utils_module.maybe_override_with_speculators is before_probe
    assert arg_utils_module.get_model_path is before_model_path


def test_integration_imports_without_gguf_package(monkeypatch) -> None:
    """Default-install tripwire: the entry-point module must import and
    register with the optional ``gguf`` package absent (vLLM loads it on every
    run).
    """
    monkeypatch.setitem(sys.modules, "gguf", None)
    monkeypatch.delitem(sys.modules, "vllm_metal.gguf.vllm_integration", raising=False)
    monkeypatch.delitem(sys.modules, "vllm_metal.gguf", raising=False)

    import vllm_metal.gguf.vllm_integration as reimported

    reimported.register()  # must not raise


def test_marker_config_survives_pickle() -> None:
    # The spawn tree pickles the constructed quant config into the
    # EngineCore/worker processes.
    config = MetalGGUFConfig()

    restored = pickle.loads(pickle.dumps(config))

    assert type(restored) is MetalGGUFConfig
    assert restored.get_name() == "gguf"


def test_loader_import_without_gguf_names_the_extra(monkeypatch) -> None:
    """The optional-dep owner gives an actionable hint when the extra is
    missing (0.24 dropped ``gguf`` from vLLM's own dependencies).
    """
    monkeypatch.setitem(sys.modules, "gguf", None)
    monkeypatch.delitem(sys.modules, "vllm_metal.gguf.loader", raising=False)
    monkeypatch.delitem(sys.modules, "vllm_metal.gguf.mlx_native", raising=False)
    monkeypatch.delitem(sys.modules, "vllm_metal.gguf.adapter", raising=False)
    monkeypatch.delitem(sys.modules, "vllm_metal.gguf.wrappers", raising=False)

    with pytest.raises(ImportError) as exc_info:
        import vllm_metal.gguf.loader  # noqa: F401
    assert str(exc_info.value) == (
        "GGUF support requires the optional 'gguf' dependency. "
        "Install it with: pip install 'vllm-metal[gguf]'"
    )


def test_probe_preserves_positional_speculative_config(gguf_file) -> None:
    # The wrap mirrors the upstream signature exactly, so a positionally
    # passed speculative config must round-trip (the official plugin's
    # kwargs-only short-circuit would silently drop it).
    speculative = {"method": "ngram"}

    result = arg_utils_module.maybe_override_with_speculators(
        gguf_file, "tokenizer-dir", False, None, speculative
    )

    assert result == (gguf_file, "tokenizer-dir", speculative)


def test_broken_official_plugin_does_not_disable_integration(
    tmp_path, monkeypatch
) -> None:
    """A vllm_gguf_plugin that is discoverable but fails to import (today's
    macOS state: its import pulls triton/CUDA) must NOT make register() defer
    to it.
    """
    broken = tmp_path / "vllm_gguf_plugin"
    broken.mkdir()
    (broken / "__init__.py").write_text("raise ImportError('No module named ...')")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "vllm_gguf_plugin", raising=False)

    sentinel = object()
    monkeypatch.setattr(arg_utils_module, "_metal_gguf_probe_patched", False)
    monkeypatch.setattr(arg_utils_module, "maybe_override_with_speculators", sentinel)

    vllm_integration.register()

    # register() proceeded past the coexistence guard and re-wrapped the probe.
    assert arg_utils_module.maybe_override_with_speculators is not sentinel


@pytest.mark.parametrize("offline_remote", [False, True])
def test_entry_point_mounts_without_manual_register(
    gguf_file, config_dir, tmp_path, offline_remote
) -> None:
    """The regression boundary #463 actually crossed: a fresh process where
    vLLM itself discovers and loads the ``vllm.general_plugins`` entry point —
    no manual import or register() — must route local and offline remote GGUFs. Uses a
    dist-info scaffold because a PYTHONPATH checkout carries no entry-point
    metadata.
    """
    dist_info = tmp_path / "vllm_metal_eptest-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: vllm-metal-eptest\nVersion: 0.0.0\n"
    )
    (dist_info / "entry_points.txt").write_text(
        "[vllm.general_plugins]\n"
        "gguf_metal = vllm_metal.gguf.vllm_integration:register\n"
    )
    repo_root = Path(vllm_integration.__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{tmp_path}"
    if offline_remote:
        gguf_file = "Qwen/Qwen3-0.6B-GGUF:Q8_0"
        env["HF_HUB_OFFLINE"] = "1"
    script = (
        "from vllm.engine.arg_utils import EngineArgs\n"
        f"mc = EngineArgs(model={gguf_file!r}, tokenizer={config_dir!r})"
        ".create_model_config()\n"
        "print('EP-OK', mc.quantization, mc.model_weights)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert f"EP-OK gguf {gguf_file}" in result.stdout


def test_remote_reference_grammar() -> None:
    """Shape-level recognition (deliberately enum-free): quant-looking tags
    match, ordinary repo:tag strings do not."""
    recognized = [
        "Qwen/Qwen3-0.6B-GGUF:Q8_0",
        "unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M",
        "unsloth/model-GGUF:UD-Q4_K_XL",
        "org/model:IQ2_M",
        "org/model:F16",
        "org/model:q4_k_m",
    ]
    not_recognized = [
        "org/model:latest",
        "org/model:main",
        "Qwen/Qwen3-0.6B",
        "/local/path/model.gguf",
    ]
    for ref in recognized:
        assert GGUFEngineIntegration.is_remote_gguf_reference(ref), ref
    for ref in not_recognized:
        assert not GGUFEngineIntegration.is_remote_gguf_reference(ref), ref


def test_remote_load_source_downloads_one_matching_gguf(tmp_path, monkeypatch) -> None:
    weight_snapshot = tmp_path / "weights"
    config_snapshot = tmp_path / "config"
    tokenizer_snapshot = tmp_path / "tokenizer"
    for directory in (weight_snapshot, config_snapshot, tokenizer_snapshot):
        directory.mkdir()
    gguf_path = weight_snapshot / "Qwen3-0.6B-Q8_0.gguf"
    gguf_path.write_text("dummy")
    list_calls: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []

    def fake_list_repo_files(**kwargs: object) -> list[str]:
        list_calls.append(kwargs)
        return ["README.md", gguf_path.name]

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        if kwargs["repo_id"] == "Qwen/Qwen3-0.6B-GGUF":
            return str(weight_snapshot)
        if kwargs["repo_id"] == "Qwen/Qwen3-0.6B":
            return str(config_snapshot)
        if kwargs["repo_id"] == "Qwen/Qwen3-0.6B-Tokenizer":
            return str(tokenizer_snapshot)
        raise AssertionError(f"unexpected repo_id={kwargs['repo_id']!r}")

    monkeypatch.setattr(
        gguf_source,
        "HfApi",
        lambda: SimpleNamespace(list_repo_files=fake_list_repo_files),
    )
    monkeypatch.setattr(gguf_source, "snapshot_download", fake_snapshot_download)
    load_config = SimpleNamespace(
        download_dir=str(tmp_path / "cache"), ignore_patterns=["*.md"]
    )
    model_config = _remote_gguf_model_config(
        tokenizer="Qwen/Qwen3-0.6B-Tokenizer",
        revision="rev-a",
        tokenizer_revision="tok-rev",
        hf_token="hf-token",
    )

    source = gguf_source.GGUFLoadSource.from_model_config(model_config, load_config)

    assert source is not None
    assert source.weights_path == str(gguf_path)
    assert source.config_dir == str(config_snapshot)
    assert source.tokenizer_dir == str(tokenizer_snapshot)
    assert list_calls == [
        {
            "repo_id": "Qwen/Qwen3-0.6B-GGUF",
            "revision": "rev-a",
            "token": "hf-token",
        }
    ]
    weight_call, config_call, tokenizer_call = calls
    assert weight_call == {
        "repo_id": "Qwen/Qwen3-0.6B-GGUF",
        "cache_dir": str(tmp_path / "cache"),
        "allow_patterns": [gguf_path.name],
        "revision": "rev-a",
        "token": "hf-token",
    }
    assert config_call == {
        "repo_id": "Qwen/Qwen3-0.6B",
        "cache_dir": str(tmp_path / "cache"),
        "allow_patterns": ["config.json", "generation_config.json"],
        "revision": "rev-a",
        "token": "hf-token",
    }
    assert tokenizer_call == {
        "repo_id": "Qwen/Qwen3-0.6B-Tokenizer",
        "cache_dir": str(tmp_path / "cache"),
        "allow_patterns": [
            "*.json",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "tiktoken.model",
            "*.txt",
            "*.jsonl",
            "*.jinja",
        ],
        "revision": "tok-rev",
        "token": "hf-token",
    }


@pytest.mark.parametrize(
    ("filenames", "error"),
    [
        (
            [],
            "No 'Q8_0' GGUF file found in remote repository 'Qwen/Qwen3-0.6B-GGUF'.",
        ),
        (
            ["model-a-Q8_0.gguf", "model-b-Q8_0.gguf"],
            "Remote GGUF reference 'Qwen/Qwen3-0.6B-GGUF:Q8_0' matched "
            "multiple files: model-a-Q8_0.gguf, model-b-Q8_0.gguf.",
        ),
        (
            ["model-Q8_0-00001-of-00002.gguf"],
            "Remote sharded GGUF files are not supported yet: "
            "'Qwen/Qwen3-0.6B-GGUF:Q8_0'.",
        ),
        # Remote selection stays N-agnostic: a lone -00001-of-00001 match is
        # still rejected here; only the local loader guard admits N==1 sets.
        (
            ["model-Q8_0-00001-of-00001.gguf"],
            "Remote sharded GGUF files are not supported yet: "
            "'Qwen/Qwen3-0.6B-GGUF:Q8_0'.",
        ),
    ],
)
def test_remote_load_source_rejects_unsupported_remote_matches(
    monkeypatch, filenames, error
) -> None:
    def fail_snapshot_download(**_: object) -> str:
        raise AssertionError("rejected remote GGUF reference must not download")

    monkeypatch.setattr(
        gguf_source,
        "HfApi",
        lambda: SimpleNamespace(list_repo_files=lambda **_: filenames),
    )
    monkeypatch.setattr(gguf_source, "snapshot_download", fail_snapshot_download)

    with pytest.raises(ValueError) as excinfo:
        gguf_source.GGUFLoadSource.from_model_config(_remote_gguf_model_config())

    assert str(excinfo.value) == error


def test_remote_load_source_rejects_unsupported_qtype_before_download(
    monkeypatch,
) -> None:
    def fail_list_repo_files(**_: object) -> list[str]:
        raise AssertionError("unsupported remote qtype must not list files")

    def fail_snapshot_download(**_: object) -> str:
        raise AssertionError("unsupported remote qtype must not download")

    monkeypatch.setattr(
        gguf_source,
        "HfApi",
        lambda: SimpleNamespace(list_repo_files=fail_list_repo_files),
    )
    monkeypatch.setattr(gguf_source, "snapshot_download", fail_snapshot_download)

    with pytest.raises(ValueError, match="Remote GGUF qtype 'Q4_K_M'"):
        gguf_source.GGUFLoadSource.from_model_config(
            _remote_gguf_model_config(
                model_weights="Qwen/Qwen3-0.6B-GGUF:Q4_K_M",
            )
        )
