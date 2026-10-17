# SPDX-License-Identifier: Apache-2.0
"""Tests for native build cache invalidation in vllm_metal.metal.build."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from vllm_metal.metal import build


@dataclass
class _Paths:
    src: Path
    patch: Path
    kernels: Path
    device: Path
    bld: Path
    consts: Path
    out: Path
    hsh: Path
    ver: Path
    versions: dict[str, str]


@pytest.fixture
def patched(tmp_path, monkeypatch) -> _Paths:
    src = tmp_path / "paged_ops.cpp"
    patch = tmp_path / "mlx_patch.cpp"
    kernels = tmp_path / "paged_attention_kernels.h"
    device = tmp_path / "metal_device.h"
    bld = tmp_path / "build.py"
    consts = tmp_path / "constants.py"
    out = tmp_path / "_paged_ops.so"
    hsh = tmp_path / "_paged_ops.so.sha256"
    ver = tmp_path / "_paged_ops.mlx-version"

    src.write_bytes(b"// source v1")
    patch.write_bytes(b"// patch v1")
    kernels.write_bytes(b"// kernel descriptions v1")
    device.write_bytes(b"// hardware detection v1")
    bld.write_bytes(b"# build v1")
    consts.write_bytes(b"PARTITION_SIZE = 256")
    ver.write_text("0.0.0\n")

    monkeypatch.setattr(build, "_SRC", src)
    monkeypatch.setattr(build, "_MLX_PATCH", patch)
    monkeypatch.setattr(build, "_KERNELS", kernels)
    monkeypatch.setattr(build, "_DEVICE", device)
    monkeypatch.setattr(build, "_BUILD", bld)
    monkeypatch.setattr(build, "_CONSTANTS", consts)
    monkeypatch.setattr(build, "_OUT", out)
    monkeypatch.setattr(build, "_HASH", hsh)
    monkeypatch.setattr(build, "_MLX_VERSION", ver)

    versions = {"mlx": "0.31.0", "nanobind": "2.10.0"}
    monkeypatch.setattr(build, "_package_version", versions.__getitem__)

    return _Paths(src, patch, kernels, device, bld, consts, out, hsh, ver, versions)


def test_needs_rebuild_when_so_missing(patched):
    assert build.needs_rebuild() is True


def test_needs_rebuild_when_hash_sidecar_missing(patched):
    patched.out.write_bytes(b"compiled")
    assert build.needs_rebuild() is True


def test_needs_rebuild_when_hash_differs(patched):
    patched.out.write_bytes(b"compiled")
    patched.hsh.write_text("deadbeef")
    assert build.needs_rebuild() is True


def test_needs_rebuild_when_hash_is_not_utf8(patched):
    patched.out.write_bytes(b"compiled")
    patched.hsh.write_bytes(b"\xff\xfe")
    assert build.needs_rebuild() is True


def test_no_rebuild_when_hash_matches(patched):
    patched.out.write_bytes(b"compiled")
    patched.hsh.write_text(build._input_hash())
    assert build.needs_rebuild() is False


def test_needs_rebuild_when_mlx_version_record_missing(patched):
    patched.out.write_bytes(b"compiled")
    patched.hsh.write_text(build._input_hash())
    patched.ver.unlink()
    assert build.needs_rebuild() is True


def test_old_content_with_newer_so_mtime_still_rebuilds(patched):
    # Regression: under the old mtime-based check, a stale .so from a previous
    # branch could shadow freshly-checked-out sources whose mtimes git set to
    # the checkout time.
    patched.out.write_bytes(b"compiled-from-prev-branch")
    patched.hsh.write_text("hash-from-prev-branch")
    older = patched.out.stat().st_mtime - 86400
    for p in (patched.src, patched.bld, patched.consts):
        os.utime(p, (older, older))
    assert build.needs_rebuild() is True


@pytest.mark.parametrize("source_name", ["src", "patch", "kernels", "device"])
def test_hash_changes_with_source_bytes(patched, source_name):
    h1 = build._input_hash()
    getattr(patched, source_name).write_bytes(b"// source v2")
    assert build._input_hash() != h1


def test_hash_changes_with_constants_bytes(patched):
    h1 = build._input_hash()
    patched.consts.write_bytes(b"PARTITION_SIZE = 512")
    assert build._input_hash() != h1


def test_hash_ignores_source_directory(patched, tmp_path, monkeypatch):
    h1 = build._input_hash()
    moved = tmp_path / "moved" / patched.src.name
    moved.parent.mkdir()
    moved.write_bytes(patched.src.read_bytes())
    monkeypatch.setattr(build, "_SRC", moved)
    assert build._input_hash() == h1


@pytest.mark.parametrize("package", ["mlx", "nanobind"])
def test_hash_changes_with_dependency_version(patched, package):
    h1 = build._input_hash()
    patched.versions[package] = "99.0.0"
    assert build._input_hash() != h1


def test_needs_rebuild_swallows_resolution_errors(patched, monkeypatch):
    patched.out.write_bytes(b"compiled")
    patched.hsh.write_text("anything")

    def boom(name):
        raise ImportError("mlx not installed")

    monkeypatch.setattr(build, "_package_version", boom)
    assert build.needs_rebuild() is True


# --------------------------------------------------------------------------
# Path contract (the runtime loader and verify_wheel_artifacts depend on these)
# --------------------------------------------------------------------------


def test_output_path_points_at_the_so(patched):
    assert build.output_path() == patched.out


def test_metallib_path_named_in_package(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "_THIS_DIR", tmp_path)
    assert build.metallib_path("gdn_kern") == tmp_path / "gdn_kern.metallib"


def test_metallib_digest_changes_with_source(patched):
    # The metallib staleness stamp is keyed on the assembled shader source, so a
    # one-byte shader edit must change the digest (else an edited kernel loads
    # silently stale).
    name = build.METALLIB_NAMES[0]
    assert build._metallib_digest(name, "shader A") != build._metallib_digest(
        name, "shader B"
    )


def test_nax_metallib_has_its_own_deployment_floor():
    required = build._metallib_flags(build.METALLIB_NAMES[0])
    nax = build._metallib_flags(build.NAX_METALLIB_NAME)
    assert required[-1] == f"-mmacosx-version-min={build.MIN_MACOS_VERSION}"
    assert nax[-1] == f"-mmacosx-version-min={build.NAX_MIN_MACOS_VERSION}"


def test_official_build_requires_nax_sdk(monkeypatch):
    monkeypatch.setattr(build, "_sdk_supports_nax", lambda: False)
    with pytest.raises(
        RuntimeError,
        match=r"Official wheel builds require macOS SDK >= 26\.2",
    ):
        build.require_nax_sdk()


# --------------------------------------------------------------------------
# is_stale: one staleness primitive for the .so and every .metallib.
# --------------------------------------------------------------------------


def test_is_stale_false_when_artifact_missing(tmp_path):
    assert build.is_stale(tmp_path / "a.metallib", "digest") is False


def test_is_stale_false_when_stamp_missing(tmp_path):
    # Wheel installs ship the artifact but NO .sha256 stamp -> never "stale", so
    # end users never hit a staleness error.
    art = tmp_path / "a.metallib"
    art.write_bytes(b"lib")
    assert build.is_stale(art, "digest") is False


def test_is_stale_false_when_stamp_matches(tmp_path):
    art = tmp_path / "a.metallib"
    art.write_bytes(b"lib")
    build._stamp_path(art).write_text("digest")
    assert build.is_stale(art, "digest") is False


def test_is_stale_true_when_stamp_differs(tmp_path):
    art = tmp_path / "a.metallib"
    art.write_bytes(b"lib")
    build._stamp_path(art).write_text("OLD")
    assert build.is_stale(art, "NEW") is True


def test_is_stale_false_when_stamp_unreadable(tmp_path):
    # A stamp we can't read must not break loading -> treated as not stale.
    art = tmp_path / "a.metallib"
    art.write_bytes(b"lib")
    build._stamp_path(art).mkdir()  # read_text() -> IsADirectoryError (OSError)
    assert build.is_stale(art, "whatever") is False


# --------------------------------------------------------------------------
# stale_artifacts: aggregates is_stale over the .so + the three .metallibs,
# with digests stubbed so no real mlx/nanobind/shader sources are touched.
# --------------------------------------------------------------------------


def _raise(exc):
    def _f(*_a, **_k):
        raise exc

    return _f


@pytest.fixture
def stale_env(tmp_path, monkeypatch):
    """Hermetic environment for stale_artifacts(): all artifacts live in
    tmp_path and the digest computations are stubbed, so the real build deps
    (mlx, nanobind, shader-source builders) are never imported."""
    monkeypatch.setattr(build, "_OUT", tmp_path / "_paged_ops.so")
    monkeypatch.setattr(build, "_THIS_DIR", tmp_path)
    monkeypatch.setattr(build, "_input_hash", lambda: "SO_DIGEST")
    monkeypatch.setattr(build, "_metallib_source", lambda name: f"SRC::{name}")
    monkeypatch.setattr(build, "_package_version", lambda name: "0.32.4")
    return tmp_path


def _seed_fresh() -> None:
    """Write every artifact with a stamp matching its stubbed digest."""
    build._OUT.write_bytes(b"so")
    build._stamp_path(build._OUT).write_text("SO_DIGEST")
    for name in build.METALLIB_NAMES:
        lib = build.metallib_path(name)
        lib.write_bytes(b"lib")
        build._stamp_path(lib).write_text(build._metallib_digest(name, f"SRC::{name}"))


def test_stale_artifacts_empty_when_all_fresh(stale_env):
    _seed_fresh()
    assert build.stale_artifacts() == []


def test_prepare_metallib_does_not_import_mlx(stale_env, monkeypatch):
    _seed_fresh()
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    name = build.METALLIB_NAMES[0]
    lib = build.metallib_path(name)
    assert build.prepare_metallib(name) == lib
    build._stamp_path(lib).write_text("STALE")
    with pytest.raises(RuntimeError, match="stale"):
        build.prepare_metallib(name)
    lib.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        build.prepare_metallib(name)


def test_stale_artifacts_empty_without_stamps_skips_build_deps(stale_env, monkeypatch):
    # Wheel install: artifacts present, no stamps. Must short-circuit to [] BEFORE
    # importing build deps -> _input_hash (which would raise here) is never hit.
    build._OUT.write_bytes(b"so")
    for name in build.METALLIB_NAMES:
        build.metallib_path(name).write_bytes(b"lib")
    monkeypatch.setattr(
        build, "_input_hash", _raise(AssertionError("touched build deps"))
    )
    monkeypatch.setattr(
        build, "_package_version", _raise(AssertionError("touched build deps"))
    )
    assert build.stale_artifacts() == []
    for name in build.METALLIB_NAMES:
        assert build.prepare_metallib(name) == build.metallib_path(name)


def test_stale_artifacts_flags_edited_so(stale_env):
    _seed_fresh()
    build._stamp_path(build._OUT).write_text("STALE")  # paged_ops.cpp edited
    assert build.stale_artifacts() == [build._OUT]


def test_stale_artifacts_flags_edited_shader(stale_env):
    _seed_fresh()
    lib = build.metallib_path(build.METALLIB_NAMES[0])
    build._stamp_path(lib).write_text("STALE")  # a .metal shader edited
    assert build.stale_artifacts() == [lib]


def test_stale_artifacts_swallows_digest_errors(stale_env, monkeypatch):
    # If digest computation blows up (e.g. mlx/nanobind missing on a dev box),
    # loading must not break -> [] even though a stamp mismatches.
    _seed_fresh()
    build._stamp_path(build._OUT).write_text("STALE")
    monkeypatch.setattr(build, "_input_hash", _raise(RuntimeError("boom")))
    assert build.stale_artifacts() == []


def test_stale_artifacts_propagates_builder_drift(stale_env, monkeypatch):
    # A KeyError means METALLIB_NAMES drifted from the source-builder map in
    # _metallib_source -- a code bug. stale_artifacts() must let it surface, not
    # swallow it to [] (which would hide the bug); only environmental failures
    # (OSError/ImportError/RuntimeError) are swallowed.
    _seed_fresh()  # stamps exist -> past the no-stamps short-circuit, into the try
    monkeypatch.setattr(build, "_metallib_source", _raise(KeyError("gdn_kern")))
    with pytest.raises(KeyError):
        build.stale_artifacts()


def test_built_mlx_version_none_when_unrecorded(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "_MLX_VERSION", tmp_path / "_paged_ops.mlx-version")
    assert build.built_mlx_version() is None


def test_mlx_version_mismatch_reports_built_and_installed(tmp_path, monkeypatch):
    record = tmp_path / "_paged_ops.mlx-version"
    record.write_text("0.32.1\n")
    monkeypatch.setattr(build, "_MLX_VERSION", record)
    monkeypatch.setattr(build, "_package_version", lambda name: "0.32.3")
    assert build.mlx_version_mismatch() == ("0.32.1", "0.32.3")


def test_mlx_version_mismatch_none_when_installed_matches(tmp_path, monkeypatch):
    record = tmp_path / "_paged_ops.mlx-version"
    record.write_text("0.32.1\n")
    monkeypatch.setattr(build, "_MLX_VERSION", record)
    monkeypatch.setattr(build, "_package_version", lambda name: "0.32.1")
    assert build.mlx_version_mismatch() is None


def test_mlx_version_mismatch_none_when_unrecorded(tmp_path, monkeypatch):
    # Artifacts built before the record existed keep loading as they did.
    monkeypatch.setattr(build, "_MLX_VERSION", tmp_path / "_paged_ops.mlx-version")
    monkeypatch.setattr(build, "_package_version", lambda name: "0.32.3")
    assert build.mlx_version_mismatch() is None


@pytest.fixture
def compiler(patched, monkeypatch, tmp_path):
    sources = {
        name: f"shader {name}"
        for name in (*build.METALLIB_NAMES, build.NAX_METALLIB_NAME)
    }
    calls = []

    def compile_artifacts(sources, build_dir):
        calls.append(build_dir)
        patched.out.write_bytes(b"compiled")
        for name in sources:
            build.metallib_path(name).write_bytes(b"metallib")

    monkeypatch.setattr(build, "_THIS_DIR", tmp_path)
    monkeypatch.setattr(build, "_sdk_supports_nax", lambda: False)
    monkeypatch.setattr(build, "_metal_toolchain_id", lambda: "tc-a")
    monkeypatch.setattr(build, "_metallib_source", sources.__getitem__)
    monkeypatch.setattr(build, "_build_extension", compile_artifacts)
    return sources, calls


@pytest.mark.parametrize("nax_supported", [False, True])
def test_build_entries_reuse_final_artifacts_without_cmake_cache(
    patched, compiler, monkeypatch, nax_supported
):
    _, calls = compiler
    monkeypatch.setattr(build, "_sdk_supports_nax", lambda: nax_supported)
    names = (
        *build.METALLIB_NAMES,
        *([build.NAX_METALLIB_NAME] if nax_supported else []),
    )

    assert build.build_metallibs() == [build.metallib_path(name) for name in names]
    assert all(build.metallib_path(name).exists() for name in names)
    assert patched.ver.read_text() == "0.31.0\n"
    assert patched.hsh.read_text() == build._input_hash()
    assert build.stale_artifacts() == []
    assert build.needs_rebuild() is False
    assert not calls[0].exists()
    assert build.build() == patched.out
    assert len(calls) == 1


@pytest.mark.parametrize("missing_stamp", [False, True])
def test_build_restores_a_missing_shader_artifact(patched, compiler, missing_stamp):
    _, calls = compiler
    build.build()
    library = build.metallib_path(build.METALLIB_NAMES[0])
    (build._stamp_path(library) if missing_stamp else library).unlink()
    assert build.build() == patched.out
    assert len(calls) == 2
    assert library.exists()
    assert build._stamp_path(library).exists()


def test_build_recompiles_edited_shader_sources(patched, compiler):
    sources, calls = compiler
    build.build()
    sources[build.METALLIB_NAMES[0]] = "changed shader"
    build.build()
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert build.stale_artifacts() == []


def test_build_adds_nax_when_sdk_support_becomes_available(
    patched, compiler, monkeypatch
):
    build.build()
    library = build.metallib_path(build.NAX_METALLIB_NAME)
    assert not library.exists()
    monkeypatch.setattr(build, "_sdk_supports_nax", lambda: True)
    build.build()
    assert library.exists()
    assert build._stamp_path(library).exists()


def test_failed_build_does_not_mark_changed_sources_current(
    patched, compiler, monkeypatch
):
    sources, _ = compiler
    build.build()
    name = build.METALLIB_NAMES[0]
    sources[name] = "changed shader"
    monkeypatch.setattr(build, "_metal_toolchain_id", lambda: "tc-b")
    monkeypatch.setattr(build, "_build_extension", _raise(RuntimeError("build failed")))

    with pytest.raises(RuntimeError, match="build failed"):
        build.build()
    assert build.stale_artifacts() == [build.metallib_path(name)]
    assert build._toolchain_path(build.metallib_path(name)).read_text() == "tc-a"


@pytest.mark.parametrize("build_from_source", ["0", "1"])
def test_source_and_wheel_modes_load_metallibs(
    tmp_path, monkeypatch, build_from_source
):
    import vllm_metal.metal as metal

    ops = SimpleNamespace(
        nax_supported=lambda: True,
        init_library_path=Mock(),
        init_nax_library_path=Mock(),
    )
    monkeypatch.setenv("VLLM_METAL_BUILD_FROM_SOURCE", build_from_source)
    monkeypatch.delenv("VLLM_METAL_DISABLE_NAX", raising=False)
    monkeypatch.setattr(metal, "_ops_module", None)
    monkeypatch.setattr(metal, "_load_native_module", lambda: ops)
    monkeypatch.setattr(build, "_THIS_DIR", tmp_path)
    for name in (*build.METALLIB_NAMES, build.NAX_METALLIB_NAME):
        build.metallib_path(name).write_bytes(b"metallib")

    assert metal.get_ops() is ops
    assert ops.init_library_path.call_args_list == [
        call(name, str(build.metallib_path(name))) for name in build.METALLIB_NAMES
    ]
    ops.init_nax_library_path.assert_called_once_with(
        str(build.metallib_path(build.NAX_METALLIB_NAME))
    )


def test_get_ops_rejects_an_extension_built_against_another_mlx(tmp_path, monkeypatch):
    import vllm_metal.metal as metal

    so = tmp_path / "_paged_ops.so"
    so.write_bytes(b"")
    monkeypatch.delenv("VLLM_METAL_BUILD_FROM_SOURCE", raising=False)
    monkeypatch.setattr(metal, "_ops_module", None)
    monkeypatch.setattr(metal, "_native_module", None)
    monkeypatch.setattr(build, "output_path", lambda: so)
    monkeypatch.setattr(build, "stale_artifacts", lambda: [])
    monkeypatch.setattr(build, "mlx_version_mismatch", lambda: ("0.32.1", "0.32.3"))
    with pytest.raises(RuntimeError, match=r"MLX 0\.32\.1.*MLX 0\.32\.3"):
        metal.get_ops()


# --- build_metallibs() skip gate (build-side only) ---------------------------

_LIB = "rope_kern"


@pytest.fixture
def metallib_env(patched, tmp_path, monkeypatch):
    """Record MetalExtension builds; CMake owns per-file incremental work."""
    state = {
        "source": "// rope v1",
        "toolchain": "tc-a",
        "compiled": [],
        "build_dirs": [],
    }

    monkeypatch.setattr(build, "_THIS_DIR", tmp_path)
    monkeypatch.setattr(build, "METALLIB_NAMES", (_LIB,))
    monkeypatch.setattr(build, "_sdk_supports_nax", lambda: True)
    monkeypatch.setattr(build, "NAX_METALLIB_NAME", "nax_fake")
    monkeypatch.setattr(build, "_metallib_source", lambda name: state["source"])
    monkeypatch.setattr(build, "_metal_toolchain_id", lambda: state["toolchain"])

    def fake_build(sources, build_dir):
        state["build_dirs"].append(build_dir)
        patched.out.write_bytes(b"compiled")
        for name in sources:
            state["compiled"].append(name)
            build.metallib_path(name).write_bytes(b"metallib")

    monkeypatch.setattr(build, "_build_extension", fake_build)
    return state


def _lib(tmp_path) -> Path:
    return tmp_path / f"{_LIB}.metallib"


def test_metallib_first_build_compiles_and_writes_sidecars(metallib_env, tmp_path):
    build.build_metallibs()
    assert set(metallib_env["compiled"]) == {_LIB, "nax_fake"}
    assert build._stamp_path(_lib(tmp_path)).is_file()
    assert build._toolchain_path(_lib(tmp_path)).read_text() == "tc-a"


def test_metallib_hit_skips_compile(metallib_env):
    build.build_metallibs()
    metallib_env["compiled"].clear()
    build.build_metallibs()
    assert metallib_env["compiled"] == []


@pytest.mark.parametrize("name", [_LIB, "nax_fake"])
@pytest.mark.parametrize("host_changed", [False, True])
def test_metallib_partial_hit_preserves_cached_library(
    patched, metallib_env, monkeypatch, name, host_changed
):
    build.build_metallibs()
    metallib_env["compiled"].clear()
    cached_name = "nax_fake" if name == _LIB else _LIB
    cached = build.metallib_path(cached_name)
    cached.write_bytes(b"restored metallib")
    monkeypatch.setattr(
        build,
        "_metallib_source",
        lambda n: "changed shader" if n == name else metallib_env["source"],
    )
    if host_changed:
        patched.src.write_bytes(b"// source v2")

    assert build.build_metallibs() == [
        build.metallib_path(_LIB),
        build.metallib_path("nax_fake"),
    ]
    assert metallib_env["compiled"] == [name]
    assert cached.read_bytes() == b"restored metallib"
    assert build.needs_rebuild() is False
    assert build._metallib_is_current(name, "changed shader")


@pytest.mark.parametrize("host_changed", ["source", "missing_artifact"])
def test_host_only_rebuild_includes_only_rope(patched, metallib_env, host_changed):
    build.build_metallibs()
    metallib_env["compiled"].clear()
    cached = build.metallib_path("nax_fake")
    cached.write_bytes(b"restored metallib")
    if host_changed == "source":
        patched.src.write_bytes(b"// source v2")
    else:
        patched.out.unlink()

    assert build.build() == patched.out
    assert metallib_env["compiled"] == [_LIB]
    assert cached.read_bytes() == b"restored metallib"
    assert build.needs_rebuild() is False
    metallib_env["compiled"].clear()
    build.build()
    assert metallib_env["compiled"] == []


def test_metallib_source_change_rebuilds(metallib_env):
    build.build_metallibs()
    metallib_env["compiled"].clear()
    metallib_env["source"] = "// rope v2"
    build.build_metallibs()
    assert set(metallib_env["compiled"]) == {_LIB, "nax_fake"}


def test_mlx_version_change_rebuilds_all_metallibs(patched, metallib_env):
    build.build_metallibs()
    metallib_env["compiled"].clear()
    patched.versions["mlx"] = "99.0.0"

    build.build_metallibs()
    assert set(metallib_env["compiled"]) == {_LIB, "nax_fake"}
    metallib_env["compiled"].clear()
    build.build_metallibs()
    assert metallib_env["compiled"] == []


def test_metallib_toolchain_change_rebuilds(metallib_env):
    build.build_metallibs()
    metallib_env["compiled"].clear()
    metallib_env["toolchain"] = "tc-b"
    build.build_metallibs()
    assert set(metallib_env["compiled"]) == {_LIB, "nax_fake"}
    old_dir, new_dir = metallib_env["build_dirs"]
    assert old_dir.parent.name == "tc-a"
    assert new_dir.parent.name == "tc-b"
    assert old_dir.name == new_dir.name == build._input_hash()


@pytest.mark.parametrize(
    "remove",
    [
        lambda p: p,
        build._stamp_path,
        build._toolchain_path,
    ],
    ids=["artifact", "stamp", "toolchain"],
)
def test_metallib_missing_file_rebuilds(metallib_env, tmp_path, remove):
    build.build_metallibs()
    metallib_env["compiled"].clear()
    remove(_lib(tmp_path)).unlink()
    assert build._metallib_is_current(_LIB, metallib_env["source"]) is False
    assert build._metallib_is_current("nax_fake", metallib_env["source"]) is True
    build.build_metallibs()
    assert metallib_env["compiled"] == [_LIB]


def test_metallib_unreadable_stamp_rebuilds(metallib_env, tmp_path):
    build.build_metallibs()
    metallib_env["compiled"].clear()
    stamp = build._stamp_path(_lib(tmp_path))
    stamp.unlink()
    stamp.mkdir()
    assert build._metallib_is_current(_LIB, metallib_env["source"]) is False
    # Runtime tolerance is unchanged: is_stale() never raises on a bad stamp.
    assert build.is_stale(_lib(tmp_path), "anything") is False


def test_metallib_undecodable_stamp_rebuilds(metallib_env, tmp_path):
    build.build_metallibs()
    metallib_env["compiled"].clear()
    build._stamp_path(_lib(tmp_path)).write_bytes(b"\xff\xfe\x00bad")
    build.build_metallibs()
    assert metallib_env["compiled"] == [_LIB]


def test_metallib_unavailable_toolchain_never_skips(metallib_env):
    build.build_metallibs()
    metallib_env["compiled"].clear()
    metallib_env["toolchain"] = "unavailable"
    build.build_metallibs()
    assert set(metallib_env["compiled"]) == {_LIB, "nax_fake"}
    first_dir = metallib_env["build_dirs"][-1]
    build.build_metallibs()
    assert metallib_env["build_dirs"][-1] != first_dir
    assert not first_dir.exists()


def test_runtime_staleness_does_not_query_toolchain(monkeypatch, tmp_path):
    lib = tmp_path / "x.metallib"
    lib.write_bytes(b"m")
    build._stamp_path(lib).write_text("d")

    def boom():
        raise AssertionError("runtime must not invoke xcrun")

    monkeypatch.setattr(build, "_metal_toolchain_id", boom)
    assert build.is_stale(lib, "d") is False


def _fake_xcrun(metal_out: str, sdk_out: str = "26.2\n"):
    def run(cmd, **kwargs):
        out = sdk_out if "--show-sdk-version" in cmd else metal_out
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    return run


def _toolchain_id_with(monkeypatch, metal_out: str) -> str:
    monkeypatch.setattr(build.subprocess, "run", _fake_xcrun(metal_out))
    build._metal_toolchain_id.cache_clear()
    try:
        return build._metal_toolchain_id()
    finally:
        build._metal_toolchain_id.cache_clear()


_METAL_V1 = "Apple metal version 32023.620 (metalfe-32023.620)\nTarget: air64-apple-darwin25.0.0\n"


def test_toolchain_id_ignores_installed_dir(monkeypatch):
    a = _toolchain_id_with(
        monkeypatch,
        _METAL_V1
        + "InstalledDir: /private/var/run/com.apple.security.cryptexd/mnt/AAA/bin\n",
    )
    b = _toolchain_id_with(
        monkeypatch,
        _METAL_V1
        + "InstalledDir: /private/var/run/com.apple.security.cryptexd/mnt/BBB/bin\n",
    )
    assert a == b != "unavailable"


def test_toolchain_id_changes_with_metal_version(monkeypatch):
    a = _toolchain_id_with(monkeypatch, _METAL_V1)
    b = _toolchain_id_with(monkeypatch, _METAL_V1.replace("32023.620", "32023.700"))
    assert a != b
