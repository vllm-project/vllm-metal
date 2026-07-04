# SPDX-License-Identifier: Apache-2.0
"""Tests for the JIT-build cache invalidation in vllm_metal.metal.build."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from vllm_metal.metal import build


@dataclass
class _Paths:
    src: Path
    bld: Path
    consts: Path
    nb_src: Path
    out: Path
    hsh: Path
    spec: build._BuildSpec


@pytest.fixture
def patched(tmp_path, monkeypatch) -> _Paths:
    src = tmp_path / "paged_ops.cpp"
    bld = tmp_path / "build.py"
    consts = tmp_path / "constants.py"
    nb_src = tmp_path / "nb_combined.cpp"
    out = tmp_path / "_paged_ops.so"
    hsh = tmp_path / "_paged_ops.so.sha256"

    src.write_bytes(b"// source v1")
    bld.write_bytes(b"# build v1")
    consts.write_bytes(b"PARTITION_SIZE = 256")
    nb_src.write_bytes(b"// nanobind combined v1")

    monkeypatch.setattr(build, "_SRC", src)
    monkeypatch.setattr(build, "_BUILD", bld)
    monkeypatch.setattr(build, "_CONSTANTS", consts)
    monkeypatch.setattr(build, "_OUT", out)
    monkeypatch.setattr(build, "_HASH", hsh)

    spec = build._BuildSpec(
        cmd=["clang++", "-O2", "-Lfake/mlx/lib", "-o", str(out)],
        nb_src=nb_src,
        mlx_lib=Path("/fake/mlx/lib"),
        mlx_include=Path("/fake/mlx/include"),
        py_include="/fake/py/include",
        mlx_version="0.31.0",
        nb_version="2.10.0",
    )
    monkeypatch.setattr(build, "_build_spec", lambda: spec)

    return _Paths(src, bld, consts, nb_src, out, hsh, spec)


def test_needs_rebuild_when_so_missing(patched):
    assert build.needs_rebuild() is True


def test_needs_rebuild_when_hash_sidecar_missing(patched):
    patched.out.write_bytes(b"compiled")
    assert build.needs_rebuild() is True


def test_needs_rebuild_when_hash_differs(patched):
    patched.out.write_bytes(b"compiled")
    patched.hsh.write_text("deadbeef")
    assert build.needs_rebuild() is True


def test_no_rebuild_when_hash_matches(patched):
    patched.out.write_bytes(b"compiled")
    patched.hsh.write_text(build._input_hash(patched.spec))
    assert build.needs_rebuild() is False


def test_old_content_with_newer_so_mtime_still_rebuilds(patched):
    # Regression: under the old mtime-based check, a stale .so from a previous
    # branch could shadow freshly-checked-out sources whose mtimes git set to
    # the checkout time.
    patched.out.write_bytes(b"compiled-from-prev-branch")
    patched.hsh.write_text("hash-from-prev-branch")
    older = patched.out.stat().st_mtime - 86400
    for p in (patched.src, patched.bld, patched.consts, patched.nb_src):
        os.utime(p, (older, older))
    assert build.needs_rebuild() is True


def test_hash_changes_with_source_bytes(patched):
    h1 = build._input_hash(patched.spec)
    patched.src.write_bytes(b"// source v2")
    assert build._input_hash(patched.spec) != h1


def test_hash_changes_with_constants_bytes(patched):
    h1 = build._input_hash(patched.spec)
    patched.consts.write_bytes(b"PARTITION_SIZE = 512")
    assert build._input_hash(patched.spec) != h1


def test_hash_changes_with_nb_src_bytes(patched):
    h1 = build._input_hash(patched.spec)
    patched.nb_src.write_bytes(b"// nanobind combined v2")
    assert build._input_hash(patched.spec) != h1


def test_hash_ignores_cmd_paths(patched):
    # The .so links `-undefined dynamic_lookup` and bakes in no absolute path, so
    # moving the venv (different -L/-I/-o paths in spec.cmd) yields an equivalent
    # binary and must NOT flag the artifact stale. Semantically meaningful inputs
    # (compile flags, PARTITION_SIZE) live in build.py/constants.py, whose bytes
    # ARE hashed below — so this excludes only machine-specific paths.
    h1 = build._input_hash(patched.spec)
    s2 = dataclasses.replace(patched.spec, cmd=patched.spec.cmd + ["-L/other/mlx/lib"])
    assert build._input_hash(s2) == h1


def test_hash_changes_with_mlx_version(patched):
    h1 = build._input_hash(patched.spec)
    s2 = dataclasses.replace(patched.spec, mlx_version="99.0.0")
    assert build._input_hash(s2) != h1


def test_hash_changes_with_nb_version(patched):
    h1 = build._input_hash(patched.spec)
    s2 = dataclasses.replace(patched.spec, nb_version="99.0.0")
    assert build._input_hash(s2) != h1


def test_needs_rebuild_swallows_resolution_errors(patched, monkeypatch):
    patched.out.write_bytes(b"compiled")
    patched.hsh.write_text("anything")

    def boom():
        raise ImportError("mlx not installed")

    monkeypatch.setattr(build, "_build_spec", boom)
    assert build.needs_rebuild() is True


# --------------------------------------------------------------------------
# Path contract (the runtime loader and verify_wheel_artifacts depend on these)
# --------------------------------------------------------------------------


def test_output_path_points_at_the_so(patched):
    assert build.output_path() == patched.out


def test_metallib_path_named_in_package(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "_THIS_DIR", tmp_path)
    assert build.metallib_path("gdn_kern") == tmp_path / "gdn_kern.metallib"


def test_metallib_digest_changes_with_source():
    # The metallib staleness stamp is keyed on the assembled shader source, so a
    # one-byte shader edit must change the digest (else an edited kernel loads
    # silently stale).
    assert build._metallib_digest("shader A") != build._metallib_digest("shader B")


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
    monkeypatch.setattr(build, "_build_spec", lambda: None)
    monkeypatch.setattr(build, "_input_hash", lambda _spec: "SO_DIGEST")
    monkeypatch.setattr(build, "_metallib_source", lambda name: f"SRC::{name}")
    return tmp_path


def _seed_fresh() -> None:
    """Write every artifact with a stamp matching its stubbed digest."""
    build._OUT.write_bytes(b"so")
    build._stamp_path(build._OUT).write_text("SO_DIGEST")
    for name in build.METALLIB_NAMES:
        lib = build.metallib_path(name)
        lib.write_bytes(b"lib")
        build._stamp_path(lib).write_text(build._metallib_digest(f"SRC::{name}"))


def test_stale_artifacts_empty_when_all_fresh(stale_env):
    _seed_fresh()
    assert build.stale_artifacts() == []


def test_stale_artifacts_empty_without_stamps_skips_build_deps(stale_env, monkeypatch):
    # Wheel install: artifacts present, no stamps. Must short-circuit to [] BEFORE
    # importing build deps -> _input_hash (which would raise here) is never hit.
    build._OUT.write_bytes(b"so")
    for name in build.METALLIB_NAMES:
        build.metallib_path(name).write_bytes(b"lib")
    monkeypatch.setattr(
        build, "_input_hash", _raise(AssertionError("touched build deps"))
    )
    assert build.stale_artifacts() == []


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
