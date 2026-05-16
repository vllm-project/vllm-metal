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


def test_hash_changes_with_cmd(patched):
    # Simulates venv switch / different MLX install path.
    h1 = build._input_hash(patched.spec)
    s2 = dataclasses.replace(patched.spec, cmd=patched.spec.cmd + ["-L/other/mlx/lib"])
    assert build._input_hash(s2) != h1


def test_hash_changes_with_mlx_version(patched):
    h1 = build._input_hash(patched.spec)
    s2 = dataclasses.replace(patched.spec, mlx_version="99.0.0")
    assert build._input_hash(s2) != h1


def test_hash_changes_with_nb_version(patched):
    h1 = build._input_hash(patched.spec)
    s2 = dataclasses.replace(patched.spec, nb_version="99.0.0")
    assert build._input_hash(s2) != h1


def test_hash_stable_across_calls(patched):
    assert build._input_hash(patched.spec) == build._input_hash(patched.spec)


def test_needs_rebuild_swallows_resolution_errors(patched, monkeypatch):
    patched.out.write_bytes(b"compiled")
    patched.hsh.write_text("anything")

    def boom():
        raise ImportError("mlx not installed")

    monkeypatch.setattr(build, "_build_spec", boom)
    assert build.needs_rebuild() is True
