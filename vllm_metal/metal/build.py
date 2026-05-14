# SPDX-License-Identifier: Apache-2.0
"""JIT build script for the native paged-attention Metal extension.

Compiles ``paged_ops.cpp`` + nanobind into a shared library that dispatches
Metal shaders through MLX's own command encoder.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sysconfig
from dataclasses import dataclass
from pathlib import Path

from vllm_metal.metal.constants import PARTITION_SIZE

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_SRC = _THIS_DIR / "paged_ops.cpp"
_BUILD = _THIS_DIR / "build.py"
_CONSTANTS = _THIS_DIR / "constants.py"
_EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
_CACHE_DIR = Path.home() / ".cache" / "vllm-metal"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_OUT = _CACHE_DIR / f"_paged_ops{_EXT_SUFFIX}"
_HASH = _OUT.with_suffix(_OUT.suffix + ".sha256")


def _find_package_path(name: str) -> Path:
    """Resolve a Python package's root directory."""
    import importlib

    mod = importlib.import_module(name)
    paths = getattr(mod, "__path__", None)
    if paths:
        return Path(list(paths)[0])
    f = getattr(mod, "__file__", None)
    if f:
        return Path(f).parent
    raise RuntimeError(f"Cannot locate package '{name}'")


def _package_version(name: str) -> str:
    import importlib

    mod = importlib.import_module(name)
    return str(getattr(mod, "__version__", ""))


@dataclass(frozen=True)
class _BuildSpec:
    cmd: list[str]
    nb_src: Path
    mlx_lib: Path
    mlx_include: Path
    py_include: str
    mlx_version: str
    nb_version: str


def _build_spec() -> _BuildSpec:
    """Resolve every input that affects the produced .so."""
    py_include = sysconfig.get_paths()["include"]
    nb_path = _find_package_path("nanobind")
    mlx_path = _find_package_path("mlx")
    mlx_include = mlx_path / "include"
    mlx_lib = mlx_path / "lib"
    metal_cpp = mlx_include / "metal_cpp"
    nb_src = nb_path / "src" / "nb_combined.cpp"

    cmd = [
        "clang++",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-O2",
        "-fvisibility=default",
        f"-I{py_include}",
        f"-I{nb_path / 'include'}",
        f"-I{nb_path / 'src'}",
        f"-I{nb_path / 'ext' / 'robin_map' / 'include'}",
        f"-I{mlx_include}",
        f"-I{metal_cpp}",
        f"-L{mlx_lib}",
        "-lmlx",
        "-framework",
        "Metal",
        "-framework",
        "Foundation",
        f"-Wl,-rpath,{mlx_lib}",
        "-D_METAL_",
        "-DACCELERATE_NEW_LAPACK",
        f"-DVLLM_METAL_PARTITION_SIZE={PARTITION_SIZE}",
        "-undefined",
        "dynamic_lookup",
        str(nb_src),
        str(_SRC),
        "-o",
        str(_OUT),
    ]

    return _BuildSpec(
        cmd=cmd,
        nb_src=nb_src,
        mlx_lib=mlx_lib,
        mlx_include=mlx_include,
        py_include=py_include,
        mlx_version=_package_version("mlx.core"),
        nb_version=_package_version("nanobind"),
    )


def _input_hash(spec: _BuildSpec) -> str:
    h = hashlib.sha256()
    # Resolved compile invocation captures MLX/nanobind/Python paths and
    # compile-time defines like PARTITION_SIZE.
    h.update("\0".join(spec.cmd).encode())
    h.update(b"\0")
    # Versions catch in-place upgrades where the install path is reused.
    h.update(f"mlx={spec.mlx_version}\0nb={spec.nb_version}\0".encode())
    for p in (_SRC, _BUILD, _CONSTANTS, spec.nb_src):
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def needs_rebuild() -> bool:
    if not _OUT.exists() or not _HASH.exists():
        return True
    try:
        spec = _build_spec()
        return _HASH.read_text().strip() != _input_hash(spec)
    except (OSError, ImportError, RuntimeError):
        return True


def build() -> Path:
    """JIT-build the native extension, returning the path to the .so."""
    spec = _build_spec()
    expected_hash = _input_hash(spec)
    if _OUT.exists() and _HASH.exists():
        try:
            if _HASH.read_text().strip() == expected_hash:
                return _OUT
        except OSError:
            pass

    logger.info("Building native paged-attention extension ...")

    for p, label in [
        (spec.py_include, "Python include"),
        (spec.mlx_include, "MLX include"),
        (spec.mlx_lib / "libmlx.dylib", "MLX lib"),
        (spec.nb_src, "nanobind source"),
    ]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    logger.info("  %s", " ".join(spec.cmd))
    result = subprocess.run(spec.cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to build paged_ops extension:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    _HASH.write_text(expected_hash)
    logger.info("Built %s", _OUT)
    return _OUT
