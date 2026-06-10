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
import tempfile
from dataclasses import dataclass
from pathlib import Path

from vllm_metal.metal.constants import PARTITION_SIZE

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_SRC = _THIS_DIR / "paged_ops.cpp"
_BUILD = _THIS_DIR / "build.py"
_CONSTANTS = _THIS_DIR / "constants.py"
_EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
# The built extension lives inside the package directory so packaging
# (maturin ``include``) bundles it into the wheel and the runtime loads it
# without ever invoking clang++ on the end-user machine.
_OUT = _THIS_DIR / f"_paged_ops{_EXT_SUFFIX}"


def _stamp_path(artifact: Path) -> Path:
    """The ``.sha256`` sidecar recording the inputs an artifact was built from."""
    return artifact.with_suffix(artifact.suffix + ".sha256")


# The .so's stamp; identical mechanism to each .metallib's (see is_stale).
_HASH = _stamp_path(_OUT)

# Names of the three precompiled Metal shader libraries.  Each is the cache-key
# name the C++ extension registers the library under (passed to
# ``init_library_path(name, path)`` in paged_ops.cpp), so a later dispatch's
# ``get_library(name)`` returns the .metallib loaded at startup.
METALLIB_NAMES = ("paged_attention_v2_kern", "gdn_kern", "paged_mla_kern")


def output_path() -> Path:
    """Path to the prebuilt native extension (whether or not it exists yet)."""
    return _OUT


def metallib_path(name: str) -> Path:
    """Path to a precompiled Metal shader library (whether or not it exists)."""
    return _THIS_DIR / f"{name}.metallib"


# Stable parts of the metallib compile command (the per-build temp input and
# output paths are appended in _compile_metallib). One constant so the actual
# compile and the staleness digest agree on the flags. We mirror the flags MLX
# uses to build its own shader libraries (share/cmake/MLX/extension.cmake:
# ``MTLLIB_COMPILE_OPTIONS = -Wall -Wextra -fno-fast-math -Wno-c++17-extensions``;
# the -W* flags only affect diagnostics). ``-fno-fast-math`` is the
# numerics-critical one: the metal compiler defaults to fast-math ON, but MLX's
# runtime ``newLibrary`` sets ``setFastMathEnabled(false)``, so without it the
# precompiled kernels diverge from the in-process source compile. We deliberately
# do NOT add ``-O3``: MLX compiles its shaders at the metal default optimization
# (already enabled), so matching keeps the prebuilt path numerically and
# behaviourally identical to what users get today.
_METALLIB_FLAGS = ("xcrun", "-sdk", "macosx", "metal", "-fno-fast-math")


def _metallib_source(name: str) -> str:
    """Assemble the concatenated Metal source for one shader library. Source
    builders are imported at call time (function-level) to avoid import-order
    coupling with :mod:`vllm_metal.metal`."""
    from vllm_metal.metal import (
        _build_gdn_source,
        _build_mla_paged_attention_source,
        _build_v2_paged_attention_source,
    )

    builders = {
        "paged_attention_v2_kern": _build_v2_paged_attention_source,
        "gdn_kern": _build_gdn_source,
        "paged_mla_kern": _build_mla_paged_attention_source,
    }
    return builders[name]()


def _metallib_digest(source: str) -> str:
    """Content hash of a metallib: its assembled ``.metal`` source + the compile
    flags. Editing any concatenated shader changes ``source`` and so the digest."""
    h = hashlib.sha256()
    h.update("\0".join(_METALLIB_FLAGS).encode())
    h.update(b"\0")
    h.update(source.encode())
    return h.hexdigest()


def _run_or_raise(cmd: list[str], what: str) -> None:
    """Log and run a compiler command, raising with both streams on failure."""
    logger.info("  %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to {what}:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _compile_metallib(name: str, source: str) -> Path:
    """Compile Metal *source* to ``<name>.metallib`` in the package dir and write
    its ``.sha256`` staleness stamp."""
    out = metallib_path(name)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".metal", delete=False, dir=_THIS_DIR
    )
    try:
        tmp.write(source)
        tmp.close()
        _run_or_raise(
            [*_METALLIB_FLAGS, tmp.name, "-o", str(out)],
            f"compile Metal library '{name}'",
        )
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    _stamp_path(out).write_text(_metallib_digest(source))
    return out


def build_metallibs() -> list[Path]:
    """Precompile the three Metal shader libraries to ``.metallib`` files.

    PACKAGING-only (CI + release.sh): the default runtime path loads these
    prebuilt libraries, but source mode (``VLLM_METAL_BUILD_FROM_SOURCE=1``)
    compiles the shaders in-process via MLX, so a source-install dev without
    the Metal toolchain never reaches this code.
    """
    logger.info("Building Metal shader libraries ...")
    return [_compile_metallib(name, _metallib_source(name)) for name in METALLIB_NAMES]


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
        # IOKit + CoreFoundation: gpu_core_count() in paged_ops.cpp queries the
        # IORegistry. Explicit -framework is load-bearing — "-undefined
        # dynamic_lookup" below resolves only against already-loaded images,
        # and nothing else loads IOKit.
        "-framework",
        "IOKit",
        "-framework",
        "CoreFoundation",
        # "-lmlx" above validates our MLX symbols at link time and records a
        # dependency on "@rpath/libmlx.dylib" (libmlx's install name). We add NO
        # "-Wl,-rpath": baking the build machine's MLX path into a prebuilt wheel
        # is wrong (it won't exist on the user's machine). So that recorded
        # dependency is resolved not via an rpath but against an already-resident
        # libmlx, which dyld matches by install name; "-undefined dynamic_lookup"
        # resolves any remaining symbols the same way. Both require libmlx to be
        # loaded first, which metal/__init__.py guarantees by importing mlx.core
        # before this .so is dlopen'd.
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
    # Hash the inputs that change the produced .so, but NOT the machine-specific
    # include/lib paths in spec.cmd: with `-undefined dynamic_lookup` the .so
    # bakes in no absolute path, so the same sources + MLX/nanobind versions give
    # an equivalent binary wherever the venv lives (moving it must not flag the
    # artifact stale). The compile flags and defines (e.g. PARTITION_SIZE) come
    # from build.py / constants.py, whose contents are hashed in the loop below,
    # so editing a flag still busts the hash.
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


def is_stale(artifact: Path, expected_digest: str) -> bool:
    """One staleness primitive for every prebuilt artifact (the ``.so`` and each
    ``.metallib``). True only if ``artifact`` and its ``.sha256`` stamp both
    exist but the stamp no longer matches ``expected_digest`` — i.e. a source
    was edited without rebuilding. A missing stamp (every wheel install, which
    ships artifacts but no stamps) returns False, so end users never see a
    staleness warning. Never raises."""
    stamp = _stamp_path(artifact)
    if not artifact.exists() or not stamp.exists():
        return False
    try:
        return stamp.read_text().strip() != expected_digest
    except OSError:
        return False


def stale_artifacts() -> list[Path]:
    """Prebuilt artifacts whose stamp no longer matches the current sources —
    the ``_paged_ops`` ``.so`` (vs ``paged_ops.cpp`` et al.) and each
    ``.metallib`` (vs its ``.metal`` shaders) — checked through the single
    :func:`is_stale` mechanism. Returns ``[]`` for wheel installs (no stamps)
    without importing the build deps.

    Environmental failures — mlx/nanobind not importable, a source file that
    cannot be read — are swallowed (logged at debug) so a staleness check never
    breaks loading. A programming error such as a ``METALLIB_NAMES`` entry with
    no source builder (a ``KeyError`` from :func:`_metallib_source`) is
    deliberately left to propagate: that is a bug to fix, not a condition to
    hide behind ``[]``."""
    artifacts = (_OUT, *(metallib_path(n) for n in METALLIB_NAMES))
    if not any(_stamp_path(a).exists() for a in artifacts):
        return []
    try:
        digests = {_OUT: _input_hash(_build_spec())}
        for name in METALLIB_NAMES:
            digests[metallib_path(name)] = _metallib_digest(_metallib_source(name))
    except (OSError, ImportError, RuntimeError) as exc:
        # The check couldn't run (not a verdict on the artifacts); don't block a
        # loadable install, but leave a breadcrumb instead of failing silently.
        logger.debug("staleness check skipped (%s): %s", type(exc).__name__, exc)
        return []
    return [a for a in artifacts if is_stale(a, digests[a])]


def build() -> Path:
    """JIT-build the native extension, returning the path to the .so.

    Skips the compile when the prebuilt .so is already current (needs_rebuild)."""
    if not needs_rebuild():
        return _OUT

    spec = _build_spec()
    expected_hash = _input_hash(spec)
    logger.info("Building native paged-attention extension ...")

    for p, label in [
        (spec.py_include, "Python include"),
        (spec.mlx_include, "MLX include"),
        (spec.mlx_lib / "libmlx.dylib", "MLX lib"),
        (spec.nb_src, "nanobind source"),
    ]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    _run_or_raise(spec.cmd, "build paged_ops extension")

    _HASH.write_text(expected_hash)
    logger.info("Built %s", _OUT)
    return _OUT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    so = build()
    print(so)
    for lib in build_metallibs():
        print(lib)
