# SPDX-License-Identifier: Apache-2.0
"""Build shared Metal shader artifacts and both native launchers."""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import subprocess
import sysconfig
import tempfile
from pathlib import Path

from vllm_metal.metal.constants import (
    PA_WINDOW_MAX_HEAD_SIZE,
    PA_WINDOW_ROWS,
    PARTITION_SIZE,
)

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_SRC = _THIS_DIR / "paged_ops.cpp"
_MLX_PATCH = _THIS_DIR / "mlx_patch.cpp"
_KERNELS = _THIS_DIR / "paged_attention_kernels.h"
_DEVICE = _THIS_DIR / "metal_device.h"
_BUILD = _THIS_DIR / "build.py"
_CONSTANTS = _THIS_DIR / "constants.py"
_EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
# The built extension lives inside the package so the wheel can include it.
# Runtime loads the prebuilt artifact without invoking clang++ on the user's Mac.
_OUT = _THIS_DIR / f"_paged_ops{_EXT_SUFFIX}"
_MPS_DIR = _THIS_DIR.parent / "pytorch_backend"
_MPS_OUT = _MPS_DIR / f"_mps_ops{_EXT_SUFFIX}"
_MPS_VERSION = _MPS_DIR / "_mps_ops.torch-version"


def _stamp_path(artifact: Path) -> Path:
    """The ``.sha256`` sidecar recording the inputs an artifact was built from."""
    return artifact.with_suffix(artifact.suffix + ".sha256")


# The .so's stamp; identical mechanism to each .metallib's (see is_stale).
_HASH = _stamp_path(_OUT)

# The MLX version build() compiled the .so against; shipped in the wheel.
_MLX_VERSION = _THIS_DIR / "_paged_ops.mlx-version"

# Required precompiled shader libraries bundled in every wheel.
METALLIB_NAMES = ("paged_attention_v2_kern", "gdn_kern", "paged_mla_kern", "rope_kern")

# Kept outside METALLIB_NAMES because wheels built with older SDKs omit it.
NAX_METALLIB_NAME = "paged_attention_nax_kern"


def output_path() -> Path:
    """Path to the prebuilt native extension (whether or not it exists yet)."""
    return _OUT


def mlx_version_path() -> Path:
    return _MLX_VERSION


def mps_output_path() -> Path:
    return _MPS_OUT


def mps_version_path() -> Path:
    return _MPS_VERSION


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
MIN_MACOS_VERSION = "15.0"
NAX_MIN_MACOS_VERSION = "26.2"
_METALLIB_BASE_FLAGS = ("xcrun", "-sdk", "macosx", "metal", "-fno-fast-math")
_METALLIB_FLAGS = (
    *_METALLIB_BASE_FLAGS,
    f"-mmacosx-version-min={MIN_MACOS_VERSION}",
)

# MPP tensor ops need a 26.2 deployment floor at compile and link time.
_NAX_MIN_SDK = tuple(int(x) for x in NAX_MIN_MACOS_VERSION.split("."))
_NAX_METALLIB_FLAGS = (
    *_METALLIB_BASE_FLAGS,
    f"-mmacosx-version-min={NAX_MIN_MACOS_VERSION}",
)


def _metallib_flags(name: str) -> tuple[str, ...]:
    return _NAX_METALLIB_FLAGS if name == NAX_METALLIB_NAME else _METALLIB_FLAGS


def _sdk_supports_nax() -> bool:
    """Whether the local macOS SDK can compile the NAX library at all."""
    try:
        out = subprocess.run(
            ["xcrun", "-sdk", "macosx", "--show-sdk-version"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        parts = tuple(int(x) for x in out.split(".")[:2])
        return parts >= _NAX_MIN_SDK
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False


def require_nax_sdk() -> None:
    """Fail unless the selected SDK can build the official NAX artifact."""
    if not _sdk_supports_nax():
        raise RuntimeError(
            f"Official wheel builds require macOS SDK >= {NAX_MIN_MACOS_VERSION} "
            f"to compile {NAX_METALLIB_NAME}."
        )


def _metallib_source(name: str) -> str:
    """Assemble the concatenated Metal source for one shader library. Source
    builders are imported at call time (function-level) to avoid import-order
    coupling with :mod:`vllm_metal.metal`."""
    from vllm_metal.metal import (
        _build_gdn_source,
        _build_mla_paged_attention_source,
        _build_nax_source,
        _build_rope_source,
        _build_v2_paged_attention_source,
    )

    builders = {
        "paged_attention_v2_kern": _build_v2_paged_attention_source,
        "gdn_kern": _build_gdn_source,
        "paged_mla_kern": _build_mla_paged_attention_source,
        NAX_METALLIB_NAME: _build_nax_source,
        "rope_kern": _build_rope_source,
    }
    return builders[name]()


def _metallib_digest(name: str, source: str) -> str:
    """Hash the assembled Metal source, compile flags, and MLX build version."""
    h = hashlib.sha256()
    h.update(f"mlx={_package_version('mlx')}\0".encode())
    h.update("\0".join(_metallib_flags(name)).encode())
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
            [*_metallib_flags(name), tmp.name, "-o", str(out)],
            f"compile Metal library '{name}'",
        )
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    _stamp_path(out).write_text(_metallib_digest(name, source))
    _toolchain_path(out).write_text(_metal_toolchain_id())
    return out


def prepare_metallib(name: str, *, build_from_source: bool = False) -> Path:
    """Resolve one shader artifact without loading either framework's extension."""
    if build_from_source:
        return _compile_metallib(name, _metallib_source(name))
    path = metallib_path(name)
    if not path.exists():
        raise RuntimeError(
            f"Prebuilt Metal library missing: {path}. Install a vllm-metal release "
            "wheel, or set VLLM_METAL_BUILD_FROM_SOURCE=1 to compile shaders."
        )
    if _stamp_path(path).exists() and is_stale(
        path, _metallib_digest(name, _metallib_source(name))
    ):
        raise RuntimeError(
            f"Prebuilt Metal library is stale: {path}. Set "
            "VLLM_METAL_BUILD_FROM_SOURCE=1 to rebuild it."
        )
    return path


def _toolchain_path(artifact: Path) -> Path:
    """Build-only sidecar recording the Metal compiler + SDK an artifact was
    compiled with. Never read at runtime (loading must not invoke ``xcrun``)."""
    return artifact.with_suffix(artifact.suffix + ".toolchain")


@functools.lru_cache(maxsize=1)
def _metal_toolchain_id() -> str:
    """Digest of ``xcrun metal --version`` + the SDK version. Build-time only.
    Any failure yields a sentinel that never matches a recorded id, forcing a
    compile (which then surfaces the real toolchain error)."""
    h = hashlib.sha256()
    for cmd in (
        ["xcrun", "-sdk", "macosx", "metal", "--version"],
        ["xcrun", "-sdk", "macosx", "--show-sdk-version"],
    ):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
        # InstalledDir names the Metal toolchain's cryptex mount, which gets a
        # random suffix per boot, so it would change the id on every runner.
        out = "\n".join(
            line for line in out.splitlines() if not line.startswith("InstalledDir:")
        )
        h.update(out.strip().encode())
        h.update(b"\0")
    return h.hexdigest()


def _read_strict(path: Path) -> str | None:
    """Read a sidecar for a build decision; any error means "not current"."""
    try:
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _metallib_is_current(name: str, source: str) -> bool:
    """Build-side skip gate: the library exists, its ``.sha256`` stamp matches
    the source, flags, and MLX version, and it was compiled by the current
    Metal toolchain. Stricter than the runtime :func:`is_stale` on purpose: an
    unreadable stamp or toolchain record means rebuild."""
    path = metallib_path(name)
    if not path.is_file():
        return False
    if _read_strict(_stamp_path(path)) != _metallib_digest(name, source):
        return False
    toolchain = _metal_toolchain_id()
    return (
        toolchain != "unavailable" and _read_strict(_toolchain_path(path)) == toolchain
    )


def _build_extension(sources: dict[str, str], build_dir: Path) -> None:
    from mlx.extension import BuildExtension, MetalExtension, MetalLibrary
    from setuptools import Distribution

    build_dir.mkdir(parents=True, exist_ok=True)
    libraries = []
    for name, source in sources.items():
        path = build_dir / f"{name}.metal"
        if not path.exists() or path.read_text() != source:
            path.write_text(source)
        libraries.append(
            MetalLibrary(
                name,
                sources=[str(path)],
                deployment_target=(
                    NAX_MIN_MACOS_VERSION
                    if name == NAX_METALLIB_NAME
                    else MIN_MACOS_VERSION
                ),
            )
        )
    extension = MetalExtension(
        "vllm_metal.metal._paged_ops",
        sources=[str(_SRC), str(_MLX_PATCH)],
        metal_libraries=libraries,
        define_macros=[
            ("_METAL_", None),
            ("ACCELERATE_NEW_LAPACK", None),
            ("VLLM_METAL_PARTITION_SIZE", str(PARTITION_SIZE)),
            ("VLLM_METAL_PA_WINDOW_ROWS", str(PA_WINDOW_ROWS)),
            ("VLLM_METAL_PA_WINDOW_MAX_HEAD", str(PA_WINDOW_MAX_HEAD_SIZE)),
        ],
        extra_compile_args={"cxx": ["-O2", "-fvisibility=default"]},
        extra_link_args=[
            "SHELL:-framework IOKit",
            "SHELL:-framework CoreFoundation",
        ],
    )
    command = BuildExtension(
        Distribution(
            {
                "ext_modules": [extension],
                "packages": ["vllm_metal.metal"],
                "package_dir": {"vllm_metal.metal": str(_THIS_DIR)},
            }
        ),
        generate_stubs=False,
    )
    command.ensure_finalized()
    command.build_lib = str(build_dir / "lib")
    command.build_temp = str(build_dir)
    command.inplace = True
    command.force = True
    previous_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = MIN_MACOS_VERSION
    try:
        command.run()
    finally:
        if previous_target is None:
            os.environ.pop("MACOSX_DEPLOYMENT_TARGET", None)
        else:
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = previous_target


def build_metallibs() -> list[Path]:
    """Build the MLX launcher and stale shader libraries."""
    names = METALLIB_NAMES
    if _sdk_supports_nax():
        names = (*names, NAX_METALLIB_NAME)
    else:
        logger.warning(
            "macOS SDK < %s: skipping %s",
            ".".join(map(str, _NAX_MIN_SDK)),
            NAX_METALLIB_NAME,
        )
    sources = {name: _metallib_source(name) for name in names}
    digests = {
        metallib_path(name): _metallib_digest(name, source)
        for name, source in sources.items()
    }
    pending_sources = {
        name: source
        for name, source in sources.items()
        if not _metallib_is_current(name, source)
    }
    if not pending_sources and not needs_rebuild():
        return list(digests)
    if not pending_sources:
        # MetalExtension requires a library even for C++-only changes.
        pending_sources["rope_kern"] = sources["rope_kern"]

    expected_hash = _input_hash()
    toolchain = _metal_toolchain_id()
    logger.info("Building MLX native extension and Metal shader libraries ...")
    if toolchain == "unavailable":
        # Unknown toolchains must not reuse intermediate outputs.
        with tempfile.TemporaryDirectory(prefix="vllm-metal-mlx-build-") as tmp:
            _build_extension(pending_sources, Path(tmp))
    else:
        _build_extension(
            pending_sources, _THIS_DIR / "build" / toolchain / expected_hash
        )
    _HASH.write_text(expected_hash)
    _MLX_VERSION.write_text(_package_version("mlx") + "\n")
    for name in pending_sources:
        path = metallib_path(name)
        _stamp_path(path).write_text(digests[path])
        _toolchain_path(path).write_text(toolchain)
    return list(digests)


def _package_version(name: str) -> str:
    from importlib.metadata import version

    return version(name)


def _input_hash() -> str:
    h = hashlib.sha256()
    # Paths are excluded so moving an environment does not invalidate artifacts.
    for name in ("mlx", "nanobind"):
        h.update(f"{name}={_package_version(name)}\0".encode())
    for p in (_SRC, _MLX_PATCH, _KERNELS, _DEVICE, _BUILD, _CONSTANTS):
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _missing_sentinels() -> list[Path]:
    """The outputs a complete build leaves behind; any missing means rebuild."""
    return [p for p in (_OUT, _HASH, _MLX_VERSION) if not p.exists()]


def needs_rebuild() -> bool:
    if _missing_sentinels():
        return True
    try:
        return _read_strict(_HASH) != _input_hash()
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


def built_mlx_version() -> str | None:
    """The MLX version the native extension was built against, if recorded."""
    try:
        return _MLX_VERSION.read_text().strip() or None
    except OSError:
        return None


def mlx_version_mismatch() -> tuple[str, str] | None:
    """``(built, installed)`` when the extension was built against another MLX.

    The ``.so`` links MLX private headers, so it is only ABI-safe against the
    exact MLX it was compiled with. ``build()`` records that version next to
    the artifact and the wheel ships it; an artifact built before the record
    existed has nothing to compare and loads as before.
    """
    built = built_mlx_version()
    if built is None:
        return None
    installed = _package_version("mlx")
    if installed == built:
        return None
    return built, installed


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
    optional = (NAX_METALLIB_NAME,) if metallib_path(NAX_METALLIB_NAME).exists() else ()
    all_names = (*METALLIB_NAMES, *optional)
    artifacts = (_OUT, *(metallib_path(n) for n in all_names))
    if not any(_stamp_path(a).exists() for a in artifacts):
        return []
    try:
        digests = {_OUT: _input_hash()}
        for name in all_names:
            digests[metallib_path(name)] = _metallib_digest(
                name, _metallib_source(name)
            )
    except (OSError, ImportError, RuntimeError) as exc:
        # The check couldn't run (not a verdict on the artifacts); don't block a
        # loadable install, but leave a breadcrumb instead of failing silently.
        logger.debug("staleness check skipped (%s): %s", type(exc).__name__, exc)
        return []
    return [a for a in artifacts if is_stale(a, digests[a])]


def build() -> Path:
    """Build with MetalExtension and return the native launcher's path."""
    build_metallibs()
    return _OUT


def _mps_input_hash() -> str:
    import torch

    h = hashlib.sha256(str(torch.__version__).encode())
    for path in (_MPS_DIR / "mps_ops.mm", _KERNELS, _DEVICE, _BUILD, _CONSTANTS):
        h.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def mps_artifact_is_stale() -> bool:
    # Wheels have no stamps; source checkouts verify the shared headers too.
    return _stamp_path(_MPS_OUT).exists() and is_stale(_MPS_OUT, _mps_input_hash())


def build_mps() -> Path:
    """Build the MPS launcher into the package using PyTorch's extension builder."""
    import torch
    from setuptools import Distribution
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    digest = _mps_input_hash()
    if all(
        p.exists() for p in (_MPS_OUT, _MPS_VERSION, _stamp_path(_MPS_OUT))
    ) and not is_stale(_MPS_OUT, digest):
        return _MPS_OUT
    extension = CppExtension(
        "_mps_ops",
        sources=[str(_MPS_DIR / "mps_ops.mm")],
        extra_compile_args=[
            "-O3",
            f"-mmacosx-version-min={MIN_MACOS_VERSION}",
            f"-DVLLM_METAL_PARTITION_SIZE={PARTITION_SIZE}",
            f"-DVLLM_METAL_PA_WINDOW_ROWS={PA_WINDOW_ROWS}",
            f"-DVLLM_METAL_PA_WINDOW_MAX_HEAD={PA_WINDOW_MAX_HEAD_SIZE}",
        ],
        extra_link_args=[
            f"-mmacosx-version-min={MIN_MACOS_VERSION}",
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
            "-framework",
            "IOKit",
            "-framework",
            "CoreFoundation",
        ],
    )
    command = BuildExtension(Distribution({"ext_modules": [extension]}))
    # A single host source needs no Ninja dependency.
    command.use_ninja = False
    command.ensure_finalized()
    command.build_lib = str(_MPS_DIR)
    command.force = True
    with tempfile.TemporaryDirectory(prefix="vllm-metal-mps-build-") as tmp:
        command.build_temp = tmp
        command.run()
    _MPS_VERSION.write_text(str(torch.__version__) + "\n")
    _stamp_path(_MPS_OUT).write_text(digest)
    return _MPS_OUT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    libraries = build_metallibs()
    print(output_path())
    for lib in libraries:
        print(lib)
    print(build_mps())
