# SPDX-License-Identifier: Apache-2.0
"""JIT build script for the native paged-attention Metal extension.

Builds paged_ops.rs + mlx_ffi.cpp via cargo into a shared library that
dispatches Metal shaders through MLX's own command encoder.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sysconfig
from pathlib import Path

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
_CACHE_DIR = Path.home() / ".cache" / "vllm-metal"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_OUT = _CACHE_DIR / f"_paged_ops{_EXT_SUFFIX}"

_RUST_SRC = _THIS_DIR / "paged_ops.rs"
_FFI_SRC = _THIS_DIR / "mlx_ffi.cpp"
_FFI_HDR = _THIS_DIR / "mlx_ffi.h"
_CRATE_DIR = _THIS_DIR / "paged_ops_crate"


def needs_rebuild() -> bool:
    """Return True if the .so is missing or older than any source file."""
    if not _OUT.exists():
        return True
    out_mtime = _OUT.stat().st_mtime
    for src in [_RUST_SRC, _FFI_SRC, _FFI_HDR, _CRATE_DIR / "Cargo.toml",
                _CRATE_DIR / "build.rs"]:
        if src.exists() and out_mtime < src.stat().st_mtime:
            return True
    return False


def build() -> Path:
    """JIT-build the native extension, returning the path to the .so."""
    if not needs_rebuild():
        return _OUT

    logger.info("Building Rust paged-attention extension ...")

    cmd = [
        "cargo", "build", "--release",
        "--manifest-path", str(_CRATE_DIR / "Cargo.toml"),
    ]

    logger.info("  %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to build Rust paged_ops extension:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    # Locate the built dylib in target/release/
    target_dir = _CRATE_DIR / "target" / "release"
    candidates = list(target_dir.glob("lib_paged_ops.*")) + \
                 list(target_dir.glob("_paged_ops.*"))
    so_candidates = [
        p for p in candidates
        if p.suffix in (".dylib", ".so") or _EXT_SUFFIX in p.name
    ]

    if not so_candidates:
        raise FileNotFoundError(
            f"Cargo build succeeded but no shared library found in {target_dir}.\n"
            f"Files: {list(target_dir.iterdir())}"
        )

    built = so_candidates[0]
    shutil.copy2(str(built), str(_OUT))
    logger.info("Built %s (from %s)", _OUT, built)
    return _OUT
