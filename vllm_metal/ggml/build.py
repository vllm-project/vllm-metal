# SPDX-License-Identifier: Apache-2.0
"""Build the Rust/ggml engine extension (``_ggml_engine``).

The crate lives in ``rust/ggml-engine`` and links the system ggml
(``brew install ggml``). The resulting cdylib is copied into this package as
``_ggml_engine{EXT_SUFFIX}`` so it can be bundled into the wheel.

Usage: ``python -m vllm_metal.ggml.build``
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
CRATE_DIR = _REPO_ROOT / "rust" / "ggml-engine"
_EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def output_path() -> Path:
    """Where the built extension is placed (whether or not it exists)."""
    return _THIS_DIR / f"_ggml_engine{_EXT_SUFFIX}"


def build(release: bool = True) -> Path:
    """Compile the crate with cargo and install the extension in-package."""
    if not (CRATE_DIR / "Cargo.toml").exists():
        raise RuntimeError(
            f"ggml engine sources not found at {CRATE_DIR}; "
            "building requires a source checkout of vllm-metal"
        )
    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError("cargo not found; install Rust (e.g. `brew install rust`)")
    env = dict(os.environ)
    env.setdefault("PYO3_PYTHON", sys.executable)
    env.setdefault("MACOSX_DEPLOYMENT_TARGET", "15.0")
    profile = "release" if release else "debug"
    cmd = [cargo, "build", "--manifest-path", str(CRATE_DIR / "Cargo.toml")]
    if release:
        cmd.append("--release")
    logger.info("Building ggml engine: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=CRATE_DIR)
    built = CRATE_DIR / "target" / profile / "lib_ggml_engine.dylib"
    if not built.exists():
        raise RuntimeError(f"cargo did not produce {built}")
    out = output_path()
    tmp = out.with_suffix(out.suffix + ".tmp")
    shutil.copy2(built, tmp)
    os.replace(tmp, out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(build())
