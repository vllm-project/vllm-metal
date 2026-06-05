#!/bin/bash
# Common library functions for vllm-metal scripts

# Print an error message
error() {
  echo -e "Error: $*" >&2
}

# Print a success message
success() {
  echo -e "✓ $*"
}

# Print a section header
section() {
  echo "=== $* ==="
}

# Check if running on Apple Silicon
is_apple_silicon() {
  [ "$(uname -m)" = "arm64" ]
}

# Require a native arm64 Python interpreter.
require_arm64_python() {
  local python_bin="${1:-python}"
  local machine

  if ! machine=$("$python_bin" -c "import platform; print(platform.machine())"); then
    error "Failed to inspect Python architecture using ${python_bin}."
    return 1
  fi

  if [ "$machine" != "arm64" ]; then
    error "vllm-metal requires native arm64 Python, got ${machine}. Remove the venv and rerun install.sh from an arm64 Python."
    return 1
  fi
}

# Ensure uv is installed
ensure_uv() {
  if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    if ! curl -LsSf "https://astral.sh/uv/0.9.18/install.sh" | sh; then
      error "Failed to install uv"
      return 1
    fi

    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

# Ensure virtual environment exists and is activated
ensure_venv() {
  if [ ! -d "$1" ]; then
    section "Creating virtual environment"
    uv venv "$1" --clear --python 3.12 --seed
  fi

  # shellcheck source=/dev/null
  source "$1/bin/activate"
}

# Install dev dependencies
install_dev_deps() {
  section "Installing dependencies"
  uv pip install -e ".[dev]"
}

# Full development environment setup
setup_dev_env() {
  ensure_uv
  ensure_venv ".venv-vllm-metal"
  install_dev_deps
}

# Get version from pyproject.toml
get_version() {
  uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
}

# Ensure `xcrun metal` can actually compile a .metallib.
#
# Producing a .metallib needs the Metal toolchain. On Xcode 26+ it is a
# separate downloadable component; older Xcode bundles it. Rather than guess,
# trial-compile a trivial shader: if that succeeds the toolchain is already
# present (no slow download), otherwise download MetalToolchain and re-check.
ensure_metal_toolchain() {
  section "Ensuring Metal toolchain"

  local tmpdir metal_src metal_lib
  tmpdir=$(mktemp -d)
  metal_src="${tmpdir}/probe.metal"
  metal_lib="${tmpdir}/probe.metallib"
  printf '[[kernel]] void _t() {}\n' > "${metal_src}"

  if xcrun -sdk macosx metal -o "${metal_lib}" "${metal_src}" &> /dev/null; then
    success "Metal toolchain present"
    rm -rf "${tmpdir}"
    return 0
  fi

  echo "Metal toolchain not available, downloading via xcodebuild..."
  xcodebuild -downloadComponent MetalToolchain

  if ! xcrun -sdk macosx metal -o "${metal_lib}" "${metal_src}" &> /dev/null; then
    error "Metal toolchain still unavailable after download; cannot compile .metallib."
    rm -rf "${tmpdir}"
    return 1
  fi

  success "Metal toolchain ready"
  rm -rf "${tmpdir}"
}

# Build the in-package native artifacts (the _paged_ops*.so and the three
# precompiled .metallib shader libraries) into vllm_metal/metal/ so that
# `uv build` can bundle them into the wheel via the maturin `include` directive.
#
# `python` here is the venv interpreter activated by setup_dev_env, so mlx and
# nanobind are importable.
build_native_artifacts() {
  section "Building native Metal artifacts"
  python -m vllm_metal.metal.build
}

# Fail unless the freshly built wheel actually bundles the prebuilt native
# artifacts: the _paged_ops*.so extension and the three .metallib shader
# libraries. maturin's `include` directive is what pulls these (gitignored)
# files in; if that ever regresses, the wheel would install fine but fail at
# first run with "Prebuilt native extension not found". The expected filenames
# are read from build.py so this guard never drifts from the runtime loader.
#
# Usage: verify_wheel_artifacts <path-to-wheel>
verify_wheel_artifacts() {
  local wheel="$1"
  section "Verifying wheel bundles native artifacts"

  local expected
  if ! expected=$(python -c "
from vllm_metal.metal.build import METALLIB_NAMES, metallib_path, output_path
print(output_path().name)
for _name in METALLIB_NAMES:
    print(metallib_path(_name).name)
"); then
    error "Failed to resolve expected native artifact names from build.py."
    return 1
  fi

  local contents name
  contents=$(unzip -l "$wheel")
  while IFS= read -r name; do
    [ -z "$name" ] && continue
    if grep -qF "$name" <<< "$contents"; then
      success "bundled: ${name}"
    else
      error "Wheel ${wheel} is missing native artifact: ${name}"
      error "maturin [tool.maturin] 'include' likely failed to bundle it."
      return 1
    fi
  done <<< "$expected"

  success "Wheel bundles all native artifacts"
}
