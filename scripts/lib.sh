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

validate_vllm_release_tag() {
  local release_tag="$1"
  if [[ ! "$release_tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-].*)?$ ]]; then
    error "Invalid vLLM release tag: ${release_tag}"
    return 1
  fi
}

# Read the upstream vLLM release paired with this source revision.
read_vllm_release_tag() {
  local path="${1:-.github/vllm-release-tag.commit}"
  local release_tag

  if [ ! -r "$path" ]; then
    error "Missing vLLM release tag file: ${path}"
    return 1
  fi

  release_tag=$(tr -d '[:space:]' < "$path")
  validate_vllm_release_tag "$release_tag" || return 1
  printf '%s\n' "$release_tag"
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
  # Keep native artifacts aligned with the wheel's macOS 15 floor.
  export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"

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

# Build the in-package native artifacts (the _paged_ops*.so and the required
# precompiled .metallib shader libraries, including NAX) into vllm_metal/metal/
# so `uv build` can bundle them as package data.
#
# `python` here is the venv interpreter activated by setup_dev_env, so mlx and
# nanobind are importable.
build_native_artifacts() {
  section "Building native Metal artifacts"
  # Official wheels require NAX, so reject an older SDK before compiling.
  if ! python -c \
    "from vllm_metal.metal.build import require_nax_sdk; require_nax_sdk()"; then
    return 1
  fi
  python -m vllm_metal.metal.build
}

# Build the optional Rust/ggml engine (VLLM_METAL_BACKEND=ggml) into
# vllm_metal/ggml/. Best effort: skipped when cargo or Homebrew ggml is absent,
# since the default MLX backend does not need it.
build_ggml_engine() {
  section "Building ggml engine (optional)"
  if ! command -v cargo &> /dev/null; then
    echo "cargo not found; skipping (install Rust to enable VLLM_METAL_BACKEND=ggml)"
    return 0
  fi
  if ! command -v brew &> /dev/null || [ ! -f "$(brew --prefix ggml)/include/ggml.h" ]; then
    echo "ggml not found; skipping (run 'brew install ggml' to enable VLLM_METAL_BACKEND=ggml)"
    return 0
  fi
  python -m vllm_metal.ggml.build
}

# Fail unless the freshly built wheel actually bundles the prebuilt native
# artifacts: the _paged_ops*.so extension, three required metallibs, and NAX.
# setup.py's package data is what pulls these (gitignored)
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
from vllm_metal.metal.build import METALLIB_NAMES, NAX_METALLIB_NAME, metallib_path, output_path
print(output_path().name)
for _name in (*METALLIB_NAMES, NAX_METALLIB_NAME):
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
      error "Check the native package data in setup.py."
      return 1
    fi
  done <<< "$expected"

  local expected_minos paged_ops_name unpack_dir native_so native_name actual_minos native_count
  expected_minos=$(python -c "from vllm_metal.metal.build import MIN_MACOS_VERSION; print(MIN_MACOS_VERSION)")
  paged_ops_name=$(python -c "from vllm_metal.metal.build import output_path; print(output_path().name)")
  unpack_dir=$(mktemp -d)
  unzip -qq "${wheel}" '*.so' -d "${unpack_dir}"
  native_count=0
  while IFS= read -r native_so; do
    native_name=$(basename "${native_so}")
    case "${native_name}" in
      "${paged_ops_name}") ;;
      *) continue ;;
    esac
    native_count=$((native_count + 1))
    actual_minos=$(otool -l "${native_so}" | awk '$1 == "minos" { print $2; exit }')
    if [ "${actual_minos}" != "${expected_minos}" ]; then
      error "${native_so} targets macOS ${actual_minos:-unknown}; expected ${expected_minos}."
      rm -rf "${unpack_dir}"
      return 1
    fi
    success "${native_name}: macOS ${actual_minos}"
  done < <(find "${unpack_dir}" -type f -name '*.so')
  rm -rf "${unpack_dir}"
  if [ "${native_count}" -lt 1 ]; then
    error "Wheel ${wheel} is missing a required native extension."
    return 1
  fi

  success "Wheel bundles all native artifacts"
}
