#!/bin/bash

_cleanup_dirs=()

register_cleanup_dir() {
  _cleanup_dirs+=("$1")
}

cleanup_tmp_dirs() {
  local dir
  if [[ ${#_cleanup_dirs[@]} -eq 0 ]]; then
    return
  fi
  for dir in "${_cleanup_dirs[@]}"; do
    rm -rf "$dir"
  done
}

fetch_latest_release() {
  local repo_owner="$1"
  local repo_name="$2"

  echo "Fetching latest release..." >&2

  local latest_release_url="https://api.github.com/repos/${repo_owner}/${repo_name}/releases/latest"
  local release_data

  if ! release_data=$(curl -fsSL "$latest_release_url" 2>&1); then
    error "Failed to fetch release information."
    echo "Please check your internet connection and try again." >&2
    exit 1
  fi

  if [[ -z "$release_data" ]] || [[ "$release_data" == *"Not Found"* ]]; then
    error "No releases found for this repository."
    echo "Please visit https://github.com/${repo_owner}/${repo_name}/releases" >&2
    exit 1
  fi

  echo "$release_data"
}

extract_wheel_url() {
  local release_data="$1"

  python3 -c "
import sys
import json
try:
    data = json.loads('''$release_data''', strict=False)
    assets = data.get('assets', [])
    for asset in assets:
        name = asset.get('name', '')
        if name.endswith('.whl'):
            print(asset.get('browser_download_url', ''))
            break
except Exception as e:
    print('', file=sys.stderr)
"
}

download_and_install_wheel() {
  local wheel_url="$1"
  local package_name="$2"

  local wheel_name
  wheel_name=$(basename "$wheel_url")
  echo "Latest release: $wheel_name"
  success "Found latest release"

  local tmp_dir
  tmp_dir=$(mktemp -d)
  register_cleanup_dir "$tmp_dir"

  echo ""
  echo "Downloading wheel..."
  local wheel_path="$tmp_dir/$wheel_name"

  if ! curl -fsSL "$wheel_url" -o "$wheel_path"; then
    error "Failed to download wheel."
    exit 1
  fi

  success "Downloaded wheel"

  # Install vllm-metal package
  if ! uv pip install "$wheel_path"; then
    error "Failed to install ${package_name}."
    exit 1
  fi

  success "Installed ${package_name}"
}

main() {
  set -eu -o pipefail
  trap cleanup_tmp_dirs EXIT

  local repo_owner="vllm-project"
  local repo_name="vllm-metal"
  local package_name="vllm-metal"

  for arg in "$@"; do
    case "$arg" in
      -h|--help)
        cat <<'EOF'
Usage: install.sh

Options:
  -h, --help        Show this help.
EOF
        exit 0
        ;;
      *)
        echo "Unknown argument: $arg" >&2
        echo "Run with --help for usage." >&2
        exit 1
        ;;
    esac
  done

  # Source shared library functions
  # Try local lib.sh first (when running ./install.sh), fall back to remote (when piped from curl)
  local local_lib=""
  if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-}")" && pwd)"
    local_lib="$script_dir/scripts/lib.sh"
  fi

  if [[ -n "$local_lib" && -f "$local_lib" ]]; then
    # shellcheck source=/dev/null
    source "$local_lib"
  else
    # Fetch from remote (curl | bash case)
    local lib_url="https://raw.githubusercontent.com/$repo_owner/$repo_name/main/scripts/lib.sh"
    local lib_tmp
    lib_tmp=$(mktemp)
    if ! curl -fsSL "$lib_url" -o "$lib_tmp"; then
      echo "Error: Failed to fetch lib.sh from $lib_url" >&2
      rm -f "$lib_tmp"
      exit 1
    fi
    # shellcheck source=/dev/null
    source "$lib_tmp"
    rm -f "$lib_tmp"
  fi

  if ! is_apple_silicon; then
    error "vllm-metal requires Apple Silicon arm64. Detected: $(uname -m)."
    exit 1
  fi

  if ! ensure_uv; then
    exit 1
  fi

  local venv="$HOME/.venv-vllm-metal"
  if [[ -n "$local_lib" && -f "$local_lib" ]]; then
    venv="$PWD/.venv-vllm-metal"
  fi

  ensure_venv "$venv"
  if ! require_arm64_python python; then
    exit 1
  fi

  local vllm_v="0.24.0"
  local url_base="https://github.com/vllm-project/vllm/releases/download"
  local filename="vllm-$vllm_v.tar.gz"
  local vllm_tmp_dir
  vllm_tmp_dir=$(mktemp -d)
  register_cleanup_dir "$vllm_tmp_dir"
  local vllm_src_dir="$vllm_tmp_dir/vllm-$vllm_v"

  curl -fSL "$url_base/v$vllm_v/$filename" -o "$vllm_tmp_dir/$filename"
  tar xf "$vllm_tmp_dir/$filename" -C "$vllm_tmp_dir"
  cd "$vllm_src_dir"

  uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
  CXXFLAGS="-Wno-parentheses" uv pip install .
  cd -

  if [[ -n "$local_lib" && -f "$local_lib" ]]; then
    # Local source install (running ./install.sh from a checkout). Prebuild the
    # native paged-attention artifacts from this tree — the _paged_ops .so and
    # the precompiled .metallib shaders — so the kernels load with no runtime
    # compile, exactly like a release wheel; otherwise get_ops() fails loud
    # ("Prebuilt native extension not found") the first time paged attention is
    # used. build_native_artifacts needs the build deps (mlx, nanobind)
    # importable, so the editable install pulls them in first and points the
    # install at this tree, where the artifacts land. The remote (curl | bash)
    # branch below installs a prebuilt release wheel instead and needs no
    # toolchain. Mirrors scripts/release.sh / scripts/test.sh.
    uv pip install -e .
    ensure_metal_toolchain
    build_native_artifacts
  else
    local release_data
    release_data=$(fetch_latest_release "$repo_owner" "$repo_name")

    local wheel_url
    wheel_url=$(extract_wheel_url "$release_data")

    if [[ -z "$wheel_url" ]]; then
      error "No wheel file found in the latest release."
      exit 1
    fi

    download_and_install_wheel "$wheel_url" "$package_name"
  fi

  echo ""
  success "Installation complete!"
  echo ""
  echo "To use vllm, activate the virtual environment:"
  echo "  source $venv/bin/activate"
  echo ""
  echo "Or add the venv to your PATH:"
  echo "  export PATH=\"$venv/bin:\$PATH\""
}

main "$@"
