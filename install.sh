#!/bin/bash

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
    data = json.loads('''$release_data''')
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
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp_dir'" EXIT

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

install_vllm_rs() {
  section "Installing vllm-rs (experimental Rust frontend)"

  if ! command -v cargo &> /dev/null || ! command -v rustup &> /dev/null; then
    error "cargo/rustup not found on PATH; install Rust from https://rustup.rs first."
    exit 1
  fi

  local repo_url="https://github.com/inferact/vllm-frontend-rs"
  local tmp_dir
  tmp_dir=$(mktemp -d)
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp_dir'" RETURN

  echo "Cloning $repo_url ..."
  if ! git clone --depth 1 "$repo_url" "$tmp_dir/vllm-frontend-rs"; then
    error "Failed to clone vllm-frontend-rs."
    exit 1
  fi

  # Run cargo from inside the tree so rustup honors rust-toolchain.toml;
  # `cargo install --git` would ignore it and use the system default toolchain.
  if ! ( cd "$tmp_dir/vllm-frontend-rs" && cargo install --path src/cmd --bin vllm-rs ); then
    error "Failed to install vllm-rs."
    exit 1
  fi

  success "Installed vllm-rs to ~/.cargo/bin"
}

main() {
  set -eu -o pipefail

  local repo_owner="vllm-project"
  local repo_name="vllm-metal"
  local package_name="vllm-metal"
  local with_vllm_rs=0

  for arg in "$@"; do
    case "$arg" in
      --with-vllm-rs)
        with_vllm_rs=1
        ;;
      -h|--help)
        cat <<'EOF'
Usage: install.sh [--with-vllm-rs]

Options:
  --with-vllm-rs    Also install vllm-rs (experimental Rust frontend) via
                    cargo from https://github.com/inferact/vllm-frontend-rs.
                    Requires the Rust toolchain on PATH (https://rustup.rs).
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

  is_apple_silicon
  if ! ensure_uv; then
    exit 1
  fi

  local venv="$HOME/.venv-vllm-metal"
  if [[ -n "$local_lib" && -f "$local_lib" ]]; then
    venv="$PWD/.venv-vllm-metal"
  fi

  ensure_venv "$venv"

  local vllm_v="0.20.0"
  local url_base="https://github.com/vllm-project/vllm/releases/download"
  local filename="vllm-$vllm_v.tar.gz"
  curl -OL $url_base/v$vllm_v/$filename
  tar xf $filename
  cd vllm-$vllm_v

  uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
  CXXFLAGS="-Wno-parentheses" uv pip install .
  cd -
  rm -rf vllm-$vllm_v*

  if [[ -n "$local_lib" && -f "$local_lib" ]]; then
    uv pip install .
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

  if [[ "$with_vllm_rs" == "1" ]]; then
    install_vllm_rs
  fi

  echo ""
  success "Installation complete!"
  echo ""
  echo "To use vllm, activate the virtual environment:"
  echo "  source $venv/bin/activate"
  echo ""
  echo "Or add the venv to your PATH:"
  echo "  export PATH=\"$venv/bin:\$PATH\""

  if [[ "$with_vllm_rs" == "1" ]]; then
    echo ""
    echo "vllm-rs is installed to ~/.cargo/bin. Make sure that directory is on your PATH."
    echo "Activate the venv, then run: vllm-rs serve <MODEL>"
  fi
}

main "$@"
