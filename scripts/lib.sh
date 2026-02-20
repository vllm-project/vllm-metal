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
    uv venv "$1" --clear --python 3.12
  fi

  # shellcheck source=/dev/null
  source "$1/bin/activate"

  # Some uv-created environments may not expose pip entrypoints depending on
  # platform/python build; ensure pip module and scripts are present.
  if ! "$1/bin/python" -m pip --version >/dev/null 2>&1; then
    section "Bootstrapping pip in virtual environment"
    if ! "$1/bin/python" -m ensurepip --upgrade >/dev/null 2>&1; then
      error "Failed to bootstrap pip in virtual environment: $1"
      return 1
    fi
  fi

  # Ensure both pip and pip3 are available as scripts in the venv bin dir.
  if [ ! -x "$1/bin/pip" ]; then
    cat > "$1/bin/pip" <<'EOF'
#!/usr/bin/env sh
exec "$(dirname "$0")/python" -m pip "$@"
EOF
    chmod +x "$1/bin/pip"
  fi

  if [ ! -x "$1/bin/pip3" ]; then
    cat > "$1/bin/pip3" <<'EOF'
#!/usr/bin/env sh
exec "$(dirname "$0")/python" -m pip "$@"
EOF
    chmod +x "$1/bin/pip3"
  fi
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
