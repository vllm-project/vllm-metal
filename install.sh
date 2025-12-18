#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# vLLM Metal Plugin Installation Script
#
# This script installs the vLLM Metal plugin for Apple Silicon Macs.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    error "This script is only for macOS"
fi

# Check if running on Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    error "This script requires Apple Silicon (M1/M2/M3/M4)"
fi

echo ""
info "=========================================="
info "   vLLM Metal Plugin Installer"
info "=========================================="
echo ""
info "System: $(uname -m) macOS $(sw_vers -productVersion)"
info "Chip: $(sysctl -n machdep.cpu.brand_string)"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)

info "Python version: $PYTHON_VERSION"

# Warn about Python 3.14
if [[ "$PYTHON_MINOR" -ge 14 ]]; then
    warn "Python 3.14 is very new - some packages may not have wheels yet."
    warn "If you encounter build issues, try Python 3.12 or 3.13."
fi

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 11 ]]; then
    error "Python 3.11+ is required. Current version: $PYTHON_VERSION"
fi

# Determine package manager
if command -v uv &> /dev/null; then
    PIP="uv pip"
    info "Using uv for package management"
else
    PIP="pip3"
    info "Using pip for package management"
fi

# Create virtual environment if not in one
if [[ -z "$VIRTUAL_ENV" ]]; then
    warn "Not in a virtual environment."

    if [[ -d ".venv" ]]; then
        info "Found existing .venv, activating..."
        source .venv/bin/activate
    else
        info "Creating virtual environment..."
        python3 -m venv .venv
        source .venv/bin/activate
        info "Virtual environment activated"
    fi
fi

echo ""
step "1/4 Installing MLX framework..."
$PIP install mlx mlx-lm

echo ""
step "2/4 Installing vLLM dependencies..."
# Install core dependencies that vLLM needs (pure Python or with macOS wheels)
$PIP install \
    torch \
    transformers \
    accelerate \
    safetensors \
    numpy \
    psutil \
    pydantic \
    cbor2 \
    msgspec \
    cloudpickle \
    prometheus-client \
    fastapi \
    uvicorn \
    uvloop \
    pillow \
    tiktoken \
    typing_extensions \
    filelock \
    py-cpuinfo \
    aiohttp \
    openai \
    einops \
    importlib_metadata \
    mistral_common \
    pyyaml \
    requests \
    tqdm \
    sentencepiece \
    gguf \
    blake3 \
    pyzmq \
    regex \
    protobuf \
    setuptools \
    depyf \
    tokenizers \
    cachetools \
    partial-json-parser \
    compressed-tensors

# Try to install optional constrained decoding packages (may fail on Python 3.14)
set +e
info "Installing optional constrained decoding packages..."
if $PIP install outlines lm-format-enforcer xgrammar 2>/dev/null; then
    info "Constrained decoding packages installed"
else
    warn "Some constrained decoding packages (outlines, xgrammar) failed to install"
    warn "This is expected on Python 3.14 - these packages are optional"
fi
set -e

echo ""
step "3/4 Installing vLLM..."
# vLLM is tricky on macOS - it needs to be built from source
# and requires cmake/ninja. We'll try multiple approaches.

VLLM_INSTALLED=false

# First, check if cmake is available (needed for source builds)
if ! command -v cmake &> /dev/null; then
    warn "cmake not found - needed to build vLLM"
    if command -v brew &> /dev/null; then
        info "Installing cmake and ninja via Homebrew..."
        brew install cmake ninja || true
    fi
fi

# Try to install vLLM
set +e  # Don't exit on error for this section
if $PIP install --only-binary=:all: vllm 2>/dev/null; then
    VLLM_INSTALLED=true
    info "vLLM installed from wheel"
else
    warn "No pre-built vLLM wheel for Python $PYTHON_VERSION on macOS arm64"

    if command -v cmake &> /dev/null; then
        info "Building vLLM from source (this may take 5-10 minutes)..."
        VLLM_TARGET_DEVICE=cpu MAX_JOBS=4 $PIP install vllm 2>&1 | tail -20
        # Check if vLLM is actually installed
        if python3 -c "import vllm" 2>/dev/null; then
            VLLM_INSTALLED=true
            info "vLLM built and installed successfully"
        fi
    fi
fi

if [ "$VLLM_INSTALLED" = false ]; then
    warn "=========================================="
    warn "vLLM installation failed."
    warn ""
    warn "Options:"
    warn "  1. Use Python 3.12 which has vLLM wheels"
    warn "  2. Install cmake: brew install cmake ninja"
    warn "  3. Continue without vLLM (plugin development only)"
    warn "=========================================="
fi
set -e  # Re-enable exit on error

echo ""
step "4/4 Installing vLLM Metal plugin..."
$PIP install -e .

echo ""
info "=========================================="
info "   Verifying Installation"
info "=========================================="
echo ""

# Verify MLX
python3 -c "import mlx.core as mx; print(f'  MLX device: {mx.default_device()}')" || warn "MLX import failed"

# Verify vllm_metal
python3 -c "import vllm_metal; print(f'  vllm_metal version: {vllm_metal.__version__}')" || warn "vllm_metal import failed"

# Verify chip detection
python3 -c "from vllm_metal.utils import get_apple_chip_name; print(f'  Chip detected: {get_apple_chip_name()}')" || true

# Try to verify vLLM
python3 -c "import vllm; print(f'  vLLM version: {vllm.__version__}')" 2>/dev/null || warn "vLLM not fully installed"

echo ""
info "=========================================="
info "   Installation Complete!"
info "=========================================="
echo ""
info "Next steps:"
info "  1. Activate the environment: source .venv/bin/activate"
info "  2. Test with: python -c 'import vllm_metal; print(vllm_metal.register())'"
info ""
info "For development:"
info "  pip install -e '.[dev]'"
info "  pytest tests/ -v"
echo ""
