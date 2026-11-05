#!/bin/bash

main() {
  set -eu -o pipefail

  local run_suffix="${VLLM_METAL_TEST_SUFFIX:-1}"
  case "$run_suffix" in
    0|1) ;;
    *)
      echo "VLLM_METAL_TEST_SUFFIX must be 0 or 1." >&2
      exit 2
      ;;
  esac

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # shellcheck source=lib.sh disable=SC1091
  source "${script_dir}/lib.sh"

  setup_dev_env

  if [ "$(uname)" == "Darwin" ]; then
    # Export the deployment target in this shell for the wheel build below.
    ensure_metal_toolchain
    # install.sh builds the native artifacts before the wheel check below.
    ./install.sh
    # shellcheck source=/dev/null
    source .venv-vllm-metal/bin/activate

    # Shift-left the release wheel guard. The pytest run below imports the source
    # tree, which shadows the installed wheel, so a package-data regression
    # (artifacts silently dropped from the wheel) would pass CI here and only
    # surface when release.sh runs on main. Build the wheel and assert it bundles
    # the prebuilt artifacts now — the same check release.sh runs before publish.
    section "Building wheel"
    wheel_build_dir=$(mktemp -d)
    trap 'rm -rf -- "$wheel_build_dir"' EXIT
    uv build --out-dir "$wheel_build_dir"
    local wheels=("$wheel_build_dir"/*.whl)
    if [ "${#wheels[@]}" -ne 1 ] || [ ! -f "${wheels[0]}" ]; then
      error "Expected exactly one wheel in ${wheel_build_dir} after uv build."
      exit 1
    fi
    verify_wheel_artifacts "${wheels[0]}"

    section "Verifying package import"
    python -c "import vllm_metal; print('vllm_metal imported successfully')"

    # Catch platform-resolution regressions without loading a model (#471).
    section "Checking Metal platform resolution"
    python -c "import sys; from vllm.platforms import current_platform; name = type(current_platform).__name__; print('Resolved platform:', name); sys.exit(0 if name == 'MetalPlatform' else 1)"

    local test_markers="not slow and not suffix"
    if [ "$run_suffix" = "1" ]; then
      section "Installing the optional Arctic suffix backend"
      install_suffix_backend "${VLLM_METAL_ARCTIC_WHEEL_DIR:-$VIRTUAL_ENV/arctic-wheels}"
      # An enabled backend must fail instead of silently skipping real-tree tests.
      python -c "from arctic_inference.suffix_decoding import SuffixDecodingCache; SuffixDecodingCache(max_tree_depth=24)"
      test_markers="not slow"
    fi

    section "Running tests"
    # Exclude perf/long-running tests by default; run them explicitly via:
    #   pytest -m slow tests/ -v --tb=short
    # VLLM_METAL_TEST_SUFFIX=0 skips the optional real-Arctic tests.
    pytest -m "$test_markers" tests/ .github/scripts/test_parity.py -v --tb=short \
      --durations=25 --durations-min=0.1
  fi
}

main "$@"
