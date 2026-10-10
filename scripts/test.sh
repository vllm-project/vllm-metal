#!/bin/bash

main() {
  set -eu -o pipefail

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # shellcheck source=lib.sh disable=SC1091
  source "${script_dir}/lib.sh"

  python3 .github/scripts/test_installation.py -v

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

    section "Running tests"
    # Exclude perf/long-running tests by default; run them explicitly via:
    #   pytest -m slow tests/ -v --tb=short
    pytest -m "not slow" tests/ .github/scripts/test_parity.py -v --tb=short
  fi
}

main "$@"
