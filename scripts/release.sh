#!/bin/bash

main() {
  set -eu -o pipefail

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # shellcheck source=lib.sh disable=SC1091
  source "${script_dir}/lib.sh"

  setup_dev_env

  local base_version version
  base_version=$(get_version)
  # Per-commit dev version, e.g. 0.3.0.dev20260603184500 — unique and PEP 440.
  version="${base_version}.dev$(date -u +%Y%m%d%H%M%S)"
  echo "Building version: $version"

  # Stamp the build version into pyproject.toml so the wheel carries it. maturin
  # reads [project].version and does not consult Cargo.toml when version is not
  # dynamic. The runner checkout is ephemeral, so the committed file is untouched.
  sed -i '' -E "s/^version = .*/version = \"${version}\"/" pyproject.toml

  section "Building wheel"
  uv build

  local tag
  tag="v${version}"
  echo "Generated tag: $tag"

  section "Creating GitHub release"
  gh release create "$tag" \
    --title "$tag" \
    --generate-notes \
    dist/*.whl
}

main "$@"
