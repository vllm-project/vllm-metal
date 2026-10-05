# Contributing to vLLM Metal

To run a released build, use the [installation guide](installation.md).
The setup below is for editing vllm-metal itself.

For the experimental backend, follow the [PyTorch MPS decision contract](pytorch-mps.md).

## Development setup

On an Apple Silicon Mac, install full
[Xcode](https://developer.apple.com/xcode/) with macOS SDK 26.2 or newer.
Select Xcode as the active developer directory. Initial source setup builds
native Metal artifacts, including for Python-only contributions.

Fork the repository on GitHub, then clone your fork (replace `YOUR_USERNAME`):

```bash
git clone https://github.com/YOUR_USERNAME/vllm-metal.git
cd vllm-metal
git remote add upstream https://github.com/vllm-project/vllm-metal.git
git switch -c my-change
./install.sh
source .venv-vllm-metal/bin/activate
uv pip install -e ".[dev]"
```

`./install.sh` creates the local environment, installs the matching vLLM core
and editable plugin, and builds the native artifacts. It downloads the Metal
toolchain if needed. Restart the server after editing Python files.

## Editing the Metal kernels

When changing `.metal` shaders or `paged_ops.cpp`, enable source builds:

```bash
VLLM_METAL_BUILD_FROM_SOURCE=1 vllm serve <model>
```

Restart the process after each edit. This mode rebuilds the selected backend's native
extension and shaders. To refresh both launchers and the shared prebuilt artifacts, run
`python -m vllm_metal.metal.build`. Stale local artifacts are rejected when
source mode is disabled.

## Checks

Run from the repository root:

```bash
scripts/lint.sh
scripts/test.sh
```

For a shorter loop, run `pytest -m "not slow" tests/` in the activated environment.
Model parity runs separately through [scheduled and requested CI](tools.md#scheduled-and-requested-ci).

## Pull requests

- **Model changes:** run the [greedy parity tool](tools.md) against the environment's native `mlx-lm`. Report `EXACT` and `TOP_K_MATCH` counts separately and investigate failures.
- **Performance claims:** include before/after [serving benchmark](https://docs.vllm.ai/en/latest/cli/bench/serve/) results, following the [macOS benchmarking guide](benchmarking-macos.md) for workload, machine-state and dispatch controls.

Sign off each commit to certify agreement with the [Developer Certificate of
Origin](https://developercertificate.org/), then push to your fork:

```bash
git commit -s -m "Describe your change"
git push -u origin my-change
```

Open a pull request against `main` in `vllm-project/vllm-metal`.

## Building documentation

```bash
uv pip install -r docs/requirements-docs.txt
mkdocs serve
```
