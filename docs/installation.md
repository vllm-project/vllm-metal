# Installation

## Requirements

- macOS on Apple Silicon
- Native arm64 Python 3.12. Rosetta/x86_64 Python is not supported.

> **Nothing is compiled on your machine.** vLLM core and the vllm-metal plugin
> both install as prebuilt wheels, and the plugin wheel bundles the native
> paged-attention extension (`_paged_ops*.so`) and its Metal shader libraries
> (`*.metallib`) — so no C++ compiler and no Metal toolchain are needed, and
> nothing is built on the first request. Xcode Command Line Tools are only
> needed to *edit* the kernels; see [Contributing](CONTRIBUTING.md).

`uv` is bootstrapped automatically.

Verify the Python architecture before installing:

```bash
python3 -c "import platform; print(platform.machine())"
file "$(which python3)"
```

The first command should print `arm64`. If it prints `x86_64`, switch to a native arm64 Python and remove `~/.venv-vllm-metal` before reinstalling.

## Install

Using the install script, the following will be installed under the `~/.venv-vllm-metal` directory (the default).
- vllm-metal plugin
- vllm core
- Related libraries

If you run `source ~/.venv-vllm-metal/bin/activate`, the `vllm` CLI becomes available and you can access the vLLM right away.

For how to use the `vllm` CLI, please refer to the [official vLLM guide](https://docs.vllm.ai/en/latest/cli/).

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

## Reinstallation and Update

If any issues occur, please use the following command to switch to the latest release version and check if the problem is resolved.
If the issue continues to occur in the latest release, please report the details of the issue.
(If you have installed it in a directory other than the default `~/.venv-vllm-metal`, substitute that path and run the command accordingly.)

```bash
rm -rf ~/.venv-vllm-metal && curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

## Uninstall

Please delete the directory that was installed by the installation script.
(If you have installed it in a directory other than the default `~/.venv-vllm-metal`, substitute that path and run the command accordingly.)

```bash
rm -rf ~/.venv-vllm-metal
```

## Building Documentation

```bash
uv pip install -r docs/requirements-docs.txt
mkdocs serve
```
