# Experimental TileLang GDN state updates

`VLLM_METAL_GDN_STATE_SCATTER=tilelang` selects TileLang-generated Metal kernels
for GDN state-cache row writes and zeroing. The default is `native`; installing
vllm-metal does not install TileLang. This is an explicit experiment, not a
replacement for the full GDN recurrence or a demonstrated serving speedup.

## Install the optional dependencies

This path currently depends on the unmerged
[TileLang MLX adapter PR](https://github.com/tile-ai/tilelang/pull/4028).
The tested revision is `2cdbb1e05efb6aee83590403274bedea31813a4b`.
Use Apple Silicon, Python 3.12, macOS 26 or later, and the SDK/toolchain required
by the [vllm-metal source setup](CONTRIBUTING.md). Both native runtimes use
MLX **0.32.4** and nanobind **3.1.0**; do not upgrade MLX independently.

After completing the vllm-metal source setup, activate that same Python
environment and build the pinned TileLang checkout in a separate directory:

```bash
git clone --filter=blob:none https://github.com/tile-ai/tilelang.git tilelang-mlx
cd tilelang-mlx
git fetch origin pull/4028/head
git checkout --detach 2cdbb1e05efb6aee83590403274bedea31813a4b
git submodule update --init --recursive
uv pip install -r requirements.txt cmake cython
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
cmake -S . -B build -DUSE_CUDA=OFF -DUSE_METAL=ON \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build -j 4
uv pip install ./tilelang/contrib/mlx_runtime
export TILELANG_CHECKOUT="$(pwd)"
export PYTHONPATH="$TILELANG_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
python -c 'from tilelang.contrib.mlx import compile; import _tilelang_mlx'
python -m pytest testing/python/metal/test_metal_mlx.py -q
```

Keep this `PYTHONPATH` when starting vLLM and its worker subprocesses. The native
TileLang libraries are discovered in the checkout's `build` directory. Restart
the server after switching the backend; each state cache captures the setting
when constructed.

From the vllm-metal checkout, with a model whose weights and state use FP16:

```bash
VLLM_METAL_GDN_STATE_SCATTER=tilelang \
  vllm serve /path/to/fp16-model
```

For a text-only converted multimodal checkpoint such as Qwen3.5, also set
`VLLM_METAL_MULTIMODAL_MODE=text-only`. The model's configured dtype must match
its actual weights. BF16 state is not supported by this adapter; use an FP16
checkpoint or keep `VLLM_METAL_GDN_STATE_SCATTER=native`. Missing optional
dependencies and unsupported state dtypes raise errors rather than silently
falling back to native.

## Scope and validation

The cache passes physical page strides to the kernel and publishes its returned
state alias through the existing `CacheViews` storage owner. Pending writes,
slot copies and zeroing use the same selection. The destination allocation is
not copied; non-contiguous read-only update rows may be materialized by MLX.
Slot IDs must be valid and distinct, as required by the native cache contract.

Kernels specialize static shape, strides, dtype, row count and zero/write mode;
up to 128 specializations are cached per process. New specializations incur
lowering/compilation overhead. Changing this switch does not replace the GDN
recurrence, attention, or the scheduling policy that can defer state writes.

Run the existing storage tests in the configured environment:

```bash
python -c 'from tilelang.contrib.mlx import compile; import _tilelang_mlx'
python -m pytest tests/attention/test_shared_cache_storage.py \
  tests/attention/test_lazy_kv_allocation.py -q
```

Native cases always run. The four TileLang cases skip only when TileLang or its
optional native extension is absent; import or execution errors in an installed
integration remain failures. Standard CI does not install this unmerged
optional dependency. Run the preflight above and confirm no skips when
validating the TileLang path.

Local testing covered real upstream allocations, FP16/FP32 state, layer offsets,
padding, deferred updates followed by copy/zero, and cross-stream consumers.
Qwen3.5-0.8B FP16 text serving also matched native MLX tokens for eight prompts
at request batch sizes one and two with both scatter backends. These checks do
not qualify all hybrid models, BF16, vision, or speculative decoding.

Repeated HTTP serving comparisons were sensitive to run order and machine
state. Keep the native default; measure your workload using the
[macOS benchmark procedure](benchmarking-macos.md) before making performance
claims. The [TileLang adapter documentation](https://github.com/tile-ai/tilelang/blob/2cdbb1e05efb6aee83590403274bedea31813a4b/examples/metal/README.mlx.md)
describes its storage, stream and version constraints.
