# TileLang GDN state scatter candidate

Backend policy lives in `vllm_metal/metal/state_scatter.py`. The cache caller
uses one scatter interface for writes, pending flushes, slot copies and zeroing.
There is no user-facing backend switch. Native remains the default based on
the measurements below; installing vllm-metal does not install TileLang.
Tests can substitute the private kernel implementation, and the benchmark
calls each candidate directly without changing serving dispatch.

The motivation is to express the operation in maintainable kernel code and
make algorithm experiments easier. This candidate does not replace the GDN
recurrence or establish a serving speedup.

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

## Code comparison and scope

| Concern | Existing Metal implementation | TileLang candidate |
| --- | --- | --- |
| Row addressing | Manual flattened-index decoding and physical-stride arithmetic | Static `T.StridedTensor` and logical tensor indices |
| Work distribution | Metal grid indices plus C++ launch configuration | `T.Kernel` and `T.Parallel` in the kernel definition |
| MLX integration | Operator-specific C++ primitive, buffer bindings and dispatch | Shared MLX adapter with `compile_mlx(..., inplace_idx=0)` |
| Dtypes and optimization | Explicit instantiations, scalar and vec4 paths; broader dtype/paged support | FP16/FP32 nonpaged writes/zeroing; static specialization |

The TileLang write body expresses the destination and source directly:

```python
with T.Kernel(T.ceildiv(width, 256), count, threads=128) as (bx, by):
    for j in T.Parallel(256):
        col = bx * 256 + j
        if col < width and ids[by] >= 0 and ids[by] < slots:
            pool[(ids[by],) + coords(col)] = rows[(by,) + coords(col)]
```

Compare `metal/tilelang_state_scatter.py` (98 lines including validation,
zeroing and caching) with `metal/kernels_v2/gdn_state_scatter.metal` (91 lines)
plus the scatter primitive and wrapper in `metal/gdn.cpp` (roughly 150 lines).
These counts are not feature-equivalent: native additionally supports BF16,
int8/uint8, paged addressing and vec4 writes. Native is retained, so this PR
increases total repository code. The maintenance benefit is avoiding a separate
operator-specific MLX primitive and Metal launch plumbing for the candidate;
the shared adapter still owns that runtime complexity upstream.

The cache passes physical strides and publishes the returned write-dependent
alias through its existing storage owner. Destination storage is shared;
non-contiguous read-only rows may be materialized. IDs must be valid and
distinct under the cache contract. Unsupported dtypes and missing dependencies
fail explicitly. Up to 128 specializations are cached per process, keyed by
shape, strides, dtype, row count and zero/write mode.

## Validation and reproducible measurements

From the vllm-metal checkout with the dependencies above:

```bash
python -m pytest tests/attention/test_shared_cache_storage.py \
  tests/attention/test_lazy_kv_allocation.py -q
python tools/benchmark/state_scatter_benchmark.py --samples 9 --iterations 32 \
  --output state-scatter-results.json
```

Native tests always run. Four optional cases skip only when TileLang or its
native extension is absent; broken installed integrations still fail. TileLang
is not added to standard installation or CI. Tests use real KVCacheStorage
allocations and cover backing bytes, padding, layer offsets, deferred updates,
copy/zero and cross-stream readers.

Local measurements used Apple M4 (8 GPU cores, 16 GiB), macOS 26.0.1,
SDK 26.2, Python 3.12.13, MLX 0.32.4 and the TileLang revision pinned above.
The benchmark alternates backend order across nine samples of 32 updates,
warms each kernel three times, and checks the full original backing after each
sample. Shapes, strides, offsets and raw samples are recorded in its JSON.

Median host wall time per update, in microseconds:

| Case | Sync native | Sync TileLang | Lazy chain native | Lazy chain TileLang |
| --- | ---: | ---: | ---: | ---: |
| Small FP32 rows | 171.74 | 185.40 | 9.01 | 11.10 |
| FP16 conv, batch 1 | 176.44 | 190.48 | 24.23 | 26.09 |
| FP16 conv, batch 2 | 189.49 | 195.43 | 23.26 | 25.58 |
| FP16 conv, padded offset | 233.62 | 201.37 | 44.12 | 28.42 |
| FP32 recurrent, batch 1 | 258.81 | 258.33 | 66.45 | 51.97 |
| FP32 recurrent, batch 2 | 291.12 | 269.26 | 105.54 | 90.14 |
| FP32 recurrent, padded zero | 263.36 | 266.45 | 91.95 | 66.99 |

Sync evaluates after every update; lazy chain evaluates after all 32 dependent
updates. These times include Python/MLX submission and completion, exclude
compilation, and are not pure GPU timings. JSON reports p10/p90 and separates
loop time from final wait; asynchronous execution overlaps those intervals.
Cold compilation cost is not characterized by this benchmark.

Small lazy updates regress about 23%, and aligned conv updates about 8–10%,
while some large or unaligned cases improve. These results do not justify
promoting TileLang universally. Historical HTTP serving measurements of the
preceding experimental dispatch also drifted with run order: paired TileLang /
native throughput ratios were 1.004/1.012/0.932 at concurrency 1 and
0.995/1.101/1.184 at concurrency 2. They do not establish stable parity or an
acceptable serving regression. Native remains the internal default; promotion
requires repeatable workload evidence and an agreed regression budget.

Qwen3.5-0.8B FP16 text parity covers eight fixed prompts, 16 generated tokens,
and request batch sizes 1 and 2. This does not qualify all hybrid models,
BF16, vision or speculative serving. See the
[macOS benchmark procedure](benchmarking-macos.md) for serving measurements and
[adapter documentation](https://github.com/tile-ai/tilelang/blob/2cdbb1e05efb6aee83590403274bedea31813a4b/examples/metal/README.mlx.md)
for storage, stream and version constraints.
