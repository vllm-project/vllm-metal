# Reproducing the state-cache budget experiment

`tools/state_cache_budget_bench.py` compares the existing compact-slot behavior
with the opt-in `state_cache_budget_mib` setting on the same source checkout.
It supports a 48 GB M5 Max, a 64 GB M5 Pro, or another Apple Silicon host;
record the actual device and memory from each report instead of comparing
machines as if their memory budgets match. The 0.8B `256` MiB and 27B `2048`
MiB values in the paste-and-run blocks are **reproducible benchmark settings
for those case sets**, not automatically selected optima for every RAM size.
A one-request minimum or a concurrency-sized budget is a separate experiment:
keep every non-budget argument identical inside that pair, and do not fold the
results into the standard 08b/27b matrix. Hosts with less planned Metal memory
may reject the published 27B `.70` / 2048 MiB / 8192 combination even though
the 0.8B case set still fits.

## Prepare an isolated source environment

Use Python 3.12 arm64 and follow the repository's source installation steps.
Install the official macOS arm64 vLLM 0.29.0+cpu wheel, this checkout in editable
mode, and its development dependencies. Build its native artifacts against the
exact pinned MLX version. Do not reuse `.so` files built against a different MLX.
The checkout pins MLX 0.32.1 and the mlx-lm Git revision in `pyproject.toml`.
Numba 0.65 cannot import with NumPy 2.5; a local `numpy<2.5` constraint avoids
that unrelated test-collection failure. Keep a package freeze with the results.

Use the same immutable model snapshot, quantization, package environment,
source revision, and background workload for both arms. An already downloaded
local model path avoids network changes between runs. No model files are
modified by the tool. The command accepts token IDs directly; it records the
chat-template hash but does not apply a chat template.
Before importing the engine, it streams every local `.safetensors` shard
through SHA256 and records relative file names, byte counts and full hashes.
Hugging Face snapshot symlinks are supported. A remote model ID or a directory
without these shards fails at `weights_identity`; download a fixed local
snapshot first. Model weights are not loaded as tensors during fingerprinting.
The tool places its checkout first on the import path, records the actual loaded
`vllm_metal.__file__` and source root, and rejects a package loaded from another
checkout or wheel. A checkout SHA therefore cannot silently describe different
Python source code used by the engine.

## Paste-and-run reproduction on a 48 GB M5 Max

Start from the tested branch/commit shown in the PR evidence, with `uv`, Rust,
and full Xcode (including its Metal toolchain) available. Obtain
`requirements-macos-arm64.lock.txt` from the same evidence bundle. It contains
182 exact dependency entries, including the official vLLM wheel URL and mlx-lm
Git revision; it excludes the original machine's editable checkout path.
The environment used for the first measurements has lock SHA256
`78debbf667d8ef8c66666f2ea09913763628a8f84ef54ad5860a422d7d3f4ba4`.
The lock belongs with the raw reports when they are published; `reports/` is
not automatically present after cloning this repository.

### Model processor compatibility in the pinned environment

The 27B reference at Hugging Face revision
[`3e6447f082e89cc7f0bc6e5441afd38dfce760ff`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/3e6447f082e89cc7f0bc6e5441afd38dfce760ff)
has conflicting processor metadata: `preprocessor_config.json` selects
`Qwen2VLImageProcessorFast`, while `processor_config.json` embeds the unregistered
`Qwen3VLImageProcessor` name. A CPU-only `AutoProcessor.from_pretrained` check
still fails with the pinned transformers **5.17.0**. Upgrading from the previous
5.12 environment did not make that fresh metadata loadable.

The 27B commands below therefore construct a dedicated model view inside the
new report directory. They symlink weight shards, copy other files, and omit
only the exact known incompatible `processor_config.json`, preserving its bytes
and hash beside the view. The existing `Qwen2VLImageProcessorFast` preprocessor
is kept unchanged. Both arms use that same view. No shared Hugging Face or
ModelScope snapshot, global processor registry, or installed package is edited.
An unexpected processor configuration fails for inspection instead of receiving
an unverified replacement. The view is checked with `AutoProcessor` before
inference; this check loads processor/tokenizer metadata, not model weights.

This compatibility recipe is for the **text-token benchmark**. It is not a
validation of image/video preprocessing equivalence. The measured local 27B
snapshot already omitted the same processor file; the official preprocessor
and that local preprocessor have equal JSON values but different whitespace,
so their raw file hashes differ. Preserve the view's file hashes and download
revision with the results rather than claiming byte-identical unmodified model
metadata. The pinned 0.8B revision already has no `processor_config.json` and
uses the recognized preprocessor name; it does not require this omission.


The block below uses a dedicated `.venv-state-budget-repro`, installs exactly
that dependency set, builds this checkout, and runs baseline/budget/compare
sequentially for both automatic and explicitly synchronous scheduling. It does
not modify another virtual environment or any model snapshot. Set the two
absolute paths first; use `STATE_BENCH_CASESET=27b` only for the quantized 27B
model, otherwise leave the 0.8B default.

```sh
export STATE_BENCH_LOCK=/absolute/path/to/requirements-macos-arm64.lock.txt
export STATE_BENCH_MODEL=/absolute/path/to/model-snapshot
export STATE_BENCH_CASESET=08b

bash <<'SH'
set -euo pipefail
test -f tools/state_cache_budget_bench.py
test -f "$STATE_BENCH_LOCK"
test -d "$STATE_BENCH_MODEL"
test "$(uname -m)" = arm64
test "$(shasum -a 256 "$STATE_BENCH_LOCK" | cut -d ' ' -f 1)" = \
  78debbf667d8ef8c66666f2ea09913763628a8f84ef54ad5860a422d7d3f4ba4
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
export MACOSX_DEPLOYMENT_TARGET=15.0
bench_venv=.venv-state-budget-repro
test -x "$bench_venv/bin/python" || uv venv "$bench_venv" --python 3.12 --seed
bench_python="$bench_venv/bin/python"
uv pip sync --python "$bench_python" "$STATE_BENCH_LOCK"
uv pip install --python "$bench_python" --no-deps -e '.[dev,stt]'
uv pip check --python "$bench_python"
bench_dir="reports/budget-bench/$(date +%Y%m%d-%H%M%S)-${STATE_BENCH_CASESET:-08b}"
mkdir -p "$bench_dir"
xcodebuild -version > "$bench_dir/xcode.txt"
xcrun --show-sdk-version > "$bench_dir/sdk.txt"
rustc --version > "$bench_dir/rust.txt"
uv --version > "$bench_dir/uv.txt"
sysctl hw.memsize iogpu.wired_limit_mb > "$bench_dir/system-memory.txt"
"$bench_python" -m vllm_metal.metal.build > "$bench_dir/native-build.log" 2>&1
uv pip freeze --python "$bench_python" > "$bench_dir/environment-freeze.txt"
git rev-parse HEAD > "$bench_dir/source-commit.txt"
git diff HEAD -- > "$bench_dir/source.patch"
export HF_HUB_OFFLINE=1
case "${STATE_BENCH_CASESET:-08b}" in
  08b)
    bench_budget=256
    bench_workload=(--gpu-memory-utilization 0.25 --max-model-len 3072
      --max-num-batched-tokens 1088 --prompt-lengths 545,2177)
    ;;
  27b)
    bench_budget=2048
    bench_workload=(--gpu-memory-utilization 0.7 --max-model-len 8192
      --max-num-batched-tokens 1568 --prompt-lengths 785,6273)
    ;;
  *) printf 'STATE_BENCH_CASESET must be 08b or 27b\n' >&2; exit 2 ;;
esac
bench_model="$STATE_BENCH_MODEL"
if test "${STATE_BENCH_CASESET:-08b}" = 27b; then
  "$bench_python" - "$STATE_BENCH_MODEL" "$bench_dir" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

from transformers import AutoProcessor, __version__ as transformers_version

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
view = output / "model-view"
view.mkdir()
known_processor_sha = "45fc17c8dd2474af6b493b52483c26c0584b0082d368c480f9fa611e73070040"
processor_file = source / "processor_config.json"
excluded = []
if processor_file.exists():
    raw = processor_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != known_processor_sha:
        raise RuntimeError("Unexpected processor_config.json; inspect compatibility before applying this 27B recipe")
    (output / "original-processor_config.json").write_bytes(raw)
    excluded.append(processor_file.name)
preprocessor = json.loads((source / "preprocessor_config.json").read_text())
if preprocessor.get("image_processor_type") not in (
    "Qwen2VLImageProcessorFast", "Qwen2VLImageProcessor"
):
    raise RuntimeError("Unexpected preprocessor; this recipe must not change its semantics")
if not list(source.glob("*.safetensors")):
    raise RuntimeError("Expected the reference snapshot's top-level safetensors shards")
for item in source.iterdir():
    if not item.is_file() or item.name in excluded:
        continue
    target = view / item.name
    if item.suffix == ".safetensors":
        target.symlink_to(item.resolve())
    else:
        shutil.copy2(item, target)
manifest = {
    "source_directory": str(source),
    "view_directory": str(view),
    "excluded_files": excluded,
    "excluded_processor_sha256": known_processor_sha if excluded else None,
    "transformers": transformers_version,
    "non_weight_files": {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in view.iterdir()
        if item.is_file() and item.suffix != ".safetensors"
    },
}
(output / "model-view-metadata.json").write_text(json.dumps(manifest, indent=2) + "\n")
processor = AutoProcessor.from_pretrained(str(view), local_files_only=True)
print("Processor CPU check:", type(processor).__name__, type(processor.image_processor).__name__)
PY
  bench_model="$bench_dir/model-view"
fi
bench_common=(--model "$bench_model" "${bench_workload[@]}"
  --max-num-seqs 4 --concurrency 1,4 --repeats 2 --max-new-tokens 32)
for bench_async in auto off; do
  "$bench_python" tools/state_cache_budget_bench.py run "${bench_common[@]}" \
    --async-scheduling "$bench_async" --output "$bench_dir/baseline-$bench_async.json" \
    > "$bench_dir/baseline-$bench_async.log" 2>&1
  "$bench_python" tools/state_cache_budget_bench.py run "${bench_common[@]}" \
    --async-scheduling "$bench_async" --state-budget-mib "$bench_budget" \
    --output "$bench_dir/budget-$bench_async.json" > "$bench_dir/budget-$bench_async.log" 2>&1
  "$bench_python" tools/state_cache_budget_bench.py compare \
    --baseline "$bench_dir/baseline-$bench_async.json" \
    --budget "$bench_dir/budget-$bench_async.json" \
    --output "$bench_dir/comparison-$bench_async.json"
done
printf 'Results: %s\n' "$bench_dir"
SH
```

The local 0.8B reference is `Qwen/Qwen3.5-0.8B`, snapshot
`2fc06364715b967f1860aea9cf38778875588b17`; obtain that snapshot with the
Hugging Face CLI or an existing download and pass its local directory. The
27B reference is `mlx-community/Qwen3.8-27B-4bit`; use the same immutable weight
snapshot and config as the evidence. A mutable `master`/`main` name alone does
not identify identical weights. The complete shard manifest in each report
identifies the actual bytes, and comparison requires nonempty matching manifests
in addition to model-config/template hashes. Retain the original model revision
as download provenance too.

The two case sets are reproducible starting workloads, not a promise that
every budget/length combination fits or reaches maximum capacity on every
machine. A failure leaves its JSON/log and stops the script before later arms.
Inspect the recorded stage and exception; do not relabel an incomplete run as
a successful comparison. A numerical mismatch likewise requires investigation.

## Run two processes sequentially

The following is a starting workload for a quantized hybrid model that fits
the host. `4096` MiB is an example stable-state budget, not a universal setting;
the required per-request working reserve depends on the model's actual state
groups and on asynchronous execution. A too-small setting produces a structured
startup failure rather than proving that the device lacks memory.

```sh
python tools/state_cache_budget_bench.py run \
  --model /absolute/path/to/model-snapshot \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 --max-num-batched-tokens 1568 --max-num-seqs 4 \
  --prompt-lengths 785,6273 --concurrency 1,4 --repeats 3 \
  --max-new-tokens 32 --output reports/budget-bench/baseline.json
```

Wait for that process to exit, then run the identical workload with the opt-in:

```sh
python tools/state_cache_budget_bench.py run \
  --model /absolute/path/to/model-snapshot \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 --max-num-batched-tokens 1568 --max-num-seqs 4 \
  --prompt-lengths 785,6273 --concurrency 1,4 --repeats 3 \
  --max-new-tokens 32 --state-budget-mib 4096 \
  --output reports/budget-bench/budget-4096.json
```

```sh
python tools/state_cache_budget_bench.py compare \
  --baseline reports/budget-bench/baseline.json \
  --budget reports/budget-bench/budget-4096.json \
  --output reports/budget-bench/comparison.json
```

The baseline omits the additional-config key completely. The budget command
passes `additional_config={"state_cache_budget_mib": 4096}` to `LLM`.
The budget run checks that a bounded runtime and scheduler were actually
installed; an ignored option cannot silently count as a successful budget run.
Run the two commands again in reverse order when timing matters, so thermal
conditions and background activity are less likely to favor one configuration.

For a short smoke test, use `--prompt-lengths 1024 --concurrency 1 --repeats 1`.
`--async-scheduling auto` is the default and passes no override to the engine.
Use `--async-scheduling off` in both processes for explicit synchronous coverage,
or `--async-scheduling on` in both for explicit asynchronous scheduling. The
comparison requires both the requested policy and the resolved policy to match.
For longer prompts, increase both `--max-model-len` and `--prompt-lengths`, keeping
prompt length plus `--max-new-tokens` within the model limit. Increase `--repeats`
for more churn and use a range of feasible budgets. A concurrency argument is
the submitted batch size; the actual scheduler running limit can be smaller.

A 32 GB-class host that cannot start the published 27B `.70` / 2048 MiB
matrix should keep that rejection as a host-specific result and add **new**
pairs instead of editing the 08b/27b case set. Two useful extra pairs, each
with identical non-budget arguments on both arms:

- one synchronous request at the 27B working-row floor (`state_cache_budget_mib=294`);
- four synchronous requests at the concurrency floor (`1175` MiB for 24 rows).

Publish those JSON/logs under a distinct directory. They do not replace the
standard matrix or the 48 GB / 64 GB 2048 MiB archives.

For cache-hit coverage, prefer `k * B + 1` tokens, where `B` is the actual
resolved cache block size: `545,2177` for `B=544`, or `785,6273` for `B=784`.
vLLM recomputes at least the last prompt token for logits, so its cache lookup
stops at `prompt_length - 1`. An exact `k * B` prompt therefore cannot reuse
its final `k * B` checkpoint; if a suitable earlier state checkpoint was not
retained, the nominal hot call can restore zero tokens. Verify positive
`admitted_num_computed_tokens` rather than treating the phase label as proof.

## Workload and evidence

Each prompt has exactly the requested token count. Batch members have stable,
distinct prefixes. A cold batch uses a new `cache_salt`; its hot batch repeats
the exact inputs and salt. Each subsequent repeat uses a fresh salt, leaving
the preceding cached checkpoints behind to create retention/eviction pressure.
Salts are deterministic across both processes. Cold and hot labels describe
the requested experiment; `admitted_num_computed_tokens` records whether a
restore actually occurred. A hot request can recompute if its checkpoint was
evicted or the prompt was at/below one block, and aligned longer prompts can
also miss when the required earlier checkpoint is absent.

Generation uses `temperature=0`, `logprobs=None`, fixed output length, and
`ignore_eos=True`. This leaves native greedy sampling and the decode pipeline
eligible. An in-process runner spy records admissions and optional scheduler
budget payloads without synchronizing per step. Both processes use the same
instrumentation. The tool synchronizes only at startup and generation-call
measurement boundaries; timings include the observer's Python overhead.

The JSON includes:

- Complete prompt and output token IDs, salts, case keys, and finish reasons.
- Source commit and tracked-diff hash, untracked Python-source hashes, tool hash, package versions/direct URLs,
  full weight-shard fingerprints, model config/hash, template hash, physical
  memory, and MLX device info.
- Startup and per-batch latency, output throughput, and actual restored-token
  counts. The actual scheduler `max_num_running_reqs` is reported separately
  from the requested `max_num_seqs` and submitted batch concurrency.
- Actual block size/count; state slot capacity, allocated/occupied slots,
  retention capacity, and optional resident/eviction/stall/sequence telemetry.
  Resolved cache policy includes `prefix_cache_retention_interval`,
  `hash_block_size` and the scheduler pool's actual hash block size; `0` and
  `None` remain distinct values.
  The baseline does not maintain the budget's authoritative resident catalog;
  its catalog `resident_blocks=0`/`sequence=0` does not mean no cached state
  exists. Use its `occupied_slots` for compact-map occupancy.
- MLX active bytes, allocator cache bytes, peak active bytes, and process RSS
  before and after startup and every generation call. Peak MLX active memory
  resets before each measured interval. RSS is sampled at boundaries; it is
  not a continuously sampled process-RSS peak and is not wired-memory usage.

State tensor capacity and runtime `stable_bytes` are reported separately from
allocator memory and RSS. They exclude forward temporaries and other allocations.
Do not add active/cache/RSS together: these are overlapping measurements with
different meanings. More configured KV blocks or successful startup does not
prove that the full KV pool was touched or that maximum concurrency was reached.
The summary deliberately makes no pool-saturation claim. To claim a capacity
gain, design and record a workload that actually reaches the relevant allocation
limits and remains correct under churn; the default short/long matrix alone
does not establish that result.

Distinguish the evidence needed for each claim:

| Claim | Required observations |
|---|---|
| Stable-state memory stays within the budget | Maximum observed `stable_bytes`/tensor capacity at or below requested MiB, occupied slots reaching the cap or positive quota evictions, and correct outputs through subsequent restores/churn. |
| More KV capacity is configured | Reported block count increases under matching block/hash/retention policy. This is a configuration result until a workload actually uses the extra capacity. |
| Longer inputs or more simultaneous work complete | A matched workload beyond the baseline's demonstrated limit completes in the budget arm, with full outputs, actual running/admission behavior, and memory data. Record baseline failure stage and distinguish allocation rejection from unrelated failure. |
| Lower overall memory | Actual allocator/RSS observations decrease for a matched workload; lower GDN state alone does not establish this because the released budget may allocate more KV memory. |
| Higher throughput | Repeated matched runs in both orders, with comparable temperature/background load. Aggregate output tokens per second includes prompt processing and observer overhead; it is not isolated decode-kernel throughput. |

The 0.8B smoke matrix demonstrates behavior on a small model, not the full
27B capacity result. Keep incomplete/exploratory reports separate from final
frozen-source evidence. A single before/after timing pair is preliminary.
Full weight fingerprinting happens before the startup timer and reads the model
files into the OS filesystem cache. Startup measurements therefore follow that
read pass and must not be presented as cold-disk model-loading time.

`compare` requires an inactive baseline and an actually active positive-budget
arm; comparing baseline with itself cannot pass as budget evidence. Nonempty
weight manifests must match by every relative shard name, byte count and SHA256.
Resolved
block size, Mamba mode, hash sizes and retention policy must match; block counts
and capacity are allowed to differ. It requires complete, nonempty evidence and checks exact inputs and
complete outputs. It also requires matching device information, physical memory,
hostname-derived host identity hash (the hostname is not stored), OS/Python,
benchmark tool hash, all installed dependency versions/direct URLs, relevant
environment settings, and strict sampling/execution policy. Different source
revisions are allowed and reported explicitly; the editable vllm-metal package
is the code under test and is excluded from the dependency equality check.
The exact seeds, temperature, logprobs setting and ignore-EOS policy are stored
and checked, including rejecting two reports that both departed from the strict
policy. Schema-v1 reports lack this identity evidence and cannot qualify as a
validated before/after comparison. The tool never waives floating-point ties. A mismatch is evidence
to investigate; it does not by itself identify state corruption as the cause.
Failed runs retain completed cases and a failure stage, exception, and traceback.
Periodic atomic checkpoints also preserve earlier results if the OS terminates
the process, but an uncatchable kill cannot record a final exception.

The tool is not in model-running CI. Its engine-free tests cover workload
validation, deterministic salts/tokens, comparison completeness, structured
failures, and the distinction between observed memory and saturation claims:

```sh
python -m pytest tests/test_state_cache_budget_bench.py -q
```

## CI-equivalent validation before publishing a PR

The repository's CI runs `scripts/lint.sh` and `scripts/test.sh` on macOS 15;
the test job selects Xcode 26.3 and a macOS 15.0 deployment target. In an already
locked environment, the equivalent checks include:

```sh
shellcheck -- *.sh scripts/*.sh
ruff check .
ruff format --check .
mypy vllm_metal
python -m pytest -m 'not slow' tests/ .github/scripts/test_parity.py -v --tb=short
```

CI also rebuilds native artifacts, builds a wheel with `uv build`, calls
`verify_wheel_artifacts` from `scripts/lib.sh` (Rust/native extensions, all
required Metal libraries including NAX, and deployment targets), and checks
that platform discovery selects `MetalPlatform`. A passing pytest result alone
does not cover those packaging checks. The scripts may resolve/install packages,
so run those steps in a dedicated environment and retain its final freeze.

## Revalidating saved reports after comparator hardening

The comparison independently rebuilds the expected case matrix from the recorded
configuration, checks its full order/metadata, validates every request's prompt
and fixed-length output, and requires matching `length` finish reasons. Explicit
`qualification_eligible=false` diagnostic metadata cannot qualify. The report
records `comparator_sha256` separately from each input report's acquisition
`tool_sha256`; both input acquisition hashes must still match. A comparison-only
change can therefore revalidate unchanged raw reports without relabeling the
code that collected them. Recompute comparisons and evidence tables after such
a change; do not edit raw run reports to make them conform.
