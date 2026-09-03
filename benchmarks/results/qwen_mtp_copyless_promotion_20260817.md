# Qwen MTP copyless GDN-state promotion decision

Date: 2026-08-17

## Decision

Promote the **copyless speculative GDN-state write path** into the native-Qwen-MTP feature branch. Do **not** promote the deferred request-local decode-state experiment or the copyless+deferred combination.

The combination is functionally correct after repairing its method-contract collision, but it did not meet the predeclared performance gate of at least 1.05x versus copyless alone. Copyless alone is the simpler, validated winner.

## Root cause of the original combined-arm failure

The original breakthrough matrix reported only client-side connection failures for the combined arm. Retained server logs showed the actual initialization failure:

```text
TypeError: _try_conv_decode() got an unexpected keyword argument 'write_slot_ids'
```

The copyless candidate extends the lazy GDN decode contract with explicit source and destination state slots. The deferred-state experiment monkeypatched the same methods with the older signature, removing that destination-slot contract. A diagnostic compatibility bridge preserved copyless handling for explicit cross-slot speculative writes and deferred handling for ordinary same-slot decode.

That bridge was used only to diagnose and evaluate the combination; it is not part of the promoted production diff.

## Four-arm functional smoke

Run: https://github.com/PhilipJohnBasile/vllm-metal/actions/runs/32026097139

Model and runner:

- `Qwen/Qwen3.5-0.8B`, MLX affine 4-bit, group size 64
- exact MLX-LM MTP head `ac6aaffd8fdfb8c8e713e17f155d83e3d72b0a0f`
- GitHub-hosted virtual Apple M1, macOS 15
- one fresh port and retained server log per launch
- explicit process-group termination and listener-release verification
- deterministic greedy output and MTP metric capture

| Arm | Successful launches | Output tok/s | Acceptance | Exact output parity | Listener released |
|---|---:|---:|---:|:---:|:---:|
| Current MTP | 1/1 | 8.356 | 33.3% | yes | yes |
| Copyless | 1/1 | 9.597 | 33.3% | yes | yes |
| Deferred | 1/1 | 7.799 | 33.3% | yes | yes |
| Copyless + deferred | 1/1 | 2.508 | 33.3% | yes | yes |

This established that the repaired combination could start, serve correctly, and shut down cleanly. The short run also showed enough variance risk to require the predeclared three-launch promotion gate rather than promoting from one measurement.

## Three-launch promotion gate

Run: https://github.com/PhilipJohnBasile/vllm-metal/actions/runs/32027165189

Identical 1,152-token prompt, 64 output tokens, one warm-up request, three independent server launches per arm:

| Arm | Successful launches | Median output tok/s | Throughput CV | Acceptance | Exact output parity | Listener released |
|---|---:|---:|---:|---:|:---:|:---:|
| Copyless | 3/3 | 12.007 | 1.7% | 31.25% | yes | 3/3 |
| Copyless + deferred | 3/3 | 11.954 | 7.7% | 31.25% | yes | 3/3 |

Promotion criteria:

- 3/3 clean launches: **pass**
- exact deterministic correctness: **pass**
- no acceptance regression: **pass**
- no stale listener/process leakage: **pass**
- throughput CV at or below 10%: **pass**
- combined throughput at least 1.05x copyless: **fail** (`0.996x`)

Decision: **keep copyless only**.

## Six-workload copyless result

Run: https://github.com/PhilipJohnBasile/vllm-metal/actions/runs/31985707678

The breakthrough matrix compared current MTP, copyless, and deferred across six serving workloads. Copyless was the fastest MTP arm on every workload, preserved output-hash parity, and improved geometric-mean throughput by **1.215x (+21.5%)** versus current MTP.

| Workload | Current MTP tok/s | Copyless tok/s | Copyless / current MTP | Copyless / non-MTP baseline |
|---|---:|---:|---:|---:|
| Interactive, concurrency 1 | 6.832 | 8.815 | 1.290x | 0.322x |
| Serving, concurrency 4 | 16.652 | 19.642 | 1.180x | 0.361x |
| Long prefix, concurrency 1 | 5.696 | 7.293 | 1.280x | 0.474x |
| Long prefix, concurrency 4 | 8.662 | 9.515 | 1.098x | 0.367x |
| Serving, concurrency 8 | 14.230 | 17.984 | 1.264x | 0.266x |
| Serving, concurrency 16 | 13.676 | 16.286 | 1.191x | 0.234x |

Across these workloads, copyless reached a geometric mean of **0.328x** the matched non-MTP baseline, with a range of **0.234x to 0.474x**. Therefore this promotion is an internal MTP-path improvement, not evidence that native MTP is yet a net serving acceleration.

Retained-prefix pressure also favored copyless: geometric mean **1.231x** versus current MTP, with output-hash parity preserved.

## Promoted source validation

Run: https://github.com/PhilipJohnBasile/vllm-metal/actions/runs/32029260290

The final formatted source diff was applied to a clean branch based on the feature head and passed:

- native Metal artifact build
- Python compilation
- `git diff --check`
- 48 focused GDN/MTP tests
- Ruff lint
- Ruff formatting

Only the direct source-to-destination speculative GDN-state changes were promoted. The diagnostic compatibility shim, deferred-state experiment, bootstrap workflow, and patch staging file were excluded from the production feature branch.

## Remaining gates

This result does not change the two remaining readiness blockers:

1. `ml-explore/mlx-lm#1740` must merge, and this branch must pin its actual merge commit.
2. The promoted copyless path must be qualified with dense Qwen3.8-27B on a real M5 Max before any positive Apple-Silicon performance claim.
