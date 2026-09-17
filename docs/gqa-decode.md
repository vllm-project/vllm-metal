# GQA decode routing

The paged attention primitive can use `paged_attention_gqa_decode` for a
limited set of single-request, long-context decode calls. Automatic routing
is intentionally narrower than the shapes supported by the shader.

## Automatic eligibility

The following geometry and KV-length bounds are inclusive:

| Query heads | KV heads | Head dimension | Minimum KV tokens | Maximum KV tokens |
|---:|---:|---:|---:|---:|
| 32 | 8 | 128 | 32,768 | 131,072 |
| 24 | 4 | 256 | 32,768 | 131,072 |
| 16 | 2 | 128 | 65,536 | 131,072 |

Every row additionally requires:

- One pure-decode request, with `num_decode_requests` equal to 1 or omitted.
- A verification window of at most 1.
- Matching FP16 or BF16 query, key-cache, and value-cache types.
- Kernel block size 16. This is the block size passed to the primitive after
  any hybrid-cache view conversion.
- No TurboQuant, attention sinks, logit soft-capping, or sliding-window
  attention.
- Enough partition threadgroups for the conservative occupancy guard:
  `ceil(KV tokens / 512) * KV heads >= 3 * detected GPU cores`.
  An unknown GPU core count falls back.

Calls outside these bounds use the established attention family. That
includes multi-request batches, even if each request individually matches a
row, and lengths above the maximum. The 16/2/128 geometry deliberately starts
at 64K; small measured gains at 32K were not used to widen automatic routing.

The gate checks geometry rather than model names and does not contain a
device-name allowlist. Its bounds and occupancy guard are empirical choices,
not a claim that GQA wins for every device or competing GPU workload.

## Disable switch

Set `VLLM_METAL_DISABLE_GQA_DECODE=1` before starting the server to keep
eligible requests on the established path. This switch provides an A/B and
operational fallback; it cannot enable an otherwise ineligible call.

There is no startup calibration or disk-cached performance threshold for
this gate. `VLLM_METAL_GQA_AUTOTUNE` and the former mutable gate-parameter
APIs are no longer used.

## Validation

`tests/test_gqa_paged_decode.py` evaluates the primitive and checks
`last_paged_dispatch()` alongside the numerical reference. Positive cases
must actually report `gqa_decode`; boundary, multi-request, verification,
feature, and disabled cases must report the appropriate established family.
`tests/test_attention_sdpa.py` checks that the environment switch reaches
the primitive, and `tests/test_native_sdpa_decode.py` provides a separate
numerical comparison.

`last_paged_dispatch()` records the last family selected in the process. It
is suitable for serial tests and isolated worker checks; it is not
per-request telemetry for concurrent serving. Evaluate the operation before
reading it, because MLX builds graphs lazily.

For performance validation, rebuild native artifacts from the tested
revision and use the real `vllm serve` process topology. Record the model,
runtime versions, hardware, warmup, KV lengths, repetitions, and background
GPU activity; compare enabled and disabled arms with actual worker-side
dispatch checks. Separate primitive timing from HTTP decode throughput,
and keep noisy measurements visible. The
[macOS benchmarking discussion](https://github.com/vllm-project/vllm-metal/issues/713)
explains why in-process engine measurements and short probes are not
interchangeable with serving results.

## Why the broader gate was removed

An earlier design combined a partition-occupancy floor with a scalar
potential-KV-reread proxy and fitted a threshold at startup. Additional
geometry, submission-mode, and load tests did not justify extending that
single threshold to all shader-supported cases. The proxy is not measured
DRAM traffic: each simdgroup still issues its own K/V loads, and the actual
traffic depends on cache residency. Grouping related heads can improve
locality without an explicit cross-simdgroup KV broadcast.

A separate experiment added paired timing, confirmation runs, numerical
checks, telemetry, and invocation-bound permits with expiry, revocation,
and native validation before encoding. It rejected known invalidation but
did not establish performance for later requests after resource conditions
changed. Tests in which requests always fell back did not validate positive
GQA performance. That experimental machinery is not part of this routing
implementation.

These limitations motivate the explicit scope above; they do not establish
that a more general optimization is impossible. Expanding automatic routing
requires its own dispatch, numerical, and real-serving benchmark evidence.
