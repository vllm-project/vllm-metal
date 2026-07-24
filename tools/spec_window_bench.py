# SPDX-License-Identifier: Apache-2.0
"""Op-level A/B for the spec-verify window mode (issue #465).

Times the paged-attention primitive with mx.synchronize fences for three
metadata shapes per (conc, ctx, K) cell:

  decodeN   conc one-token segments             (ideal 1x-traffic floor)
  expanded  conc*(K+1) one-token segments       (per-token dispatch)
  windowed  conc merged segments, window mode   (this branch)

exp/win > 1 means the window mode wins.  The win grows with context length
and concurrency; per-chip measured grids live in the RFC #465 thread and
the PR that landed window mode, not here, so this header cannot go stale.

Run from the repo root:

    PYTHONPATH=$PWD VLLM_METAL_BUILD_FROM_SOURCE=1 \
        python tools/spec_window_bench.py
"""

from __future__ import annotations

import gc
import statistics
import time

import mlx.core as mx
import numpy as np

from vllm_metal.metal import get_ops

BLOCK_SIZE = 16
HEAD_SIZE = 128
DTYPE = mx.float16
LAYERS_PER_REP = 32
WARMUP_REPS = 5
TIMED_REPS = 20

CONFIGS = {
    "0.6B-like (16q/8kv)": (16, 8),
    "8B-like (32q/8kv)": (32, 8),
}
CELLS = [
    (1, 2048, 5),
    (1, 8192, 5),
    (1, 16384, 5),
    (1, 8192, 3),
    (4, 8192, 5),
    (16, 2048, 5),
    (16, 8192, 5),
    (32, 2048, 5),
    (32, 8192, 5),
]


def _time_shape(ops, query, kc, vc, nkv, tables, seq_lens, cu, max_kv, window_seqlen_q):
    scale = HEAD_SIZE**-0.5
    tables_a = mx.array(tables, dtype=mx.int32)
    seq_a = mx.array(seq_lens, dtype=mx.int32)
    cu_a = mx.array(cu, dtype=mx.int32)
    mx.eval(tables_a, seq_a, cu_a)

    def run_once():
        outs = []
        for _ in range(LAYERS_PER_REP):
            out = mx.array(0)
            ops.paged_attention_primitive(
                query,
                kc,
                vc,
                nkv,
                scale,
                0.0,
                tables_a,
                seq_a,
                cu_a,
                BLOCK_SIZE,
                max_kv,
                -1,
                out,
                window_seqlen_q=window_seqlen_q,
            )
            outs.append(out)
        mx.eval(*outs)
        return outs

    for _ in range(WARMUP_REPS):
        outs = run_once()
    mx.synchronize()
    times = []
    for _ in range(TIMED_REPS):
        t0 = time.perf_counter()
        outs = run_once()
        mx.synchronize()
        times.append((time.perf_counter() - t0) / LAYERS_PER_REP * 1e6)
    return statistics.median(times), outs[0]


def main() -> None:
    ops = get_ops()
    print(
        f"device: {mx.device_info()['device_name']}  "
        f"min_decode_grid: {ops.min_decode_grid()}"
    )
    print(f"dtype: {DTYPE}  layers/rep: {LAYERS_PER_REP}  reps: {TIMED_REPS}")
    print()
    mx.random.seed(0)

    for cfg_name, (nq, nkv) in CONFIGS.items():
        print(f"=== {cfg_name} ===")
        header = (
            f"{'conc':>4} {'ctx':>6} {'K':>2} | {'decodeN us':>11} "
            f"| {'expanded us':>12} | {'windowed us':>12} "
            f"| {'exp/dec':>7} {'win/dec':>7} {'exp/win':>7} | bitwise"
        )
        print(header)
        print("-" * len(header))

        for conc, ctx, k_spec in CELLS:
            window = k_spec + 1
            total_len = ctx + window
            bps = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
            kc = mx.random.normal(
                shape=(bps * conc, BLOCK_SIZE, nkv, HEAD_SIZE)
            ).astype(DTYPE)
            vc = mx.random.normal(
                shape=(bps * conc, BLOCK_SIZE, nkv, HEAD_SIZE)
            ).astype(DTYPE)
            query = mx.random.normal(shape=(conc * window, nq, HEAD_SIZE)).astype(DTYPE)
            mx.eval(kc, vc, query)
            tables = [list(range(r * bps, (r + 1) * bps)) for r in range(conc)]

            d_query = query.reshape(conc, window, nq, HEAD_SIZE)[:, -1, :, :]
            mx.eval(d_query)
            d_med, _ = _time_shape(
                ops,
                d_query,
                kc,
                vc,
                nkv,
                tables,
                [total_len] * conc,
                list(range(conc + 1)),
                total_len,
                1,
            )

            e_tables = [t for t in tables for _ in range(window)]
            e_seq = [ctx + j + 1 for _ in range(conc) for j in range(window)]
            e_med, e_out = _time_shape(
                ops,
                query,
                kc,
                vc,
                nkv,
                e_tables,
                e_seq,
                list(range(conc * window + 1)),
                total_len,
                1,
            )

            w_med, w_out = _time_shape(
                ops,
                query,
                kc,
                vc,
                nkv,
                tables,
                [total_len] * conc,
                [i * window for i in range(conc + 1)],
                total_len,
                window,
            )

            if bool(mx.array_equal(e_out, w_out)):
                bw = "1"
            else:
                close = np.allclose(
                    np.array(e_out.astype(mx.float32)),
                    np.array(w_out.astype(mx.float32)),
                    atol=1.5e-2,
                    rtol=2e-2,
                )
                bw = f"0(close={int(close)})"

            print(
                f"{conc:>4} {ctx:>6} {k_spec:>2} | {d_med:>11.1f} "
                f"| {e_med:>12.1f} | {w_med:>12.1f} "
                f"| {e_med / d_med:>7.2f} {w_med / d_med:>7.2f} "
                f"{e_med / w_med:>7.2f} | {bw}"
            )
            del kc, vc, query, d_query, e_out, w_out
            gc.collect()
            mx.clear_cache()
        print()


if __name__ == "__main__":
    main()
