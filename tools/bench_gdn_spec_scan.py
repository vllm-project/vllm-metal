# SPDX-License-Identifier: Apache-2.0
"""Phase-0 microbenchmark: GDN state-scan cost for speculative verify spans.

Times the two lazy-kernel paths at Qwen3.8-27B recurrent geometry
(Hk=16, Hv=48, Dk=Dv=128, conv_dim=10240, conv kernel 4):

1. ``try_recurrent_decode`` — today's T=1-per-request decode update.
2. ``try_recurrent_prefill`` — the variable-length scan (chunked prefill
   path) at T=1..6, the proposed shape for a fused spec-verify span
   ``[last_token, *drafts]`` and for the partial-accept fixup re-scan.

The T-curve answers go/no-go: if a T-token scan costs ~T x a T=1 decode,
fused verification cannot pay on Metal; if it costs ~1 x (weights/state
traffic amortized), it can.
"""

from __future__ import annotations

import time

import mlx.core as mx

from vllm_metal.attention.caches.state_cache import PagedStateCache
from vllm_metal.attention.impls.gdn_lazy import (
    GDNLazyKernels,
    GDNRecurrentDecodeRequest,
    GDNRecurrentPrefillRequest,
)


def make_state_cache(**kwargs):
    num_layers = kwargs.get("num_layers", 1)
    max_seqs = kwargs["max_seqs"]
    conv_states = [
        mx.zeros(
            (max_seqs, kwargs["conv_kernel_dim"] - 1, kwargs["conv_dim"]),
            dtype=mx.float32,
        )
        for _ in range(num_layers)
    ]
    recurrent_states = [
        mx.zeros(
            (
                max_seqs,
                kwargs["num_v_heads"],
                kwargs["value_head_dim"],
                kwargs["key_head_dim"],
            ),
            dtype=mx.float32,
        )
        for _ in range(num_layers)
    ]
    return PagedStateCache([conv_states, recurrent_states])


# Qwen3.8-27B GDN geometry (matches the 2026-10-08 30k bench appendix).
N_HK, N_HV, D_K, D_V = 16, 48, 128, 128
CONV_DIM = 2 * N_HK * D_K + N_HV * D_V  # 10240
CONV_KERNEL = 4
DTYPE = mx.float32  # conservative serving policy: fp32 compute, fp32 state


def make_inputs(total_tokens: int):
    q = mx.random.normal((1, total_tokens, N_HK, D_K)).astype(DTYPE)
    k = mx.random.normal((1, total_tokens, N_HK, D_K)).astype(DTYPE)
    v = mx.random.normal((1, total_tokens, N_HV, D_V)).astype(DTYPE)
    g = mx.random.normal((1, total_tokens, N_HV)).astype(DTYPE)
    beta = mx.random.normal((1, total_tokens, N_HV)).astype(DTYPE)
    return q, k, v, g, beta


def bench(fn, iters: int = 50, warmup: int = 10) -> float:
    """Time ``fn``; it returns the arrays to evaluate.

    Both kernel paths defer the state scatter, parking it as a pending lazy
    array beside the scan output — evaluate both so the state update is
    actually timed (evaluating the output alone can miss it).
    """
    for _ in range(warmup):
        mx.eval(*fn())
    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(*fn())
    return (time.perf_counter() - start) / iters * 1e3  # ms


def main() -> None:
    mx.random.seed(0)
    kernels = GDNLazyKernels(enabled=True)
    assert kernels.enabled, "lazy GDN kernels unavailable"

    print(
        f"geometry: Hk={N_HK} Hv={N_HV} Dk={D_K} Dv={D_V} "
        f"conv_dim={CONV_DIM} recurrent/slot/layer="
        f"{N_HV * D_V * D_K * 4 / 1e6:.2f}MB fp32"
    )

    # --- 1) decode kernel: B requests x 1 token each (today's path) ---
    for num_requests in (1, 8):
        cache = make_state_cache(
            num_layers=1,
            max_seqs=num_requests,
            conv_kernel_dim=CONV_KERNEL,
            conv_dim=CONV_DIM,
            num_v_heads=N_HV,
            value_head_dim=D_V,
            key_head_dim=D_K,
        )
        q, k, v, g, beta = make_inputs(num_requests)
        slot_ids = list(range(num_requests))
        req = GDNRecurrentDecodeRequest(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            state_cache=cache,
            cache_idx=0,
            slot_ids=slot_ids,
            output_dtype=DTYPE,
            threadgroup_dv=4,
        )
        ms = bench(
            lambda req=req, cache=cache: (
                kernels.try_recurrent_decode(req),
                cache.pending_recurrent_states[0],
            )
        )
        print(
            f"decode kernel  B={num_requests:<2} T=1      : {ms:7.3f} ms "
            f"({ms / num_requests:.3f} ms/req)"
        )

    # --- 2) prefill scan kernel: one segment of T tokens (verify span) ---
    cache = make_state_cache(
        num_layers=1,
        max_seqs=16,
        conv_kernel_dim=CONV_KERNEL,
        conv_dim=CONV_DIM,
        num_v_heads=N_HV,
        value_head_dim=D_V,
        key_head_dim=D_K,
    )
    for num_reqs in (1, 4):
        for t in range(1, 7):
            total = num_reqs * t
            q, k, v, g, beta = make_inputs(total)
            cu = [i * t for i in range(num_reqs + 1)]
            req = GDNRecurrentPrefillRequest(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                state_cache=cache,
                cache_idx=0,
                slot_ids=list(range(num_reqs)),
                output_dtype=DTYPE,
                cu_seqlens=cu,
                compute_dtype=DTYPE,
                defer_state_scatter=True,
            )
            ms = bench(
                lambda req=req, cache=cache: (
                    kernels.try_recurrent_prefill(req),
                    cache.pending_recurrent_states[0],
                )
            )
            print(
                f"scan kernel    B={num_reqs} segT={t}    : {ms:7.3f} ms ({total} tok)"
            )

    # Conv state at spec spans is a tiny [B, T, conv_dim] window; its cost is
    # dominated by the recurrent scan above and is exercised end-to-end by the
    # parity tests, so it is not timed separately here.
    print("done")


if __name__ == "__main__":
    main()
