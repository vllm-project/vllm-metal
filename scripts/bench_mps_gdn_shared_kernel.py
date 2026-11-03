#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manual GDN core benchmark; run only after acquiring the machine's load lock.

Build first (no editable install needed):
    python -c 'from vllm_metal.metal.build import build_mps, build_gdn_metallib; \
build_mps(); build_gdn_metallib()'
Run from the worktree with PYTORCH_ENABLE_MPS_FALLBACK=0:
    python scripts/bench_mps_gdn_shared_kernel.py

Inputs are synthetic. Only the local Qwen3.5-0.8B config is read. State resets,
head expansion and gate conversion are outside the timed recurrence. This is
a core microbenchmark, not an end-to-end serving measurement.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from functools import cache, partial
from itertools import accumulate
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
# A direct script invocation must use this worktree, not another editable tree.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@cache
def qwen_linear_dims() -> tuple[int, int, int, int]:
    snapshots = (
        Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots"
    )
    configs = sorted(snapshots.glob("*/config.json"))
    if not configs:
        return 16, 16, 128, 128
    config = json.loads(configs[-1].read_text())
    text = config.get("text_config", config)
    return tuple(
        int(text[key])
        for key in (
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
        )
    )


@cache
def reference_functions():
    """Load the exact PR #1241 core without vLLM/Transformers initialization.

    Compile the two unmodified function ASTs and their chunk-size constant
    directly from the worktree. No copied recurrence or installed vLLM core
    is used; the model module's serving imports are unnecessary for this PoC.
    """
    path = ROOT / "vllm_metal/pytorch_backend/models/qwen3_5.py"
    names = {"delta_rule_step", "delta_rule_chunked"}
    nodes = [
        node
        for node in ast.parse(path.read_text(), filename=str(path)).body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_CHUNK_SIZE" for t in node.targets
            )
        )
    ]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["delta_rule_step"], namespace["delta_rule_chunked"]


@dataclass
class GDNCase:
    lengths: tuple[int, ...]
    slots: tuple[int, ...]
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    state: torch.Tensor
    cu: torch.Tensor
    slot_mapping: torch.Tensor
    y: torch.Tensor
    state_storage: torch.Tensor

    def args(self):
        return (
            self.q,
            self.k,
            self.v,
            self.g,
            self.beta,
            self.state,
            self.cu,
            self.slot_mapping,
            self.y,
        )

    def reference_inputs(self):
        group = self.v.shape[1] // self.q.shape[1]
        q, k = (t.float().repeat_interleave(group, dim=1) for t in (self.q, self.k))
        # Compare identical quantized inputs: the shader takes decay, while
        # the reference takes log(decay). Its arithmetic/state stay fp32.
        return q, k, self.v.float(), self.g.float().log(), self.beta.float()


def make_case(
    lengths, slots, dtype, *, nonzero_state=True, padded_state=False, device="mps"
) -> GDNCase:
    lengths, slots = tuple(lengths), tuple(slots)
    hk, hv, dk, dv = qwen_linear_dims()
    total = sum(lengths)
    rng = torch.Generator().manual_seed(1241)

    def randn(*shape):
        return torch.randn(*shape, generator=rng)

    q, k = (
        torch.nn.functional.normalize(randn(total, hk, dk), dim=-1) for _ in range(2)
    )
    q = q * dk**-0.5
    v = randn(total, hv, dv) * 0.25
    g = (-0.01 - 0.2 * torch.rand(total, hv, generator=rng)).exp()
    beta = torch.rand(total, hv, generator=rng)
    q, k, v, g, beta = (t.to(device=device, dtype=dtype) for t in (q, k, v, g, beta))
    num_slots = max(slots) + 2  # Include an unused physical slot.
    row_size = hv * dv * dk
    # Prefix and row gaps catch negative-slot addressing and byte/element stride
    # confusion. Even a dense case has a nonzero storage offset.
    storage = (randn(num_slots + 1, row_size + (17 if padded_state else 0)) * 0.1).to(
        device
    )
    state = storage[1:, :row_size].view(num_slots, hv, dv, dk)
    if not nonzero_state:
        state.zero_()
    cu = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device)
    mapping = torch.tensor(slots, dtype=torch.int32, device=device)
    y = torch.full_like(v, 13.0)
    return GDNCase(lengths, slots, q, k, v, g, beta, state, cu, mapping, y, storage)


def reference_result(case: GDNCase, *, decode: bool):
    step, chunked = reference_functions()
    q, k, v, g, beta = case.reference_inputs()
    # This is a transpose, never a reshape: kernel [Hv,Dv,Dk] -> [Hv,Dk,Dv].
    state = case.state.transpose(-1, -2).contiguous().clone()
    y = case.y.float().clone()
    if decode:
        indices = case.slot_mapping.long()
        active = state[indices]
        y.copy_(step(q, k, v, g, beta, active))
        state[indices] = active
    else:
        start = 0
        for length, slot in zip(case.lengths, case.slots, strict=True):
            end = start + length
            if slot >= 0 and length:
                y[start:end], state[slot] = chunked(
                    q[start:end],
                    k[start:end],
                    v[start:end],
                    g[start:end],
                    beta[start:end],
                    state[slot],
                )
            start = end
    return y, state.transpose(-1, -2)


def measure_ms(run, reset, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        reset()
        run()
    torch.mps.synchronize()
    samples = []
    for _ in range(repeats):
        reset()
        torch.mps.synchronize()
        start = time.perf_counter()
        result = run()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
        del result
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=DTYPES, default="fp32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("Set PYTORCH_ENABLE_MPS_FALLBACK=0 before starting Python")
    if not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    from vllm_metal.pytorch_backend.mps_ops import gdn_linear_attention

    step, chunked = reference_functions()
    print(f"Qwen3.5-0.8B Hk,Hv,Dk,Dv={qwen_linear_dims()}; {args.dtype}; fp32 state")
    print("Median synchronized core latency; preparation and state resets excluded.")
    print("| Case | Kernel (ms) | PyTorch reference (ms) | Reference / kernel |")
    print("|---|---:|---:|---:|")
    with torch.inference_mode():
        for label, lengths in (("decode B=8", (1,) * 8), ("prefill T=512", (512,))):
            case = make_case(lengths, range(len(lengths)), DTYPES[args.dtype])
            initial = case.state.clone()
            ref_state = initial.transpose(-1, -2).contiguous()
            ref_initial = ref_state.clone()
            q, k, v, g, beta = case.reference_inputs()
            if len(lengths) == 8:
                # make_case reserves a final inactive slot; decode uses eight.
                run_ref = partial(step, q, k, v, g, beta, ref_state[:8])
            else:
                run_ref = partial(chunked, q, k, v, g, beta, ref_state[0])
            kernel_ms = measure_ms(
                partial(gdn_linear_attention, *case.args()),
                partial(case.state.copy_, initial),
                args.warmup,
                args.repeats,
            )
            reference_ms = measure_ms(
                run_ref,
                partial(ref_state.copy_, ref_initial),
                args.warmup,
                args.repeats,
            )
            print(
                f"| {label} | {kernel_ms:.4f} | {reference_ms:.4f} | "
                f"{reference_ms / kernel_ms:.2f}x |"
            )


if __name__ == "__main__":
    main()
