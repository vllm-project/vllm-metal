#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate two-Mac JACCL collectives, activation transfers, and model parity.

Run once on each Mac with the same rank-ordered peer IPs and transport config:
    python tools/jaccl_pp_smoke.py --rank 0 --peer-ips IP0,IP1 --config pp.json
    python tools/jaccl_pp_smoke.py --rank 1 --peer-ips IP0,IP1 --config pp.json

No model downloads: parity uses a small, seeded Qwen3 model by default.
Use --model gpt-oss for alternating attention and routed experts. With that
model, VLLM_PP_LAYER_PARTITION=3,5 tests an asymmetric split at an odd layer.
Add --expert-parallel to keep every layer on both Macs and split the routed
experts instead of the pipeline stages. This checks correctness, not
performance or vLLM's Ray serving path.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import mlx.core as mx

from vllm_metal.distributed.pipeline import (
    PipelinedModel,
    PipelineGroup,
    apply_pipeline_split,
    pipeline_recv,
    pipeline_send,
)
from vllm_metal.distributed.transport import PipelineTransportConfig

ITERATIONS = 20
PARITY_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--peer-ips", required=True, help="rank-ordered IPv4s: IP0,IP1")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen3", "gpt-oss"), default="qwen3")
    parser.add_argument(
        "--expert-parallel",
        action="store_true",
        help="split routed experts across ranks instead of pipeline stages",
    )
    args = parser.parse_args()
    if args.expert_parallel and args.model != "gpt-oss":
        parser.error("--expert-parallel requires --model gpt-oss")
    return args


def load_config(args: argparse.Namespace) -> tuple[list[str], PipelineTransportConfig]:
    peers = [ip.strip() for ip in args.peer_ips.split(",")]
    if len(peers) != 2:
        raise ValueError("this smoke test requires exactly two peers")
    options = json.loads(args.config.read_text())
    config = PipelineTransportConfig.from_additional_config(options, 2)
    if config.backend != "jaccl":
        raise ValueError("this smoke test requires pipeline_transport.backend=jaccl")
    return peers, config


def send(array: mx.array, peer: int, pp: PipelineGroup) -> None:
    mx.eval(mx.distributed.send(array, peer, group=pp.group, stream=mx.cpu))


def recv(shape: tuple[int, ...], dtype: mx.Dtype, peer: int, pp: PipelineGroup):
    value = mx.distributed.recv(shape, dtype, peer, group=pp.group, stream=mx.cpu)
    mx.eval(value)
    return value


def agree_check(pp: PipelineGroup, error: str | None, label: str) -> None:
    """Send last-rank status back, then acknowledge the combined result.

    Both peers finish the data operation before this exchange. A comparison
    failure on rank 1 therefore reaches rank 0 before either exits.
    """
    if pp.rank == 1:
        send(mx.array([int(error is None)], dtype=mx.int32), 0, pp)
        passed = bool(recv((1,), mx.int32, 0, pp).item())
    else:
        peer_passed = bool(recv((1,), mx.int32, 1, pp).item())
        passed = peer_passed and error is None
        send(mx.array([int(passed)], dtype=mx.int32), 1, pp)
    if not passed:
        raise RuntimeError(f"{label}: {error or 'peer reported failure'}")


def check_all_sum(pp: PipelineGroup, summary: dict) -> None:
    base = mx.arange(257, dtype=mx.int32)
    for iteration in range(1, ITERATIONS + 1):
        values = base + (pp.rank + 1) * iteration
        summed = mx.distributed.all_sum(values, group=pp.group, stream=mx.cpu)
        mx.eval(summed)
        valid = bool(mx.all(summed == 2 * base + 3 * iteration).item())
        agree_check(pp, None if valid else "values differ", f"all_sum {iteration}")
        summary["checks_passed"] += 1
    summary["all_sum_iterations"] = ITERATIONS


def check_activations(pp: PipelineGroup, summary: dict) -> None:
    """Plugin forward handoff, then a changed payload returned over native JACCL.

    Pipeline helpers encode increasing stage order, so the return direction
    deliberately uses native send/recv with explicit peers.
    """
    cases = (("small", 1, 64), ("4MiB", 1024, 1024))
    summary["activation_cases"] = []
    for name, tokens, hidden in cases:
        shape = (1, tokens, hidden)
        base = (mx.arange(tokens * hidden) % 1021).astype(mx.float32).reshape(shape)
        for iteration in range(1, ITERATIONS + 1):
            payload = base + iteration * 7
            offset = iteration + 100
            if pp.rank == 0:
                # Retain the handle until peer receipt and the status exchange.
                send_handle = pipeline_send(payload, pp)
                mx.async_eval(send_handle)
                returned = recv(shape, mx.float32, 1, pp)
                valid = bool(mx.all(returned == payload + offset).item())
            else:
                received = pipeline_recv(pp, tokens, hidden, mx.float32)
                mx.eval(received)
                valid = bool(mx.all(received == payload).item())
                send(received + offset, 0, pp)
            agree_check(
                pp, None if valid else "values differ", f"{name} activation {iteration}"
            )
            summary["checks_passed"] += 1
        summary["activation_cases"].append(
            {
                "name": name,
                "bytes_each_direction": tokens * hidden * 4,
                "iterations": ITERATIONS,
                "forward": "plugin",
                "return": "native_jaccl",
            }
        )


def tiny_model(model_name: str = "qwen3"):
    if model_name == "gpt-oss":
        from gpt_oss_smoke_model import tiny_model as gpt_oss_model

        return gpt_oss_model()
    from mlx_lm.models.qwen3 import Model, ModelArgs

    mx.random.seed(20260917)
    model = Model(
        ModelArgs(
            model_type="qwen3",
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=97,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=10000.0,
            head_dim=16,
            tie_word_embeddings=False,
        )
    )
    mx.eval(model.parameters())
    return model


def parity_models(pp: PipelineGroup, batches: list[mx.array], model_name: str):
    """Build references before pipeline I/O so local setup failures can be shared."""
    from mlx_lm.models.cache import make_prompt_cache

    reference = []
    if pp.is_last:
        full = tiny_model(model_name)
        cache = make_prompt_cache(full)
        for batch in batches:
            logits = full(batch, cache=cache)
            mx.eval(logits)
            reference.append(logits)
    model = tiny_model(model_name)
    if model_name == "gpt-oss":
        # Honor explicit partitions: an odd boundary reverses the stage's
        # starting attention type and catches incorrect local cache indexing.
        span = apply_pipeline_split(model, pp)
    else:
        # Preserve the Qwen3 smoke's fixed split regardless of serving config.
        previous = os.environ.pop("VLLM_PP_LAYER_PARTITION", None)
        try:
            span = apply_pipeline_split(model, pp)
        finally:
            if previous is not None:
                os.environ["VLLM_PP_LAYER_PARTITION"] = previous
    local_cache = make_prompt_cache(model)
    if len(local_cache) != span[1] - span[0]:
        raise ValueError("stage cache count does not match its owned layers")
    return PipelinedModel(model, pp), local_cache, reference, span


def model_batches(model_name: str) -> list[tuple[str, mx.array]]:
    if model_name == "gpt-oss":
        from gpt_oss_smoke_model import parity_batches

        return parity_batches()
    batches = [("prefill", mx.array([[1, 7, 11, 23, 42, 3, 9, 17]], dtype=mx.int32))]
    batches.extend(
        (f"cached_decode_{step}", mx.array([[token]], dtype=mx.int32))
        for step, token in enumerate((19, 31, 5, 8), 1)
    )
    return batches


def check_model_parity(pp: PipelineGroup, summary: dict, model_name: str) -> None:
    batches = model_batches(model_name)
    setup_error = None
    prepared = None
    try:
        prepared = parity_models(pp, [batch for _, batch in batches], model_name)
    except Exception as exc:
        setup_error = f"{type(exc).__name__}: {exc}"
    agree_check(pp, setup_error, f"tiny {model_name} setup")
    assert prepared is not None
    model, cache, reference, span = prepared
    record = {"layers": span, "steps": [], "tolerance": PARITY_TOLERANCE}
    summary[f"tiny_{model_name.replace('-', '_')}"] = record
    if model_name == "gpt-oss":
        from gpt_oss_smoke_model import SLIDING_WINDOW

        record["sliding_window"] = SLIDING_WINDOW
    record["cache_types"] = [type(entry).__name__ for entry in cache]
    context_tokens = 0
    for step, (label, batch) in enumerate(batches):
        output = model(batch, cache=cache)
        context_tokens += batch.shape[1]
        error = None
        details = {
            "phase": label,
            "input_tokens": batch.shape[1],
            "context_tokens": context_tokens,
        }
        if pp.is_last:
            mx.eval(output)
            # Catch comparison errors after the receive has completed, so rank 0
            # can finish its send and receive this step's explicit status.
            try:
                expected = reference[step]
                difference = float(mx.max(mx.abs(output - expected)).item())
                argmax_equal = bool(
                    mx.all(
                        mx.argmax(output, axis=-1) == mx.argmax(expected, axis=-1)
                    ).item()
                )
                finite = bool(mx.all(mx.isfinite(output)).item())
                details.update(max_abs_diff=difference, argmax_equal=argmax_equal)
                if not finite or difference > PARITY_TOLERANCE or not argmax_equal:
                    error = f"logits differ: max_abs_diff={difference}, {argmax_equal=}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        else:
            send_handle = pipeline_send(output, pp)
            mx.async_eval(send_handle)
        offsets = [entry.offset for entry in cache]
        details["cache_offsets"] = offsets
        if any(offset != context_tokens for offset in offsets):
            error = error or f"cache offsets {offsets} differ from {context_tokens}"
        agree_check(pp, error, label)
        summary["checks_passed"] += 1
        record["steps"].append(details)


def check_expert_parity(
    peer_ips: list[str], config: PipelineTransportConfig, rank: int, summary: dict
) -> None:
    """Tiny GPT-OSS under expert parallelism: both ranks compare to reference."""
    from gpt_oss_smoke_model import parity_batches, tiny_model
    from mlx_lm.models.cache import make_prompt_cache

    from vllm_metal.distributed.experts import apply_expert_shard
    from vllm_metal.distributed.tensor import TensorGroup

    # mx.distributed.init returns the process-wide JACCL group, so this reuses
    # the connection the collective/activation checks already formed.
    tg = TensorGroup(config.bootstrap_jaccl(rank, peer_ips))
    batches = parity_batches()
    reference = []
    full = tiny_model()
    ref_cache = make_prompt_cache(full)
    for _, batch in batches:
        logits = full(batch, cache=ref_cache)
        mx.eval(logits)
        reference.append(logits)
    model = tiny_model()
    apply_expert_shard(model, tg)
    cache = make_prompt_cache(model)
    record = {"steps": [], "tolerance": PARITY_TOLERANCE}
    summary["tiny_gpt_oss_expert"] = record
    for step, (label, batch) in enumerate(batches):
        output = model(batch, cache=cache)
        # Collectives inside the sharded forward are lazy: evaluate before
        # comparing so both ranks contribute their expert partials.
        mx.eval(output)
        expected = reference[step]
        difference = float(mx.max(mx.abs(output - expected)).item())
        argmax_equal = bool(
            mx.all(mx.argmax(output, axis=-1) == mx.argmax(expected, axis=-1)).item()
        )
        error = (
            None
            if difference <= PARITY_TOLERANCE and argmax_equal
            else f"logits differ: max_abs_diff={difference}, {argmax_equal=}"
        )
        agree_check(tg, error, label)
        summary["checks_passed"] += 1
        record["steps"].append(
            {"phase": label, "max_abs_diff": difference, "argmax_equal": argmax_equal}
        )


def versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "macos": platform.mac_ver()[0]}
    for package in ("mlx", "mlx-lm", "vllm", "vllm-metal"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed as a distribution"
    return result


def main() -> int:
    args = parse_args()
    summary = {
        "rank": args.rank,
        "backend": "jaccl",
        "model": args.model,
        "world_size": 2,
        "forward_submission": "mx.async_eval",
        "checks_passed": 0,
        "passed": False,
        "versions": versions(),
    }
    try:
        peers, config = load_config(args)
        entry = config.device_matrix[args.rank][1 - args.rank]
        summary["rails"] = [entry] if isinstance(entry, str) else list(entry)
        summary["peer_ips"] = peers
        pp = PipelineGroup.bootstrap(args.rank, peers, config)
        print(
            f"Rank {pp.rank}: JACCL ready; validating collectives and activations",
            flush=True,
        )
        check_all_sum(pp, summary)
        check_activations(pp, summary)
        if args.expert_parallel:
            print(
                f"Rank {pp.rank}: transfers passed; checking expert-parallel parity",
                flush=True,
            )
            check_expert_parity(peers, config, args.rank, summary)
        else:
            print(
                f"Rank {pp.rank}: transfers passed; checking tiny {args.model} parity",
                flush=True,
            )
            check_model_parity(pp, summary, args.model)
        summary["passed"] = True
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
