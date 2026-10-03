# SPDX-License-Identifier: Apache-2.0
"""Compare DSpark paged proposals to the native dense forward at the same dtype.

This isolates cache/attention changes; it does not qualify cross-backend reduced
precision or generated-sequence/serving equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

if __name__ == "__main__":
    os.environ["MLX_ENABLE_TF32"] = "0"

import mlx.core as mx
import numpy as np
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

from tools.attention_bench_utils import native_source_hashes, package_versions
from tools.dspark_parity import capture_samples, check_tokens
from vllm_metal.attention.block_tables import build_block_tables
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.v1.block_draft_paged import BlockDraftPagedCache
from vllm_metal.v1.dflash import DFlashTargetCapture
from vllm_metal.v1.draft_checkpoint import load_draft_weights
from vllm_metal.v1.dspark import DSparkConfig, load_dspark
from vllm_metal.v1.dspark_paged import DSparkPagedCache
from vllm_metal.v1.proposer import validate_scheduler_blocks

NATIVE_SOURCES = (
    load_dspark,
    load_draft_weights,
    DFlashTargetCapture.run,
    BlockDraftPagedCache.compile_block,
    DSparkPagedCache.compile_draft,
    capture_samples,
    KVCacheStorage.__init__,
    MetalPagedKVCache.__init__,
    build_block_tables,
    validate_scheduler_blocks,
)


def compare_outputs(actual, expected, *, atol, rtol):
    """Record every failed gate, including near ties, without stopping the matrix."""
    checks = {}
    for name, observed, reference in zip(
        ("logits", "confidence"), actual[1:], expected[1:], strict=True
    ):
        if observed is None or reference is None:
            checks[name] = {"passed": observed is None and reference is None}
            continue
        a, b = (np.array(x.astype(mx.float32)) for x in (observed, reference))
        if (
            a.shape != b.shape
            or not a.size
            or not np.isfinite(a).all()
            or not np.isfinite(b).all()
        ):
            checks[name] = {"passed": False, "error": "Incomplete/non-finite output"}
            continue
        checks[name] = {
            "passed": bool(np.allclose(a, b, atol=atol, rtol=rtol)),
            "max_abs_error": float(np.max(np.abs(a - b))),
        }
    try:
        if (
            not actual[0].size
            or actual[0].shape != actual[1].shape[:2]
            or expected[0].shape != expected[1].shape[:2]
        ):
            raise AssertionError("Incomplete DSpark proposal IDs")
        check_tokens(
            actual[0],
            torch.from_numpy(np.array(expected[0])),
            actual[1],
            torch.from_numpy(np.array(expected[1].astype(mx.float32))),
        )
        checks["tokens"] = {"passed": True}
    except AssertionError as exc:
        checks["tokens"] = {"passed": False, "error": str(exc)}
    return checks


def make_cache(model, max_length, block_size):
    cfg = model.config.backbone
    pages_per_row = (max_length + block_size - 1) // block_size
    num_blocks = pages_per_row * 2 + 1
    names = [f"dspark_layers.{i}.self_attn" for i in range(cfg.num_hidden_layers)]
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=cfg.num_key_value_heads,
        head_size=cfg.head_dim,
        dtype=torch.float16
        if model.embed_tokens.weight.dtype == mx.float16
        else torch.bfloat16,
    )
    size = num_blocks * spec.page_size_bytes
    storage = KVCacheStorage(
        KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_groups=[KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec)],
            kv_cache_tensors=[
                KVCacheTensor(
                    size=len(names) * size,
                    layers=names,
                    layer_stride=size,
                    block_stride=spec.page_size_bytes,
                )
            ],
            kv_cache_layout="LBNHC",
        )
    )
    for tensor in storage.tensors.values():
        tensor.fill_(float("nan"))
    tables = [list(range(1 + row, num_blocks, 2)) for row in range(2)]
    tables[1].reverse()
    return DSparkPagedCache(
        model, storage, tuple(names), max_model_len=max_length
    ), tables


def qualify(args):
    raw = json.loads((args.draft / "config.json").read_text())
    config = DSparkConfig.from_dict(raw)
    target = json.loads((args.target / "config.json").read_text())
    config.backbone.validate_target(target)
    samples = capture_samples(args.target, config.backbone, args.context_lengths)
    draft = load_dspark(args.draft, target_config=target)
    dtype = getattr(mx, args.dtype)
    draft.set_dtype(dtype)
    mx.eval(draft.parameters())
    width_max = config.backbone.block_size
    cache, tables = make_cache(
        draft, max(args.context_lengths) + width_max, args.cache_block_size
    )
    widths = sorted({1, min(3, width_max), width_max})
    compiled = {width: cache.compile_draft(num_draft_tokens=width) for width in widths}
    atol, rtol = (0.015, 0.02) if args.dtype == "float16" else (0.25, 0.02)
    rows = []
    for batch, length, anchor_values, feature_values in samples:
        anchors = mx.array(anchor_values)
        features = [mx.array(f).astype(dtype) for f in feature_values]
        cache.write_context(
            [f.reshape(-1, f.shape[-1]) for f in features],
            [(tables[row], 0, length) for row in range(batch)],
        )
        for width in widths:
            expected = draft.draft(anchors, features, num_draft_tokens=width)
            actual = compiled[width](
                anchors, [(tables[row], length) for row in range(batch)]
            )
            mx.eval(actual, expected, *cache.storage.buffers)
            checks = compare_outputs(actual, expected, atol=atol, rtol=rtol)
            passed = all(check["passed"] for check in checks.values())
            rows.append(
                {
                    "batch": batch,
                    "context_length": length,
                    "width": width,
                    "passed": passed,
                    "checks": checks,
                }
            )
            print(
                f"{'PASS' if passed else 'FAIL'} B={batch} context={length} K={width}",
                flush=True,
            )
    metal = Path(__file__).resolve().parents[1] / "vllm_metal/metal"
    return {
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "reference": "native dense DSpark at the same compute precision",
        "target": str(args.target.resolve()),
        "draft": str(args.draft.resolve()),
        "dtype": args.dtype,
        "cache_block_size": args.cache_block_size,
        "atol": atol,
        "rtol": rtol,
        "native_source_sha256": native_source_hashes(*NATIVE_SOURCES, qualify),
        "metal_source_sha256": {
            str(p.relative_to(metal)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(metal.rglob("*"))
            if p.suffix in (".metal", ".cpp", ".h")
        },
        "versions": package_versions("mlx", "mlx-lm", "vllm"),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target", "draft", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--cache-block-size", type=int, choices=[8, 16, 32], default=16)
    parser.add_argument(
        "--context-lengths", type=int, nargs="+", default=[1, 15, 16, 17, 65, 257, 1025]
    )
    args = parser.parse_args()
    if args.output.exists() or min(args.context_lengths) < 1:
        parser.error("Use a new output path and positive context lengths")
    result = qualify(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit("Paged DSpark qualification failed; see the recorded checks")


if __name__ == "__main__":
    main()
