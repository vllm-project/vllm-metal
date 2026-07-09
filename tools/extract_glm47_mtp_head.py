#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract the GLM-4.7-Flash nextn (MTP) head into a standalone MLX checkpoint.

``zai-org/GLM-4.7-Flash`` ships one trained nextn layer as
``model.layers.<num_hidden_layers>.*`` (dedicated ``embed_tokens``, ``enorm`` /
``hnorm`` / ``eh_proj``, an MLA ``self_attn``, a 64-expert MoE + shared expert,
and a ``shared_head`` norm + untied head). The 4-bit ``mlx-community`` mirror
strips it, so the head must be extracted from the bf16 master. Only the 3 shards
that hold the nextn tensors are downloaded (~4 GB, not the full ~60 GB repo).

The extracted checkpoint uses the flat, post-``sanitize`` layout that
``vllm_metal.v1.mtp_heads.glm4_moe_lite_mtp.Glm4MoeLiteMTPModel`` loads:

    model.embed_tokens.weight        <- model.layers.<N>.embed_tokens.weight
    model.enorm/hnorm/eh_proj.weight <- model.layers.<N>.{enorm,hnorm,eh_proj}
    model.mtp_block.*                <- model.layers.<N>.{input_layernorm,
                                          post_attention_layernorm,self_attn.*,mlp.*}
    model.shared_head_norm.weight    <- model.layers.<N>.shared_head.norm.weight
    lm_head.weight                   <- model.layers.<N>.shared_head.head.weight

The shared ``convert_nextn_weights`` transform (absorbed-MLA ``kv_b_proj`` split +
MoE expert stacking) is applied so the model loads without further sanitizing.

Usage:
    # bf16 head
    python tools/extract_glm47_mtp_head.py --out ~/.cache/vllm-metal-dev/glm47-mtp-head/bf16

    # 4-bit (group size 64) head
    python tools/extract_glm47_mtp_head.py --q-bits 4 --q-group-size 64 \\
        --out ~/.cache/vllm-metal-dev/glm47-mtp-head/4bit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from vllm_metal.v1.mtp_heads.glm4_moe_lite_mtp import (
    GLM4_MOE_LITE_MTP_ARCHITECTURE,
    GLM4_MOE_LITE_MTP_MODEL_TYPE,
    Glm4MoeLiteMTPArgs,
    Glm4MoeLiteMTPHeadLoader,
    convert_nextn_weights,
)

DEFAULT_REPO = "zai-org/GLM-4.7-Flash"
_SHARED_HEAD_NORM = "shared_head.norm.weight"
_SHARED_HEAD_HEAD = "shared_head.head.weight"
_TOP_LEVEL = ("enorm.weight", "hnorm.weight", "eh_proj.weight")


def _hf_download(repo: str, filename: str, revision: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, revision=revision))


def _nextn_shards(index: dict[str, Any], nextn_prefix: str) -> tuple[list[str], int]:
    """Return the shard filenames that hold the nextn layer tensors + tensor count."""
    weight_map = index["weight_map"]
    nextn_keys = [k for k in weight_map if k.startswith(nextn_prefix)]
    if not nextn_keys:
        raise ValueError(
            f"No tensors with prefix {nextn_prefix!r} found in the weight index; "
            "does this repo ship a nextn layer?"
        )
    shards = sorted({weight_map[k] for k in nextn_keys})
    return shards, len(nextn_keys)


def _flat_name(rest: str) -> str:
    """Map a ``model.layers.<N>.`` suffix to the flat head layout."""
    if rest == "embed_tokens.weight":
        return "model.embed_tokens.weight"
    if rest in _TOP_LEVEL:
        return f"model.{rest}"
    if rest == _SHARED_HEAD_NORM:
        return "model.shared_head_norm.weight"
    if rest == _SHARED_HEAD_HEAD:
        return "lm_head.weight"
    # input_layernorm / post_attention_layernorm / self_attn.* / mlp.*
    return f"model.mtp_block.{rest}"


# Keys under the nextn prefix that are registered buffers, not trained
# parameters. Upstream skips them; they have no home in the flat head module, so
# a strict load would reject them if they leaked through.
_SKIP_BUFFER_SUFFIXES = ("rotary_emb.inv_freq",)


def _is_nonparameter_buffer(rest: str) -> bool:
    return any(rest.endswith(suffix) for suffix in _SKIP_BUFFER_SUFFIXES)


def _collect_flat_weights(
    repo: str,
    revision: str | None,
    shards: list[str],
    nextn_prefix: str,
) -> dict[str, mx.array]:
    from huggingface_hub import snapshot_download

    # One download for all nextn shards (allow_patterns accepts the exact shard
    # filenames) rather than a sequential per-shard hf_hub_download loop.
    snapshot_dir = Path(
        snapshot_download(repo, revision=revision, allow_patterns=shards)
    )
    flat: dict[str, mx.array] = {}
    skipped: list[str] = []
    for shard in shards:
        shard_weights = mx.load(str(snapshot_dir / shard))
        for key, tensor in shard_weights.items():
            if not key.startswith(nextn_prefix):
                continue
            rest = key[len(nextn_prefix) :]
            if _is_nonparameter_buffer(rest):
                skipped.append(key)
                continue
            flat_key = _flat_name(rest)
            # e_score_correction_bias is fp32 in the source shard and must stay
            # fp32 (upstream cast_predicate excludes it; vLLM keeps the noaux_tc
            # bias fp32). Everything else is cast to bf16.
            if "e_score_correction_bias" in flat_key:
                flat[flat_key] = tensor
            else:
                flat[flat_key] = tensor.astype(mx.bfloat16)
        # Drop the shard's lazy arrays we did not keep; keep peak memory low.
        del shard_weights
    if skipped:
        print(f"  skipped {len(skipped)} non-parameter buffer(s): {sorted(skipped)}")
    return flat


def _build_head_config(src_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(src_config)
    config["model_type"] = GLM4_MOE_LITE_MTP_MODEL_TYPE
    config["architectures"] = [GLM4_MOE_LITE_MTP_ARCHITECTURE]
    config["num_hidden_layers"] = 0
    config["num_nextn_predict_layers"] = 1
    config["n_predict"] = 1
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    return config


def _quant_predicate(path: str, module: nn.Module) -> bool:
    # Router gate weights stay full precision for routing stability; norms and
    # the MoE gate have no ``to_quantized`` so this is belt-and-suspenders.
    if path.endswith(".mlp.gate"):
        return False
    return True


def _quantize(
    args: Glm4MoeLiteMTPArgs,
    weights: dict[str, mx.array],
    config: dict[str, Any],
    q_bits: int,
    q_group_size: int,
) -> tuple[dict[str, mx.array], dict[str, Any]]:
    from mlx_lm.utils import quantize_model

    model = _load_into_model(args, weights)
    _, quantized_config = quantize_model(
        model,
        config,
        group_size=q_group_size,
        bits=q_bits,
        quant_predicate=_quant_predicate,
    )
    mx.eval(model.parameters())
    quant_weights = dict(tree_flatten(model.parameters()))
    config = dict(config)
    config["quantization"] = quantized_config["quantization"]
    config["quantization_config"] = quantized_config["quantization"]
    return quant_weights, config


def _load_into_model(
    args: Glm4MoeLiteMTPArgs,
    weights: dict[str, mx.array],
) -> Any:
    from vllm_metal.v1.mtp_heads.glm4_moe_lite_mtp import Glm4MoeLiteMTPModel

    model = Glm4MoeLiteMTPModel(args)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def _write_checkpoint(
    out_dir: Path,
    weights: dict[str, mx.array],
    config: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "model.safetensors"), weights)
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.glob("*") if f.is_file()) / (1024 * 1024)


def _smoke_test(out_dir: Path, src_config: dict[str, Any], seed: int) -> None:
    """Load through the head loader and run one dummy forward, failing loudly."""
    from mlx_lm.models.cache import KVCache

    spec = SimpleNamespace(
        method="mtp",
        draft_model_config=SimpleNamespace(
            model=str(out_dir),
            revision=None,
            hf_config=SimpleNamespace(model_type=GLM4_MOE_LITE_MTP_MODEL_TYPE),
        ),
    )
    loader = Glm4MoeLiteMTPHeadLoader(model_path_resolver=lambda name: name)
    runtime = loader.load_if_needed(speculative_config=spec, target_config=src_config)
    model = runtime.model

    mx.random.seed(seed)
    num_slots = 4
    hidden = int(src_config["hidden_size"])
    vocab = int(src_config["vocab_size"])
    token_ids = mx.array([3, 5, 7, 11])
    hidden_rows = mx.random.normal((num_slots, hidden)).astype(mx.bfloat16)

    cache = KVCache()
    x = model.build_slot_inputs(token_ids, hidden_rows, first_position=0)
    h = model.forward_slots(x, cache)
    logits = model.compute_logits(h)
    mx.eval(logits)

    if tuple(logits.shape) != (num_slots, vocab):
        raise RuntimeError(
            f"Smoke test logits shape {tuple(logits.shape)} != {(num_slots, vocab)}"
        )
    argmax = mx.argmax(logits, axis=-1).tolist()
    print(f"  smoke forward: logits shape {tuple(logits.shape)}, argmax {argmax}")


def _report(out_dir: Path, weights: dict[str, mx.array]) -> None:
    lm_head = weights.get("lm_head.weight")
    lm_head_shape = tuple(lm_head.shape) if lm_head is not None else None
    print(f"  wrote {len(weights)} tensors to {out_dir}")
    print(f"  lm_head.weight shape: {lm_head_shape}")
    print(f"  on-disk size: {_dir_size_mb(out_dir):.1f} MB")


def extract(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).expanduser()
    print(f"Extracting GLM nextn head from {args.repo} -> {out_dir}")

    config_path = _hf_download(args.repo, "config.json", args.revision)
    src_config = json.loads(config_path.read_text(encoding="utf-8"))
    nextn_idx = int(src_config["num_hidden_layers"])
    nextn_prefix = f"model.layers.{nextn_idx}."

    index_path = _hf_download(args.repo, "model.safetensors.index.json", args.revision)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards, num_nextn_tensors = _nextn_shards(index, nextn_prefix)
    print(f"  nextn layer {nextn_idx}: {num_nextn_tensors} tensors in shards {shards}")

    flat = _collect_flat_weights(args.repo, args.revision, shards, nextn_prefix)
    head_args = Glm4MoeLiteMTPArgs.from_dict(_build_head_config(src_config))
    weights = convert_nextn_weights(flat, head_args)
    mx.eval(list(weights.values()))
    del flat

    config = _build_head_config(src_config)
    if args.q_bits is not None:
        print(f"  quantizing: {args.q_bits}-bit, group size {args.q_group_size}")
        weights, config = _quantize(
            head_args, weights, config, args.q_bits, args.q_group_size
        )

    _write_checkpoint(out_dir, weights, config)
    _report(out_dir, weights)
    print("  running smoke test...")
    _smoke_test(out_dir, src_config, args.seed)
    print("Done.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", required=True, help="Output checkpoint directory")
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help=f"Source repo (default: {DEFAULT_REPO})"
    )
    parser.add_argument("--revision", default=None, help="Source repo revision")
    parser.add_argument(
        "--q-bits",
        type=int,
        default=None,
        help="Quantize to this many bits (omit for bf16)",
    )
    parser.add_argument(
        "--q-group-size", type=int, default=64, help="Quantization group size"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the smoke-forward dummy inputs"
    )
    return extract(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
