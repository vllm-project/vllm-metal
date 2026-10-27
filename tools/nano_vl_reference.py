# SPDX-License-Identifier: Apache-2.0
"""Fresh Nano VL V2 greedy reference using MLX-VLM and NVIDIA preprocessing.

The Metal adapter, processor, and paged runtime are intentionally not used.
Only checkpoint loader compatibility is shared. All inputs must
already be local; the processor source is verified before it is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
from importlib.machinery import SourceFileLoader
from importlib.metadata import version
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_images(case: dict, fixture: dict) -> list:
    from PIL import Image, ImageDraw

    images = []
    for name in case["images"]:
        spec = fixture["images"][name]
        image = Image.new("RGB", tuple(spec["size"]), tuple(spec["color"]))
        for rectangle in spec.get("rectangles", []):
            ImageDraw.Draw(image).rectangle(
                rectangle["box"], fill=tuple(rectangle["color"])
            )
        images.append(image)
    return images


def load_processor(source: Path, expected_sha256: str):
    if sha256(source) != expected_sha256:
        raise ValueError("NVIDIA processor source hash does not match the pin")
    # HF snapshot symlinks resolve to content-addressed blobs without a suffix.
    # Select the Python loader explicitly, after checking the source hash.
    spec = importlib.util.spec_from_file_location(
        "nvidia_vl_v2_reference",
        source,
        loader=SourceFileLoader("nvidia_vl_v2_reference", str(source)),
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the verified NVIDIA processor source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_input(case: dict, fixture: dict, tokenizer, config: dict, nvidia):
    """Return independent prompt IDs and per-image BF16 PyTorch pixels."""
    import torch
    from torchvision.transforms import ToTensor

    images = make_images(case, fixture)
    mean = torch.tensor(config["norm_mean"]).reshape(1, 3, 1, 1)
    std = torch.tensor(config["norm_std"]).reshape(1, 3, 1, 1)
    pixels = []
    for image in images:
        tiles = nvidia.dynamic_preprocess(
            ToTensor()(image),
            image_size=512,
            max_num_tiles=fixture["max_num_tiles"],
            use_thumbnail=True,
        )
        pixels.append(((torch.stack(tiles) - mean) / std).bfloat16())
    rendered = tokenizer.apply_chat_template(
        case["messages"], tokenize=False, add_generation_prompt=True
    )
    parts = rendered.split("<image>")
    if len(parts) != len(pixels) + 1:
        raise ValueError("Image markers and image payloads do not agree")
    expanded = parts[0]
    for tensor, suffix in zip(pixels, parts[1:], strict=True):
        expanded += "<img>" + "<image>" * (len(tensor) * 256) + "</img>" + suffix
    ids = tokenizer.encode(expanded, add_special_tokens=False)
    if len(ids) + fixture["max_tokens"] > fixture["max_model_len"]:
        raise ValueError("Reference prompt plus output exceeds the context limit")
    return images, pixels, rendered, expanded, ids


def mlx_pixels(pixels: list):
    import mlx.core as mx

    return (
        mx.concatenate(
            [mx.array(tensor.float().numpy()).astype(mx.bfloat16) for tensor in pixels]
        )
        if pixels
        else None
    )


def greedy_tokens(
    model,
    ids: list[int],
    pixels,
    *,
    max_tokens: int,
    seed: int,
    eos_token_ids: list[int],
    record_logits=None,
) -> list[int]:
    """Decode with a fresh hybrid cache; retain terminal EOS in the result."""
    import mlx.core as mx

    cache = model.language_model.make_cache()
    mx.random.seed(seed)
    current = mx.array([ids], dtype=mx.int32)
    tokens = []
    for step in range(max_tokens):
        result = model(current, pixel_values=pixels if step == 0 else None, cache=cache)
        logits = result.logits[:, -1, :]
        if record_logits is not None:
            record_logits(current, logits)
        token = int(mx.argmax(logits, axis=-1).item())
        tokens.append(token)
        if token in eos_token_ids:
            break
        current = mx.array([[token]], dtype=mx.int32)
    return tokens


def run_reference(model_path: Path, fixture_path: Path, source: Path, output: Path):
    fixture = json.loads(fixture_path.read_text())
    reference = fixture["reference"]
    for name, value in reference["environment"].items():
        if os.environ.get(name) != value:
            raise ValueError(f"Reference requires explicit {name}={value}")
    for name, expected in fixture["model"]["files_sha256"].items():
        if sha256(model_path / name) != expected:
            raise ValueError(f"Checkpoint mismatch: {name}")
    for package in ("mlx", "mlx-vlm"):
        if version(package) != reference[package.replace("-", "_")]:
            raise ValueError(f"Reference requires the pinned {package} version")
    if output.exists():
        raise ValueError("Refusing to overwrite a previous reference")
    nvidia = load_processor(source, reference["nvidia_processor_sha256"])

    import mlx.core as mx
    from mlx_vlm import utils
    from transformers import AutoTokenizer

    from vllm_metal.compat import _patch_mlx_vlm_nemotron_nano_vl_v2

    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("The pinned reference requires Metal")
    _patch_mlx_vlm_nemotron_nano_vl_v2()
    model = utils.load_model(model_path, lazy=False, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    config = json.loads((model_path / "config.json").read_text())
    eos = config["eos_token_id"]
    eos = [eos] if isinstance(eos, int) else eos
    cases = {}
    for case in fixture["cases"]:
        _, tensors, prompt, _, ids = prepare_input(
            case, fixture, tokenizer, config, nvidia
        )
        pixels = mlx_pixels(tensors)
        tokens = greedy_tokens(
            model,
            ids,
            pixels,
            max_tokens=fixture["max_tokens"],
            seed=fixture["seed"],
            eos_token_ids=eos,
        )
        cases[case["name"]] = {
            "prompt": prompt,
            "prompt_token_ids": ids,
            "output_token_ids": tokens,
            "finish_reason": "stop" if tokens[-1] in eos else "length",
        }
        del tensors, pixels
        mx.clear_cache()
    report = {
        "fixture_sha256": sha256(fixture_path),
        "reference_source_sha256": sha256(Path(__file__)),
        "model": fixture["model"],
        "nvidia_processor_sha256": sha256(source),
        "environment": reference["environment"],
        "packages": {
            name: version(name)
            for name in ("mlx", "mlx-vlm", "torch", "torchvision", "transformers")
        },
        "machine": platform.platform(),
        "strict_load": True,
        "cases": cases,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--processor-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_reference(args.model, args.fixture, args.processor_source, args.output)
