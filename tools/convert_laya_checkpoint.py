# SPDX-License-Identifier: Apache-2.0
"""Flatten an original ModernBERT Laya checkpoint for Metal token pooling."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def convert_checkpoint(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    encoder = json.loads((source / "encoder/config.json").read_text())
    agent = json.loads((source / "rl_agent_config.json").read_text())
    if encoder["model_type"] != "modernbert":
        raise NotImplementedError("Metal Laya currently supports ModernBERT only.")
    # Validate required files before creating the output directory.
    weight_file = source / "model.safetensors"
    if not weight_file.is_file():
        raise FileNotFoundError(weight_file)
    tokenizer_files = list((source / "tokenizer").iterdir())
    destination.mkdir(parents=True)
    for file in tokenizer_files:
        if file.is_file():
            shutil.copyfile(file, destination / file.name)
    tokenizer_config = destination / "tokenizer_config.json"
    config = json.loads(tokenizer_config.read_text())
    if config.get("tokenizer_class") in (None, "TokenizersBackend"):
        config["tokenizer_class"] = "PreTrainedTokenizerFast"
        config.pop("backend", None)
        config.pop("is_local", None)
    if isinstance(config.get("extra_special_tokens"), list):
        config["extra_special_tokens"] = {
            f"extra_{i}": token
            for i, token in enumerate(config["extra_special_tokens"])
        }
    tokenizer_config.write_text(json.dumps(config, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(destination)
    qtype_ids = [
        tokenizer(f"{name} question: ", add_special_tokens=False)["input_ids"][0]
        for name in ("choice", "score", "noul")
    ]
    if tokenizer.mask_token_id is None or len(set(qtype_ids)) != 3:
        raise ValueError("Laya requires a MASK token and three distinct type tokens.")
    encoder["architectures"] = ["LayaForDecision"]
    encoder["dtype"] = "float32"
    encoder["laya_config"] = {
        key: agent[key]
        for key in (
            "head_layers",
            "max_len",
            "head_max_len",
            "act_costs",
            "temperature",
            "temperature_by_options",
        )
        if key in agent
    }
    encoder["laya_config"].update(
        mask_token_id=tokenizer.mask_token_id, qtype_token_ids=qtype_ids
    )
    (destination / "config.json").write_text(json.dumps(encoder, indent=2) + "\n")
    try:
        os.link(weight_file.resolve(), destination / weight_file.name)
    except OSError:
        shutil.copyfile(weight_file, destination / weight_file.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", help="Original local checkpoint directory or HF repo ID"
    )
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--revision", help="HF checkpoint revision")
    args = parser.parse_args()
    source = Path(args.model)
    if not source.exists():
        source = Path(
            snapshot_download(
                args.model,
                revision=args.revision,
                allow_patterns=[
                    "encoder/config.json",
                    "rl_agent_config.json",
                    "model.safetensors",
                    "tokenizer/*",
                ],
            )
        )
    convert_checkpoint(source, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
