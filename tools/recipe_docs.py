# SPDX-License-Identifier: Apache-2.0
"""Render the editor's picks from local YAML during the MkDocs build."""

import shlex
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined

RECIPES = Path(__file__).resolve().parents[1] / "recipes"


def _command(model_id: str, args: list[str]) -> str:
    lines = [["vllm", "serve", model_id], []]
    for arg in args:
        if arg in {"--enable-auto-tool-choice", "--reasoning-parser"}:
            lines.append([])
        lines[-1].append(arg)
    return " \\\n  ".join(shlex.join(line) for line in lines)


def _option(args, flag, default=None):
    for i in range(len(args) - 1, -1, -1):
        if args[i] == flag:
            return args[i + 1]
    return default


def on_config(config):
    config.watch.append(str(RECIPES))
    return config


def on_page_markdown(markdown, *, page, config, files):
    if page.file.src_uri != "recipes.md":
        return markdown

    picks = yaml.safe_load((RECIPES / "featured.yaml").read_text())["picks"]
    for pick in picks:
        recipe = yaml.safe_load(
            (RECIPES / "overlays" / f"{pick['recipe']}.yaml").read_text()
        )
        variant = recipe["variants"][pick["variant"]]
        model_id = variant["model_id"]
        args = variant["base_args"] + variant.get("extra_args", [])
        for feature in pick["features"]:
            if feature not in variant["supported_features"]:
                raise ValueError(f"{pick['recipe']}: unsupported feature {feature}")
            args += recipe["features"][feature]["hardware_overrides"]["metal"]["args"]
        context = _option(args, "--max-model-len")
        if context is None:
            raise ValueError(f"{pick['recipe']}: missing --max-model-len")
        host = _option(args, "--host", "127.0.0.1")
        port = _option(args, "--port", "8000")
        hardware = f"apple_{pick['memory_tier_gb']}gb"
        notes = [
            note["body"].rstrip()
            for note in recipe.get("notes", [])
            if hardware in note["when"]["hardware"]
            and (
                not note["when"].get("variants")
                or pick["variant"] in note["when"]["variants"]
            )
        ]
        pick.update(
            {
                "model_id": model_id,
                "description": variant["description"],
                "command": _command(model_id, args),
                "guide": "\n\n".join(notes),
                "base_url": f"http://{host}:{port}",
                "context": context,
                "batch_tokens": _option(args, "--max-num-batched-tokens"),
            }
        )
    env = Environment(
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.from_string(markdown).render(picks=picks)
