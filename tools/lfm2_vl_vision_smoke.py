# SPDX-License-Identifier: Apache-2.0
"""LFM2-VL image smoke test against the real engine and mlx-vlm.

Starts ``vllm.LLM`` with the Metal plugin on an LFM2-VL checkpoint, sends one
chat request per image, and checks three things:

1. **Parity** - the engine's first-token log-probs match native mlx-vlm's for
   the same image and prompt (EXACT top-1, plus TOP_K_MATCH membership).
2. **The image is actually read** - two different images must not produce the
   same distribution.  Before image support existed the engine accepted the
   image, dropped it, and answered from the text alone; that failure is silent,
   so parity alone would not catch a regression back to it.
3. **Generation runs** - a short greedy completion returns non-empty text.

Prints ``SMOKE PASS`` on success.  Run from the repository root:

    PYTORCH_ENABLE_MPS_FALLBACK=0 python tools/lfm2_vl_vision_smoke.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Generic over two log-prob dicts despite the module name; not Gemma 4 specific.
from gemma4_mask_modes import kl_over_support  # noqa: E402

# The MLX conversions are packaged for mlx-vlm rather than Transformers: the
# 450M declares model_type "lfm2-vl", which Transformers does not register,
# and the 3B names an image processor class Transformers does not have.  The
# 1.6B checkpoint is packaged for Transformers and loads.
DEFAULT_CHECKPOINT = "LiquidAI/LFM2.5-VL-1.6B"
PROMPT = "Describe the image."
# AGENTS.md requires --top-k 5 for model parity.
TOP_K = 5

# bf16 noise between the Metal paged path and MLX's own SDPA.  The Gemma 4
# smoke uses the same scale for the same comparison.
KL_TOLERANCE = 5e-3
# Two different images must separate by more than the noise above, or the
# engine is not reading them.
KL_DISTINCT_MIN = 1e-2


def _image(seed: int, size: tuple[int, int] = (448, 336)) -> Image.Image:
    """A deterministic image: smooth colour ramps, not noise.

    Random pixels give a near-flat first-token distribution, which makes the
    top-1 comparison a coin flip; structured content does not.
    """
    width, height = size
    ys, xs = np.mgrid[0:height, 0:width]
    rng = np.random.default_rng(seed)
    phase = rng.integers(0, 255, size=3)
    red = (xs * 255 // width + phase[0]) % 256
    green = (ys * 255 // height + phase[1]) % 256
    blue = ((xs + ys) * 255 // (width + height) + phase[2]) % 256
    return Image.fromarray(np.stack([red, green, blue], axis=-1).astype(np.uint8))


def _content(image: Image.Image, text: str) -> list[dict[str, object]]:
    return [{"type": "image_pil", "image_pil": image}, {"type": "text", "text": text}]


def _start(checkpoint: str):
    from vllm import LLM

    return LLM(
        model=checkpoint,
        max_model_len=4096,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.3,
        limit_mm_per_prompt={"image": 1},
    )


def _engine_logprobs(llm, image: Image.Image) -> dict[int, float]:
    """Top-K first-token log-probs from the engine for one image."""
    from vllm import SamplingParams

    outputs = llm.chat(
        [{"role": "user", "content": _content(image, PROMPT)}],
        sampling_params=SamplingParams(max_tokens=1, temperature=0, logprobs=TOP_K),
    )
    return {tid: lp.logprob for tid, lp in outputs[0].outputs[0].logprobs[0].items()}


def _engine_text(llm, image: Image.Image) -> str:
    from vllm import SamplingParams

    outputs = llm.chat(
        [{"role": "user", "content": _content(image, PROMPT)}],
        sampling_params=SamplingParams(max_tokens=16, temperature=0),
    )
    return outputs[0].outputs[0].text


def _reference_logprobs(checkpoint: str, image: Image.Image) -> dict[int, float]:
    """Top-K first-token log-probs from native mlx-vlm for the same request."""
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs

    model, processor = load(checkpoint)
    prompt = apply_chat_template(processor, model.config, PROMPT, num_images=1)
    inputs = prepare_inputs(
        processor,
        images=[image],
        prompts=prompt,
        image_token_index=None,
        add_special_tokens=False,
    )
    # mlx-vlm 0.7.6 takes (input_ids, pixel_values, mask, cache=None, **kwargs);
    # the mask is positional, so it cannot ride in kwargs.
    kwargs = {
        key: value
        for key, value in inputs.items()
        if key not in ("input_ids", "pixel_values", "attention_mask")
    }
    out = model(
        inputs["input_ids"],
        inputs.get("pixel_values"),
        inputs.get("attention_mask"),
        **kwargs,
    )
    logits = out.logits if hasattr(out, "logits") else out
    row = logits[0, -1].astype(mx.float32)
    logprobs = row - mx.logsumexp(row)
    top = mx.argpartition(-logprobs, TOP_K)[:TOP_K]
    return {int(i): float(logprobs[i]) for i in top.tolist()}


def _top1(dist: dict[int, float]) -> int:
    return max(dist, key=dist.get)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    image_a, image_b = _image(1), _image(2)
    failures: list[str] = []

    print(f"checkpoint: {args.checkpoint}")
    llm = _start(args.checkpoint)

    # 3. Generation runs at all.
    text = _engine_text(llm, image_a)
    print(f"generation: {text!r}")
    if not text.strip():
        failures.append("engine returned empty text for an image request")

    engine_a = _engine_logprobs(llm, image_a)
    engine_b = _engine_logprobs(llm, image_b)

    # 2. The image reaches the model.
    kl_images = kl_over_support(engine_a, engine_b)
    print(f"distinct images: KL(A||B)={kl_images}")
    if kl_images is None or kl_images < KL_DISTINCT_MIN:
        failures.append(
            f"two different images gave the same distribution (KL={kl_images}); "
            "the image is probably being dropped"
        )

    # 1. Parity against mlx-vlm.
    reference = _reference_logprobs(args.checkpoint, image_a)
    kl_parity = kl_over_support(engine_a, reference)
    engine_top1, reference_top1 = _top1(engine_a), _top1(reference)
    exact = int(engine_top1 == reference_top1)
    top_k_match = int(engine_top1 in reference)
    print(
        f"parity: KL(engine||mlx-vlm)={kl_parity} "
        f"engine_top1={engine_top1} reference_top1={reference_top1}"
    )
    print(f"EXACT={exact}/1  TOP_K_MATCH={top_k_match}/1  (top_k={TOP_K})")
    if not top_k_match:
        failures.append("engine top-1 is not in the mlx-vlm top-K")
    if kl_parity is None or kl_parity > KL_TOLERANCE:
        failures.append(f"KL(engine||mlx-vlm)={kl_parity} over {KL_TOLERANCE}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
