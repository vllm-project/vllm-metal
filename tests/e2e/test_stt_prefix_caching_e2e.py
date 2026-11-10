# SPDX-License-Identifier: Apache-2.0
"""End-to-end: an identical repeat of a transcription request is served.

vLLM enables prefix caching for Qwen3-ASR by default (to vLLM it is a
decoder-only model). A repeat of an earlier request finds its prompt blocks
cached, and since vLLM 0.29 ``NewRequestData`` strips the audio features of
every placeholder inside those blocks (``strip_covered_mm_data``). The
one-shot STT runner keeps no KV cache and transcribes each request from its
audio, so the repeat reached it without features, raised, and killed the
engine.

The audio is three seconds of seeded noise: its 39 audio tokens end inside
the first three 16-token blocks of the 49-token prompt, which is what a
repeat finds cached. The test checks that geometry before relying on it.

The LLM body runs in a spawned child process, as in
``test_paged_prefix_caching_e2e.py``: Metal is not fork-safe.
"""

from __future__ import annotations

import os

import pytest

MODEL_NAME = "Qwen/Qwen3-ASR-0.6B"
SAMPLE_RATE = 16000
AUDIO_SECONDS = 3.0


def _run_repeated_transcription() -> None:
    """Body of the e2e test — runs in a spawned child process."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import numpy as np
    import torch
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )

    from vllm_metal.stt.qwen3_asr.adapter import Qwen3ASRRuntimeAdapter

    llm = LLM(
        model=MODEL_NAME,
        max_model_len=2048,
        limit_mm_per_prompt={"audio": 1},
    )
    rng = np.random.default_rng(0)
    audio = 0.1 * rng.standard_normal(int(SAMPLE_RATE * AUDIO_SECONDS))
    placeholder = Qwen3ASRForConditionalGeneration.get_placeholder_str("audio", 0)
    prompt = {
        "prompt": (
            f"<|im_start|>user\n{placeholder}<|im_end|>\n<|im_start|>assistant\n"
        ),
        "multi_modal_data": {"audio": (audio.astype(np.float32), SAMPLE_RATE)},
    }
    sp = SamplingParams(temperature=0, max_tokens=64)

    # Installed after the engine is built, so the warm-up encode is not seen.
    encoded: list[torch.Tensor] = []
    orig_extract = Qwen3ASRRuntimeAdapter.extract_audio_features

    def spy_extract(self, input_features):
        encoded.append(torch.as_tensor(input_features).clone())
        return orig_extract(self, input_features)

    Qwen3ASRRuntimeAdapter.extract_audio_features = spy_extract

    try:
        first = llm.generate(prompt, sp, use_tqdm=False)[0]

        # vLLM never serves the last prompt token from the cache, so a repeat
        # hits at most the full blocks before it.
        prompt_ids = list(first.prompt_token_ids)
        audio_token_id = llm.get_tokenizer().convert_tokens_to_ids("<|audio_pad|>")
        audio_end = 1 + max(i for i, t in enumerate(prompt_ids) if t == audio_token_id)
        block_size = llm.llm_engine.vllm_config.cache_config.block_size
        cacheable = (len(prompt_ids) - 1) // block_size * block_size
        if audio_end > cacheable:
            raise AssertionError(
                f"The audio span ends at token {audio_end}, past the {cacheable} "
                f"tokens a repeat could find cached; pick another AUDIO_SECONDS"
            )

        repeat = llm.generate(prompt, sp, use_tqdm=False)[0]
    finally:
        Qwen3ASRRuntimeAdapter.extract_audio_features = orig_extract

    if len(encoded) != 2:
        raise AssertionError(
            f"The runner should encode audio for both requests, but encoded "
            f"{len(encoded)} time(s)"
        )
    if not torch.equal(encoded[0], encoded[1]):
        raise AssertionError("The repeat was encoded from different audio features")
    first_ids = list(first.outputs[0].token_ids)
    repeat_ids = list(repeat.outputs[0].token_ids)
    if first_ids != repeat_ids:
        raise AssertionError(
            f"The repeat produced different tokens: first {first_ids}, "
            f"repeat {repeat_ids}"
        )


@pytest.mark.slow
@pytest.mark.network
def test_repeated_transcription_is_served_from_its_audio(run_in_spawn_process) -> None:
    run_in_spawn_process(_run_repeated_transcription)
