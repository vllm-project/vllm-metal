# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text model loading and orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vllm_metal.stt.config import (
    QWEN3_ASR_MAX_DECODE_TOKENS,
    SpeechToTextConfig,
)
from vllm_metal.stt.protocol import TranscriptionResult
from vllm_metal.stt.qwen3_asr import Qwen3ASRConfig, Qwen3ASRModel
from vllm_metal.stt.whisper import WhisperConfig, WhisperModel, WhisperTranscriber

# Tag used by Qwen3-ASR to wrap transcription text
_ASR_TEXT_TAG = "<asr_text>"

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    snapshot_download = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None  # type: ignore[assignment]

# Supported floating-point dtypes for STT model loading.
_SUPPORTED_LOAD_DTYPES = frozenset({mx.float16, mx.float32, mx.bfloat16})


# ===========================================================================
# Qwen3-ASR Transcriber
# ===========================================================================


class Qwen3ASRTranscriber:
    """Transcriber for Qwen3-ASR models.

    Handles prompt construction with audio_pad tokens,
    embedding injection, and greedy autoregressive decoding.

    Args:
        model: Loaded :class:`Qwen3ASRModel`.
        model_path: Path to model directory (for tokenizer loading).
    """

    def __init__(self, model, model_path: str | None = None):
        self.model = model
        self.model_path = model_path
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazily-loaded tokenizer (Qwen2Tokenizer via AutoTokenizer)."""
        if self._tokenizer is None:
            self._tokenizer = _load_qwen3_asr_tokenizer(self.model_path)
        return self._tokenizer

    def greedy_decode_tokens(
        self,
        audio_features: mx.array,
        prompt_token_ids: list[int],
        max_tokens: int | None = None,
    ) -> list[int]:
        """Greedy autoregressive decode with audio embedding injection.

        Args:
            audio_features: Audio embeddings from the encoder.
            prompt_token_ids: Full prompt with audio_pad placeholders.
            max_tokens: Maximum decode steps
                (default: :data:`QWEN3_ASR_MAX_DECODE_TOKENS`).

        Returns:
            Decoded token IDs (excluding prompt prefix, excluding EOS).
        """
        if max_tokens is None:
            max_tokens = QWEN3_ASR_MAX_DECODE_TOKENS

        if not prompt_token_ids:
            logger.warning("Empty prompt_token_ids; returning no tokens")
            return []

        eos_token = self.model.config.eos_token_id
        tokens = mx.array([prompt_token_ids], dtype=mx.int32)

        # Prefill with audio embedding injection
        logits, cache = self.model.prefill(tokens, audio_features)
        mx.eval(logits)

        output_tokens: list[int] = []
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        if next_token == eos_token:
            return output_tokens
        output_tokens.append(next_token)

        # Autoregressive decode
        for _ in range(max_tokens - 1):
            token_input = mx.array([[next_token]], dtype=mx.int32)
            logits, cache = self.model.decode_step(token_input, cache)
            mx.eval(logits)
            next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            if next_token == eos_token:
                break
            output_tokens.append(next_token)

        return output_tokens

    def build_prompt_tokens(self, n_audio_frames: int) -> list[int]:
        """Build prompt token IDs with audio placeholders.

        Format: ``<|im_start|>user\\n<|audio_start|>{N*audio_pad}<|audio_end|>\\n<|im_end|>\\n<|im_start|>assistant\\n``

        Args:
            n_audio_frames: Number of audio_pad tokens to insert.

        Returns:
            List of token IDs.
        """
        tok = self.tokenizer
        audio_pad_id = self.model.config.audio_token_id
        audio_start_id = self.model.config.audio_start_token_id
        audio_end_id = self.model.config.audio_end_token_id

        # Encode structural tokens
        im_start = tok.encode("<|im_start|>", add_special_tokens=False)
        im_end = tok.encode("<|im_end|>", add_special_tokens=False)
        user = tok.encode("user\n", add_special_tokens=False)
        assistant = tok.encode("assistant\n", add_special_tokens=False)
        newline = tok.encode("\n", add_special_tokens=False)

        prompt = (
            im_start
            + user
            + [audio_start_id]
            + [audio_pad_id] * n_audio_frames
            + [audio_end_id]
            + newline
            + im_end
            + newline
            + im_start
            + assistant
        )
        return prompt

    @staticmethod
    def post_process_output(text: str) -> str:
        """Strip ``language {lang}<asr_text>`` prefix and trailing tags."""
        if not text:
            return ""
        if _ASR_TEXT_TAG not in text:
            return text
        _, text_part = text.rsplit(_ASR_TEXT_TAG, 1)
        # Truncate at first special token marker
        for marker in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
            idx = text_part.find(marker)
            if idx >= 0:
                text_part = text_part[:idx]
        return text_part.strip()


# ===========================================================================
# Model loading
# ===========================================================================


def _read_config(model_path: Path) -> dict:
    """Read and return config.json from a model directory."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with open(config_path) as f:
        return json.load(f)


def _load_weights(model_path: Path) -> dict[str, mx.array]:
    """Load model weights from safetensors or npz files."""
    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        weight_files = sorted(model_path.glob("*.npz"))
    if not weight_files:
        raise FileNotFoundError(f"No weight files in {model_path}")

    weights: dict[str, mx.array] = {}
    for wf in weight_files:
        weights.update(mx.load(str(wf)))
    return weights


def _resolve_model_path(model_path: str | Path) -> Path:
    """Resolve model path, downloading from HF if needed."""
    model_path = Path(model_path)
    if not model_path.exists():
        if snapshot_download is None:
            raise ValueError(
                f"Could not download model {model_path}: huggingface_hub is not installed"
            )
        try:
            model_path = Path(snapshot_download(repo_id=str(model_path)))
        except OSError as e:
            raise ValueError(f"Could not download model: {model_path}") from e
    return model_path


def _validate_load_dtype(dtype: mx.Dtype) -> None:
    """Validate the floating-point dtype used for model loading."""
    if dtype not in _SUPPORTED_LOAD_DTYPES:
        names = ", ".join(sorted(str(d) for d in _SUPPORTED_LOAD_DTYPES))
        raise TypeError(
            f"Unsupported STT model dtype: {dtype!r}. Must be one of {names}."
        )


def load_model(model_path: str | Path, dtype: mx.Dtype = mx.float16):
    """Load an STT model from a local directory or HuggingFace repo.

    Auto-detects model type from config.json and dispatches to the
    appropriate loader (Whisper or Qwen3-ASR).

    Args:
        model_path: Local path or HuggingFace repo ID.
        dtype: Model dtype (default: float16).

    Returns:
        Loaded model ready for inference.

    Raises:
        ValueError: If the model type is unsupported or download fails.
        FileNotFoundError: If config.json or weight files are missing.
    """
    if isinstance(model_path, str) and not model_path.strip():
        raise ValueError(
            "model_path must be a non-empty local path or HuggingFace repo ID."
        )
    _validate_load_dtype(dtype)
    model_path = _resolve_model_path(model_path)
    config_dict = _read_config(model_path)
    model_type = config_dict.get("model_type", "").lower()

    if model_type == "qwen3_asr":
        return _load_qwen3_asr_model(model_path, config_dict, dtype)
    if model_type in ("", "whisper"):
        # Default to Whisper for backward compatibility
        return _load_whisper_model(model_path, config_dict, dtype)
    raise ValueError(
        f"Unsupported STT model_type: {model_type!r}. "
        "Expected 'whisper' or 'qwen3_asr'."
    )


def _load_and_init_model(model, model_path: Path, config_dict: dict):
    """Shared loader: quantize, sanitize, load weights, and eval.

    Args:
        model: Instantiated model with a ``sanitize`` method.
        model_path: Path to weight files.
        config_dict: Raw config.json dict (checked for ``quantization``).

    Returns:
        The model with weights loaded and evaluated.
    """
    weights = _load_weights(model_path)

    quantization = config_dict.get("quantization")
    if quantization is not None:

        def class_predicate(p, m):
            return isinstance(m, (nn.Linear, nn.Embedding)) and f"{p}.scales" in weights

        nn.quantize(model, **quantization, class_predicate=class_predicate)

    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model


def _load_whisper_model(
    model_path: Path, config_dict: dict, dtype: mx.Dtype
) -> WhisperModel:
    """Load a Whisper model from config and weights."""
    config = WhisperConfig.from_dict(config_dict)
    model = WhisperModel(config, dtype)
    return _load_and_init_model(model, model_path, config_dict)


def _load_qwen3_asr_model(model_path: Path, config_dict: dict, dtype: mx.Dtype):
    """Load a Qwen3-ASR model from config and weights."""
    config = Qwen3ASRConfig.from_dict(config_dict)
    model = Qwen3ASRModel(config, dtype)
    return _load_and_init_model(model, model_path, config_dict)


# ===========================================================================
# Module-level helpers
# ===========================================================================


def _load_qwen3_asr_tokenizer(model_path: str | None = None):
    """Load a Qwen3-ASR tokenizer (Qwen2Tokenizer via AutoTokenizer).

    Args:
        model_path: Local model directory with tokenizer files.

    Returns:
        A tokenizer instance.
    """
    if AutoTokenizer is None:
        raise ImportError("Qwen3-ASR tokenizer requires transformers to be installed.")

    if model_path:
        try:
            return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except (OSError, ValueError) as e:
            logger.debug("Local tokenizer load failed for %s: %s", model_path, e)

    raise ValueError(
        "Qwen3-ASR requires a local tokenizer. "
        "Provide a model_path with tokenizer files."
    )


# ===========================================================================
# Convenience entrypoint
# ===========================================================================


def transcribe(
    model: WhisperModel,
    audio: str | np.ndarray | mx.array,
    language: str | None = None,
    task: str = "transcribe",
    prompt: str | None = None,
    with_timestamps: bool = False,
    model_path: str | None = None,
    stt_config: SpeechToTextConfig | None = None,
) -> TranscriptionResult:
    """Transcribe audio to text (convenience wrapper).

    Creates a :class:`WhisperTranscriber` and delegates. For repeated
    calls, prefer constructing a transcriber directly to reuse the
    tokenizer.

    Args:
        model: Loaded :class:`WhisperModel`.
        audio: File path, numpy array, or MLX array.
        language: Language code.
        task: ``"transcribe"`` or ``"translate"``.
        prompt: Optional context prompt.
        with_timestamps: Emit timestamped segments.
        model_path: Path to model directory (for tokenizer loading).
        stt_config: Optional config overrides.

    Returns:
        :class:`TranscriptionResult`.
    """
    transcriber = WhisperTranscriber(model, model_path=model_path, config=stt_config)
    return transcriber.transcribe(audio, language, task, prompt, with_timestamps)
