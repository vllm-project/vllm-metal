# SPDX-License-Identifier: Apache-2.0
"""Audio processing for Whisper STT."""

from __future__ import annotations

import math
import subprocess

import mlx.core as mx
import numpy as np

# Whisper audio constants (matches OpenAI spec)
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES = N_SAMPLES // HOP_LENGTH
N_MELS_DEFAULT = 80  # 128 for large-v3


def load_audio(file_path: str, sample_rate: int = SAMPLE_RATE) -> mx.array:
    """Load audio file using ffmpeg (must be in PATH)."""
    import shutil

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found")

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-i",
        file_path,
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-hide_banner",
        "-loglevel",
        "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")
    return mx.array(np.frombuffer(result.stdout, np.float32), mx.float32)


def pad_or_trim(array: mx.array, length: int = N_SAMPLES, axis: int = -1) -> mx.array:
    """Pad or trim array to specified length."""
    if array.shape[axis] > length:
        sl = [slice(None)] * array.ndim
        sl[axis] = slice(0, length)
        array = array[tuple(sl)]
    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = mx.pad(array, pad_widths)
    return array


def _hanning(size: int) -> mx.array:
    return mx.array([0.5 - 0.5 * math.cos(2 * math.pi * n / size) for n in range(size)])


def _stft(audio: mx.array, window: mx.array, n_fft: int, hop_length: int) -> mx.array:
    pad_amount = n_fft // 2
    audio = mx.pad(audio, [(pad_amount, pad_amount)])
    num_frames = (len(audio) - n_fft) // hop_length + 1
    frames = mx.stack(
        [
            audio[i * hop_length : i * hop_length + n_fft] * window
            for i in range(num_frames)
        ]
    )
    return mx.fft.rfft(frames).T


def _mel_filters(sample_rate: int, n_fft: int, n_mels: int) -> mx.array:
    def hz_to_mel(hz: float) -> float:
        return 2595 * math.log10(1 + hz / 700)

    def mel_to_hz(mel: float) -> float:
        return 700 * (10 ** (mel / 2595) - 1)

    mel_points = mx.linspace(hz_to_mel(0), hz_to_mel(sample_rate / 2), n_mels + 2)
    hz_points = mx.array([mel_to_hz(m.item()) for m in mel_points])
    bin_points = mx.floor((n_fft + 1) * hz_points / sample_rate).astype(mx.int32)

    filters = mx.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = (bin_points[j].item() for j in (i, i + 1, i + 2))
        for j in range(left, center):
            if center != left:
                filters[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filters[i, j] = (right - j) / (right - center)

    # Slaney normalization
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    return filters * enorm[:, None]


def log_mel_spectrogram(
    audio: str | np.ndarray | mx.array,
    n_mels: int = N_MELS_DEFAULT,
) -> mx.array:
    """Compute log-Mel spectrogram."""
    if isinstance(audio, str):
        audio = load_audio(audio)
    elif isinstance(audio, np.ndarray):
        audio = mx.array(audio, mx.float32)

    window = _hanning(N_FFT)
    freqs = _stft(audio, window, N_FFT, HOP_LENGTH)
    magnitudes = mx.abs(freqs).square()

    filters = _mel_filters(SAMPLE_RATE, N_FFT, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = mx.maximum(mel_spec, 1e-10).log10()
    log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def audio_duration(audio: mx.array, sample_rate: int = SAMPLE_RATE) -> float:
    """Return the duration of *audio* in seconds."""
    return audio.shape[0] / sample_rate


def _rms_energy(audio: mx.array, window_size: int) -> mx.array:
    """Compute sliding-window RMS energy over *audio*.

    Returns an array of length ``ceil(len(audio) / window_size)`` where
    each element is the RMS energy of the corresponding window.
    """
    n = audio.shape[0]
    n_windows = math.ceil(n / window_size)
    energies = []
    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, n)
        chunk = audio[start:end]
        rms = mx.sqrt(mx.mean(chunk * chunk))
        energies.append(rms.item())
    return mx.array(energies)


def _find_split_point(
    audio: mx.array,
    center: int,
    window_size: int,
    search_radius: int | None = None,
) -> int:
    """Find a low-energy split point near *center*.

    Searches within *search_radius* samples (default: ``window_size * 4``)
    around *center* and returns the sample index at the start of the
    quietest window.
    """
    if search_radius is None:
        search_radius = window_size * 4
    n = audio.shape[0]
    lo = max(0, center - search_radius)
    hi = min(n, center + search_radius)
    region = audio[lo:hi]

    energies = _rms_energy(region, window_size)
    best_window = int(mx.argmin(energies).item())
    return lo + best_window * window_size


def split_audio(
    audio: mx.array,
    max_clip_s: float = CHUNK_LENGTH,
    overlap_s: float = 1.0,
    window_size: int = 1600,
    sample_rate: int = SAMPLE_RATE,
) -> list[tuple[mx.array, float]]:
    """Split long audio at quiet points into clips of at most *max_clip_s*.

    Returns a list of ``(chunk, start_seconds)`` tuples.  Each chunk is
    an ``mx.array`` of audio samples.  Short audio that already fits in
    one clip is returned as-is.
    """
    max_samples = int(max_clip_s * sample_rate)
    overlap_samples = int(overlap_s * sample_rate)
    n = audio.shape[0]

    if n <= max_samples:
        return [(audio, 0.0)]

    chunks: list[tuple[mx.array, float]] = []
    pos = 0

    while pos < n:
        end = pos + max_samples
        if end >= n:
            chunks.append((audio[pos:], pos / sample_rate))
            break

        split = _find_split_point(audio, end, window_size)
        # Ensure forward progress
        if split <= pos:
            split = end

        chunks.append((audio[pos:split], pos / sample_rate))
        pos = max(split - overlap_samples, pos + 1)

    return chunks
