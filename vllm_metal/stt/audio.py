# SPDX-License-Identifier: Apache-2.0
"""Audio processing for Whisper STT.

Provides mel-spectrogram computation, audio loading via ffmpeg / librosa,
and energy-based long-audio splitting.
"""

from __future__ import annotations

import math
import subprocess
from functools import lru_cache

import mlx.core as mx
import numpy as np

# ===========================================================================
# Whisper audio constants (matches OpenAI spec)
# ===========================================================================

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES = N_SAMPLES // HOP_LENGTH
N_MELS_DEFAULT = 80  # 128 for large-v3

# Log-mel spectrogram normalisation constants.
#   floor  : minimum value before log10 to avoid log(0)
#   dynamic: max dynamic range in log10 scale (8.0 ≈ 80 dB)
#   offset : shift applied after clamping
#   scale  : divisor for final normalisation
_LOG_FLOOR = 1e-10
_LOG_DYNAMIC_RANGE = 8.0
_LOG_OFFSET = 4.0
_LOG_SCALE = 4.0

# When searching for a quiet split point, look this many times the
# window size on each side of the target boundary.  4× gives enough
# room to find a natural pause without jumping too far from the ideal
# chunk length.
_SPLIT_SEARCH_MULTIPLIER = 4


# ===========================================================================
# Audio I/O
# ===========================================================================


def load_audio(file_path: str, sample_rate: int = SAMPLE_RATE) -> mx.array:
    """Load an audio file and return mono samples at *sample_rate* Hz.

    Tries ``librosa`` first (handles WAV natively without ffmpeg).
    Falls back to ffmpeg for non-WAV formats.

    Args:
        file_path: Path to the audio file.
        sample_rate: Target sample rate in Hz.

    Returns:
        1-D ``mx.array`` of float32 samples.

    Raises:
        RuntimeError: If the file cannot be decoded.
    """
    # Fast path: try librosa (handles WAV without ffmpeg)
    try:
        import librosa

        audio_np, _ = librosa.load(file_path, sr=sample_rate, mono=True)
        return mx.array(audio_np, mx.float32)
    except (ImportError, OSError, ValueError):
        pass

    # Fallback: ffmpeg
    return _load_audio_ffmpeg(file_path, sample_rate)


def _load_audio_ffmpeg(file_path: str, sample_rate: int) -> mx.array:
    """Load audio via ffmpeg subprocess.

    Args:
        file_path: Path to the audio file.
        sample_rate: Target sample rate in Hz.

    Returns:
        1-D ``mx.array`` of float32 samples.

    Raises:
        RuntimeError: If ffmpeg is missing or returns an error.
    """
    import shutil

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install it with: brew install ffmpeg")

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


def pad_or_trim(
    array: mx.array,
    length: int = N_SAMPLES,
    axis: int = -1,
) -> mx.array:
    """Pad (with zeros) or trim *array* to exactly *length* along *axis*.

    Args:
        array: Input array.
        length: Desired length.
        axis: Axis to pad/trim.

    Returns:
        Array with ``shape[axis] == length``.
    """
    if array.shape[axis] > length:
        sl = [slice(None)] * array.ndim
        sl[axis] = slice(0, length)
        array = array[tuple(sl)]
    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = mx.pad(array, pad_widths)
    return array


# ===========================================================================
# Spectrogram helpers
# ===========================================================================


def _hanning(size: int) -> mx.array:
    """Return a Hann window of *size* samples.

    Args:
        size: Window length.

    Returns:
        1-D ``mx.array`` of float32 window values.
    """
    return mx.array([0.5 - 0.5 * math.cos(2 * math.pi * n / size) for n in range(size)])


def _stft(
    audio: mx.array,
    window: mx.array,
    n_fft: int,
    hop_length: int,
) -> mx.array:
    """Compute the Short-Time Fourier Transform.

    Uses ``mx.as_strided`` to build the frame matrix in one shot,
    avoiding a Python loop over thousands of frames.

    Args:
        audio: 1-D input signal.
        window: Hann window of length *n_fft*.
        n_fft: FFT size.
        hop_length: Hop between frames.

    Returns:
        Complex spectrogram of shape ``(n_fft // 2 + 1, num_frames)``.
    """
    pad_amount = n_fft // 2
    audio = mx.pad(audio, [(pad_amount, pad_amount)])

    num_frames = (audio.shape[0] - n_fft) // hop_length + 1

    # Vectorised framing via as_strided (replaces Python loop).
    # mx.as_strided strides are in elements (not bytes).
    frames = mx.as_strided(audio, shape=(num_frames, n_fft), strides=(hop_length, 1))
    frames = frames * window

    return mx.fft.rfft(frames).T


@lru_cache(maxsize=4)
def _mel_filters(sample_rate: int, n_fft: int, n_mels: int) -> mx.array:
    """Build a Mel filter-bank matrix with Slaney normalisation.

    Results are cached so repeated calls with the same parameters
    (e.g. every chunk of a long file) skip recomputation.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_fft: FFT size.
        n_mels: Number of Mel bands.

    Returns:
        Filter-bank of shape ``(n_mels, n_fft // 2 + 1)``.
    """

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

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

    # Slaney normalisation
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    return filters * enorm[:, None]


def log_mel_spectrogram(
    audio: str | np.ndarray | mx.array,
    n_mels: int = N_MELS_DEFAULT,
) -> mx.array:
    """Compute a log-Mel spectrogram from raw audio.

    Args:
        audio: File path, numpy array, or MLX array of audio samples.
        n_mels: Number of Mel frequency bands.

    Returns:
        Log-Mel spectrogram of shape ``(n_mels, num_frames)``.
    """
    if isinstance(audio, str):
        audio = load_audio(audio)
    elif isinstance(audio, np.ndarray):
        audio = mx.array(audio, mx.float32)

    window = _hanning(N_FFT)
    freqs = _stft(audio, window, N_FFT, HOP_LENGTH)
    magnitudes = mx.abs(freqs).square()

    filters = _mel_filters(SAMPLE_RATE, N_FFT, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = mx.maximum(mel_spec, _LOG_FLOOR).log10()
    log_spec = mx.maximum(log_spec, log_spec.max() - _LOG_DYNAMIC_RANGE)
    return (log_spec + _LOG_OFFSET) / _LOG_SCALE


# ===========================================================================
# Audio duration & splitting
# ===========================================================================


def audio_duration(audio: mx.array, sample_rate: int = SAMPLE_RATE) -> float:
    """Return the duration of *audio* in seconds.

    Args:
        audio: 1-D audio samples.
        sample_rate: Sample rate in Hz.

    Returns:
        Duration in seconds.
    """
    return audio.shape[0] / sample_rate


def _rms_energy(audio: mx.array, window_size: int) -> mx.array:
    """Compute sliding-window RMS energy over *audio*.

    Args:
        audio: 1-D audio samples.
        window_size: Number of samples per window.

    Returns:
        Array of length ``ceil(len(audio) / window_size)`` with the
        RMS energy of each window.
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

    Searches within *search_radius* samples around *center* and returns
    the sample index at the start of the quietest window.

    Args:
        audio: 1-D audio samples.
        center: Target split position (sample index).
        window_size: RMS window size in samples.
        search_radius: Search range in samples.  Defaults to
            ``window_size * _SPLIT_SEARCH_MULTIPLIER``.

    Returns:
        Sample index of the best split point.
    """
    if search_radius is None:
        search_radius = window_size * _SPLIT_SEARCH_MULTIPLIER
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

    Short audio that already fits in one clip is returned as-is.

    Args:
        audio: 1-D audio samples.
        max_clip_s: Maximum clip duration in seconds.
        overlap_s: Overlap between consecutive clips in seconds.
        window_size: RMS energy window size in samples.
        sample_rate: Audio sample rate in Hz.

    Returns:
        List of ``(chunk, start_seconds)`` tuples.
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
