# SPDX-License-Identifier: Apache-2.0
"""Environment variable definitions for the vLLM Metal plugin.

This module is the single source of truth for all ``VLLM_METAL_*`` (and
``VLLM_MLX_*``) environment variables.  It mirrors the lazy-evaluation
pattern used by ``vllm/envs.py``: each variable is read from
``os.environ`` on access via ``__getattr__``, so values are never stale
and ``monkeypatch.setenv`` works in tests without extra resets.

During plugin registration (``vllm_metal._register``), the
``environment_variables`` dict is merged into
``vllm.envs.environment_variables`` so that ``validate_environ()``
recognises our variables and does not emit spurious "Unknown vLLM
environment variable" warnings.

``validate_environment`` parses every variable once at startup (from
``MetalPlatform.check_and_update_config``) and reports every bad value
together, so a typo fails the startup rather than the first request that
reads it.  Boolean switches treat ``"1"`` as on and anything else as off.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

MLX_DEVICES = ("gpu", "cpu")
MULTIMODAL_MODES = ("auto", "multimodal-native", "text-only")
MM_PREFIX_PATHS = ("kernel", "recompute")
TQ_PREFILL_MODES = ("auto", "0", "1")


def _choice(
    name: str, default: str | None, choices: tuple[str, ...]
) -> Callable[[], str | None]:
    """A variable limited to ``choices``; a bad value lists them."""

    def parse() -> str | None:
        raw = os.getenv(name)
        if raw is None:
            return default
        if raw not in choices:
            raise ValueError(f"{name} must be one of {', '.join(choices)}, got {raw!r}")
        return raw

    return parse


def _int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    note: str = "",
) -> Callable[[], int]:
    """An integer variable; a bad value names the variable and the value."""

    def parse() -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from None
        too_low = minimum is not None and value < minimum
        too_high = maximum is not None and value > maximum
        if too_low or too_high:
            if maximum is None:
                bound = f"at least {minimum}"
            elif minimum is None:
                bound = f"at most {maximum}"
            else:
                bound = f"in [{minimum}, {maximum}]"
            raise ValueError(f"{name} must be {bound}{note}, got {raw!r}")
        return value

    return parse


def _bool(name: str, default: bool) -> Callable[[], bool]:
    """A switch: ``"1"`` is on, anything else is off; unset is ``default``."""

    def parse() -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw == "1"

    return parse


def _auto_or_nonnegative_int(name: str, *, unit: str) -> Callable[[], int | str]:
    """``auto`` (the default) or an integer of at least 0, in ``unit``."""

    def parse() -> int | str:
        raw = os.getenv(name, "auto")
        if raw == "auto":
            return raw
        try:
            value: int | None = int(raw)
        except ValueError:
            value = None
        if value is None or value < 0:
            raise ValueError(
                f"{name} must be auto or a nonnegative integer in {unit}, got {raw!r}"
            )
        return value

    return parse


if TYPE_CHECKING:
    VLLM_METAL_BACKEND: str = "mlx"
    VLLM_MLX_DEVICE: str = "gpu"
    VLLM_METAL_MULTIMODAL_MODE: str = "auto"
    VLLM_METAL_MM_PREFIX_PATH: str | None = None
    VLLM_METAL_MODELSCOPE_CACHE: str | None = None
    VLLM_METAL_GDN_LAZY_KERNELS: bool = True
    VLLM_METAL_DECODE_PIPELINE: bool = True
    VLLM_METAL_COMPILED_MLP: bool = False
    VLLM_METAL_NATIVE_SAMPLING: bool = False
    VLLM_METAL_MLA_KERNEL: bool = False
    VLLM_METAL_DISABLE_NAX: bool = False
    VLLM_METAL_DISABLE_GQA_DECODE: bool = False
    VLLM_METAL_TQ_PREFILL: str = "auto"
    VLLM_METAL_TQ_PREFILL_MAX_MIB: int | str = "auto"
    VLLM_METAL_SPEC_VERIFY_WINDOW: bool = False
    VLLM_METAL_SPEC_INGEST_CHUNK: int = 1024
    VLLM_METAL_BUILD_FROM_SOURCE: bool = False
    VLLM_METAL_VISIBLE_DEVICES: str | None = None
    VLLM_METAL_RING_BASE_PORT: int = 32323

environment_variables: dict[str, Callable[[], Any]] = {
    # Opt-in PyTorch MPS execution backend.
    "VLLM_METAL_BACKEND": lambda: os.getenv("VLLM_METAL_BACKEND", "mlx"),
    # MLX device type: "gpu" (default) or "cpu".
    "VLLM_MLX_DEVICE": _choice("VLLM_MLX_DEVICE", "gpu", MLX_DEVICES),
    # Multimodal serving mode:
    # - "auto": known-incompatible multimodal checkpoints fall back to the
    #   text-only compatibility path; Gemma 4 serves images through the
    #   vision sidecar on the mlx_lm text backbone when the checkpoint allows.
    # - "multimodal-native": keep native multimodal loading enabled.
    # - "text-only": force the text-only path for every multimodal checkpoint.
    "VLLM_METAL_MULTIMODAL_MODE": _choice(
        "VLLM_METAL_MULTIMODAL_MODE", "auto", MULTIMODAL_MODES
    ),
    # Gemma 4 vision image-block attention path: "kernel" (default) hands the
    # per-row block ranges to the tiled Metal prefill kernel; "recompute"
    # keeps the MLX SDPA recompute of the block rows after the kernel
    # (the reference path).  Read per forward.
    "VLLM_METAL_MM_PREFIX_PATH": _choice(
        "VLLM_METAL_MM_PREFIX_PATH", None, MM_PREFIX_PATHS
    ),
    # Custom cache directory for ModelScope downloads (None if unset).
    "VLLM_METAL_MODELSCOPE_CACHE": lambda: os.getenv("VLLM_METAL_MODELSCOPE_CACHE"),
    # Enable lazy GDN kernels by default.
    # Set to "0" to force the eager conv / C++ recurrent fallback path.
    "VLLM_METAL_GDN_LAZY_KERNELS": _bool("VLLM_METAL_GDN_LAZY_KERNELS", True),
    # One-step-ahead decode pipelining (default on). Eligible pure-decode
    # greedy steps defer the sampling sync one step so the next step's graph
    # build and submit overlap the in-flight GPU forward. Set to "0" to
    # force the fully synchronous per-step sample path.
    "VLLM_METAL_DECODE_PIPELINE": _bool("VLLM_METAL_DECODE_PIPELINE", True),
    # Compiled stateless-MLP dispatch (opt-in): decode-shaped MLP/MoE
    # block calls run through an mx.compile trace, fusing the per-layer
    # elementwise glue. Bitwise-identical on the quantized serving path;
    # set to "1" to enable, the default keeps the eager per-op dispatch.
    "VLLM_METAL_COMPILED_MLP": _bool("VLLM_METAL_COMPILED_MLP", False),
    # MLX-native non-greedy sampling for the decode pipeline (opt-in): the
    # pipeline's deferred sampler learns a temperature/top-k/top-p graph, so
    # eligible non-greedy pure-decode steps defer like greedy ones instead
    # of dropping to the synchronous torch path. Requests with seeds,
    # penalties, logprobs, or token constraints keep the torch path
    # unchanged. Set to "1" to enable; intended to flip default-on once the
    # path has serve mileage, with this var remaining as the kill switch.
    "VLLM_METAL_NATIVE_SAMPLING": _bool("VLLM_METAL_NATIVE_SAMPLING", False),
    # Experimental MLA Metal decode kernel (RFC #360). Off by default —
    # the MLA wrapper uses the MLX SDPA per-request slow path unless
    # this opt-in is set. Set to "1" to route absorbed-MLA decode
    # through the single-pass Metal kernel when the workload matches
    # the kernel's instantiated specialization (kv_lora_rank=512,
    # qk_rope_head_dim=64, block_size ∈ {16, 32}, fp16/bf16,
    # decode-only).
    "VLLM_METAL_MLA_KERNEL": _bool("VLLM_METAL_MLA_KERNEL", False),
    # Emergency override for automatic M5 NAX prefill attention.
    "VLLM_METAL_DISABLE_NAX": _bool("VLLM_METAL_DISABLE_NAX", False),
    # Emergency override for the GQA-shared flash-decode dispatch gate
    # (issue #713): force long-context decode back to the established
    # per-token / split-KV kernels. Mainly a benchmarking/A-B escape
    # hatch so controlled comparisons can run under the server topology.
    "VLLM_METAL_DISABLE_GQA_DECODE": _bool("VLLM_METAL_DISABLE_GQA_DECODE", False),
    # TQ materialized prefill: auto enables only when NAX is available;
    # 1 explicitly opts into tiled prefill on older GPUs, 0 disables it.
    "VLLM_METAL_TQ_PREFILL": _choice("VLLM_METAL_TQ_PREFILL", "auto", TQ_PREFILL_MODES),
    # Temporary-workspace allowance, deducted before KV sizing. Auto takes
    # 2% of the recommended working set (256 MiB to 2 GiB). A numeric value
    # sets an explicit MiB limit; 0 disables. Set before worker startup.
    "VLLM_METAL_TQ_PREFILL_MAX_MIB": _auto_or_nonnegative_int(
        "VLLM_METAL_TQ_PREFILL_MAX_MIB", unit="MiB"
    ),
    # Spec-decode verification window mode (issue #465). Off by default —
    # verify windows keep the expanded per-token layout (main behavior)
    # unless this opt-in is set. Set to "1" to merge K+1 verify windows
    # into one segment and share each KV block load across the window
    # rows. Profitability is chip- and shape-dependent: measured wins at
    # conc >= 4 with 8k+ context (M2 Ultra / M3 Ultra / M4 Pro, up to
    # +40% e2e at conc 16-32), measured losses single-stream on M4 Pro
    # and at conc 32 on M2 Max. Outputs are bitwise identical either way.
    "VLLM_METAL_SPEC_VERIFY_WINDOW": _bool("VLLM_METAL_SPEC_VERIFY_WINDOW", False),
    # Max tokens of cold draft KV ingested per forward (issue #482,
    # direction 3). The first propose of a fresh prefix ingests the whole
    # prompt into the draft model's KV in one tiled prefill forward;
    # chunking bounds the stall at any single dispatch and the logits peak
    # allocation (draft_vocab x chunk instead of x prompt length). 1024
    # tokens is ~2 ms of draft-model work on a modern M-series chip; a
    # multiple of the block size is recommended. Set to "0" to restore the
    # single-forward behavior.
    "VLLM_METAL_SPEC_INGEST_CHUNK": _int(
        "VLLM_METAL_SPEC_INGEST_CHUNK",
        1024,
        minimum=0,
        note=" (0 means single-forward ingest)",
    ),
    # When set, compile the native _paged_ops extension from source at runtime
    # instead of loading the prebuilt artifact shipped in the wheel. Intended
    # for kernel developers / source installs; requires Xcode command-line
    # tools (clang++). Default off — release wheels ship the .so prebuilt.
    "VLLM_METAL_BUILD_FROM_SOURCE": _bool("VLLM_METAL_BUILD_FROM_SOURCE", False),
    # Per-worker visible-device list set by vLLM's Ray executor (the
    # CUDA_VISIBLE_DEVICES analog for Metal; see MetalPlatform.device_control_env_var).
    # Registered here only so validate_environ() does not warn — vLLM reads it
    # from os.environ directly.
    "VLLM_METAL_VISIBLE_DEVICES": lambda: os.getenv("VLLM_METAL_VISIBLE_DEVICES"),
    # Base TCP port for the MLX ring data plane under pipeline parallelism;
    # stage r binds base + r (default 32323/32324 for two stages). Set the same
    # value on every node to move the ring off a busy port. Default matches
    # mlx.launch's starting_port. See distributed.md#pipeline-parallelism.
    "VLLM_METAL_RING_BASE_PORT": _int(
        "VLLM_METAL_RING_BASE_PORT",
        32323,
        minimum=1024,
        maximum=65535,
        note=" (the user-port range)",
    ),
}


def validate_environment() -> None:
    """Parse every variable once and report all bad values together."""
    errors = []
    for name, parse in environment_variables.items():
        try:
            parse()
        except ValueError as exc:
            message = str(exc)
            errors.append(message if message.startswith(name) else f"{name}: {message}")
    if errors:
        raise ValueError("Invalid vllm-metal environment: " + "; ".join(errors))


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Mirrors vllm/envs.py; enables tab-completion and introspection.
    return list(environment_variables.keys())
