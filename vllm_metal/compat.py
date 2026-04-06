# SPDX-License-Identifier: Apache-2.0
"""Compatibility patches for vLLM + transformers version mismatches.

Applied once at platform registration time.  Each patch is guarded by
try/except so it degrades silently if the target module changes.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_compat_patches() -> None:
    """Apply all known compatibility patches (idempotent)."""
    global _APPLIED  # noqa: PLW0603
    if _APPLIED:
        return
    _APPLIED = True
    _patch_qwen35_rope_validation()


def _patch_qwen35_rope_validation() -> None:
    """Fix vLLM 0.17.1 Qwen3.5 config vs transformers >=5.4 rope validation.

    vLLM's ``Qwen3_5TextConfig.__init__`` hardcodes
    ``kwargs["ignore_keys_at_rope_validation"] = [...]`` (a list), but
    transformers 5.4+ does ``received_keys -= ignore_keys`` which requires
    a set.

    Upstream: vllm-project/vllm#34604 fixed this but was reverted in #34610.
    Remove this patch when vllm-metal upgrades to a vLLM version with the fix.
    """
    from importlib.util import find_spec

    if find_spec("vllm.transformers_utils.configs.qwen3_5") is None:
        return

    try:
        from transformers.modeling_rope_utils import RopeConfigBase

        rope_config_base = RopeConfigBase
    except ImportError:
        rope_config_base = None

    if rope_config_base is None:
        # Try the direct path
        try:
            import transformers.modeling_rope_utils as _rope

            _orig_check = _rope._check_received_keys

            def _safe_check(
                rope_type,
                received_keys,
                required_keys,
                optional_keys=None,
                ignore_keys=None,
            ):
                if ignore_keys is not None and isinstance(ignore_keys, list):
                    ignore_keys = set(ignore_keys)
                return _orig_check(
                    rope_type, received_keys, required_keys, optional_keys, ignore_keys
                )

            _rope._check_received_keys = _safe_check
            logger.debug("Patched _check_received_keys for rope validation compat")
            return
        except (ImportError, AttributeError):
            pass

    # Fallback: patch the static method on PreTrainedConfig if available
    try:
        from transformers import PreTrainedConfig

        if hasattr(PreTrainedConfig, "_check_received_keys"):
            _orig_check = PreTrainedConfig._check_received_keys

            @staticmethod
            def _safe_check(
                rope_type,
                received_keys,
                required_keys,
                optional_keys=None,
                ignore_keys=None,
            ):
                if ignore_keys is not None and isinstance(ignore_keys, list):
                    ignore_keys = set(ignore_keys)
                return _orig_check(
                    rope_type, received_keys, required_keys, optional_keys, ignore_keys
                )

            PreTrainedConfig._check_received_keys = _safe_check
            logger.debug(
                "Patched PreTrainedConfig._check_received_keys for rope compat"
            )
    except (ImportError, AttributeError):
        pass
