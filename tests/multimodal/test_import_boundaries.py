# SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal helper import boundaries."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_multimodal_helpers_do_not_import_v1_worker() -> None:
    code = textwrap.dedent(
        """
        import sys

        import vllm_metal.multimodal.embeddings
        import vllm_metal.multimodal.gemma4
        import vllm_metal.multimodal.gemma4.adapter
        import vllm_metal.multimodal.gemma4.geometry
        import vllm_metal.multimodal.gemma4.sidecar
        import vllm_metal.multimodal.qwen3_vl
        import vllm_metal.multimodal.qwen3_vl.adapter
        import vllm_metal.multimodal.nemotron_nano_vl.adapter
        import vllm_metal.v1.mm.encoder_cache

        if "vllm_metal.v1.worker" in sys.modules:
            raise SystemExit("vllm_metal.v1.worker was imported")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_spawned_worker_registers_nano_before_receiver_cache(tmp_path) -> None:
    """The wrapper reads configuration before it constructs MetalWorker."""
    (tmp_path / "config.json").write_text(
        '{"model_type":"NemotronH_Nano_VL_V2",'
        '"architectures":["NemotronH_Nano_VL_V2"],'
        '"auto_map":{"AutoConfig":"configuration.NemotronH_Nano_VL_V2_Config"},'
        '"llm_config":{"hidden_size":5120}}'
    )
    code = textwrap.dedent(
        """
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch

        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from vllm.platforms import current_platform
        from vllm.transformers_utils.config import _CONFIG_REGISTRY, get_config
        from vllm.v1.worker.worker_base import WorkerWrapperBase
        from vllm_metal.platform import MetalPlatform

        assert isinstance(current_platform, MetalPlatform)
        assert "vllm_metal.v1.worker" not in sys.modules
        model_type = "NemotronH_Nano_VL_V2"
        CONFIG_MAPPING._extra_content.pop(model_type, None)
        _CONFIG_REGISTRY.pop(model_type, None)

        class CacheReached(Exception):
            pass

        def check_cache(vllm_config, lock):
            assert model_type in CONFIG_MAPPING, "config missing before receiver cache"
            assert model_type in _CONFIG_REGISTRY
            config = get_config(sys.argv[1], trust_remote_code=False)
            assert config.text_config.hidden_size == 5120
            # Stop before any worker, device, model, or real cache is created.
            raise CacheReached

        config = SimpleNamespace(
            enable_trace_function_call_for_thread=lambda: None,
            parallel_config=SimpleNamespace(
                worker_cls="vllm_metal.v1.worker.MetalWorker",
                worker_extension_cls="",
            ),
        )
        with patch(
            "vllm.v1.worker.worker_base.worker_receiver_cache_from_config", check_cache
        ):
            try:
                WorkerWrapperBase().init_worker([{
                    "vllm_config": config, "shared_worker_lock": object()
                }])
            except CacheReached:
                pass
            else:
                raise AssertionError("Wrapper skipped receiver cache construction")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
