# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal platform."""

import importlib
import platform
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_capacity,
    get_kv_cache_configs,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

import vllm_metal.compat as compat
from tests.stub_runner import make_gemma4_mixed_mha_runner
from vllm_metal.config import reset_config
from vllm_metal.platform import MetalPlatform
from vllm_metal.v1.cache_policy import WorkerCachePlanner


class TestMetalPlatform:
    """Tests for MetalPlatform class."""

    @staticmethod
    def _patch_stt_resolution(
        monkeypatch: pytest.MonkeyPatch,
        *,
        is_stt: bool,
    ) -> None:
        monkeypatch.setattr(
            "vllm_metal.utils.get_model_download_path",
            lambda model: model,
        )
        monkeypatch.setattr(
            "vllm_metal.stt.detection.is_stt_model", lambda _model: is_stt
        )

    def test_device_name(self) -> None:
        """Test device name retrieval."""
        name = MetalPlatform.get_device_name()
        assert "Apple Silicon" in name

    def test_set_device_valid(self) -> None:
        """Test setting valid device."""
        MetalPlatform.set_device(0)  # Should not raise

    def test_set_device_invalid(self) -> None:
        """Test setting invalid device."""
        with pytest.raises(ValueError, match="only supports device 0"):
            MetalPlatform.set_device(1)

    def test_set_device_accepts_torch_device(self) -> None:
        """Ray's compiled-DAG path passes a torch.device, not an int."""
        MetalPlatform.set_device(torch.device("cpu"))  # index None -> ok
        MetalPlatform.set_device(torch.device("cpu", 0))  # index 0 -> ok
        with pytest.raises(ValueError, match="only supports device 0"):
            MetalPlatform.set_device(torch.device("cpu", 1))

    @pytest.mark.parametrize(
        ("sampling_params", "control"),
        [
            (SamplingParams(min_p=0.1), "min_p"),
            (SamplingParams(logit_bias={1: 2.0}), "logit_bias"),
            (SamplingParams(min_tokens=2), "min_tokens"),
        ],
        ids=["min-p", "logit-bias", "min-tokens"],
    )
    def test_validate_request_rejects_logits_processor_controls(
        self,
        sampling_params: SamplingParams,
        control: str,
    ) -> None:
        with pytest.raises(VLLMValidationError, match=control) as exc_info:
            MetalPlatform.validate_request({}, sampling_params)
        assert exc_info.value.parameter == control

    def test_check_and_update_config_rejects_pipeline_with_tensor_parallel(
        self,
    ) -> None:
        """PP is allowed, but combining PP>1 with TP>1 is rejected at config time."""
        vllm_config = SimpleNamespace(
            speculative_config=None,
            cache_config=SimpleNamespace(kv_cache_dtype_skip_layers=[]),
            parallel_config=SimpleNamespace(
                worker_cls="auto",
                # "mp", not "uni": uni+PP>1 short-circuits to the uni-executor
                # guard; "mp" reaches the PP+TP check this test targets.
                distributed_executor_backend="mp",
                pipeline_parallel_size=2,
                tensor_parallel_size=2,
                disable_custom_all_reduce=False,
            ),
            model_config=None,
        )
        with pytest.raises(
            NotImplementedError, match="alone or combined with pipeline"
        ):
            MetalPlatform.check_and_update_config(vllm_config)

    @pytest.mark.parametrize(
        ("quantization", "multimodal_config", "rejects_pp"),
        [
            (None, None, False),
            ("auto_awq", None, True),
            ("gguf", None, True),
            ("auto_awq", SimpleNamespace(), False),
        ],
        ids=["safetensors", "text-awq", "text-gguf", "multimodal-awq"],
    )
    def test_check_and_update_config_handles_pipeline_loader_routes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        quantization: str | None,
        multimodal_config: SimpleNamespace | None,
        rejects_pp: bool,
    ) -> None:
        """PP admits lazy loaders and rejects proven eager text loaders."""
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        reset_config()
        try:
            vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=2,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    kv_cache_dtype_skip_layers=[],
                    block_size=None,
                ),
                model_config=SimpleNamespace(
                    model="test-model",
                    disable_cascade_attn=False,
                    tokenizer=None,
                    max_model_len=32768,
                    multimodal_config=multimodal_config,
                    hf_config=(
                        SimpleNamespace(
                            model_type="qwen3_vl",
                            architectures=["Qwen3VLForConditionalGeneration"],
                        )
                        if multimodal_config is not None
                        else SimpleNamespace(model_type="qwen3")
                    ),
                    is_hybrid=False,
                    quantization=quantization,
                ),
                scheduler_config=SimpleNamespace(
                    async_scheduling=False,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=2048,
                    max_num_scheduled_tokens=None,
                ),
                speculative_config=None,
                lora_config=None,
            )

            if rejects_pp:
                with pytest.raises(
                    NotImplementedError, match=r"AWQ or GGUF.*pipeline_parallel_size=1"
                ):
                    MetalPlatform.check_and_update_config(vllm_config)
                return

            MetalPlatform.check_and_update_config(vllm_config)

            assert vllm_config.parallel_config.distributed_executor_backend == "mp"
            assert (
                vllm_config.parallel_config.worker_cls
                == "vllm_metal.v1.worker.MetalWorker"
            )
        finally:
            reset_config()

    def test_check_and_update_config_fails_if_bytelevel_retry_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_import_error() -> None:
            raise ImportError("tokenizer registry unavailable")

        monkeypatch.setattr(
            compat,
            "ensure_vllm_bytelevel_tokenizer_patch",
            _raise_import_error,
        )

        with pytest.raises(ImportError, match="tokenizer registry unavailable"):
            MetalPlatform.check_and_update_config(SimpleNamespace())

    def test_check_and_update_config_rejects_pipeline_ring_port_overflow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_RING_BASE_PORT", "65535")
        vllm_config = SimpleNamespace(
            speculative_config=None,
            cache_config=SimpleNamespace(kv_cache_dtype_skip_layers=[]),
            parallel_config=SimpleNamespace(
                worker_cls="auto",
                distributed_executor_backend="auto",
                pipeline_parallel_size=2,
                tensor_parallel_size=1,
                disable_custom_all_reduce=False,
            ),
            model_config=None,
        )
        with pytest.raises(ValueError, match="too high for pipeline_parallel_size"):
            MetalPlatform.check_and_update_config(vllm_config)

    def test_check_and_update_config_rejects_pipeline_with_async_scheduling(
        self,
    ) -> None:
        """PP>1 requires synchronous scheduling; async scheduling is rejected.

        The first stage has no sampler and rebuilds the token stream from the
        scheduler's new_token_ids, which async scheduling leaves empty (sampled
        tokens would travel a GPU broadcast we do not implement). Fail loud
        rather than silently flip the user's scheduler config.
        """
        vllm_config = SimpleNamespace(
            speculative_config=None,
            cache_config=SimpleNamespace(kv_cache_dtype_skip_layers=[]),
            parallel_config=SimpleNamespace(
                worker_cls="auto",
                distributed_executor_backend="auto",
                pipeline_parallel_size=2,
                tensor_parallel_size=1,
                disable_custom_all_reduce=False,
            ),
            model_config=None,
            scheduler_config=SimpleNamespace(async_scheduling=True),
        )
        with pytest.raises(NotImplementedError, match="synchronous scheduling"):
            MetalPlatform.check_and_update_config(vllm_config)

    def test_check_and_update_config_rejects_pipeline_with_speculative_decoding(
        self,
    ) -> None:
        """PP>1 with speculative decoding is rejected at config time.

        The PP forward path produces no target hidden states and draft proposal
        runs only on the sampling (last) stage, so no speculative method is
        implemented under PP. Reject loudly rather than run it unvalidated.
        """
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(kv_cache_dtype_skip_layers=[]),
            parallel_config=SimpleNamespace(
                worker_cls="auto",
                distributed_executor_backend="auto",
                pipeline_parallel_size=2,
                tensor_parallel_size=1,
                disable_custom_all_reduce=False,
            ),
            model_config=None,
            scheduler_config=SimpleNamespace(
                async_scheduling=False,
                long_prefill_token_threshold=0,
            ),
            speculative_config=SimpleNamespace(
                use_heterogeneous_vocab=False,
                num_speculative_tokens=3,
                method="ngram",
            ),
        )
        with pytest.raises(NotImplementedError, match="speculative decoding"):
            MetalPlatform.check_and_update_config(vllm_config)

    def test_check_and_update_config_rejects_pipeline_with_stt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PP>1 with an STT model is rejected at config time.

        STT checkpoints use a dedicated runner with no pipeline-split path, so
        reject before any worker spawns rather than fail after startup.
        """
        self._patch_stt_resolution(monkeypatch, is_stt=True)
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(kv_cache_dtype_skip_layers=[]),
            parallel_config=SimpleNamespace(
                worker_cls="auto",
                distributed_executor_backend="auto",
                pipeline_parallel_size=2,
                tensor_parallel_size=1,
                disable_custom_all_reduce=False,
            ),
            model_config=SimpleNamespace(
                model="openai/whisper-tiny",
                disable_cascade_attn=False,
                tokenizer=None,
                multimodal_config=None,
                hf_config=SimpleNamespace(model_type="whisper"),
                is_hybrid=False,
            ),
            scheduler_config=SimpleNamespace(
                async_scheduling=False,
                enable_chunked_prefill=False,
            ),
            speculative_config=None,
            lora_config=None,
        )
        with pytest.raises(NotImplementedError, match="speech-to-text"):
            MetalPlatform.check_and_update_config(vllm_config)

    def test_check_and_update_config_rejects_uni_executor_with_pipeline_parallel(
        self,
    ) -> None:
        """The single-process 'uni' executor cannot host PP's per-stage workers.

        vLLM's UniProcExecutor builds only rank 0, so without this guard the lone
        worker hangs in gloo/ring rendezvous waiting for a stage that never
        spawns. Reject the explicit combination rather than flip it silently.
        """
        vllm_config = SimpleNamespace(
            speculative_config=None,
            cache_config=SimpleNamespace(kv_cache_dtype_skip_layers=[]),
            parallel_config=SimpleNamespace(
                worker_cls="auto",
                distributed_executor_backend="uni",
                pipeline_parallel_size=2,
                tensor_parallel_size=1,
                disable_custom_all_reduce=False,
            ),
            model_config=None,
        )
        with pytest.raises(NotImplementedError, match="single process"):
            MetalPlatform.check_and_update_config(vllm_config)

    def test_check_and_update_config_rejects_tensor_parallel(self) -> None:
        """Tensor parallelism is unsupported on Metal yet; reject it at config time."""
        vllm_config = SimpleNamespace(
            speculative_config=None,
            cache_config=SimpleNamespace(kv_cache_dtype_skip_layers=[]),
            parallel_config=SimpleNamespace(
                worker_cls="auto",
                distributed_executor_backend="uni",
                pipeline_parallel_size=1,
                tensor_parallel_size=2,
                disable_custom_all_reduce=False,
            ),
            model_config=None,
        )
        with pytest.raises(NotImplementedError, match="tensor parallelism"):
            MetalPlatform.check_and_update_config(vllm_config)

    @staticmethod
    def _dp_parallel_config(**overrides: object) -> SimpleNamespace:
        """A valid dense data-parallel-over-Ray parallel_config, with overrides.

        Defaults to the one supported shape (dense + ray backend + local==1 +
        internal LB); reject tests override the field they exercise. Executor
        backend defaults to ``mp`` so the executor branch falls through without
        importing Ray; the ALLOW test overrides it to ``ray``.
        """
        base: dict[str, object] = {
            "worker_cls": "auto",
            "distributed_executor_backend": "mp",
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 1,
            "disable_custom_all_reduce": False,
            "data_parallel_size": 2,
            "data_parallel_backend": "ray",
            "data_parallel_size_local": 1,
            "data_parallel_external_lb": False,
            "data_parallel_hybrid_lb": False,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    @classmethod
    def _dp_vllm_config(
        cls,
        *,
        parallel: dict | None = None,
        parallel_config: object = None,
        model: dict | None = None,
        speculative_config: object = None,
        lora_config: object = None,
    ) -> SimpleNamespace:
        """A complete vllm_config for a dense-DP run, with field overrides.

        Reject tests override only the field under test on top of the full scaffold
        so a guard fires for the right reason (not a missing attribute). Pass
        ``parallel_config`` to inject a prebuilt (e.g. real ``ParallelConfig``)
        object instead of the SimpleNamespace stand-in.
        """
        model_fields: dict[str, object] = {
            "model": "test-model",
            "is_moe": False,
            "multimodal_config": None,
            "disable_cascade_attn": False,
            "tokenizer": None,
            "max_model_len": 32768,
            "hf_config": SimpleNamespace(model_type="qwen3"),
            "is_hybrid": False,
        }
        model_fields.update(model or {})
        return SimpleNamespace(
            parallel_config=(
                parallel_config
                if parallel_config is not None
                else cls._dp_parallel_config(**(parallel or {}))
            ),
            cache_config=SimpleNamespace(
                kv_cache_dtype_skip_layers=[],
                block_size=None,
            ),
            model_config=SimpleNamespace(**model_fields),
            scheduler_config=SimpleNamespace(
                long_prefill_token_threshold=0,
                async_scheduling=False,
                enable_chunked_prefill=True,
                max_num_batched_tokens=2048,
                max_num_scheduled_tokens=None,
            ),
            speculative_config=speculative_config,
            lora_config=lora_config,
        )

    @staticmethod
    def _stub_ray(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
        """Stub ray.init / is_initialized and reset the DP-hook flag so a unit test
        exercising the DP admission never contacts a real cluster. Returns the list
        that captures ray.init kwargs."""
        ray = pytest.importorskip("ray")
        monkeypatch.setattr(MetalPlatform, "_dp_ray_hook_registered", False)
        init_calls: list[dict] = []
        monkeypatch.setattr(ray, "is_initialized", lambda: False)
        monkeypatch.setattr(ray, "init", lambda **kwargs: init_calls.append(kwargs))
        return init_calls

    def test_check_and_update_config_allows_dense_data_parallel_ray(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dense DP over the Ray backend (one replica/node, internal LB) is allowed.

        Also pins the job-level hook registration: DP registers the
        worker_process_setup_hook via ray.init (Ray does not honor it from the
        per-actor runtime_env the DP engine manager uses), and the registered hook
        string resolves to a real callable. ray.init is stubbed (no real cluster).
        """
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        init_calls = self._stub_ray(monkeypatch)
        reset_config()
        try:
            vllm_config = self._dp_vllm_config(
                parallel={
                    "distributed_executor_backend": "ray",
                    "ray_runtime_env": None,
                }
            )
            MetalPlatform.check_and_update_config(vllm_config)
            assert vllm_config.parallel_config.distributed_executor_backend == "ray"
            assert init_calls, "DP must register the worker hook via ray.init"
            # Pin the full cluster-connect contract: the documented RAY_ADDRESS=auto
            # launch only works if we connect to the existing cluster, not a private
            # local Ray, so address must be "auto".
            assert init_calls[0]["address"] == "auto"
            hook = init_calls[0]["runtime_env"]["worker_process_setup_hook"]
            assert hook == MetalPlatform._RAY_WORKER_SETUP_HOOK
            # The registered hook string must resolve to a real callable so a
            # rename of compat._patch_ray_distributed breaks this unit test, not
            # only a live cluster run.
            module_path, _, attr = hook.rpartition(".")
            resolved = getattr(importlib.import_module(module_path), attr)
            assert callable(resolved)
        finally:
            reset_config()

    def test_check_and_update_config_allows_dp_for_text_only_backbone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model whose multimodal_config is cleared by normalize (served on the
        text-only backbone) is NOT rejected under DP — the DP multimodal guard runs
        AFTER normalize_model_config, not before."""
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        self._stub_ray(monkeypatch)
        # normalize clears multimodal_config (text-only backbone).
        monkeypatch.setattr(
            "vllm_metal.v1.model_adapter.DefaultModelAdapter.normalize_model_config",
            lambda _self, mc: setattr(mc, "multimodal_config", None),
        )
        reset_config()
        try:
            vllm_config = self._dp_vllm_config(
                parallel={
                    "distributed_executor_backend": "ray",
                    "ray_runtime_env": None,
                },
                model={"multimodal_config": SimpleNamespace()},
            )
            # Does not raise: multimodal_config is None after normalize.
            MetalPlatform.check_and_update_config(vllm_config)
        finally:
            reset_config()

    def test_check_and_update_config_rejects_dp_moe(self) -> None:
        """MoE data parallelism (expert-parallel all-to-all) is unsupported."""
        with pytest.raises(NotImplementedError, match="dense models only"):
            MetalPlatform.check_and_update_config(
                self._dp_vllm_config(model={"is_moe": True})
            )

    def test_check_and_update_config_rejects_dp_mp_backend(self) -> None:
        """DP across Macs requires the Ray DP backend; mp cannot span nodes."""
        with pytest.raises(NotImplementedError, match="Ray DP backend"):
            MetalPlatform.check_and_update_config(
                self._dp_vllm_config(parallel={"data_parallel_backend": "mp"})
            )

    def test_check_and_update_config_rejects_dp_size_local(self) -> None:
        """One Apple GPU per node: exactly one DP replica per node (reject > 1)."""
        with pytest.raises(
            NotImplementedError, match="exactly one data-parallel replica"
        ):
            MetalPlatform.check_and_update_config(
                self._dp_vllm_config(parallel={"data_parallel_size_local": 2})
            )

    def test_check_and_update_config_rejects_dp_size_local_external_sentinel(
        self,
    ) -> None:
        """data_parallel_size_local==0 is upstream's externally-specified-DP
        sentinel (headless / front-end-only). Metal never validated that topology,
        so the guard must reject 0 too, not only > 1."""
        with pytest.raises(
            NotImplementedError, match="exactly one data-parallel replica"
        ):
            MetalPlatform.check_and_update_config(
                self._dp_vllm_config(parallel={"data_parallel_size_local": 0})
            )

    def test_check_and_update_config_rejects_dp_external_lb(self) -> None:
        """Only the default internal LB is supported under DP."""
        with pytest.raises(NotImplementedError, match="internal load balancer"):
            MetalPlatform.check_and_update_config(
                self._dp_vllm_config(parallel={"data_parallel_external_lb": True})
            )

    def test_check_and_update_config_rejects_dp_hybrid_lb(self) -> None:
        """The hybrid load balancer is also rejected under DP."""
        with pytest.raises(NotImplementedError, match="internal load balancer"):
            MetalPlatform.check_and_update_config(
                self._dp_vllm_config(parallel={"data_parallel_hybrid_lb": True})
            )

    def test_check_and_update_config_rejects_dp_with_pipeline_parallel(self) -> None:
        """DP combined with PP is not validated (per-replica ring scoping/ports)."""
        with pytest.raises(
            NotImplementedError, match="combining data parallelism with"
        ):
            MetalPlatform.check_and_update_config(
                self._dp_vllm_config(parallel={"pipeline_parallel_size": 2})
            )

    def test_check_and_update_config_rejects_dp_speculative_decoding(self) -> None:
        """DP with speculative decoding is unvalidated; reject at config time."""
        with pytest.raises(NotImplementedError, match="speculative decoding"):
            MetalPlatform.check_and_update_config(
                self._dp_vllm_config(
                    speculative_config=SimpleNamespace(
                        method="ngram",
                        use_heterogeneous_vocab=False,
                        num_speculative_tokens=3,
                    )
                )
            )

    def test_check_and_update_config_rejects_dp_lora(self) -> None:
        """DP with LoRA is unvalidated; reject at config time."""
        with pytest.raises(NotImplementedError, match="data parallelism with LoRA"):
            MetalPlatform.check_and_update_config(
                self._dp_vllm_config(lora_config=SimpleNamespace(max_loras=1))
            )

    def test_check_and_update_config_rejects_heterogeneous_kv_cache_dtypes(
        self,
    ) -> None:
        """Skip-layer KV dtypes are unsupported; reject at config time.

        Upstream would otherwise resize the block pool behind Metal's own
        TurboQuant sizing, from inside a spawned worker.
        """
        vllm_config = self._dp_vllm_config()
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.cache_config.kv_cache_dtype_skip_layers = ["0", "31"]

        with pytest.raises(NotImplementedError, match="heterogeneous KV cache dtypes"):
            MetalPlatform.check_and_update_config(vllm_config)

    def test_check_and_update_config_rejects_heterogeneous_draft_vocab(self) -> None:
        """A draft vocabulary that differs from the target is unsupported.

        Upstream stopped verifying equal vocab sizes when the flag is set, so the
        proposer would otherwise verify draft ids against the target vocabulary
        with no mapping.
        """
        vllm_config = self._dp_vllm_config()
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.speculative_config = SimpleNamespace(use_heterogeneous_vocab=True)

        with pytest.raises(NotImplementedError, match="heterogeneous draft vocabulary"):
            MetalPlatform.check_and_update_config(vllm_config)

    def test_check_and_update_config_rejects_clipped_spec_decode_padding(
        self,
    ) -> None:
        """A long-prefill threshold below the speculative width is unsupported.

        The scheduler clips a padded decode request to the threshold while still
        attaching the full placeholder draft list, so the handoff's token
        accounting no longer balances.
        """
        vllm_config = self._dp_vllm_config(
            speculative_config=SimpleNamespace(
                method="ngram",
                use_heterogeneous_vocab=False,
                num_speculative_tokens=5,
            )
        )
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.scheduler_config.long_prefill_token_threshold = 3

        with pytest.raises(NotImplementedError, match="long-prefill-token-threshold"):
            MetalPlatform.check_and_update_config(vllm_config)

    def test_check_and_update_config_rejects_dp_multimodal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A genuine multimodal model (multimodal_config survives normalize) is
        rejected under DP — the tensor-IPC path is DP=1 only."""
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        init_calls = self._stub_ray(monkeypatch)
        # normalize leaves multimodal_config in place (genuine multimodal model).
        monkeypatch.setattr(
            "vllm_metal.v1.model_adapter.DefaultModelAdapter.normalize_model_config",
            lambda _self, _mc: None,
        )
        reset_config()
        try:
            vllm_config = self._dp_vllm_config(
                parallel={
                    "distributed_executor_backend": "ray",
                    "ray_runtime_env": None,
                },
                model={"multimodal_config": SimpleNamespace()},
            )
            with pytest.raises(NotImplementedError, match="multimodal models"):
                MetalPlatform.check_and_update_config(vllm_config)
            # Fail fast before any ray.init side effect.
            assert init_calls == []
        finally:
            reset_config()

    def test_check_and_update_config_rejects_dp_stt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """STT models use a dedicated runner with no DP path; reject DP."""
        self._patch_stt_resolution(monkeypatch, is_stt=True)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        init_calls = self._stub_ray(monkeypatch)
        reset_config()
        try:
            vllm_config = self._dp_vllm_config(
                parallel={
                    "distributed_executor_backend": "ray",
                    "ray_runtime_env": None,
                }
            )
            with pytest.raises(NotImplementedError, match="speech-to-text"):
                MetalPlatform.check_and_update_config(vllm_config)
            # Fail fast before any ray.init side effect.
            assert init_calls == []
        finally:
            reset_config()

    def test_register_dp_hook_is_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With our hook already registered and the Ray session still live,
        re-registering is a no-op."""
        ray = pytest.importorskip("ray")
        monkeypatch.setattr(MetalPlatform, "_dp_ray_hook_registered", True)
        monkeypatch.setattr(ray, "is_initialized", lambda: True)
        init_calls: list[dict] = []
        monkeypatch.setattr(ray, "init", lambda **kwargs: init_calls.append(kwargs))
        MetalPlatform._register_dp_ray_worker_setup_hook()
        assert init_calls == []

    def test_register_dp_hook_reregisters_after_ray_shutdown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A registered flag left True after ray.shutdown() is stale: if our Ray
        session is gone, re-init with the hook so a second in-process DP engine still
        gets the Metal worker patch — otherwise the DP manager rebuilds Ray with a
        hook-less ray.init and the workers KeyError on the mlx resource."""
        ray = pytest.importorskip("ray")
        monkeypatch.setattr(MetalPlatform, "_dp_ray_hook_registered", True)
        monkeypatch.setattr(ray, "is_initialized", lambda: False)
        init_calls: list[dict] = []
        monkeypatch.setattr(ray, "init", lambda **kwargs: init_calls.append(kwargs))
        MetalPlatform._register_dp_ray_worker_setup_hook()
        assert init_calls, "stale flag + shut-down Ray must trigger a re-init"
        assert (
            init_calls[0]["runtime_env"]["worker_process_setup_hook"]
            == MetalPlatform._RAY_WORKER_SETUP_HOOK
        )
        assert MetalPlatform._dp_ray_hook_registered is True

    def test_register_dp_hook_rejects_foreign_ray_init(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If Ray was already initialized by something else (no setup hook), the
        DP workers would miss the patch — fail loud instead of silently proceeding."""
        ray = pytest.importorskip("ray")
        monkeypatch.setattr(MetalPlatform, "_dp_ray_hook_registered", False)
        monkeypatch.setattr(ray, "is_initialized", lambda: True)
        monkeypatch.setattr(ray, "init", lambda **kwargs: None)
        with pytest.raises(RuntimeError, match="already initialized"):
            MetalPlatform._register_dp_ray_worker_setup_hook()

    def test_register_dp_hook_rejects_foreign_worker_hook(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A foreign worker_process_setup_hook in ray_runtime_env is rejected
        (chaining unsupported), matching the single-stage Ray path — the DP job-level
        registration must not silently replace a user-set hook."""
        ray = pytest.importorskip("ray")
        monkeypatch.setattr(MetalPlatform, "_dp_ray_hook_registered", False)
        monkeypatch.setattr(ray, "is_initialized", lambda: False)
        init_calls: list[dict] = []
        monkeypatch.setattr(ray, "init", lambda **kwargs: init_calls.append(kwargs))
        with pytest.raises(ValueError, match="chaining is not supported"):
            MetalPlatform._register_dp_ray_worker_setup_hook(
                {"worker_process_setup_hook": "some.other.hook"}
            )
        # Rejected before any ray.init side effect.
        assert init_calls == []

    def test_dp_hook_registration_preserves_user_ray_runtime_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A user-provided ray_runtime_env (env_vars / working_dir / py_modules the
        remote Macs need) is merged with the worker hook, not replaced by a
        hook-only env: the DP manager reuses this job session without re-applying
        ray_runtime_env, so dropping the user's keys would start remote actors
        without the requested environment."""
        pytest.importorskip("ray")
        from ray.runtime_env import RuntimeEnv

        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        init_calls = self._stub_ray(monkeypatch)
        reset_config()
        try:
            vllm_config = self._dp_vllm_config(
                parallel={
                    "distributed_executor_backend": "ray",
                    "ray_runtime_env": RuntimeEnv(env_vars={"VLLM_METAL_DP": "1"}),
                }
            )
            MetalPlatform.check_and_update_config(vllm_config)
            assert init_calls, "DP must register the worker hook via ray.init"
            runtime_env = init_calls[0]["runtime_env"]
            # Both the user's env and our worker hook reach the Ray job.
            assert runtime_env["env_vars"] == {"VLLM_METAL_DP": "1"}
            assert (
                runtime_env["worker_process_setup_hook"]
                == MetalPlatform._RAY_WORKER_SETUP_HOOK
            )
        finally:
            reset_config()

    def test_check_and_update_config_dp_binds_real_parallel_config_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The DP admission reads real ``vllm.config.ParallelConfig`` fields, not
        only SimpleNamespace stand-ins: a real DP-over-Ray config in the supported
        shape is admitted (and registers the job-level hook), while a real config
        with the external LB flag is rejected. Pins the field-name contract so an
        upstream rename of the ``data_parallel_*`` fields fails this unit test, not
        only a live cluster run."""
        pytest.importorskip("ray")
        # Reject path: a real config with external LB must fail fast at the guard.
        reject_pc = ParallelConfig(
            data_parallel_size=2,
            data_parallel_backend="ray",
            data_parallel_size_local=1,
            data_parallel_external_lb=True,
        )
        with pytest.raises(NotImplementedError, match="internal load balancer"):
            MetalPlatform.check_and_update_config(
                self._dp_vllm_config(parallel_config=reject_pc)
            )

        # Admit path: the supported shape on a real config is accepted and
        # registers the job-level Ray worker hook.
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        init_calls = self._stub_ray(monkeypatch)
        admit_pc = ParallelConfig(
            data_parallel_size=2,
            data_parallel_backend="ray",
            data_parallel_size_local=1,
        )
        reset_config()
        try:
            MetalPlatform.check_and_update_config(
                self._dp_vllm_config(parallel_config=admit_pc)
            )
            assert init_calls, "DP must register the worker hook via ray.init"
        finally:
            reset_config()

    def test_get_attn_backend_cls_returns_cpu_backend(self) -> None:
        """Metal platform should return a concrete backend path."""
        cfg = AttentionSelectorConfig(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
        )
        backend = MetalPlatform.get_attn_backend_cls(AttentionBackendEnum.CPU_ATTN, cfg)
        assert backend == AttentionBackendEnum.CPU_ATTN.get_path()

    def test_get_attn_backend_cls_accepts_mla(self) -> None:
        """MLA is handled by the vllm-metal model runner; CPU_ATTN is returned."""
        cfg = AttentionSelectorConfig(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=True,
        )
        backend = MetalPlatform.get_attn_backend_cls(AttentionBackendEnum.CPU_ATTN, cfg)
        assert backend == AttentionBackendEnum.CPU_ATTN.get_path()

    def test_get_attn_backend_cls_rejects_sparse(self) -> None:
        """Sparse attention is not supported on Metal/MLX."""
        cfg = AttentionSelectorConfig(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_sparse=True,
        )
        with pytest.raises(
            NotImplementedError, match="Sparse Attention is not supported"
        ):
            MetalPlatform.get_attn_backend_cls(AttentionBackendEnum.CPU_ATTN, cfg)

    def test_memory_info(self) -> None:
        """Test memory information."""
        total = MetalPlatform.get_device_total_memory()
        available = MetalPlatform.get_device_available_memory()

        assert total > 0
        assert available > 0
        assert available <= total

    @pytest.mark.skipif(
        platform.machine() != "arm64" or platform.system() != "Darwin",
        reason="Only runs on Apple Silicon",
    )
    def test_is_available(self) -> None:
        """Test platform availability on Apple Silicon."""
        assert MetalPlatform.is_available() is True

    def test_is_available_does_not_mutate_default_device(self) -> None:
        """Availability check should not change the MLX default device."""
        mx = pytest.importorskip("mlx.core")

        before = mx.default_device()
        MetalPlatform.is_available()
        after = mx.default_device()

        assert before == after

    def test_is_available_propagates_unexpected_mlx_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unexpected MLX errors should surface instead of looking unavailable."""
        monkeypatch.setattr("vllm_metal.platform.py_platform.machine", lambda: "arm64")
        monkeypatch.setattr("vllm_metal.platform.py_platform.system", lambda: "Darwin")

        mlx_module = ModuleType("mlx")
        mlx_core = ModuleType("mlx.core")

        class _BrokenMetal:
            @staticmethod
            def is_available() -> bool:
                raise ValueError("unexpected mlx regression")

        mlx_core.metal = _BrokenMetal()
        mlx_module.core = mlx_core
        monkeypatch.setitem(sys.modules, "mlx", mlx_module)
        monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

        with pytest.raises(ValueError, match="unexpected mlx regression"):
            MetalPlatform.is_available()

    def test_torch_device(self) -> None:
        """Test PyTorch device retrieval."""

        device = MetalPlatform.get_torch_device()
        assert device.type in ("mps", "cpu")

    def test_check_and_update_config_disables_chunked_prefill_non_paged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-paged path should disable chunked prefill.

        When chunked prefill is disabled, max_num_batched_tokens must be at
        least max_model_len so the scheduler can schedule the entire prompt
        in a single step.
        """
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "0")
        reset_config()
        try:
            vllm_config = SimpleNamespace(
                speculative_config=None,
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    kv_cache_dtype_skip_layers=[],
                    block_size=None,
                ),
                model_config=SimpleNamespace(
                    model="test-model",
                    disable_cascade_attn=False,
                    tokenizer=None,
                    max_model_len=32768,
                    multimodal_config=None,
                    hf_config=SimpleNamespace(model_type="qwen3"),
                    is_hybrid=False,
                ),
                scheduler_config=SimpleNamespace(
                    async_scheduling=True,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=2048,
                    max_num_scheduled_tokens=None,
                ),
            )

            MetalPlatform.check_and_update_config(vllm_config)

            assert vllm_config.scheduler_config.enable_chunked_prefill is False
            assert vllm_config.scheduler_config.max_num_batched_tokens == 32768
            assert (
                vllm_config.parallel_config.worker_cls
                == "vllm_metal.v1.worker.MetalWorker"
            )
            assert vllm_config.parallel_config.distributed_executor_backend == "uni"
            assert vllm_config.parallel_config.disable_custom_all_reduce is True
        finally:
            reset_config()

    def test_check_and_update_config_keeps_chunked_prefill_for_paged_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Paged path should keep chunked prefill enabled.

        The unified varlen Metal kernel handles mixed prefill + decode,
        so chunked prefill works correctly on the paged path.
        """
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        reset_config()
        try:
            vllm_config = SimpleNamespace(
                speculative_config=None,
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    kv_cache_dtype_skip_layers=[],
                    block_size=None,
                    enable_prefix_caching=False,
                ),
                model_config=SimpleNamespace(
                    model="test-model",
                    disable_cascade_attn=False,
                    tokenizer=None,
                    max_model_len=32768,
                    multimodal_config=None,
                    hf_config=SimpleNamespace(model_type="qwen3"),
                    is_hybrid=False,
                ),
                scheduler_config=SimpleNamespace(
                    async_scheduling=True,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=2048,
                    max_num_scheduled_tokens=None,
                ),
            )

            MetalPlatform.check_and_update_config(vllm_config)

            assert vllm_config.scheduler_config.enable_chunked_prefill is True
            # max_num_batched_tokens should NOT be bumped (chunked prefill handles it)
            assert vllm_config.scheduler_config.max_num_batched_tokens == 2048
        finally:
            reset_config()

    def _hybrid_vllm_config(
        self,
        cache_config: SimpleNamespace,
        speculative_config: SimpleNamespace | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            speculative_config=speculative_config,
            lora_config=None,
            parallel_config=SimpleNamespace(
                worker_cls="auto",
                distributed_executor_backend="auto",
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
                disable_custom_all_reduce=False,
            ),
            cache_config=cache_config,
            model_config=SimpleNamespace(
                model="test-model",
                disable_cascade_attn=False,
                tokenizer=None,
                max_model_len=32768,
                multimodal_config=None,
                hf_config=SimpleNamespace(model_type="qwen3"),
                is_hybrid=True,
            ),
            scheduler_config=SimpleNamespace(
                async_scheduling=True,
                enable_chunked_prefill=True,
                max_num_batched_tokens=2048,
                max_num_scheduled_tokens=None,
                long_prefill_token_threshold=0,
            ),
        )

    @pytest.mark.parametrize(
        "paged,cache_config,speculative,err_match",
        [
            (
                "0",
                SimpleNamespace(
                    block_size=None,
                    kv_cache_dtype_skip_layers=[],
                    enable_prefix_caching=True,
                    mamba_cache_mode="none",
                    mamba_ssm_cache_dtype="float32",
                ),
                None,
                "requires paged",
            ),
            (
                "1",
                SimpleNamespace(
                    block_size=None,
                    kv_cache_dtype_skip_layers=[],
                    enable_prefix_caching=False,
                    mamba_cache_mode="all",
                    mamba_ssm_cache_dtype="float32",
                ),
                None,
                "mamba_cache_mode='all'",
            ),
            (
                "1",
                SimpleNamespace(
                    block_size=None,
                    kv_cache_dtype_skip_layers=[],
                    enable_prefix_caching=True,
                    mamba_cache_mode="align",
                    mamba_ssm_cache_dtype="float32",
                ),
                SimpleNamespace(
                    use_heterogeneous_vocab=False,
                    num_speculative_tokens=2,
                ),
                "speculative decoding",
            ),
        ],
        ids=["prefix_caching_non_paged", "mamba_cache_mode_all", "prefix_and_sd"],
    )
    def test_check_and_update_config_rejects_unsupported_hybrid_cache_modes(
        self,
        paged: str,
        cache_config: SimpleNamespace,
        speculative: SimpleNamespace | None,
        err_match: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", paged)
        reset_config()
        try:
            vllm_config = self._hybrid_vllm_config(
                cache_config, speculative_config=speculative
            )
            with pytest.raises(NotImplementedError, match=err_match):
                MetalPlatform.check_and_update_config(vllm_config)
        finally:
            reset_config()

    def test_check_and_update_config_accepts_hybrid_align_prefix_caching(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Paged hybrid + prefix caching (align mode) passes config checks."""
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        reset_config()
        try:
            vllm_config = self._hybrid_vllm_config(
                SimpleNamespace(
                    block_size=None,
                    kv_cache_dtype_skip_layers=[],
                    enable_prefix_caching=True,
                    mamba_cache_mode="align",
                    mamba_ssm_cache_dtype="float32",
                ),
            )
            MetalPlatform.check_and_update_config(vllm_config)
        finally:
            reset_config()

    @pytest.mark.parametrize(
        ("dtype", "reject"),
        [("auto", False), ("float16", True), ("bfloat16", True)],
    )
    def test_check_and_update_config_hybrid_gdn_state_dtype(
        self, dtype: str, reject: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_config = CacheConfig(
            enable_prefix_caching=False, mamba_ssm_cache_dtype=dtype
        )
        vllm_config = self._hybrid_vllm_config(cache_config)

        if reject:
            with pytest.raises(NotImplementedError, match="require.*float32"):
                MetalPlatform.check_and_update_config(vllm_config)
        else:
            self._patch_stt_resolution(monkeypatch, is_stt=False)
            MetalPlatform.check_and_update_config(vllm_config)
            assert cache_config.mamba_ssm_cache_dtype == "float32"

    def test_check_and_update_config_increases_max_num_scheduled_tokens_below_max_model_len(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """max_num_scheduled_tokens below max_model_len should be bumped up to max_model_len.

        When max_num_scheduled_tokens is explicitly set to a value smaller
        than max_model_len, it must be raised to match max_model_len so that
        the scheduler can schedule the full prompt in a single step.
        """
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "0")
        reset_config()
        try:
            vllm_config = SimpleNamespace(
                speculative_config=None,
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    kv_cache_dtype_skip_layers=[],
                    block_size=None,
                ),
                model_config=SimpleNamespace(
                    model="test-model",
                    disable_cascade_attn=False,
                    tokenizer=None,
                    max_model_len=32768,
                    multimodal_config=None,
                    hf_config=SimpleNamespace(model_type="qwen3"),
                    is_hybrid=False,
                ),
                scheduler_config=SimpleNamespace(
                    async_scheduling=True,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=2048,
                    max_num_scheduled_tokens=2048,
                ),
            )

            MetalPlatform.check_and_update_config(vllm_config)

            assert vllm_config.scheduler_config.enable_chunked_prefill is False
            assert vllm_config.scheduler_config.max_num_batched_tokens == 32768
            assert vllm_config.scheduler_config.max_num_scheduled_tokens == 32768
        finally:
            reset_config()

    def test_check_and_update_config_does_not_reduce_large_max_num_batched_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """max_num_batched_tokens must not be lowered when already >= max_model_len.

        If the user has explicitly set a token budget larger than max_model_len,
        that setting must be preserved.
        """
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "0")
        reset_config()
        try:
            vllm_config = SimpleNamespace(
                speculative_config=None,
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    kv_cache_dtype_skip_layers=[],
                    block_size=None,
                ),
                model_config=SimpleNamespace(
                    model="test-model",
                    disable_cascade_attn=False,
                    tokenizer=None,
                    max_model_len=32768,
                    multimodal_config=None,
                    hf_config=SimpleNamespace(model_type="qwen3"),
                    is_hybrid=False,
                ),
                scheduler_config=SimpleNamespace(
                    async_scheduling=True,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=65536,
                    max_num_scheduled_tokens=None,
                ),
            )

            MetalPlatform.check_and_update_config(vllm_config)

            assert vllm_config.scheduler_config.enable_chunked_prefill is False
            # 65536 > 32768, so the value must stay at 65536
            assert vllm_config.scheduler_config.max_num_batched_tokens == 65536
        finally:
            reset_config()

    @pytest.mark.parametrize("max_num_scheduled_tokens", [32768, 65536])
    def test_check_and_update_config_does_not_reduce_max_num_scheduled_tokens_when_at_least_max_model_len(
        self,
        monkeypatch: pytest.MonkeyPatch,
        max_num_scheduled_tokens: int,
    ) -> None:
        """max_num_scheduled_tokens must not be lowered when already >= max_model_len.

        If the user has explicitly set a scheduled-token budget at least
        max_model_len, that setting must be preserved (only values strictly
        below max_model_len are bumped up).
        """
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "0")
        reset_config()
        try:
            vllm_config = SimpleNamespace(
                speculative_config=None,
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    kv_cache_dtype_skip_layers=[],
                    block_size=None,
                ),
                model_config=SimpleNamespace(
                    model="test-model",
                    disable_cascade_attn=False,
                    tokenizer=None,
                    max_model_len=32768,
                    multimodal_config=None,
                    hf_config=SimpleNamespace(model_type="qwen3"),
                    is_hybrid=False,
                ),
                scheduler_config=SimpleNamespace(
                    async_scheduling=True,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=65536,
                    max_num_scheduled_tokens=max_num_scheduled_tokens,
                ),
            )

            MetalPlatform.check_and_update_config(vllm_config)

            assert vllm_config.scheduler_config.enable_chunked_prefill is False
            assert vllm_config.scheduler_config.max_num_batched_tokens == 65536
            assert (
                vllm_config.scheduler_config.max_num_scheduled_tokens
                == max_num_scheduled_tokens
            )
        finally:
            reset_config()

    def test_check_and_update_config_applies_stt_scheduler_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """STT models should get tokenizer fallback and async scheduling disabled."""
        self._patch_stt_resolution(monkeypatch, is_stt=True)
        vllm_config = SimpleNamespace(
            speculative_config=None,
            parallel_config=SimpleNamespace(
                worker_cls="auto",
                distributed_executor_backend="auto",
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
                disable_custom_all_reduce=False,
            ),
            cache_config=SimpleNamespace(
                kv_cache_dtype_skip_layers=[],
                block_size=None,
            ),
            model_config=SimpleNamespace(
                model="openai/whisper-tiny",
                disable_cascade_attn=False,
                tokenizer=None,
                multimodal_config=None,
                hf_config=SimpleNamespace(model_type="whisper"),
                is_hybrid=False,
            ),
            scheduler_config=SimpleNamespace(
                async_scheduling=True,
                enable_chunked_prefill=False,
            ),
        )

        MetalPlatform.check_and_update_config(vllm_config)

        assert vllm_config.model_config.tokenizer == "openai/whisper-tiny"
        assert vllm_config.scheduler_config.async_scheduling is False

    def test_check_and_update_config_preserves_existing_tokenizer_for_stt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """STT policy should not overwrite an explicitly configured tokenizer."""
        self._patch_stt_resolution(monkeypatch, is_stt=True)
        vllm_config = SimpleNamespace(
            speculative_config=None,
            parallel_config=SimpleNamespace(
                worker_cls="auto",
                distributed_executor_backend="auto",
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
                disable_custom_all_reduce=False,
            ),
            cache_config=SimpleNamespace(
                kv_cache_dtype_skip_layers=[],
                block_size=None,
            ),
            model_config=SimpleNamespace(
                model="openai/whisper-tiny",
                disable_cascade_attn=False,
                tokenizer="custom-tokenizer",
                multimodal_config=None,
                hf_config=SimpleNamespace(model_type="whisper"),
                is_hybrid=False,
            ),
            scheduler_config=SimpleNamespace(
                async_scheduling=True,
                enable_chunked_prefill=False,
            ),
        )

        MetalPlatform.check_and_update_config(vllm_config)

        assert vllm_config.model_config.tokenizer == "custom-tokenizer"
        assert vllm_config.scheduler_config.async_scheduling is False

    def test_check_and_update_config_clears_multimodal_for_text_backbone_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Gemma4-style multimodal configs must be cleared for the text-only path.

        Gemma4 MLX checkpoints are flagged multimodal in the HF config but
        ship without the vision/audio preprocessor files that vLLM's input
        processor tries to load. Clearing ``multimodal_config`` at the
        platform layer makes ``is_multimodal_model`` False so the input
        processor skips feature-extractor loading.
        """
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        reset_config()
        try:
            model_config = SimpleNamespace(
                model="test-model",
                disable_cascade_attn=False,
                tokenizer=None,
                max_model_len=128,
                multimodal_config=SimpleNamespace(language_model_only=False),
                hf_config=SimpleNamespace(model_type="gemma4"),
                is_hybrid=False,
            )
            vllm_config = SimpleNamespace(
                speculative_config=None,
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    kv_cache_dtype_skip_layers=[],
                    block_size=None,
                    enable_prefix_caching=False,
                ),
                model_config=model_config,
                scheduler_config=SimpleNamespace(
                    async_scheduling=False,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=2048,
                    max_num_scheduled_tokens=None,
                ),
            )

            MetalPlatform.check_and_update_config(vllm_config)

            assert model_config.multimodal_config is None

        finally:
            reset_config()

    def test_check_and_update_config_auto_mode_clears_qwen35_fp8_wrapper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        reset_config()
        try:
            model_config = SimpleNamespace(
                model="test-model",
                disable_cascade_attn=False,
                tokenizer=None,
                max_model_len=128,
                multimodal_config=SimpleNamespace(language_model_only=False),
                hf_config=SimpleNamespace(
                    model_type="qwen3_5",
                    architectures=["Qwen3_5ForConditionalGeneration"],
                    quantization_config={"quant_method": "fp8"},
                ),
                is_hybrid=False,
            )
            vllm_config = SimpleNamespace(
                speculative_config=None,
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    kv_cache_dtype_skip_layers=[],
                    block_size=None,
                    enable_prefix_caching=False,
                ),
                model_config=model_config,
                scheduler_config=SimpleNamespace(
                    async_scheduling=False,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=2048,
                    max_num_scheduled_tokens=None,
                ),
            )

            MetalPlatform.check_and_update_config(vllm_config)

            assert model_config.multimodal_config is None

        finally:
            reset_config()

    def test_check_and_update_config_preserves_multimodal_for_non_gemma4_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-overridden multimodal models must keep multimodal_config set."""
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        reset_config()
        try:
            sentinel = SimpleNamespace(language_model_only=False)
            model_config = SimpleNamespace(
                model="test-model",
                disable_cascade_attn=False,
                tokenizer=None,
                max_model_len=128,
                multimodal_config=sentinel,
                hf_config=SimpleNamespace(model_type="qwen3_vl", architectures=[]),
                is_hybrid=False,
            )
            vllm_config = SimpleNamespace(
                speculative_config=None,
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    kv_cache_dtype_skip_layers=[],
                    block_size=None,
                    enable_prefix_caching=False,
                ),
                model_config=model_config,
                scheduler_config=SimpleNamespace(
                    async_scheduling=False,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=2048,
                    max_num_scheduled_tokens=None,
                ),
            )

            MetalPlatform.check_and_update_config(vllm_config)

            assert model_config.multimodal_config is sentinel

        finally:
            reset_config()

    def test_check_and_update_config_multimodal_native_preserves_qwen35_fp8(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "multimodal-native")
        reset_config()
        try:
            sentinel = SimpleNamespace(language_model_only=False)
            model_config = SimpleNamespace(
                model="test-model",
                disable_cascade_attn=False,
                tokenizer=None,
                max_model_len=128,
                multimodal_config=sentinel,
                hf_config=SimpleNamespace(
                    model_type="qwen3_5",
                    architectures=["Qwen3_5ForConditionalGeneration"],
                    quantization_config={"quant_method": "fp8"},
                ),
                is_hybrid=False,
            )
            vllm_config = SimpleNamespace(
                speculative_config=None,
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    kv_cache_dtype_skip_layers=[],
                    block_size=None,
                    enable_prefix_caching=False,
                ),
                model_config=model_config,
                scheduler_config=SimpleNamespace(
                    async_scheduling=False,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=2048,
                    max_num_scheduled_tokens=None,
                ),
            )

            MetalPlatform.check_and_update_config(vllm_config)

            assert model_config.multimodal_config is sentinel

        finally:
            reset_config()

    def test_synchronize_runs_mlx_barrier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Platform synchronize should use the pinned MLX barrier."""
        mx = pytest.importorskip("mlx.core")

        called = False

        def fake_sync() -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(mx, "synchronize", fake_sync)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        MetalPlatform.synchronize()
        assert called is True


class TestKvBudgetBytes:
    """Tests for paged-attention base KV budget calculation.

    Numbers mirror a real M2 Max with GLM-4.7-Flash-4bit loaded:
      metal_limit = 22.9 GB (max_recommended_working_set_size)
      model_memory = 16.85 GB (mx.get_active_memory() after load)
    """

    _METAL_LIMIT = int(22.9e9)
    _MODEL_MEM = int(16.85e9)
    # Simulated measured overhead (matches what profile_run would return).
    _OVERHEAD = 200 * 1024 * 1024  # 200 MB

    def test_normal_case(self) -> None:
        budget = WorkerCachePlanner.base_kv_budget_bytes(
            self._METAL_LIMIT,
            self._MODEL_MEM,
            fraction=0.9,
            overhead=self._OVERHEAD,
        )

        assert budget == int(self._METAL_LIMIT * 0.9) - self._MODEL_MEM - self._OVERHEAD
        assert budget > 0

    def test_fraction_too_low_yields_negative_budget(self) -> None:
        # fraction=0.3 → usable=6.9 GB < model(16.85 GB) → negative
        budget = WorkerCachePlanner.base_kv_budget_bytes(
            self._METAL_LIMIT,
            self._MODEL_MEM,
            fraction=0.3,
            overhead=self._OVERHEAD,
        )

        assert budget < 0

    def test_boundary_zero(self) -> None:
        # Craft inputs so budget lands exactly at zero.
        limit = self._MODEL_MEM + self._OVERHEAD

        budget = WorkerCachePlanner.base_kv_budget_bytes(
            limit, self._MODEL_MEM, fraction=1.0, overhead=self._OVERHEAD
        )

        assert budget == 0

    def test_custom_overhead(self) -> None:
        budget_zero_overhead = WorkerCachePlanner.base_kv_budget_bytes(
            self._METAL_LIMIT, self._MODEL_MEM, fraction=0.9, overhead=0
        )
        budget_with_overhead = WorkerCachePlanner.base_kv_budget_bytes(
            self._METAL_LIMIT,
            self._MODEL_MEM,
            fraction=0.9,
            overhead=self._OVERHEAD,
        )

        assert budget_zero_overhead - budget_with_overhead == self._OVERHEAD

    def test_large_model_has_positive_budget_at_default_fraction(self) -> None:
        # GLM-4.7-Flash-4bit at fraction=0.9 must yield > 1 GB for KV cache.
        budget = WorkerCachePlanner.base_kv_budget_bytes(
            self._METAL_LIMIT,
            self._MODEL_MEM,
            fraction=0.9,
            overhead=self._OVERHEAD,
        )

        assert budget > 1e9


class TestAutoFitMaxModelLenChain:
    """The -1 sentinel drives the Metal null-block auto-fit contract.

    Builds the gemma-4-31B mixed-MHA KV shape and runs vLLM's
    ``get_kv_cache_configs`` against fixed synthetic memory budgets. These
    tests do not re-test upstream's fitting algorithm; they check Metal's
    null-block reservation, too-small-pool failure, and current mixed-layout
    budget shape for issue #505.
    """

    _NUM_LAYERS = 60
    _MAX_BATCH_TOKENS = 2048

    @pytest.fixture(autouse=True)
    def _ensure_compat_patches(self) -> None:
        """The null-block auto-fit patch (compat.py) is part of the contract
        under test; ensure it directly because plugin activation can skip it
        while vLLM is partially imported."""
        compat.ensure_vllm_auto_fit_null_block_patch()

    # 16 tokens x (50 x 16 x 256 + 10 x 4 x 512) heads*dims x K/V x bf16 —
    # equals the packed per-block bytes the Metal pool reports.
    _PACKED_BLOCK_BYTES = 14_417_920
    _DERIVED_MAX_LEN = 262_144
    _FULL_CONTEXT_BUDGET_BYTES = int(22.5 * 1024**3)

    def gemma4_config_and_specs(
        self, *, original_max_model_len: int | None
    ) -> tuple[VllmConfig, dict[str, KVCacheSpec]]:
        runner = make_gemma4_mixed_mha_runner(
            num_layers=self._NUM_LAYERS,
            sliding_kv_heads=16,
            full_kv_heads=4,
            max_model_len=self._DERIVED_MAX_LEN,
            original_max_model_len=original_max_model_len,
            max_in_flight_tokens=self._MAX_BATCH_TOKENS,
        )
        return runner.vllm_config, runner.get_kv_cache_spec()

    def test_reduced_length_reserves_the_null_block(self) -> None:
        """The fitted length must never need the whole pool: BlockPool keeps
        block 0 as the null placeholder, so a request at the fitted length
        would otherwise starve in the scheduler forever."""
        num_pool_blocks = 1500
        available = num_pool_blocks * self._PACKED_BLOCK_BYTES
        vllm_config, specs = self.gemma4_config_and_specs(original_max_model_len=-1)

        config = get_kv_cache_configs(vllm_config, [specs], [available])[0]

        capacity, _ = get_kv_cache_capacity(vllm_config, config)
        assert vllm_config.model_config.max_model_len < self._DERIVED_MAX_LEN
        assert capacity > vllm_config.model_config.max_model_len

    def test_mixed_layout_keeps_full_context_under_sufficient_budget(self) -> None:
        available = self._FULL_CONTEXT_BUDGET_BYTES
        vllm_config, specs = self.gemma4_config_and_specs(
            original_max_model_len=self._DERIVED_MAX_LEN
        )

        config = get_kv_cache_configs(vllm_config, [specs], [available])[0]

        full_groups = [
            group
            for group in config.kv_cache_groups
            if type(group.kv_cache_spec) is FullAttentionSpec
        ]
        sliding_groups = [
            group
            for group in config.kv_cache_groups
            if type(group.kv_cache_spec) is SlidingWindowSpec
        ]
        assert len(sliding_groups) == 5
        assert len(full_groups) == 1
        assert sum(len(group.layer_names) for group in sliding_groups) == 50
        assert sum(len(group.layer_names) for group in full_groups) == 10
        assert len(config.kv_cache_tensors) == 10
        capacity, _ = get_kv_cache_capacity(vllm_config, config)
        assert vllm_config.model_config.max_model_len == self._DERIVED_MAX_LEN
        assert capacity >= self._DERIVED_MAX_LEN
        assert sum(tensor.size for tensor in config.kv_cache_tensors) <= available

    def test_insufficient_memory_for_one_block_raises(self) -> None:
        available = self._PACKED_BLOCK_BYTES - 1
        vllm_config, specs = self.gemma4_config_and_specs(original_max_model_len=-1)

        with pytest.raises(ValueError, match="Cannot auto-fit max_model_len"):
            get_kv_cache_configs(vllm_config, [specs], [available])
