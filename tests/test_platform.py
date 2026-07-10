# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal platform."""

import importlib
import platform
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.config import ParallelConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig

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

    def test_check_and_update_config_rejects_pipeline_with_tensor_parallel(
        self,
    ) -> None:
        """PP is allowed, but combining PP>1 with TP>1 is rejected at config time."""
        vllm_config = SimpleNamespace(
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

    def test_check_and_update_config_allows_pipeline_parallel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PP>1 with TP=1 is allowed and selects the multiproc executor."""
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
                cache_config=SimpleNamespace(block_size=None),
                model_config=SimpleNamespace(
                    model="test-model",
                    disable_cascade_attn=False,
                    tokenizer=None,
                    max_model_len=32768,
                    multimodal_config=None,
                    hf_config=SimpleNamespace(model_type="qwen3"),
                    is_hybrid=False,
                ),
                # PP requires synchronous scheduling: the first stage has no
                # sampler and rebuilds tokens from the scheduler, so a valid PP
                # config sets async_scheduling False (see the reject test below).
                scheduler_config=SimpleNamespace(
                    async_scheduling=False,
                    enable_chunked_prefill=True,
                    max_num_batched_tokens=2048,
                    max_num_scheduled_tokens=None,
                ),
                speculative_config=None,
                lora_config=None,
            )

            # Does not raise: PP>1 with TP=1 and sync scheduling falls through.
            MetalPlatform.check_and_update_config(vllm_config)

            # "uni" cannot host two stages, so PP>1 defaults to the mp executor.
            assert vllm_config.parallel_config.distributed_executor_backend == "mp"
            assert (
                vllm_config.parallel_config.worker_cls
                == "vllm_metal.v1.worker.MetalWorker"
            )
        finally:
            reset_config()

    def test_check_and_update_config_rejects_pipeline_ring_port_overflow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_RING_BASE_PORT", "65535")
        vllm_config = SimpleNamespace(
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
            parallel_config=SimpleNamespace(
                worker_cls="auto",
                distributed_executor_backend="auto",
                pipeline_parallel_size=2,
                tensor_parallel_size=1,
                disable_custom_all_reduce=False,
            ),
            model_config=None,
            scheduler_config=SimpleNamespace(async_scheduling=False),
            speculative_config=SimpleNamespace(method="ngram"),
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
            cache_config=SimpleNamespace(block_size=None),
            model_config=SimpleNamespace(**model_fields),
            scheduler_config=SimpleNamespace(
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
                self._dp_vllm_config(speculative_config=SimpleNamespace(method="ngram"))
            )

    def test_check_and_update_config_rejects_dp_lora(self) -> None:
        """DP with LoRA is unvalidated; reject at config time."""
        with pytest.raises(NotImplementedError, match="data parallelism with LoRA"):
            MetalPlatform.check_and_update_config(
                self._dp_vllm_config(lora_config=SimpleNamespace(max_loras=1))
            )

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

    def test_device_capability(self) -> None:
        """Test device capability."""
        major, minor = MetalPlatform.get_device_capability()
        assert isinstance(major, int)
        assert isinstance(minor, int)

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
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(block_size=None),
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
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    block_size=None, enable_prefix_caching=False
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

    def test_check_and_update_config_rejects_hybrid_prefix_caching(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        reset_config()
        try:
            vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    block_size=None,
                    enable_prefix_caching=True,
                ),
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
                ),
            )

            with pytest.raises(NotImplementedError, match="Prefix caching"):
                MetalPlatform.check_and_update_config(vllm_config)
        finally:
            reset_config()

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
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(block_size=None),
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
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(block_size=None),
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
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(block_size=None),
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
            parallel_config=SimpleNamespace(
                worker_cls="auto",
                distributed_executor_backend="auto",
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
                disable_custom_all_reduce=False,
            ),
            cache_config=SimpleNamespace(block_size=None),
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
            parallel_config=SimpleNamespace(
                worker_cls="auto",
                distributed_executor_backend="auto",
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
                disable_custom_all_reduce=False,
            ),
            cache_config=SimpleNamespace(block_size=None),
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
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    block_size=None, enable_prefix_caching=False
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
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    block_size=None, enable_prefix_caching=False
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
                hf_config=SimpleNamespace(model_type="qwen3_vl"),
                is_hybrid=False,
            )
            vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    block_size=None, enable_prefix_caching=False
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
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    block_size=None, enable_prefix_caching=False
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

    def test_check_and_update_config_text_only_compat_preserves_generic_vlm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "text-only-compat")
        reset_config()
        try:
            sentinel = SimpleNamespace(language_model_only=False)
            model_config = SimpleNamespace(
                model="test-model",
                disable_cascade_attn=False,
                tokenizer=None,
                max_model_len=128,
                multimodal_config=sentinel,
                hf_config=SimpleNamespace(model_type="phi3_v"),
                is_hybrid=False,
            )
            vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    worker_cls="auto",
                    distributed_executor_backend="auto",
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                    disable_custom_all_reduce=False,
                ),
                cache_config=SimpleNamespace(
                    block_size=None, enable_prefix_caching=False
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
