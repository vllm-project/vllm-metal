# SPDX-License-Identifier: Apache-2.0
"""Metal Platform implementation for vLLM."""

import logging
import platform as py_platform
from typing import TYPE_CHECKING

import psutil
import torch
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

import vllm_metal.envs as envs
from vllm_metal.config import get_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = logging.getLogger(__name__)


class MetalPlatform(Platform):
    """Platform implementation for Apple Silicon Metal/MLX.

    This class provides vLLM with information about the Metal platform
    capabilities and handles device management.
    """

    _enum: PlatformEnum = PlatformEnum.OOT  # Out-of-tree platform
    device_name: str = "cpu"  # PyTorch device name (use CPU for compatibility)
    device_type: str = "cpu"  # PyTorch device type (use CPU for compatibility)
    dispatch_key: str = "CPU"  # PyTorch dispatch key

    # --- Ray distributed executor support (Phase 1) ---
    # Advertise the Apple GPU as a custom Ray resource named "mlx".  Because this
    # is not "GPU", vLLM's Ray executor uses the generic
    # resources={ray_device_key: n} placement path instead of the CUDA num_gpus
    # path.  Each node must be launched with `ray start --resources='{"mlx": 1}'`.
    ray_device_key: str = "mlx"
    # Per-worker visible-device env var (the CUDA_VISIBLE_DEVICES analog).  With
    # one Apple GPU per node this is effectively a no-op, but it must be a unique
    # name that does not collide with CUDA_VISIBLE_DEVICES.
    device_control_env_var: str = "VLLM_METAL_VISIBLE_DEVICES"

    # Dotted path of the worker setup hook that patches RayWorkerProc to read the
    # custom "mlx" resource (see compat._patch_ray_distributed). Installed both via
    # the executor ray_runtime_env (single-stage) and, for data parallelism, at the
    # Ray job level (_register_dp_ray_worker_setup_hook).
    _RAY_WORKER_SETUP_HOOK: str = "vllm_metal.compat._patch_ray_distributed"
    # Set once this process has initialized Ray with the job-level worker hook, so
    # the DP registration is idempotent across repeated check_and_update_config calls.
    _dp_ray_hook_registered: bool = False

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of the Metal device.

        Args:
            device_id: Device index (ignored for Metal, single GPU)

        Returns:
            Device name string
        """
        try:
            import mlx.core as mx

            device = mx.default_device()
            return f"Apple Silicon ({device})"
        except ImportError:
            return "Apple Silicon (MLX not available)"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get total memory available for the device.

        On Apple Silicon, this returns the fraction of unified memory
        configured for use by the plugin.

        Args:
            device_id: Device index (ignored for Metal)

        Returns:
            Total memory in bytes
        """
        config = get_config()
        total_memory = psutil.virtual_memory().total
        # In auto mode, report full memory - actual allocation is dynamic
        if config.is_auto_memory:
            return total_memory
        return int(total_memory * config.memory_fraction)

    @classmethod
    def get_device_available_memory(cls, device_id: int = 0) -> int:
        """Get available memory for the device.

        Args:
            device_id: Device index (ignored for Metal)

        Returns:
            Available memory in bytes
        """
        config = get_config()
        available = psutil.virtual_memory().available
        # In auto mode, report full available memory - actual allocation is dynamic
        if config.is_auto_memory:
            return available
        return int(available * config.memory_fraction)

    @classmethod
    def is_available(cls) -> bool:
        """Check if Metal platform is available.

        Returns:
            True if running on Apple Silicon with MLX support
        """
        # Check architecture
        if py_platform.machine() != "arm64":
            return False

        # Check OS
        if py_platform.system() != "Darwin":
            return False

        # Check MLX availability without mutating global device state
        try:
            import mlx.core as mx

            return bool(mx.metal.is_available())
        except (ImportError, AttributeError, RuntimeError):
            return False

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        """Get device compute capability.

        Returns a fake capability for compatibility with CUDA-centric code.

        Args:
            device_id: Device index (ignored)

        Returns:
            DeviceCapability with (major, minor) version
        """
        # Return a reasonable value for compatibility
        return DeviceCapability(major=8, minor=0)

    @classmethod
    def get_device_count(cls) -> int:
        """Get number of available devices.

        Apple Silicon has unified memory, so we expose a single device.

        Returns:
            Always 1 for Metal
        """
        return 1

    @classmethod
    def set_device(cls, device: "torch.device | int | None" = 0) -> None:
        """Set the current device.

        vLLM's base contract (and the Ray compiled-DAG path) passes a
        ``torch.device``; some internal callers pass an int index.  Apple
        Silicon exposes a single GPU, so anything other than index 0 (or a
        deviceless ``torch.device("cpu")`` whose index is ``None``) is invalid.

        Args:
            device: A ``torch.device`` or integer index; must resolve to 0.
        """
        index = device.index if isinstance(device, torch.device) else device
        if index not in (0, None):
            msg = f"Metal only supports device 0, got {device!r}"
            raise ValueError(msg)

        config = get_config()
        if config.use_mlx:
            import mlx.core as mx

            device_type = (
                mx.DeviceType.gpu if config.mlx_device == "gpu" else mx.DeviceType.cpu
            )
            mx.set_default_device(mx.Device(device_type))

    @classmethod
    def current_device(cls) -> int:
        """Get the current device index.

        Returns:
            Always 0 for Metal
        """
        return 0

    @classmethod
    def synchronize(cls, device_id: int = 0) -> None:
        """Synchronize the device.

        Args:
            device_id: Device index (ignored)
        """
        import mlx.core as mx

        mx.synchronize()

        if torch.backends.mps.is_available():
            torch.mps.synchronize()

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        """Seed the Metal-side RNG (MLX) for this platform.

        Called from ``vllm.utils.torch_utils.set_random_seed`` after Python
        ``random``, NumPy, and PyTorch (which reaches MPS via its default
        generator) have all been seeded.  MLX maintains its own global PRNG
        that does not auto-seed and is not reached by ``torch.manual_seed``,
        so we seed it explicitly here.
        """
        import mlx.core as mx

        mx.random.seed(seed)

    @classmethod
    def get_torch_device(cls, device_id: int = 0) -> torch.device:
        """Get the corresponding PyTorch device.

        Args:
            device_id: Device index (ignored)

        Returns:
            PyTorch device (MPS or CPU)
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @classmethod
    def _ray_runtime_env_with_metal_hook(
        cls, ray_runtime_env: object = None
    ) -> dict[str, object]:
        """Return a runtime_env dict that installs the Apple-GPU worker setup hook.

        The single source of the hook-install rule, shared by the single-stage Ray
        executor and the DP job-level ``ray.init`` so the two cannot drift: it
        preserves the caller's ``ray_runtime_env`` keys (``working_dir`` /
        ``py_modules`` / ``env_vars``) and rejects a conflicting foreign
        ``worker_process_setup_hook`` (chaining is unsupported) rather than silently
        replacing it.
        """
        runtime_env: dict[str, object] = (
            dict(ray_runtime_env) if isinstance(ray_runtime_env, dict) else {}
        )
        existing = runtime_env.get("worker_process_setup_hook")
        if existing not in (None, cls._RAY_WORKER_SETUP_HOOK):
            raise ValueError(
                "Metal must install a Ray worker_process_setup_hook, but one is "
                f"already set ({existing!r}); chaining is not supported. Unset "
                "ray_runtime_env's worker_process_setup_hook."
            )
        runtime_env["worker_process_setup_hook"] = cls._RAY_WORKER_SETUP_HOOK
        return runtime_env

    @classmethod
    def _register_dp_ray_worker_setup_hook(cls, ray_runtime_env: object = None) -> None:
        """Register the Apple-GPU worker patch at the Ray JOB level for DP.

        ``ray_runtime_env`` is the user's ``parallel_config.ray_runtime_env`` (or
        ``None``): its keys are preserved and merged with the worker hook so a DP
        serve that ships a ``working_dir`` / ``py_modules`` / ``env_vars`` still
        reaches the remote Macs.

        Ray only runs a ``worker_process_setup_hook`` from the JOB runtime_env
        (``ray.init``), not from a per-actor runtime_env. The data-parallel engine
        manager connects to Ray without forwarding ``parallel_config.ray_runtime_env``
        to ``ray.init`` (see ``vllm/v1/engine/utils.py``), so the hook wired into
        ``ray_runtime_env`` for the single-stage Ray executor never reaches the
        per-replica ``RayWorkerProc`` workers, and ``get_node_and_gpu_ids`` would
        ``KeyError`` on the custom "mlx" resource. We initialize Ray ourselves with
        the hook in the job runtime_env before the engine connects; the engine's
        later ``ray.init`` reuses this session.

        Ordering contract: this must run before anything else initializes Ray in
        this process. ``address="auto"`` connects to the cluster the documented
        launch points at via ``RAY_ADDRESS=auto`` and fails loud (ConnectionError)
        if no cluster is reachable — DP requires a running Ray cluster.

        Fails loud if Ray was already initialized by something other than this hook
        (its job runtime_env is then fixed without our setup hook, which would
        silently re-break the DP workers). Idempotent while our Ray session is live;
        if that session was shut down, the now-stale flag is cleared and the hook is
        re-registered so a second in-process DP engine still gets the patch.
        """
        # SCAFFOLDING: remove when upstream CoreEngineActorManager forwards
        # parallel_config.ray_runtime_env into its ray.init (vllm/v1/engine/utils.py),
        # mirroring the single-stage initialize_ray_cluster path; the executor-level
        # hook set in check_and_update_config would then reach the DP RayWorkerProc
        # workers and this job-level registration could be dropped.
        import ray

        if ray.is_initialized():
            if cls._dp_ray_hook_registered:
                # Our hook is already in the live Ray session — idempotent no-op.
                return
            raise RuntimeError(
                "Ray is already initialized before vllm-metal could register the "
                "Apple-GPU worker setup hook at the Ray job level, which data "
                "parallelism requires (Ray honors worker_process_setup_hook only "
                "from the job runtime_env). Do not initialize Ray before serving — "
                "let vllm-metal own ray.init (remove any manual ray.init)."
            )
        # Ray is not initialized: a fresh start, or our previous session was shut
        # down in this process (a now-stale registered flag). Clear the flag so a
        # second in-process DP engine re-registers instead of skipping the hook.
        cls._dp_ray_hook_registered = False
        # Install the worker hook by the same rule as the single-stage Ray path
        # (_ray_runtime_env_with_metal_hook): preserve the user's ray_runtime_env
        # (working_dir / py_modules / env_vars the remote Macs may need) and reject a
        # conflicting foreign hook rather than silently replacing it. The DP manager
        # reuses this job session without re-applying ray_runtime_env, so this is the
        # one place the user's env reaches the remote actors.
        ray.init(
            address="auto",
            runtime_env=cls._ray_runtime_env_with_metal_hook(ray_runtime_env),
        )
        cls._dp_ray_hook_registered = True

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Check and update vLLM configuration for Metal compatibility.

        Args:
            vllm_config: vLLM configuration object
        """
        config = get_config()
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config

        # Apply TurboQuant config from --additional-config
        # Example: --additional-config '{"turboquant": true, "k_quant": "q4_0"}'
        add = getattr(vllm_config, "additional_config", None) or {}
        if add.get("turboquant"):
            config.turboquant = True
            config.k_quant = add.get("k_quant", "q8_0")
            config.v_quant = add.get("v_quant", "q3_0")
            config._validate_turboquant()
            logger.info(
                f"TurboQuant enabled via --additional-config: "
                f"k_quant={config.k_quant}, v_quant={config.v_quant}"
            )

        if config.debug:
            logger.info(f"Metal config: {config}")

        # Set worker class for Metal
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_metal.v1.worker.MetalWorker"

        # Resolve the executor backend. vLLM's ParallelConfig already defaults an
        # unset backend to "mp" when world_size > 1 on non-CUDA/Ray/TPU (see
        # config/parallel.py), so for the common PP>1 case mp is selected upstream
        # before this hook runs. This branch only covers what vLLM leaves unset:
        # the PP==1 default of "uni" (a single in-process worker), and the
        # explicit "auto" string ("uni" cannot host PP's per-stage processes).
        if parallel_config.distributed_executor_backend in ("auto", None):
            if parallel_config.pipeline_parallel_size > 1:
                parallel_config.distributed_executor_backend = "mp"
            else:
                parallel_config.distributed_executor_backend = "uni"
        elif (
            parallel_config.distributed_executor_backend == "uni"
            and parallel_config.pipeline_parallel_size > 1
        ):
            # uni builds only rank 0, but PP needs one worker per stage. vLLM's
            # UniProcExecutor does not reject this, so the lone worker would hang
            # in gloo/ring rendezvous waiting for a stage that never spawns. Fail
            # loud rather than silently flip the user's explicit executor choice.
            raise NotImplementedError(
                "Pipeline parallelism (pipeline_parallel_size > 1) requires the "
                "'mp' or 'ray' executor; 'uni' runs a single process and cannot "
                "place per-stage workers. Re-run with "
                "--distributed-executor-backend mp (single node) or ray."
            )
        elif parallel_config.distributed_executor_backend == "ray":
            # Apple GPUs are not a Ray accelerator family, so the Ray worker
            # actor's get_node_and_gpu_ids would KeyError on
            # get_accelerator_ids()[ray_device_key].  Install our override (see
            # vllm_metal.compat._patch_ray_distributed) in every Ray worker via a
            # worker_process_setup_hook, which runs at worker startup before the
            # first actor call (the lazy plugin-load path is too late).  Fail
            # loud rather than warn-and-continue: the user asked for Ray.
            from ray.runtime_env import RuntimeEnv

            parallel_config.ray_runtime_env = RuntimeEnv(
                **cls._ray_runtime_env_with_metal_hook(parallel_config.ray_runtime_env)
            )

        # Disable features not supported on Metal
        parallel_config.disable_custom_all_reduce = True

        if (
            model_config is not None
            and model_config.is_hybrid
            and vllm_config.cache_config.enable_prefix_caching
        ):
            raise NotImplementedError(
                "Prefix caching is not supported for hybrid GDN models on Metal "
                "because GDN recurrent state cannot be restored from KV blocks."
            )

        # Pipeline parallelism is supported on Metal/MLX: each stage runs in its
        # own worker process and the inter-stage activations cross the
        # mx.distributed data plane (point-to-point send/recv), wired in the
        # model runner (see MetalModelRunner._start_paged_forward). The control
        # plane stays on vLLM's gloo group; the two transports coexist.
        if parallel_config.pipeline_parallel_size > 1:
            base_port = envs.VLLM_METAL_RING_BASE_PORT
            max_port = base_port + parallel_config.pipeline_parallel_size - 1
            if max_port > 65535:
                raise ValueError(
                    "VLLM_METAL_RING_BASE_PORT is too high for "
                    "pipeline_parallel_size: "
                    f"base {base_port} with pipeline_parallel_size="
                    f"{parallel_config.pipeline_parallel_size} would use port "
                    f"{max_port}"
                )

        # Tensor parallelism is not supported on Metal/MLX yet: a single Apple
        # GPU per node cannot shard a TP>1 model, and there is no cross-device
        # collective wired in (mx.distributed). Only TP=1 is validated. Reject
        # at config time rather than hang on Ray placement-group creation (one
        # "mlx" resource per node) or silently misbehave. This is executor-
        # agnostic: uni/mp are equally unable to run TP>1 on one device. This
        # guard also rejects PP+TP — which must STAY rejected when TP support
        # lands: PipelineGroup relies on global_rank == pipeline stage index,
        # which only holds while world_size = PP * TP has TP == 1.
        if parallel_config.tensor_parallel_size > 1:
            raise NotImplementedError(
                "Metal/MLX does not support tensor parallelism yet "
                "(tensor_parallel_size > 1), alone or combined with pipeline "
                "parallelism; only TP=1 is validated."
            )

        # Data parallelism (dense models, one full replica per Mac via the Ray DP
        # backend). DP replicas are independent engines, not part of a single
        # engine's world_size, so the tensor-parallel guard above does not cover
        # them. Dense DP needs NO cross-device collective: upstream runs each
        # replica as an independent
        # EngineCoreActor (data_parallel_size reset to 1, no dp_group — see
        # vllm/v1/engine/core.py), placed on the "mlx" resource, and each replica
        # spawns the same RayExecutorV2/RayWorkerProc, so the Ray worker hook
        # installed above covers it unchanged. We only relax admission: allow that
        # validated dense-DP-over-Ray shape and fail fast on every other DP
        # combination this reachability newly admits (guard-widening audit).
        if getattr(parallel_config, "data_parallel_size", 1) > 1:
            # MoE DP routes to DPMoEEngineCoreActor + an expert-parallel all-to-all
            # that mx.distributed has no equivalent for; only dense DP is validated.
            if model_config is not None and model_config.is_moe:
                raise NotImplementedError(
                    "Metal supports data parallelism for dense models only; MoE "
                    "data parallelism (expert-parallel all-to-all) is not supported."
                )
            # The 'mp' DP backend spawns local subprocesses only and cannot place a
            # replica on a second Mac — it would silently overcommit one node.
            # Cross-Mac DP requires the Ray DP backend.
            if parallel_config.data_parallel_backend != "ray":
                raise NotImplementedError(
                    "Metal data parallelism across Macs requires the Ray DP "
                    "backend; re-run with --data-parallel-backend ray (the default "
                    "'mp' backend spawns local subprocesses and cannot place "
                    "replicas on other nodes)."
                )
            # One Apple GPU per node: exactly one full-model replica per node.
            # Reject != 1, not just > 1: upstream treats data_parallel_size_local==0
            # as a sentinel for externally-specified (headless / front-end-only) DP
            # (see ParallelConfig.data_parallel_size_local), a topology Metal never
            # validated and which the > 1 check alone would silently admit.
            if parallel_config.data_parallel_size_local != 1:
                raise NotImplementedError(
                    "Metal allows exactly one data-parallel replica per node (one "
                    "Apple GPU per Mac); set --data-parallel-size-local 1 "
                    "(the external-DP sentinel 0 and values > 1 are not supported)."
                )
            # External/hybrid LB change the front-end topology (a per-rank API
            # server) and are documented for MoE / wide-EP serving; only the
            # default internal load balancer is validated.
            if (
                parallel_config.data_parallel_external_lb
                or parallel_config.data_parallel_hybrid_lb
            ):
                raise NotImplementedError(
                    "Metal data parallelism supports only the default internal load "
                    "balancer; --data-parallel-external-lb / --data-parallel-hybrid-lb "
                    "are not supported."
                )
            # DP+PP (each replica its own PP group) needs the mx.distributed ring
            # bootstrap scoped per-replica and per-replica ring ports; never
            # validated. (DP+TP is already rejected by the tensor-parallel guard.)
            if parallel_config.pipeline_parallel_size > 1:
                raise NotImplementedError(
                    "Metal does not support combining data parallelism with "
                    "pipeline parallelism yet; use --data-parallel-size or "
                    "--pipeline-parallel-size > 1, not both."
                )
            # DP + speculative decoding / LoRA were never validated under data
            # parallelism (each replica is an independent dense engine); mirror the
            # pipeline-parallel rejections below rather than run an untested path.
            # (DP + multimodal and DP + STT are rejected later, after the model
            # config is normalized / the STT model is detected.)
            if vllm_config.speculative_config is not None:
                raise NotImplementedError(
                    "Metal does not support data parallelism with speculative "
                    "decoding; remove --speculative-config or use "
                    "--data-parallel-size 1."
                )
            if vllm_config.lora_config is not None:
                raise NotImplementedError(
                    "Metal does not support data parallelism with LoRA; remove "
                    "--enable-lora or use --data-parallel-size 1."
                )
            # NB: the Ray job-level hook is registered only AFTER the remaining DP
            # rejections (multimodal, STT) further below, so an unsupported DP
            # config fails fast before any ray.init side effect.

        scheduler_config = vllm_config.scheduler_config

        # Pipeline parallelism relays each sampled token to the first stage via the
        # scheduler's CachedRequestData.new_token_ids — the first stage has no
        # sampler and rebuilds the token stream from the scheduler (see
        # MetalModelRunner._update_pp_stage_states). Async scheduling instead routes
        # those tokens through a GPU broadcast we do not implement, which leaves
        # new_token_ids empty. Fail loud rather than silently flip the user's
        # scheduler config.
        if (
            parallel_config.pipeline_parallel_size > 1
            and scheduler_config.async_scheduling
        ):
            raise NotImplementedError(
                "Pipeline parallelism on Metal requires synchronous scheduling. "
                "Async scheduling delivers each sampled token to the first stage "
                "through a GPU broadcast that is not implemented, so the stage "
                "would starve. Re-run with --no-async-scheduling."
            )

        # Speculative decoding is not implemented under pipeline parallelism:
        # the PP forward path produces no target hidden states and draft
        # proposal runs only on the sampling (last) stage; no method has been
        # validated. Reject loudly rather than run an untested combination.
        if (
            parallel_config.pipeline_parallel_size > 1
            and vllm_config.speculative_config is not None
        ):
            raise NotImplementedError(
                "Pipeline parallelism on Metal does not support speculative "
                "decoding; remove --speculative-config or run with "
                "pipeline_parallel_size=1."
            )

        # LoRA is not supported under pipeline parallelism: non-last stages
        # rebuild RequestState from the scheduler broadcast without lora_id
        # (see MetalModelRunner._update_pp_stage_states), so per-step LoRA
        # decode routing would silently run their layer slices without the
        # adapter while the last stage applies it. The LoRA runtime is also
        # wired to the full model before the stage slice. Reject loudly.
        if (
            parallel_config.pipeline_parallel_size > 1
            and vllm_config.lora_config is not None
        ):
            raise NotImplementedError(
                "Pipeline parallelism on Metal does not support LoRA; "
                "remove --enable-lora or run with pipeline_parallel_size=1."
            )

        if getattr(scheduler_config, "enable_chunked_prefill", False):
            if config.use_paged_attention:
                # The paged path uses a unified varlen Metal kernel that
                # handles mixed prefill + decode in a single forward pass,
                # so chunked prefill works correctly.
                logger.info(
                    "Metal: chunked prefill enabled (paged attention), "
                    "max_num_batched_tokens=%d",
                    scheduler_config.max_num_batched_tokens,
                )
            else:
                # The non-paged MLX path does not honor chunked-prefill
                # scheduler boundaries.  Disable so the scheduler only
                # requests full prefills.
                scheduler_config.enable_chunked_prefill = False

                # Without chunked prefill, the scheduler must fit the
                # entire prompt in a single step.  Ensure
                # max_num_batched_tokens (and max_num_scheduled_tokens)
                # are at least max_model_len; otherwise the scheduler
                # silently refuses to schedule any prompt that exceeds
                # the budget.
                if model_config is not None:
                    model_max = model_config.max_model_len
                    if scheduler_config.max_num_batched_tokens < model_max:
                        scheduler_config.max_num_batched_tokens = model_max
                    if (
                        scheduler_config.max_num_scheduled_tokens is not None
                        and scheduler_config.max_num_scheduled_tokens < model_max
                    ):
                        scheduler_config.max_num_scheduled_tokens = model_max

                logger.info(
                    "Metal: disabled chunked prefill (non-paged path), "
                    "max_num_batched_tokens=%d",
                    scheduler_config.max_num_batched_tokens,
                )

        # Disable cascade attention (not supported), then let the adapter
        # apply any model-specific normalisations (e.g. clearing
        # ``multimodal_config`` for model types served on the text-only
        # backbone — see ``DefaultModelAdapter.normalize_model_config``).
        if model_config is not None:
            model_config.disable_cascade_attn = True
            from vllm_metal.v1.model_adapter import DefaultModelAdapter

            DefaultModelAdapter().normalize_model_config(model_config)

            # DP + multimodal: the multimodal tensor-IPC queue only supports DP=1
            # (vllm/v1/engine/utils.py). Checked AFTER normalize_model_config so a
            # model served on the text-only backbone (multimodal_config cleared by
            # the adapter) is not wrongly rejected — only a genuine multimodal model.
            if (
                getattr(parallel_config, "data_parallel_size", 1) > 1
                and model_config.multimodal_config is not None
            ):
                raise NotImplementedError(
                    "Metal does not support data parallelism for multimodal models "
                    "(the multimodal tensor-IPC path is DP=1 only)."
                )

        # STT model detection — set tokenizer fallback if not already configured.
        # Lazy imports to avoid circular import: platform.py is loaded during
        # vllm.config init, and stt.detection imports from vllm.config.
        from vllm_metal.stt.detection import is_stt_model
        from vllm_metal.stt.policy import apply_stt_scheduler_policy
        from vllm_metal.utils import get_model_download_path

        resolved_model = (
            get_model_download_path(model_config.model)
            if model_config is not None
            else None
        )
        if resolved_model is not None and is_stt_model(resolved_model):
            # STT checkpoints use a dedicated STTModelRunner with no pipeline-
            # split path. Reject PP here, with the other config-time PP guards,
            # before any worker spawns.
            if parallel_config.pipeline_parallel_size > 1:
                raise NotImplementedError(
                    "Pipeline parallelism (pipeline_parallel_size > 1) is not "
                    "supported for speech-to-text models."
                )
            if getattr(parallel_config, "data_parallel_size", 1) > 1:
                raise NotImplementedError(
                    "Data parallelism (data_parallel_size > 1) is not "
                    "supported for speech-to-text models."
                )
            was_async_scheduling = bool(scheduler_config.async_scheduling)
            apply_stt_scheduler_policy(model_config, scheduler_config)
            if was_async_scheduling and not scheduler_config.async_scheduling:
                logger.info("STT: disabled async_scheduling")
            logger.info("STT model detected")

        # Data parallelism passed every admission guard above (including the
        # multimodal and STT rejections). Only now — after all fail-fasts, before
        # the engine connects — register the Apple-GPU worker patch at the Ray job
        # level so it reaches the per-replica RayWorkerProc workers (the
        # executor-level ray_runtime_env hook does not propagate on the DP path).
        if getattr(parallel_config, "data_parallel_size", 1) > 1:
            cls._register_dp_ray_worker_setup_hook(parallel_config.ray_runtime_env)

        # Log memory configuration
        total_mem = cls.get_device_total_memory()
        available_mem = cls.get_device_available_memory()
        logger.info(
            f"Metal memory: {total_mem / 1e9:.1f}GB total, "
            f"{available_mem / 1e9:.1f}GB available"
        )

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        """Metal supports hybrid KV cache for models like Qwen3.5 (SDPA + GDN)."""
        return True

    @classmethod
    def _find_non_ssm_backend(
        cls, vllm_config: "VllmConfig"
    ) -> "type[AttentionBackend] | None":
        """Return a Metal-specific backend for block_size calculation.

        Since MLX models don't populate static_forward_context, the default
        Platform._find_non_ssm_backend (which walks attention layers via
        get_layers_from_vllm_config) returns nothing. We override to return
        the synthetic MetalBackend, which advertises Metal's MultipleOf(16)
        kernel alignment to the framework's hybrid-block-size math.
        """
        from vllm_metal.attention.synthetic_backend import MetalBackend

        return MetalBackend

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: "VllmConfig") -> None:
        """Update block_size for Metal platform.

        Delegates to vLLM's base implementation, which reads the Metal kernel
        alignment (MultipleOf(16)) from our :meth:`_find_non_ssm_backend`
        override. Adds a one-time warning when paged attention is enabled for
        a hybrid model, explaining the cache-block-size translation mechanism
        (PR #235).
        """
        from vllm_metal.config import get_config

        metal_config = get_config()
        model_config = vllm_config.model_config

        if not model_config:
            return

        # For hybrid models with paged attention, log a warning explaining the
        # block-size translation mechanism.
        #
        # Background:
        # - vLLM requires block_size=160 (or larger) for hybrid models to satisfy
        #   page size divisibility validation between SDPA and Mamba layers.
        #
        # Solution (PR #235):
        # - vLLM sees a large block_size (e.g., 144 = 16 * 9) for its scheduler
        #   validation.
        # - The Metal kernel uses a translated block_size (16, the kernel sweet
        #   spot) that it supports.
        # - Each vLLM block is split into ratio = cache_block_size / kernel_block_size
        #   kernel blocks. For example, one vLLM block of 144 tokens becomes 9 kernel
        #   blocks of 16 tokens each.
        # - The KV cache is reshaped (zero-copy) to match: [num_blocks, 144, ...] →
        #   [num_blocks*9, 16, ...]. The physical memory layout is unchanged.
        # - Block tables are expanded so the kernel reads the correct blocks.
        #
        # This is a logical transformation only — the computation is identical, just
        # the kernel sees more, smaller blocks.
        if model_config.is_hybrid and metal_config.use_paged_attention:
            logger.warning(
                "Hybrid model (e.g., Qwen3.5) with paged attention enabled. "
                "Using block-size translation (PR #235) to convert vLLM's large "
                "block_size to a Metal kernel-compatible size.\n"
                "  Mechanism: Each vLLM block is split into multiple kernel blocks.\n"
                "  Example: vLLM block_size=144 → kernel block_size=16 (ratio=9).\n"
                "  The KV cache is reshaped (zero-copy) and block tables are expanded.\n"
                "  This is a logical transformation — physical memory is unchanged."
            )

        # Delegate the rest to upstream. With our ``_find_non_ssm_backend``
        # returning :class:`MetalBackend` (which advertises ``MultipleOf(16)``),
        # vLLM's Phase 1 picks a kernel-aligned default of 16 for non-hybrid
        # models (matching the kernel sweet spot), and Phase 2
        # (``_align_hybrid_block_size``) handles hybrid alignment. The kernel
        # layer (``_pick_kernel_block_size``) validates the final
        # ``block_size`` at request time.
        super().update_block_size_for_backend(vllm_config)

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        """Get the attention backend class for Metal."""
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if selected_backend and selected_backend != AttentionBackendEnum.CPU_ATTN:
            logger.info(f"Cannot use {selected_backend} backend on Metal/MLX.")
        if attn_selector_config.use_mla:
            # MLA attention is handled by the vllm-metal model runner (MLAPagedAttentionWrapper),
            # not by vLLM's attention backend selector. Continue to return CPU_ATTN below.
            logger.info(
                "MLA model detected; attention handled by vllm-metal model runner"
            )
        if attn_selector_config.use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on Metal/MLX.")
        return AttentionBackendEnum.CPU_ATTN.get_path()

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        """Verify that quantization method is supported.

        Args:
            quant: Quantization method name

        Raises:
            ValueError: If quantization is not supported
        """
        # Allow all quantization methods to pass through - actual support
        # depends on the model implementation. This avoids blocking models
        # that use quantization formats we may be able to handle.
        pass

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Check if pin_memory is available for Metal platform.

        Returns:
            False - pin_memory is not needed/supported on Metal/MLX

        Note:
            Although MLX uses unified memory (which theoretically could benefit
            from pin_memory), we disable it because:
            1. PyTorch's pin_memory is primarily designed for CUDA
            2. In our architecture, PyTorch tensors are on CPU for MLX interop
            3. pin_memory on CPU can cause issues or errors
            4. Unified memory already provides fast CPU-GPU transfers without pinning
        """
        return False
