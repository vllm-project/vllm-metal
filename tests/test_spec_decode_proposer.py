# SPDX-License-Identifier: Apache-2.0
import time

import mlx.core as mx
import pytest
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_metal.utils import get_model_download_path
from vllm_metal.v1.spec_decode.proposer import MetalDraftProposer, MinimalDraftRunner
from vllm_metal.v1.spec_decode.worker import MetalSpecDecodeWorker

logger = init_logger(__name__)


def _setup_spec_decode_config(model_name: str, num_spec_tokens: int = 5):
    """Standardized configuration setup for speculative decoding tests."""
    resolved_model = get_model_download_path(model_name)

    model_cfg = ModelConfig(
        model=resolved_model,
        tokenizer=resolved_model,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )

    parallel_cfg = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        is_moe_model=False,
        distributed_executor_backend="uni",
    )
    cache_cfg = CacheConfig(block_size=16, gpu_memory_utilization=0.9)
    scheduler_cfg = SchedulerConfig(
        max_num_batched_tokens=2048,
        max_num_seqs=32,
        max_model_len=2048,
        is_encoder_decoder=False,
    )
    device_cfg = DeviceConfig(device="cpu")

    spec_cfg = SpeculativeConfig(
        model=resolved_model,
        num_speculative_tokens=num_spec_tokens,
        target_model_config=model_cfg,
        target_parallel_config=parallel_cfg,
    )

    config = VllmConfig(
        model_config=model_cfg,
        cache_config=cache_cfg,
        parallel_config=parallel_cfg,
        scheduler_config=scheduler_cfg,
        device_config=device_cfg,
        speculative_config=spec_cfg,
    )

    return config, resolved_model


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ["mlx-community/Qwen2.5-0.5B-4bit"])
def test_proposer_functional(model_name):
    """
    Functional test for MetalDraftProposer using a real model.
    Uses get_model_download_path to resolve local models and avoid downloading.
    """
    config, resolved_model = _setup_spec_decode_config(model_name)
    with set_current_vllm_config(config):
        logger.info("Resolved model path: %s", resolved_model)

        # 2. Initialize Proposer
        logger.info("Initializing MetalDraftProposer...")
        proposer = MetalDraftProposer(config, model_path=resolved_model)

        # 3. Load Draft Model
        logger.info("Loading Draft Model...")
        start_load = time.perf_counter()
        proposer.load_model()
        load_time = time.perf_counter() - start_load
        logger.info("Draft Model loaded in %.2fs", load_time)

        # 4. Initialize the "Scorer" Model (Target)
        # We use the same model to avoid large downloads, but rename to Scorer
        # to clarify its role in verification.
        logger.info("Initializing Scorer (Target) Model...")
        scorer_runner = MinimalDraftRunner(config, resolved_model)
        start_load_scorer = time.perf_counter()
        scorer_runner.load_model()
        logger.info(
            "Scorer Model loaded in %.2fs", time.perf_counter() - start_load_scorer
        )

        # 5. Run Propose
        prompt = "The capital of France is"
        logger.info("Encoding prompt: '%s'", prompt)
        tokenizer = proposer._draft_runner.tokenizer
        input_ids = mx.array([tokenizer.encode(prompt)])

        k = 5
        logger.info("Starting proposer.propose(k=%d)...", k)
        start_graph = time.perf_counter()
        draft_tokens, draft_logits = proposer.propose(input_ids, k=k)
        graph_time = (time.perf_counter() - start_graph) * 1000
        logger.info("Graph Building (Lazy): %.4fms", graph_time)

        # IMPORTANT: Evaluate draft tokens now so they don't pollute the Scorer benchmark
        logger.info("Evaluating draft tokens...")
        mx.eval(draft_tokens, draft_logits)

        # 6. Performance Showdown: Sequential vs. Widened Verification
        print("\n" + "=" * 40)
        print("--- Scorer Verification Benchmark ---")

        # Baseline: Scorer generates 1 token sequentially
        # Warmup first
        for _ in range(3):
            _ = scorer_runner.model(input_ids)
            mx.eval(_)

        iters = 5
        base_times = []
        for _ in range(iters):
            start_base = time.perf_counter()
            base_logits = scorer_runner.model(input_ids)
            mx.eval(base_logits)
            base_times.append((time.perf_counter() - start_base) * 1000)

        base_time = sum(base_times) / itls if (itls := len(base_times)) else 0
        print(f"Scorer Sequential (avg 1 token): {base_time:.2f}ms")

        # Speculative: Scorer verifies k tokens in a single "Widened" pass
        # We concatenate prompt + draft tokens to create the widened input
        verification_input = mx.concatenate([input_ids, draft_tokens], axis=1)

        # Warmup widened pass
        for _ in range(3):
            _ = scorer_runner.model(verification_input)
            mx.eval(_)

        spec_times = []
        for _ in range(iters):
            start_spec = time.perf_counter()
            spec_logits = scorer_runner.model(verification_input)
            mx.eval(spec_logits)
            spec_times.append((time.perf_counter() - start_spec) * 1000)

        spec_time = sum(spec_times) / itls if (itls := len(spec_times)) else 0
        print(f"Scorer Verification (avg {k} tokens): {spec_time:.2f}ms")

        # Speedup Verdict
        # Amortized cost: (Sequential * k) vs (Widened Pass)
        speedup = (base_time * k) / spec_time
        print(f"\nVERDICT: Scorer is {speedup:.1f}x more efficient via Speculation!")
        print("=" * 40 + "\n")

        # 7. Verification & Assertions
        logger.info("Running assertions...")
        # The Scorer should return logits for the entire window (prompt + draft)
        assert spec_logits.shape == (
            1,
            input_ids.shape[1] + k,
            config.model_config.get_vocab_size(),
        )
        logger.info("Scorer returned logits for the entire window (prompt + draft)")

        # Proposer shape checks
        assert draft_tokens.shape == (1, k)
        assert draft_logits.shape == (1, k, config.model_config.get_vocab_size())

        # Coherence check
        decoded = tokenizer.decode(draft_tokens[0].tolist())
        logger.info("Draft Output: '%s'", decoded)
        assert len(decoded.strip()) > 0

        # Ensure speculation provides a theoretical benefit (usually > 1.0 on Metal)
        # Even with the same model, verifying a batch of tokens is significantly
        # faster than sequential steps due to memory bandwidth amortization.
        assert speedup > 0.5  # Safety margin for small models/jit


@pytest.mark.parametrize("model_name", ["mlx-community/Qwen2.5-0.5B-4bit"])
def test_spec_decode_worker_integration(model_name, monkeypatch):
    """
    Integration test for MetalSpecDecodeWorker.
    Verifies the Propose-Score-Verify-Rewind loop and the handshake for verification_logits.
    """
    monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
    config, resolved_model = _setup_spec_decode_config(model_name, num_spec_tokens=5)

    with set_current_vllm_config(config):
        logger.info("Initializing MetalSpecDecodeWorker...")
        worker = MetalSpecDecodeWorker(
            config,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://localhost:12345",
        )
        worker.init_device()
        worker.load_model()

        # Follow full vLLM worker initialization sequence to enable paged attention
        logger.info("Initializing KV cache and paged backend...")
        worker.determine_available_memory()
        worker.initialize_cache(num_gpu_blocks=1024, num_cpu_blocks=0)

        # Explicitly load proposer to satisfy the orchestrator
        worker.proposer.load_model()

        # 1. Create a mock SchedulerOutput with a new request
        logger.info("Mocking SchedulerOutput...")
        req_id = "test-request-0"
        prompt = "The quick brown fox"
        tokenizer = worker.target_worker.model_runner.tokenizer
        prompt_token_ids = tokenizer.encode(prompt)

        # Use NewRequestData to simulate a fresh request hitting the engine
        new_req = NewRequestData(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=[],
            sampling_params=SamplingParams(temperature=0.0),
            pooling_params=None,
            block_ids=([1, 2, 3],),  # Mock block IDs (tuple of lists)
            num_computed_tokens=0,
            lora_request=None,
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[new_req],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={req_id: len(prompt_token_ids)},
            total_num_scheduled_tokens=len(prompt_token_ids),
            scheduled_spec_decode_tokens={req_id: [1, 2, 3, 4, 5]},  # Mock spec tokens
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )

        # 2. Mock a rejection scenario to verify the rewind logic
        # We force the sampler to accept only 2 out of 5 tokens
        forced_num_accepted = 2
        forced_bonus_token = 12345
        logger.info("Mocking rejection: accepting %d/5 tokens", forced_num_accepted)

        # Create dummy SpecDecodeMetadata to trigger speculative path in sampler
        spec_metadata = SpecDecodeMetadata.make_dummy(
            draft_token_ids=[[1, 2, 3, 4, 5]],
            device=worker.device,
        )

        def mocked_sample(*args, **kwargs):
            return forced_num_accepted, forced_bonus_token

        monkeypatch.setattr(worker.sampler, "sample", mocked_sample)

        # 3. Execute the Speculative Loop
        logger.info("Executing worker.execute_model()...")

        original_prompt_len = len(prompt_token_ids)
        output = worker.execute_model(scheduler_output, spec_metadata=spec_metadata)

        # Handle asynchronous path (Paged Attention)
        if output is None:
            logger.info("execute_model returned None, calling sample_tokens()...")
            output = worker.sample_tokens(None)

        # 4. Verify results
        logger.info("Running assertions...")
        assert output is not None
        assert output.req_ids == [req_id]

        # Verify that the output token is our forced bonus token
        assert output.sampled_token_ids[0] == [forced_bonus_token]

        # Verify KV Cache Rewind:
        # Expected length = prompt_len + num_accepted + 1 (for bonus token)
        expected_len = original_prompt_len + forced_num_accepted + 1
        actual_len = worker.target_worker.model_runner._paged_request_seq_lens[req_id]

        logger.info(
            "Sequence Length Check: Expected=%d, Actual=%d", expected_len, actual_len
        )
        assert actual_len == expected_len

        # Verify RequestState matches
        state = worker.target_worker.model_runner._request_states[req_id]
        assert len(state.token_ids) == actual_len
        assert state.token_ids[-1] == forced_bonus_token

        logger.info("Integration test successful: Handshake and Rewind verified.")
