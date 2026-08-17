# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest
import torch
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

import vllm_metal.envs as metal_envs
import vllm_metal.v1.model_runner as mr
from tests.stub_runner import make_stub_runner
from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.attention.runtime.mha import MHAPagedAttentionRuntime
from vllm_metal.attention.state import HybridGDNStateManager
from vllm_metal.distributed.pipeline import PipelineGroup
from vllm_metal.multimodal.qwen3_vl import Qwen3VLMultimodalAdapter
from vllm_metal.v1.gemma4_mtp import Gemma4MTPDraftSeed
from vllm_metal.v1.proposer import Gemma4MTPProposer
from vllm_metal.v1.spec_decode import PagedDecodeSegment


class HybridRuntimeStub:
    def __init__(self, state_cache: GDNPagedStateCache) -> None:
        self._gdn_state_manager = HybridGDNStateManager(state_cache)

    def needs_step_context(self) -> bool:
        return True

    @property
    def gdn_state_manager(self) -> HybridGDNStateManager:
        return self._gdn_state_manager

    def populate_step_context(
        self, *, req_ids: list[str], ctx, state_block_ids=None, step_positions=None
    ) -> None:
        del state_block_ids, step_positions
        self._gdn_state_manager.populate_step_context(req_ids=req_ids, ctx=ctx)

    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        self._gdn_state_manager.extend_forward_eval_outputs(outputs)

    def release_requests(self, req_ids: set[str]) -> None:
        self._gdn_state_manager.release_requests(req_ids)

    def materialize_pending_state(self) -> None:
        self._gdn_state_manager.materialize_pending_state()


class ForwardOutputRuntimeStub:
    def __init__(self, arrays: list[mx.array]) -> None:
        self._arrays = arrays

    def needs_step_context(self) -> bool:
        return False

    def populate_step_context(
        self, *, req_ids: list[str], ctx, state_block_ids=None, step_positions=None
    ) -> None:
        del req_ids, ctx, state_block_ids, step_positions

    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        outputs.extend(self._arrays)

    def release_requests(self, req_ids: set[str]) -> None:
        del req_ids

    def materialize_pending_state(self) -> None:
        return None


class PoolingForwardBackendStub:
    def __init__(self, hidden_states: mx.array) -> None:
        self.hidden_states = hidden_states

    def forward_packed(self, input_ids, offset_caches):
        del input_ids, offset_caches
        return self.hidden_states


def test_gemma4_mtp_config_installs_gemma4_proposer() -> None:
    runner = make_stub_runner(tokenizer=object())
    runner.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="mtp",
            uses_draft_model=lambda: False,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="gemma4_mtp"),
            ),
        ),
    )

    runner.install_drafter(num_blocks=1, block_size=16)

    assert isinstance(runner._drafter, Gemma4MTPProposer)


class TestDrafterReleaseOnLifecycle:
    def test_reconcile_releases_drafter_state_for_invalidated_requests(self) -> None:
        released: list[set[str]] = []

        class _RecordingDrafter:
            def needs_target_hidden_states(self, *args, **kwargs) -> bool:
                return False

            def propose(self, ctx) -> None:
                return None

            def release_requests(self, req_ids: set[str]) -> None:
                released.append(set(req_ids))

        runner = make_stub_runner(tokenizer=object())
        runner._drafter = _RecordingDrafter()

        # Eviction + preemption + resume all invalidate the drafter's per-request
        # state, mirroring the runtime recurrent-state release path.
        runner._reconcile_request_lifecycle(
            {"done"},
            preempted_req_ids={"paused"},
            resumed_req_ids={"back"},
            materialize_runtime_state=False,
        )

        assert released == [{"done", "paused", "back"}]


class TestV1MetalModelRunnerGenerate:
    def _make_runner(self) -> mr.MetalModelRunner:
        return make_stub_runner(tokenizer=object())

    @pytest.mark.parametrize(
        ("configured_processors", "installed_processors"),
        [
            ([object], ()),
            (None, (object(),)),
        ],
        ids=["configured", "installed-plugin"],
    )
    def test_init_rejects_custom_logits_processors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        configured_processors: list[type] | None,
        installed_processors: tuple[object, ...],
    ) -> None:
        monkeypatch.setattr(
            mr, "entry_points", lambda **_: installed_processors, raising=False
        )
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                logits_processors=configured_processors,
                runner_type="generate",
            ),
            cache_config=SimpleNamespace(),
            scheduler_config=SimpleNamespace(async_scheduling=False),
            speculative_config=None,
        )

        with pytest.raises(NotImplementedError, match="custom logits processors"):
            mr.MetalModelRunner(vllm_config, torch.device("cpu"))

    def test_warm_up_propagates_dummy_forward_failure(self) -> None:
        runner = self._make_runner()
        runner._dummy_forward_outputs = Mock(
            side_effect=RuntimeError("dummy forward failed")
        )

        with pytest.raises(RuntimeError, match="dummy forward failed"):
            runner.warm_up()

    def test_accumulates_streamed_segments(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_stream_generate(model, tokenizer, prompt, max_tokens=256, **kwargs):
            captured["model"] = model
            captured["prompt"] = prompt
            captured["max_tokens"] = max_tokens
            captured["kwargs"] = kwargs
            yield SimpleNamespace(text="hello")
            yield SimpleNamespace(text=" ")
            yield SimpleNamespace(text="world")

        monkeypatch.setattr(mr, "stream_generate", fake_stream_generate)

        runner = self._make_runner()
        out = runner.generate("p", max_tokens=3, temperature=0.0)

        assert out == "hello world"
        assert captured["model"] is runner.model
        assert captured["prompt"] == "p"
        assert captured["max_tokens"] == 3
        kwargs = captured.get("kwargs")
        assert isinstance(kwargs, dict)
        # mlx_lm 0.29+ uses sampler parameter instead of temp
        assert "sampler" in kwargs
        assert callable(kwargs["sampler"])

    def test_passes_sampler_for_temperature_sampling(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_stream_generate(model, tokenizer, prompt, max_tokens=256, **kwargs):
            captured["kwargs"] = kwargs
            assert "sampler" in kwargs
            assert callable(kwargs["sampler"])
            yield SimpleNamespace(text="a")
            yield SimpleNamespace(text="b")

        monkeypatch.setattr(mr, "stream_generate", fake_stream_generate)

        runner = self._make_runner()
        out = runner.generate("p", max_tokens=2, temperature=0.5)

        assert out == "ab"
        kwargs = captured.get("kwargs")
        assert isinstance(kwargs, dict)
        assert "sampler" in kwargs

    def test_uses_forward_model_for_vlm_composite(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_stream_generate(model, tokenizer, prompt, max_tokens=256, **kwargs):
            captured["model"] = model
            yield SimpleNamespace(text="ok")

        monkeypatch.setattr(mr, "stream_generate", fake_stream_generate)

        language_model = object()
        runner = self._make_runner()
        runner.model = SimpleNamespace(language_model=object())
        runner._multimodal_adapter = Qwen3VLMultimodalAdapter(
            spatial_merge_size=2,
            language_model=language_model,
        )
        runner._is_vlm = True

        out = runner.generate("p", max_tokens=1)

        assert out == "ok"
        assert captured["model"] is language_model


class TestV1MetalModelRunnerSampleTokens:
    """Tests for `MetalModelRunner.sample_tokens`.

    vLLM v1 may call `sample_tokens()` even if `execute_model()` failed before
    producing output. In that case, `sample_tokens()` must return `None` so vLLM
    can surface the original `execute_model()` exception (instead of raising a
    misleading error from `sample_tokens()` itself).
    """

    def _make_runner(self) -> mr.MetalModelRunner:
        return make_stub_runner()

    def test_returns_pending_output_and_clears_state(self) -> None:
        runner = self._make_runner()
        pending = ModelRunnerOutput(
            req_ids=["req-0"],
            req_id_to_index={"req-0": 0},
            sampled_token_ids=[[123]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None],
        )
        runner._pending_output = pending

        out = runner.sample_tokens(grammar_output=None)

        assert out is pending
        assert runner._pending_output is None

    def test_take_draft_token_ids_returns_and_clears_state(self) -> None:
        runner = self._make_runner()
        draft_token_ids = DraftTokenIds(["req-0"], [[123]])
        runner._draft_token_ids = draft_token_ids

        out = runner.take_draft_token_ids()

        assert out is draft_token_ids
        assert runner._draft_token_ids is None

    def test_returns_none_when_no_pending_output(self) -> None:
        runner = self._make_runner()
        out = runner.sample_tokens(grammar_output=None)

        assert out is None


class TestPagedBlockIds:
    def test_copies_only_runtime_scheduler_groups(self) -> None:
        runner = make_stub_runner(_paged_block_size=4)

        block_ids = runner._copy_paged_block_ids(([1, 2], [9]))

        assert block_ids == [[1, 2]]

    def test_rejects_missing_runtime_scheduler_group(self) -> None:
        runner = make_stub_runner(_paged_block_size=4)
        runner._paged_scheduler_group_indices = (1,)

        with pytest.raises(ValueError, match="required cache groups"):
            runner._copy_paged_block_ids(([1, 2],))


class TestV1MetalModelRunnerSpecDecodeVerification:
    def _make_runner(self) -> mr.MetalModelRunner:
        return make_stub_runner(
            model_args={"vocab_size": 16},
            _paged_block_size=4,
        )

    def _make_state(
        self,
        token_ids: list[int],
        *,
        temperature: float = 0.0,
    ) -> mr.RequestState:
        return mr.RequestState(
            token_ids=token_ids,
            prompt_len=1,
            cache=[],
            sampling_params=SamplingParams(temperature=temperature),
            generator=None,
            generated_tokens=len(token_ids) - 1,
        )

    def _make_logits(self, token_ids: list[int]) -> mx.array:
        rows = []
        for token_id in token_ids:
            row = [0.0] * 16
            row[token_id] = 10.0
            rows.append(row)
        return mx.array([rows])

    def _make_scheduler_output(
        self,
        num_scheduled_tokens: dict[str, int],
        scheduled_spec_decode_tokens: dict[str, list[int]],
        num_invalid_spec_tokens: dict[str, int] | None = None,
        num_spec_tokens_to_schedule: int = 1,
    ) -> SchedulerOutput:
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            num_invalid_spec_tokens=num_invalid_spec_tokens,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
        )

    def _make_gemma4_mtp_config(self) -> SimpleNamespace:
        return SimpleNamespace(
            parallel_config=SimpleNamespace(distributed_executor_backend=None),
            speculative_config=SimpleNamespace(
                method="mtp",
                draft_model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(
                        model_type="gemma4_assistant",
                        architectures=["Gemma4AssistantForCausalLM"],
                    )
                ),
            ),
        )

    def _make_grammar_output(
        self,
        req_ids: list[str],
        allowed_token_id: int,
    ) -> SimpleNamespace:
        return self._make_grammar_rows(
            req_ids,
            [allowed_token_id for _ in req_ids],
        )

    def _make_grammar_rows(
        self,
        req_ids: list[str],
        allowed_token_ids: list[int],
    ) -> SimpleNamespace:
        bitmask = np.zeros((len(allowed_token_ids), 1), dtype=np.int32)
        for row, allowed_token_id in enumerate(allowed_token_ids):
            bitmask[row, 0] = 1 << allowed_token_id
        return SimpleNamespace(
            structured_output_request_ids=req_ids,
            grammar_bitmask=bitmask,
        )

    def _install_paged_state(
        self,
        runner: mr.MetalModelRunner,
        decode_reqs: list[tuple[str, mr.RequestState]],
        decode_segments: tuple[mr.PagedDecodeSegment, ...],
        logits: mx.array,
        scheduler_output: SimpleNamespace,
        target_hidden_states: mx.array | None = None,
    ) -> None:
        batch = mr._ExecutionBatch()
        batch.paged_decode_reqs = decode_reqs
        runner._execute_model_state = mr._PagedForwardState(
            batch=batch,
            prefill_reqs=[],
            decode_reqs=decode_reqs,
            scheduler_output=scheduler_output,
            logits=logits,
            target_hidden_states=target_hidden_states,
            cu_seqlens=[
                0,
                *[s.start_row + s.num_query_tokens for s in decode_segments],
            ],
            decode_segments=decode_segments,
            num_decode_tokens=sum(s.num_query_tokens for s in decode_segments),
            mm_prefill_deltas={},
        )

    def test_start_paged_forward_includes_scheduled_drafts(self, monkeypatch) -> None:
        # Opt into window mode so the captured kwarg pins the full
        # flag -> merge_verify_windows -> prepare_grouped chain.
        monkeypatch.setenv("VLLM_METAL_SPEC_VERIFY_WINDOW", "1")
        runner = self._make_runner()
        runner.vllm_config = self._make_gemma4_mtp_config()
        runner._drafter = Gemma4MTPProposer(runner)
        runner.num_layers = 0
        runner._paged_block_size = 4
        runner._paged_request_seq_lens["r0"] = 1

        captured: dict[str, object] = {}

        def capture_prepare_grouped(
            decode_info, prefill_info, block_sizes, *, merge_verify_windows
        ):
            captured["decode_info"] = decode_info
            captured["prefill_info"] = prefill_info
            captured["block_size"] = block_sizes[0]
            captured["merge_verify_windows"] = merge_verify_windows

        def fake_target_forward(input_ids, *, cache, collect_hidden_states):
            del cache
            captured["input_ids"] = input_ids.tolist()
            captured["collect_hidden_states"] = collect_hidden_states
            return mr.TargetModelForwardOutput(
                logits=mx.zeros((1, 3, 16)),
                hidden_states=mx.ones((3, 4)),
            )

        monkeypatch.setattr(mr, "prepare_grouped", capture_prepare_grouped)
        monkeypatch.setattr(runner, "_target_forward", fake_target_forward)

        req_state = self._make_state([1, 6])
        req_state.block_ids = [[0, 1]]
        scheduler_output = self._make_scheduler_output(
            {"r0": 3},
            {"r0": [7, 8]},
        )

        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[],
            decode_reqs=[("r0", req_state)],
            scheduler_output=scheduler_output,
        )

        assert captured["input_ids"] == [[6, 7, 8]]
        assert captured["collect_hidden_states"] is True
        assert captured["decode_info"] == [([[0, 1]], 1, 3)]
        assert captured["prefill_info"] == []
        assert captured["block_size"] == 4
        assert captured["merge_verify_windows"] is True
        assert runner._execute_model_state is not None
        assert runner._execute_model_state.target_hidden_states is not None
        assert runner._execute_model_state.cu_seqlens == [0, 3]

    def test_start_paged_forward_drops_scheduler_padded_drafts(
        self, monkeypatch
    ) -> None:
        """vLLM 0.25 pads a newly admitted decode request with placeholder drafts.

        The runner must hand ``build_decode_segments`` the filtered map, so the
        request gets a single query row instead of splicing ``-1`` into the
        embedding lookup.
        """
        runner = self._make_runner()
        runner.vllm_config = self._make_gemma4_mtp_config()
        runner._drafter = Gemma4MTPProposer(runner)
        runner.num_layers = 0
        runner._paged_block_size = 4
        runner._paged_request_seq_lens["r0"] = 1

        captured: dict[str, object] = {}

        def capture_prepare_grouped(
            decode_info, prefill_info, block_sizes, *, merge_verify_windows
        ):
            del prefill_info, block_sizes, merge_verify_windows
            captured["decode_info"] = decode_info

        def fake_target_forward(input_ids, *, cache, collect_hidden_states):
            del cache, collect_hidden_states
            captured["input_ids"] = input_ids.tolist()
            return mr.TargetModelForwardOutput(
                logits=mx.zeros((1, 1, 16)),
                hidden_states=mx.ones((1, 4)),
            )

        monkeypatch.setattr(mr, "prepare_grouped", capture_prepare_grouped)
        monkeypatch.setattr(runner, "_target_forward", fake_target_forward)

        req_state = self._make_state([1, 6])
        req_state.block_ids = [[0, 1]]
        scheduler_output = self._make_scheduler_output(
            {"r0": 3},
            {"r0": [-1, -1]},
        )

        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[],
            decode_reqs=[("r0", req_state)],
            scheduler_output=scheduler_output,
        )

        assert captured["input_ids"] == [[6]]
        assert captured["decode_info"] == [([[0, 1]], 1, 1)]
        assert runner._execute_model_state is not None
        assert runner._execute_model_state.cu_seqlens == [0, 1]

    def test_start_paged_forward_skips_hidden_states_without_drafts(
        self, monkeypatch
    ) -> None:
        runner = self._make_runner()
        runner.num_layers = 0
        runner._paged_block_size = 4
        runner._paged_request_seq_lens["r0"] = 1

        captured: dict[str, object] = {}

        def capture_prepare_grouped(
            decode_info, prefill_info, block_sizes, *, merge_verify_windows
        ):
            del merge_verify_windows
            captured["decode_info"] = decode_info
            captured["prefill_info"] = prefill_info
            captured["block_size"] = block_sizes[0]

        def fake_target_forward(input_ids, *, cache, collect_hidden_states):
            del cache
            captured["input_ids"] = input_ids.tolist()
            captured["collect_hidden_states"] = collect_hidden_states
            return mr.TargetModelForwardOutput(logits=mx.zeros((1, 1, 16)))

        monkeypatch.setattr(mr, "prepare_grouped", capture_prepare_grouped)
        monkeypatch.setattr(runner, "_target_forward", fake_target_forward)

        req_state = self._make_state([1, 6])
        req_state.block_ids = [[0, 1]]
        scheduler_output = self._make_scheduler_output({"r0": 1}, {})

        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[],
            decode_reqs=[("r0", req_state)],
            scheduler_output=scheduler_output,
        )

        assert captured["input_ids"] == [[6]]
        assert captured["collect_hidden_states"] is False
        assert captured["decode_info"] == [([[0, 1]], 1, 1)]
        assert captured["prefill_info"] == []
        assert captured["block_size"] == 4
        assert runner._execute_model_state is not None
        assert runner._execute_model_state.target_hidden_states is None
        assert runner._execute_model_state.cu_seqlens == [0, 1]

    def test_start_paged_forward_applies_cow_before_step_context(
        self, monkeypatch
    ) -> None:
        runtime = Mock()
        runtime.needs_step_context.return_value = True
        runner = self._make_runner()
        runner._paged_attention_runtime = runtime
        runner.num_layers = 0
        runner._paged_block_size = 4
        runner._paged_group_block_sizes = (4,)
        runner._paged_request_seq_lens["r0"] = 1
        monkeypatch.setattr(
            runner,
            "_target_forward",
            lambda *args, **kwargs: mr.TargetModelForwardOutput(
                logits=mx.zeros((1, 1, 16))
            ),
        )

        req_state = self._make_state([1, 6])
        req_state.block_ids = [[0, 1]]
        scheduler_output = self._make_scheduler_output({"r0": 1}, {})
        scheduler_output.kv_cache_block_copies = [(2, 6)]

        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[],
            decode_reqs=[("r0", req_state)],
            scheduler_output=scheduler_output,
        )

        runtime.copy_blocks.assert_called_once_with([(2, 6)])
        method_order = [call[0] for call in runtime.mock_calls]
        assert method_order.index("copy_blocks") < method_order.index(
            "populate_step_context"
        )

    def test_start_paged_forward_clears_context_on_gdn_slot_error(self) -> None:
        state_cache = GDNPagedStateCache(
            num_layers=1,
            max_seqs=1,
            conv_kernel_dim=2,
            conv_dim=4,
            num_v_heads=1,
            value_head_dim=4,
            key_head_dim=32,
            initial_seqs=0,
            dtype=mx.float32,
        )
        backend = HybridRuntimeStub(state_cache)
        runner = make_stub_runner(
            tokenizer=object(),
            model_args={"full_attention_interval": 2},
            _paged_attention_runtime=backend,
        )
        runner.num_layers = 0
        runner._paged_block_size = 4
        runner._paged_group_block_sizes = (4,)
        runner._paged_scheduler_group_indices = (0,)
        scheduler_output = self._make_scheduler_output({"p0": 1, "p1": 1}, {})
        prefill_reqs = [
            mr.PrefillRequest(
                req_id="p0",
                token_ids=[5],
                sampling_params=SamplingParams(),
                block_ids=[[0]],
                generator=None,
                prompt_len=1,
                start_pos=0,
                full_prompt_token_ids=None,
            ),
            mr.PrefillRequest(
                req_id="p1",
                token_ids=[6],
                sampling_params=SamplingParams(),
                block_ids=[[1]],
                generator=None,
                prompt_len=1,
                start_pos=0,
                full_prompt_token_ids=None,
            ),
        ]

        mr.clear_context()
        with pytest.raises(RuntimeError, match="more slots than max_num_seqs"):
            runner._start_paged_forward(
                mr._ExecutionBatch(),
                prefill_reqs=prefill_reqs,
                decode_reqs=[],
                scheduler_output=scheduler_output,
            )

        assert mr.get_context() is None
        assert backend.gdn_state_manager.request_slots == {}

    def test_start_paged_forward_collects_hidden_states_for_gemma4_mtp(
        self, monkeypatch
    ) -> None:
        runner = self._make_runner()
        runner.vllm_config = self._make_gemma4_mtp_config()
        runner._drafter = Gemma4MTPProposer(runner)
        runner.num_layers = 0
        runner._paged_block_size = 4
        runner._paged_request_seq_lens["r0"] = 1

        captured: dict[str, object] = {}

        def capture_prepare_grouped(
            decode_info, prefill_info, block_sizes, *, merge_verify_windows
        ):
            del merge_verify_windows
            captured["decode_info"] = decode_info
            captured["prefill_info"] = prefill_info
            captured["block_size"] = block_sizes[0]

        def fake_target_forward(input_ids, *, cache, collect_hidden_states):
            del cache
            captured["input_ids"] = input_ids.tolist()
            captured["collect_hidden_states"] = collect_hidden_states
            return mr.TargetModelForwardOutput(
                logits=mx.zeros((1, 1, 16)),
                hidden_states=mx.ones((1, 4)),
            )

        monkeypatch.setattr(mr, "prepare_grouped", capture_prepare_grouped)
        monkeypatch.setattr(runner, "_target_forward", fake_target_forward)

        req_state = self._make_state([1, 6])
        req_state.block_ids = [[0, 1]]
        scheduler_output = self._make_scheduler_output({"r0": 1}, {})

        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[],
            decode_reqs=[("r0", req_state)],
            scheduler_output=scheduler_output,
        )

        assert captured["input_ids"] == [[6]]
        assert captured["collect_hidden_states"] is True
        assert captured["decode_info"] == [([[0, 1]], 1, 1)]
        assert captured["prefill_info"] == []
        assert captured["block_size"] == 4
        assert runner._execute_model_state is not None
        assert runner._execute_model_state.target_hidden_states is not None
        assert runner._execute_model_state.cu_seqlens == [0, 1]

    def test_start_paged_forward_collects_hidden_states_for_gemma4_mtp_prefill(
        self, monkeypatch
    ) -> None:
        runner = self._make_runner()
        runner.vllm_config = self._make_gemma4_mtp_config()
        runner._drafter = Gemma4MTPProposer(runner)
        runner.num_layers = 0
        runner._paged_block_size = 4

        captured: dict[str, object] = {}

        def capture_prepare_grouped(
            decode_info, prefill_info, block_sizes, *, merge_verify_windows
        ):
            del merge_verify_windows
            captured["decode_info"] = decode_info
            captured["prefill_info"] = prefill_info
            captured["block_size"] = block_sizes[0]

        def fake_target_forward(input_ids, *, cache, collect_hidden_states):
            del cache
            captured["input_ids"] = input_ids.tolist()
            captured["collect_hidden_states"] = collect_hidden_states
            return mr.TargetModelForwardOutput(
                logits=mx.zeros((1, 2, 16)),
                hidden_states=mx.ones((2, 4)),
            )

        monkeypatch.setattr(mr, "prepare_grouped", capture_prepare_grouped)
        monkeypatch.setattr(runner, "_target_forward", fake_target_forward)

        scheduler_output = self._make_scheduler_output({"r0": 2}, {})

        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[
                mr.PrefillRequest(
                    req_id="r0",
                    token_ids=[5, 6],
                    sampling_params=SamplingParams(),
                    block_ids=[[0]],
                    generator=None,
                    prompt_len=2,
                    start_pos=0,
                    full_prompt_token_ids=None,
                )
            ],
            decode_reqs=[],
            scheduler_output=scheduler_output,
        )

        assert captured["input_ids"] == [[5, 6]]
        assert captured["collect_hidden_states"] is True
        assert captured["decode_info"] == []
        assert captured["prefill_info"] == [([[0]], 2, 0)]
        assert captured["block_size"] == 4
        assert runner._execute_model_state is not None
        assert runner._execute_model_state.target_hidden_states is not None
        assert runner._execute_model_state.cu_seqlens == [0, 2]

    def test_start_paged_forward_skips_hidden_states_for_intermediate_prefill(
        self, monkeypatch
    ) -> None:
        runner = self._make_runner()
        runner.vllm_config = self._make_gemma4_mtp_config()
        runner._drafter = Gemma4MTPProposer(runner)
        runner.num_layers = 0
        runner._paged_block_size = 4

        captured: dict[str, object] = {}

        def capture_prepare_grouped(
            decode_info, prefill_info, block_sizes, *, merge_verify_windows
        ):
            del merge_verify_windows
            captured["decode_info"] = decode_info
            captured["prefill_info"] = prefill_info
            captured["block_size"] = block_sizes[0]

        def fake_target_forward(input_ids, *, cache, collect_hidden_states):
            del cache
            captured["input_ids"] = input_ids.tolist()
            captured["collect_hidden_states"] = collect_hidden_states
            return mr.TargetModelForwardOutput(logits=mx.zeros((1, 2, 16)))

        monkeypatch.setattr(mr, "prepare_grouped", capture_prepare_grouped)
        monkeypatch.setattr(runner, "_target_forward", fake_target_forward)

        scheduler_output = self._make_scheduler_output({"r0": 2}, {})

        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[
                mr.PrefillRequest(
                    req_id="r0",
                    token_ids=[5, 6],
                    sampling_params=SamplingParams(),
                    block_ids=[[0]],
                    generator=None,
                    prompt_len=None,
                    start_pos=0,
                    full_prompt_token_ids=None,
                )
            ],
            decode_reqs=[],
            scheduler_output=scheduler_output,
        )

        assert captured["input_ids"] == [[5, 6]]
        assert captured["collect_hidden_states"] is False
        assert captured["decode_info"] == []
        assert captured["prefill_info"] == [([[0]], 2, 0)]
        assert captured["block_size"] == 4
        assert runner._execute_model_state is not None
        assert runner._execute_model_state.target_hidden_states is None
        assert runner._execute_model_state.cu_seqlens == [0, 2]

    def test_accepts_all_drafts_and_emits_bonus_token(self) -> None:
        runner = self._make_runner()
        req_state = self._make_state([1, 6])
        decode_reqs = [("r0", req_state)]
        segment = mr.PagedDecodeSegment(
            req_id="r0",
            input_token_ids=(6, 7, 8),
            start_row=0,
            num_query_tokens=3,
            draft_token_ids=(7, 8),
            cache_start_pos=1,
            block_ids=((0,),),
        )
        scheduler_output = self._make_scheduler_output(
            {"r0": 3},
            {"r0": [7, 8]},
        )
        self._install_paged_state(
            runner,
            decode_reqs,
            (segment,),
            self._make_logits([7, 8, 9]),
            scheduler_output,
        )

        output = runner.sample_tokens(grammar_output=None)

        assert output is not None
        assert output.sampled_token_ids == [[7, 8, 9]]
        assert req_state.token_ids == [1, 6, 7, 8, 9]
        assert req_state.generated_tokens == 4
        assert runner._paged_request_seq_lens["r0"] == 4

    def test_sample_paged_batch_stashes_gemma4_decode_drafts(self) -> None:
        captured: dict[str, object] = {}

        class Assistant:
            forward_ready = True

            def propose_draft_token_ids(
                self,
                *,
                seeds,
                target_hidden_states,
                target_input_embeddings,
            ):
                captured["seeds"] = seeds
                captured["hidden_states"] = target_hidden_states.tolist()
                captured["embeddings"] = target_input_embeddings.tolist()
                return [[42]]

        class Adapter:
            def target_input_embeddings(self, model, input_ids):
                del model
                captured["input_ids"] = input_ids.tolist()
                return mx.ones((*input_ids.shape, 4))

        runner = self._make_runner()
        runner._gemma4_mtp_assistant = Assistant()
        runner._model_adapter = Adapter()
        runner._drafter = Gemma4MTPProposer(runner)
        req_state = self._make_state([1, 6])
        decode_reqs = [("r0", req_state)]
        segment = mr.PagedDecodeSegment(
            req_id="r0",
            input_token_ids=(6, 7),
            start_row=0,
            num_query_tokens=2,
            draft_token_ids=(7,),
            cache_start_pos=1,
            block_ids=((0,),),
        )
        scheduler_output = self._make_scheduler_output(
            {"r0": 2},
            {"r0": [7]},
        )
        self._install_paged_state(
            runner,
            decode_reqs,
            (segment,),
            self._make_logits([7, 9]),
            scheduler_output,
            target_hidden_states=mx.array([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]),
        )

        output = runner.sample_tokens(grammar_output=None)
        draft_token_ids = runner.take_draft_token_ids()

        assert output is not None
        assert output.sampled_token_ids == [[7, 9]]
        assert captured["input_ids"] == [[9]]
        assert captured["hidden_states"] == [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ]
        assert captured["embeddings"] == [[[1.0, 1.0, 1.0, 1.0]]]
        seeds = captured["seeds"]
        assert len(seeds) == 1
        assert seeds[0] == Gemma4MTPDraftSeed(
            req_id="r0",
            token_id=9,
            target_hidden_row=1,
            target_position=2,
            block_ids=((0,),),
        )
        assert draft_token_ids == DraftTokenIds(["r0"], [[42]])

    def test_gemma4_mtp_honors_scheduler_selected_zero_drafts(self) -> None:
        class Assistant:
            forward_ready = True

            def propose_draft_token_ids(
                self,
                *,
                seeds,
                target_hidden_states,
                target_input_embeddings,
            ):
                del seeds, target_hidden_states, target_input_embeddings
                raise AssertionError("assistant should not draft when K=0")

        class Adapter:
            def target_input_embeddings(self, model, input_ids):
                del model, input_ids
                raise AssertionError("draft embeddings should not be requested")

        runner = self._make_runner()
        runner._gemma4_mtp_assistant = Assistant()
        runner._model_adapter = Adapter()
        runner._drafter = Gemma4MTPProposer(runner)
        req_state = self._make_state([1, 6])
        decode_reqs = [("r0", req_state)]
        segment = mr.PagedDecodeSegment(
            req_id="r0",
            input_token_ids=(6,),
            start_row=0,
            num_query_tokens=1,
            draft_token_ids=(),
            cache_start_pos=1,
            block_ids=((0,),),
        )
        scheduler_output = self._make_scheduler_output(
            {"r0": 1},
            {},
            num_spec_tokens_to_schedule=0,
        )
        self._install_paged_state(
            runner,
            decode_reqs,
            (segment,),
            self._make_logits([9]),
            scheduler_output,
            target_hidden_states=mx.array([[1.0, 0.0, 0.0, 0.0]]),
        )

        output = runner.sample_tokens(grammar_output=None)

        assert output is not None
        assert output.sampled_token_ids == [[9]]
        assert req_state.token_ids == [1, 6, 9]
        assert runner.take_draft_token_ids() is None

    def test_sample_paged_batch_stashes_gemma4_prefill_drafts(self) -> None:
        captured: dict[str, object] = {}

        class Assistant:
            forward_ready = True

            def propose_draft_token_ids(
                self,
                *,
                seeds,
                target_hidden_states,
                target_input_embeddings,
            ):
                del target_hidden_states, target_input_embeddings
                captured["seeds"] = seeds
                return [[43]]

        class Adapter:
            def target_input_embeddings(self, model, input_ids):
                del model
                captured["input_ids"] = input_ids.tolist()
                return mx.ones((*input_ids.shape, 4))

        runner = self._make_runner()
        runner._gemma4_mtp_assistant = Assistant()
        runner._model_adapter = Adapter()
        runner._drafter = Gemma4MTPProposer(runner)
        prefill = mr.PrefillRequest(
            req_id="p0",
            token_ids=[5, 6],
            sampling_params=SamplingParams(temperature=0.0),
            block_ids=[[0]],
            generator=None,
            prompt_len=2,
            start_pos=0,
            full_prompt_token_ids=None,
        )
        batch = mr._ExecutionBatch()
        output_idx = batch.add_output("p0", [])
        batch.paged_prefill_entries = [
            mr._PendingPrefillEntry(output_idx, prefill, "new_final")
        ]
        runner._execute_model_state = mr._PagedForwardState(
            batch=batch,
            prefill_reqs=[prefill],
            decode_reqs=[],
            scheduler_output=self._make_scheduler_output({"p0": 2}, {}),
            logits=self._make_logits([0, 7]),
            target_hidden_states=mx.array([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]),
            cu_seqlens=[0, 2],
            decode_segments=(),
            num_decode_tokens=0,
            mm_prefill_deltas={},
        )

        output = runner.sample_tokens(grammar_output=None)
        draft_token_ids = runner.take_draft_token_ids()

        assert output is not None
        assert output.sampled_token_ids == [[7]]
        assert captured["input_ids"] == [[7]]
        seeds = captured["seeds"]
        assert len(seeds) == 1
        assert seeds[0] == Gemma4MTPDraftSeed(
            req_id="p0",
            token_id=7,
            target_hidden_row=1,
            target_position=1,
            block_ids=((0,),),
        )
        assert draft_token_ids == DraftTokenIds(["p0"], [[43]])

    def test_rejects_first_mismatched_draft_and_stops_before_bonus(self) -> None:
        runner = self._make_runner()
        req_state = self._make_state([1, 6])
        decode_reqs = [("r0", req_state)]
        segment = mr.PagedDecodeSegment(
            req_id="r0",
            input_token_ids=(6, 7, 8),
            start_row=0,
            num_query_tokens=3,
            draft_token_ids=(7, 8),
            cache_start_pos=1,
            block_ids=((0,),),
        )
        scheduler_output = self._make_scheduler_output(
            {"r0": 3},
            {"r0": [7, 8]},
        )
        self._install_paged_state(
            runner,
            decode_reqs,
            (segment,),
            self._make_logits([7, 5, 9]),
            scheduler_output,
        )

        output = runner.sample_tokens(grammar_output=None)

        assert output is not None
        assert output.sampled_token_ids == [[7, 5]]
        assert req_state.token_ids == [1, 6, 7, 5]
        assert req_state.generated_tokens == 3
        assert runner._paged_request_seq_lens["r0"] == 3

    def test_mixed_batch_keeps_plain_decode_request(self) -> None:
        runner = self._make_runner()
        draft_state = self._make_state([1, 6])
        plain_state = self._make_state([2, 3])
        decode_reqs = [("draft", draft_state), ("plain", plain_state)]
        segments = (
            mr.PagedDecodeSegment(
                req_id="draft",
                input_token_ids=(6, 7),
                start_row=0,
                num_query_tokens=2,
                draft_token_ids=(7,),
                cache_start_pos=1,
                block_ids=((0,),),
            ),
            mr.PagedDecodeSegment(
                req_id="plain",
                input_token_ids=(3,),
                start_row=2,
                num_query_tokens=1,
                draft_token_ids=(),
                cache_start_pos=1,
                block_ids=((1,),),
            ),
        )
        scheduler_output = self._make_scheduler_output(
            {"draft": 2, "plain": 1},
            {"draft": [7]},
        )
        self._install_paged_state(
            runner,
            decode_reqs,
            segments,
            self._make_logits([7, 9, 4]),
            scheduler_output,
        )

        output = runner.sample_tokens(grammar_output=None)

        assert output is not None
        assert output.req_ids == ["draft", "plain"]
        assert output.sampled_token_ids == [[7, 9], [4]]
        assert draft_state.token_ids == [1, 6, 7, 9]
        assert plain_state.token_ids == [2, 3, 4]

    def test_mixed_batch_routes_plain_request_through_sampler(
        self, monkeypatch
    ) -> None:
        runner = self._make_runner()
        draft_state = self._make_state([1, 6])
        plain_state = self._make_state([2, 3], temperature=0.7)
        decode_reqs = [("draft", draft_state), ("plain", plain_state)]
        segments = (
            mr.PagedDecodeSegment(
                req_id="draft",
                input_token_ids=(6, 7),
                start_row=0,
                num_query_tokens=2,
                draft_token_ids=(7,),
                cache_start_pos=1,
                block_ids=((0,),),
            ),
            mr.PagedDecodeSegment(
                req_id="plain",
                input_token_ids=(3,),
                start_row=2,
                num_query_tokens=1,
                draft_token_ids=(),
                cache_start_pos=1,
                block_ids=((1,),),
            ),
        )
        scheduler_output = self._make_scheduler_output(
            {"draft": 2, "plain": 1},
            {"draft": [7]},
        )
        self._install_paged_state(
            runner,
            decode_reqs,
            segments,
            self._make_logits([7, 9, 4]),
            scheduler_output,
        )
        sampled_rows = []

        def fake_sample_from_logits(logits_2d, batch, sampler, device):
            del sampler, device
            sampled_rows.append(logits_2d.tolist())
            assert [sp.temperature for sp in batch.sampling_params_list] == [0.7]
            return mr._SamplingResult([4])

        monkeypatch.setattr(mr, "sample_from_logits", fake_sample_from_logits)

        output = runner.sample_tokens(grammar_output=None)

        assert output is not None
        assert output.sampled_token_ids == [[7, 9], [4]]
        assert sampled_rows == [[[0.0] * 4 + [10.0] + [0.0] * 11]]

    def test_structured_output_plain_spec_decode_request_is_allowed(self) -> None:
        runner = self._make_runner()
        structured_state = self._make_state([1, 6])
        draft_state = self._make_state([2, 3])
        decode_reqs = [("structured", structured_state), ("draft", draft_state)]
        segments = (
            mr.PagedDecodeSegment(
                req_id="structured",
                input_token_ids=(6,),
                start_row=0,
                num_query_tokens=1,
                draft_token_ids=(),
                cache_start_pos=1,
                block_ids=((0,),),
            ),
            mr.PagedDecodeSegment(
                req_id="draft",
                input_token_ids=(3, 7),
                start_row=1,
                num_query_tokens=2,
                draft_token_ids=(7,),
                cache_start_pos=1,
                block_ids=((1,),),
            ),
        )
        scheduler_output = self._make_scheduler_output(
            {"structured": 1, "draft": 2},
            {"draft": [7]},
        )
        self._install_paged_state(
            runner,
            decode_reqs,
            segments,
            self._make_logits([0, 7, 9]),
            scheduler_output,
        )

        output = runner.sample_tokens(
            grammar_output=self._make_grammar_output(["structured"], 5),
        )

        assert output is not None
        assert output.req_ids == ["structured", "draft"]
        assert output.sampled_token_ids == [[5], [7, 9]]
        assert structured_state.token_ids == [1, 6, 5]
        assert draft_state.token_ids == [2, 3, 7, 9]

    def test_structured_output_after_spec_decode_uses_segment_start_row(self) -> None:
        runner = self._make_runner()
        draft_state = self._make_state([1, 6])
        structured_state = self._make_state([2, 3])
        decode_reqs = [("draft", draft_state), ("structured", structured_state)]
        segments = (
            mr.PagedDecodeSegment(
                req_id="draft",
                input_token_ids=(6, 7),
                start_row=0,
                num_query_tokens=2,
                draft_token_ids=(7,),
                cache_start_pos=1,
                block_ids=((0,),),
            ),
            mr.PagedDecodeSegment(
                req_id="structured",
                input_token_ids=(3,),
                start_row=2,
                num_query_tokens=1,
                draft_token_ids=(),
                cache_start_pos=1,
                block_ids=((1,),),
            ),
        )
        scheduler_output = self._make_scheduler_output(
            {"draft": 2, "structured": 1},
            {"draft": [7]},
        )
        self._install_paged_state(
            runner,
            decode_reqs,
            segments,
            self._make_logits([7, 9, 0]),
            scheduler_output,
        )

        output = runner.sample_tokens(
            grammar_output=self._make_grammar_output(["structured"], 5),
        )

        assert output is not None
        assert output.req_ids == ["draft", "structured"]
        assert output.sampled_token_ids == [[7, 9], [5]]
        assert draft_state.token_ids == [1, 6, 7, 9]
        assert structured_state.token_ids == [2, 3, 5]

    def test_structured_output_masks_same_request_spec_decode_rows(self) -> None:
        runner = self._make_runner()
        req_state = self._make_state([1, 6])
        decode_reqs = [("r0", req_state)]
        segment = mr.PagedDecodeSegment(
            req_id="r0",
            input_token_ids=(6, 7),
            start_row=0,
            num_query_tokens=2,
            draft_token_ids=(7,),
            cache_start_pos=1,
            block_ids=((0,),),
        )
        scheduler_output = self._make_scheduler_output(
            {"r0": 2},
            {"r0": [7]},
        )
        self._install_paged_state(
            runner,
            decode_reqs,
            (segment,),
            self._make_logits([0, 0]),
            scheduler_output,
        )

        output = runner.sample_tokens(
            grammar_output=self._make_grammar_rows(["r0"], [7, 9]),
        )

        assert output is not None
        assert output.sampled_token_ids == [[7, 9]]
        assert req_state.token_ids == [1, 6, 7, 9]

    def test_rejects_non_greedy_spec_decode_verification(self) -> None:
        runner = self._make_runner()
        req_state = self._make_state([1, 6], temperature=0.7)
        decode_reqs = [("r0", req_state)]
        segment = mr.PagedDecodeSegment(
            req_id="r0",
            input_token_ids=(6, 7),
            start_row=0,
            num_query_tokens=2,
            draft_token_ids=(7,),
            cache_start_pos=1,
            block_ids=((0,),),
        )
        scheduler_output = self._make_scheduler_output(
            {"r0": 2},
            {"r0": [7]},
        )
        self._install_paged_state(
            runner,
            decode_reqs,
            (segment,),
            self._make_logits([7, 9]),
            scheduler_output,
        )

        with pytest.raises(NotImplementedError, match="greedy sampling"):
            runner.sample_tokens(grammar_output=None)

        assert req_state.token_ids == [1, 6]


class TestV1MetalModelRunnerExecuteModel:
    def _make_runner(self) -> mr.MetalModelRunner:
        return make_stub_runner()

    def _make_scheduler_output(
        self,
        cached_req_ids: list[str] | None = None,
        *,
        finished_req_ids: set[str] | None = None,
        scheduled_spec_decode_tokens: dict[str, list[int]] | None = None,
        num_invalid_spec_tokens: dict[str, int] | None = None,
        scheduled_new_reqs: list[NewRequestData] | None = None,
    ) -> SchedulerOutput:
        req_ids = cached_req_ids or []
        return SchedulerOutput(
            scheduled_new_reqs=scheduled_new_reqs or [],
            scheduled_cached_reqs=CachedRequestData(
                req_ids=req_ids,
                resumed_req_ids=set(),
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=[None] * len(req_ids),
                num_computed_tokens=[0] * len(req_ids),
                num_output_tokens=[0] * len(req_ids),
            ),
            num_scheduled_tokens=dict.fromkeys(req_ids, 1),
            total_num_scheduled_tokens=len(req_ids),
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens or {},
            num_invalid_spec_tokens=num_invalid_spec_tokens,
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=finished_req_ids or set(),
            free_encoder_mm_hashes=[],
            preempted_req_ids=set(),
            has_structured_output_requests=False,
        )

    def _make_new_request(
        self,
        req_id: str = "new",
    ) -> NewRequestData:
        return NewRequestData(
            req_id=req_id,
            prompt_token_ids=[1],
            mm_features=[],
            sampling_params=SamplingParams(),
            pooling_params=None,
            block_ids=([0],),
            num_computed_tokens=0,
            lora_request=None,
        )

    def test_returns_empty_output_directly_for_empty_batch(self) -> None:
        runner = self._make_runner()

        out = runner.execute_model(self._make_scheduler_output())

        assert out is not None
        assert out.req_ids == []
        assert out.req_id_to_index == {}
        assert out.sampled_token_ids == []
        assert runner._pending_output is None

    def test_non_paged_cached_request_without_state_raises(self) -> None:
        runner = self._make_runner()

        with pytest.raises(RuntimeError, match="req-0"):
            runner.execute_model(self._make_scheduler_output(["req-0"]))

        assert runner._pending_output is None

    def test_paged_cached_request_without_state_raises(self) -> None:
        runner = self._make_runner()
        runner._paged_attention_runtime = MHAPagedAttentionRuntime(
            num_layers=1,
            num_kv_heads=1,
            head_dim=4,
            block_size=4,
            dtype=mx.float32,
        )

        with pytest.raises(RuntimeError, match="req-0"):
            runner.execute_model(self._make_scheduler_output(["req-0"]))

        assert runner._pending_output is None

    def test_missing_cached_request_fails_before_new_prefill(self, monkeypatch) -> None:
        runner = self._make_runner()
        monkeypatch.setattr(
            runner,
            "_prefill_single",
            lambda *args, **kwargs: pytest.fail("prefill should not run"),
        )
        new_req = self._make_new_request()

        with pytest.raises(RuntimeError, match="missing"):
            runner.execute_model(
                self._make_scheduler_output(
                    ["missing"],
                    scheduled_new_reqs=[new_req],
                )
            )

        assert "new" not in runner._request_states
        assert runner._pending_output is None

    def test_missing_cached_request_materializes_released_gdn_state(self) -> None:
        cache = GDNPagedStateCache(
            num_layers=1,
            max_seqs=2,
            conv_kernel_dim=2,
            conv_dim=4,
            num_v_heads=1,
            value_head_dim=4,
            key_head_dim=32,
            initial_seqs=0,
            dtype=mx.float32,
        )
        runtime = HybridRuntimeStub(cache)
        runner = make_stub_runner(_paged_attention_runtime=runtime)
        runner._request_states["done"] = mr.RequestState(
            token_ids=[1],
            prompt_len=1,
            cache=[],
            sampling_params=SamplingParams(),
            generator=None,
            generated_tokens=0,
        )
        slot = runtime.gdn_state_manager.assign_step_slots(["done"])[0]
        cache.set_pending_conv_state(0, [slot], mx.full((1, 1, 4), 7, dtype=mx.float32))
        cache.set_pending_recurrent_state(
            0,
            [slot],
            mx.full((1, 1, 4, 32), 9, dtype=mx.float32),
        )

        with pytest.raises(RuntimeError, match="missing"):
            runner.execute_model(
                self._make_scheduler_output(
                    ["missing"],
                    finished_req_ids={"done"},
                )
            )

        assert not cache.has_pending_conv_state(0)
        assert not cache.has_pending_recurrent_state(0)
        mx.eval(cache.conv_states[0], cache.recurrent_states[0])
        np.testing.assert_array_equal(np.array(cache.conv_states[0][slot]), 7)
        np.testing.assert_array_equal(np.array(cache.recurrent_states[0][slot]), 9)
        assert runtime.gdn_state_manager.needs_materialize is False

    def test_non_paged_spec_decode_fails_after_cleanup_before_new_state(self) -> None:
        runner = self._make_runner()
        runner._request_states["done"] = mr.RequestState(
            token_ids=[1],
            prompt_len=1,
            cache=[],
            sampling_params=SamplingParams(),
            generator=None,
            generated_tokens=0,
        )
        scheduler_output = self._make_scheduler_output(
            finished_req_ids={"done"},
            scheduled_spec_decode_tokens={"req-0": [7]},
            scheduled_new_reqs=[self._make_new_request()],
        )

        with pytest.raises(NotImplementedError, match="requires paged attention"):
            runner.execute_model(scheduler_output)

        assert "done" not in runner._request_states
        assert "new" not in runner._request_states

    def test_paged_spec_decode_failure_does_not_mutate_request_setup(self) -> None:
        runner = self._make_runner()
        runner._paged_attention_runtime = MHAPagedAttentionRuntime(
            num_layers=1,
            num_kv_heads=1,
            head_dim=4,
            block_size=4,
            dtype=mx.float32,
        )
        req_state = mr.RequestState(
            token_ids=[1, 6],
            prompt_len=1,
            cache=[],
            sampling_params=SamplingParams(),
            generator=None,
            generated_tokens=1,
            block_ids=[[0]],
        )
        runner._request_states["r0"] = req_state
        scheduler_output = self._make_scheduler_output(
            ["r0"],
            scheduled_spec_decode_tokens={"r0": [-1]},
            num_invalid_spec_tokens={"r0": 1},
            scheduled_new_reqs=[self._make_new_request()],
        )
        scheduler_output.num_scheduled_tokens = {"r0": 2, "new": 1}
        scheduler_output.total_num_scheduled_tokens = 3
        scheduler_output.scheduled_cached_reqs.new_block_ids = [([99],)]

        with pytest.raises(NotImplementedError, match="scheduler-invalid"):
            runner.execute_model(scheduler_output)

        assert req_state.block_ids == [[0]]
        assert "new" not in runner._request_states

    def test_gemma4_mtp_async_scheduling_fails_before_request_setup(self) -> None:
        runner = self._make_runner()
        runner.use_async_scheduling = True
        runner.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(distributed_executor_backend=None),
            speculative_config=SimpleNamespace(
                method="mtp",
                draft_model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(
                        model_type="gemma4_assistant",
                        architectures=["Gemma4AssistantForCausalLM"],
                    )
                ),
            ),
        )
        scheduler_output = self._make_scheduler_output(
            scheduled_new_reqs=[self._make_new_request()]
        )

        with pytest.raises(NotImplementedError, match="no-async-scheduling"):
            runner.execute_model(scheduler_output)

        assert "new" not in runner._request_states


class TestV1MetalModelRunnerGDNSubmit:
    def make_gdn_cache(self) -> GDNPagedStateCache:
        return GDNPagedStateCache(
            num_layers=1,
            max_seqs=2,
            conv_kernel_dim=2,
            conv_dim=4,
            num_v_heads=1,
            value_head_dim=4,
            key_head_dim=32,
            initial_seqs=0,
            dtype=mx.float32,
        )

    def make_runtime_with_side_effects(self) -> ForwardOutputRuntimeStub:
        conv_states = [
            mx.array([1], dtype=mx.float32),
            mx.array([2], dtype=mx.float32),
        ]
        recurrent_states = [
            mx.array([3], dtype=mx.float32),
            mx.array([4], dtype=mx.float32),
        ]
        return ForwardOutputRuntimeStub([*conv_states, *recurrent_states])

    def test_prefill_hybrid_submits_pending_compact_gdn_states(
        self, monkeypatch
    ) -> None:
        submitted: list[tuple[object, ...]] = []
        cache = self.make_gdn_cache()
        pending_conv = mx.full((1, 1, 4), 7, dtype=mx.float32)
        pending_recurrent = mx.full((1, 1, 4, 32), 9, dtype=mx.float32)
        cache.ensure_capacity(2)
        cache.set_pending_conv_state(0, [1], pending_conv)
        cache.set_pending_recurrent_state(0, [1], pending_recurrent)
        backend = HybridRuntimeStub(cache)
        runner = make_stub_runner(_paged_attention_runtime=backend)
        logits = mx.array([0], dtype=mx.float32)
        monkeypatch.setattr(mr.mx, "async_eval", lambda *args: submitted.append(args))

        runner._submit_paged_forward_outputs(logits)

        assert len(submitted) == 1
        # Pending compact updates ride the submission; stable pool arrays may
        # accompany them (shared pools carry sibling layers' state).
        assert any(a is pending_conv for a in submitted[0])
        assert any(a is pending_recurrent for a in submitted[0])
        assert cache.has_pending_conv_state(0)
        assert cache.has_pending_recurrent_state(0)

    def test_hybrid_submits_primary_outputs_before_gdn_states(
        self, monkeypatch
    ) -> None:
        submitted: list[tuple[object, ...]] = []
        runtime = self.make_runtime_with_side_effects()
        runner = make_stub_runner(_paged_attention_runtime=runtime)
        logits = mx.array([0], dtype=mx.float32)
        target_hidden_states = mx.array([5], dtype=mx.float32)
        monkeypatch.setattr(mr.mx, "async_eval", lambda *args: submitted.append(args))

        runner._submit_paged_forward_outputs(logits, target_hidden_states)

        assert len(submitted) == 1
        assert submitted[0][0] is logits
        assert submitted[0][1] is target_hidden_states
        for actual, expected in zip(submitted[0][2:], runtime._arrays, strict=True):
            assert actual is expected

    def test_pooling_forward_submits_runtime_outputs(self, monkeypatch) -> None:
        submitted: list[tuple[object, ...]] = []
        runtime = self.make_runtime_with_side_effects()
        pooling_hidden_states = mx.array([[[1.0]]], dtype=mx.float32)
        runner = make_stub_runner(
            _paged_attention_runtime=runtime,
            _is_pooling=True,
            _pooling_backend=PoolingForwardBackendStub(pooling_hidden_states),
            _paged_block_size=4,
            num_layers=0,
        )
        monkeypatch.setattr(mr.mx, "async_eval", lambda *args: submitted.append(args))

        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[
                mr.PrefillRequest(
                    req_id="pool-0",
                    token_ids=[1],
                    sampling_params=SamplingParams(),
                    block_ids=[[0]],
                    generator=None,
                    prompt_len=1,
                    start_pos=0,
                    full_prompt_token_ids=[1],
                    pooling_params=PoolingParams(),
                )
            ],
            decode_reqs=[],
            scheduler_output=SchedulerOutput.make_empty(),
        )

        assert len(submitted) == 1
        assert submitted[0][0] is pooling_hidden_states
        for actual, expected in zip(submitted[0][1:], runtime._arrays, strict=True):
            assert actual is expected

    def test_non_last_pp_send_submits_runtime_outputs(self, monkeypatch) -> None:
        submitted: list[tuple[object, ...]] = []
        runtime = self.make_runtime_with_side_effects()
        runner = make_stub_runner(
            _paged_attention_runtime=runtime,
            _paged_block_size=4,
            num_layers=0,
        )
        runner.pp = SimpleNamespace(size=2, is_last=False)
        stage_output = mx.array([[[1.0]]], dtype=mx.float32)
        send_handle = mx.array([2.0], dtype=mx.float32)
        runner._pp_model = lambda input_ids, cache: stage_output
        monkeypatch.setattr(mr, "pipeline_send", lambda output, pp: send_handle)
        monkeypatch.setattr(mr.mx, "async_eval", lambda *args: submitted.append(args))

        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[
                mr.PrefillRequest(
                    req_id="pp-0",
                    token_ids=[1],
                    sampling_params=SamplingParams(),
                    block_ids=[[0]],
                    generator=None,
                    prompt_len=1,
                    start_pos=0,
                    full_prompt_token_ids=[1],
                )
            ],
            decode_reqs=[],
            scheduler_output=SchedulerOutput.make_empty(),
        )

        assert len(submitted) == 1
        assert submitted[0][0] is send_handle
        for actual, expected in zip(submitted[0][1:], runtime._arrays, strict=True):
            assert actual is expected

    def test_prefill_non_hybrid_submits_logits_only(self, monkeypatch) -> None:
        submitted: list[tuple[object, ...]] = []
        runner = make_stub_runner(
            _paged_attention_runtime=MHAPagedAttentionRuntime(
                num_layers=1,
                num_kv_heads=1,
                head_dim=4,
                block_size=4,
                dtype=mx.float32,
            )
        )
        logits = mx.array([0], dtype=mx.float32)
        monkeypatch.setattr(mr.mx, "async_eval", lambda *args: submitted.append(args))

        runner._submit_paged_forward_outputs(logits)

        assert submitted == [(logits,)]

    def test_non_last_pp_sample_materializes_reused_slot_state(self) -> None:
        cache = self.make_gdn_cache()
        runtime = HybridRuntimeStub(cache)
        runner = make_stub_runner(_paged_attention_runtime=runtime)
        runner.pp = SimpleNamespace(size=2, is_last=False)
        runner._execute_model_state = object()
        runner._request_states["done"] = mr.RequestState(
            token_ids=[1],
            prompt_len=1,
            cache=[],
            sampling_params=SamplingParams(),
            generator=None,
            generated_tokens=0,
        )
        runner._paged_request_seq_lens["done"] = 1

        released_slot = runtime.gdn_state_manager.assign_step_slots(["done"])[0]
        runner._reconcile_request_lifecycle({"done"}, materialize_runtime_state=False)
        reused_slot = runtime.gdn_state_manager.assign_step_slots(["next"])[0]
        assert reused_slot == released_slot

        cache.set_pending_conv_state(
            0, [reused_slot], mx.full((1, 1, 4), 7, dtype=mx.float32)
        )
        cache.set_pending_recurrent_state(
            0,
            [reused_slot],
            mx.full((1, 1, 4, 32), 9, dtype=mx.float32),
        )

        output = runner.sample_tokens(None)

        assert output is mr.EMPTY_MODEL_RUNNER_OUTPUT
        assert runner._execute_model_state is None
        assert not cache.has_pending_conv_state(0)
        assert not cache.has_pending_recurrent_state(0)
        mx.eval(cache.conv_states[0], cache.recurrent_states[0])
        np.testing.assert_array_equal(np.array(cache.conv_states[0][reused_slot]), 7)
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0][reused_slot]),
            9,
        )
        assert runtime.gdn_state_manager.needs_materialize is False


class TestV1MetalModelRunnerGDNLifecycle:
    def _make_runner(
        self,
    ) -> tuple[mr.MetalModelRunner, HybridRuntimeStub, GDNPagedStateCache]:
        cache = GDNPagedStateCache(
            num_layers=1,
            max_seqs=2,
            conv_kernel_dim=2,
            conv_dim=4,
            num_v_heads=1,
            value_head_dim=4,
            key_head_dim=32,
            initial_seqs=0,
            dtype=mx.float32,
        )
        runtime = HybridRuntimeStub(cache)
        runner = make_stub_runner(_paged_attention_runtime=runtime)
        return runner, runtime, cache

    def _make_scheduler_output(
        self,
        *,
        resumed_req_ids: set[str] | None = None,
        preempted_req_ids: set[str] | None = None,
        scheduled_encoder_inputs: dict[str, list[int]] | None = None,
    ) -> SchedulerOutput:
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData(
                req_ids=list(resumed_req_ids or ()),
                resumed_req_ids=resumed_req_ids or set(),
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=[None] * len(resumed_req_ids or ()),
                num_computed_tokens=[0] * len(resumed_req_ids or ()),
                num_output_tokens=[0] * len(resumed_req_ids or ()),
            ),
            num_scheduled_tokens=dict.fromkeys(resumed_req_ids or (), 1),
            total_num_scheduled_tokens=len(resumed_req_ids or ()),
            scheduled_spec_decode_tokens={},
            num_invalid_spec_tokens=None,
            scheduled_encoder_inputs=scheduled_encoder_inputs or {},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            preempted_req_ids=preempted_req_ids or set(),
            has_structured_output_requests=False,
        )

    @pytest.mark.parametrize(
        ("event_kwargs", "req_id"),
        [
            ({"preempted_req_ids": {"req-0"}}, "req-0"),
            ({"resumed_req_ids": {"req-0"}}, "req-0"),
        ],
        ids=["preempted", "resumed"],
    )
    def test_preempt_or_resume_releases_runtime_state_not_runner_metadata(
        self,
        event_kwargs: dict[str, set[str]],
        req_id: str,
    ) -> None:
        runner, runtime, cache = self._make_runner()
        state = mr.RequestState(
            token_ids=[1, 2],
            prompt_len=2,
            cache=[],
            sampling_params=SamplingParams(),
            generator=None,
            generated_tokens=0,
        )
        runner._request_states[req_id] = state
        runner._paged_request_seq_lens[req_id] = 2
        slot = runtime.gdn_state_manager.assign_step_slots([req_id])[0]

        cache.set_pending_conv_state(
            0,
            [slot],
            mx.full((1, 1, 4), 7, dtype=mx.float32),
        )
        cache.set_pending_recurrent_state(
            0,
            [slot],
            mx.full((1, 1, 4, 32), 9, dtype=mx.float32),
        )
        scheduler_output = self._make_scheduler_output(
            scheduled_encoder_inputs={req_id: [0]},
            **event_kwargs,
        )

        with pytest.raises(RuntimeError, match="Multimodal encoder dispatch"):
            runner.execute_model(scheduler_output)

        assert runner._request_states[req_id] is state
        assert runner._paged_request_seq_lens[req_id] == 2
        assert runtime.gdn_state_manager.request_slots == {}
        assert runtime.gdn_state_manager.free_slots == (slot,)
        assert runtime.gdn_state_manager.needs_materialize is False
        mx.eval(cache.conv_states[0], cache.recurrent_states[0])
        np.testing.assert_array_equal(np.array(cache.conv_states[0][slot]), 7)
        np.testing.assert_array_equal(np.array(cache.recurrent_states[0][slot]), 9)

    def test_start_paged_forward_assigns_hybrid_slots_in_batch_order(
        self, monkeypatch
    ) -> None:
        cache = GDNPagedStateCache(
            num_layers=1,
            max_seqs=4,
            conv_kernel_dim=2,
            conv_dim=4,
            num_v_heads=1,
            value_head_dim=4,
            key_head_dim=32,
            initial_seqs=0,
            dtype=mx.float32,
        )
        runtime = HybridRuntimeStub(cache)
        runner = make_stub_runner(_paged_attention_runtime=runtime)
        runner.num_layers = 0
        runner._paged_block_size = 4
        runner._paged_group_block_sizes = (4,)
        runner._paged_scheduler_group_indices = (0,)
        runner._paged_request_seq_lens["decode-0"] = 1

        captured: dict[str, object] = {}

        def fake_target_forward(input_ids, *, cache, collect_hidden_states):
            del cache, collect_hidden_states
            ctx = mr.get_context()
            assert ctx is not None
            captured["input_ids"] = input_ids.tolist()
            captured["gdn_slot_mapping"] = list(ctx.gdn_slot_mapping or [])
            return mr.TargetModelForwardOutput(logits=mx.zeros((1, 2, 16)))

        monkeypatch.setattr(runner, "_target_forward", fake_target_forward)

        decode_state = mr.RequestState(
            token_ids=[5, 6],
            prompt_len=1,
            cache=[],
            sampling_params=SamplingParams(),
            generator=None,
            generated_tokens=1,
        )
        decode_state.block_ids = [[0]]
        prefill = mr.PrefillRequest(
            req_id="prefill-0",
            token_ids=[9],
            sampling_params=SamplingParams(),
            block_ids=[[1]],
            generator=None,
            prompt_len=1,
            start_pos=0,
            full_prompt_token_ids=[9],
        )
        scheduler_output = SchedulerOutput.make_empty()

        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[prefill],
            decode_reqs=[("decode-0", decode_state)],
            scheduler_output=scheduler_output,
        )

        assert captured["input_ids"] == [[6, 9]]
        assert captured["gdn_slot_mapping"] == [0, 1]
        assert runtime.gdn_state_manager.request_slots == {
            "decode-0": 0,
            "prefill-0": 1,
        }

    def test_sample_tokens_materializes_reused_slot_state(self, monkeypatch) -> None:
        cache = GDNPagedStateCache(
            num_layers=1,
            max_seqs=2,
            conv_kernel_dim=2,
            conv_dim=4,
            num_v_heads=1,
            value_head_dim=4,
            key_head_dim=32,
            initial_seqs=0,
            dtype=mx.float32,
        )
        runtime = HybridRuntimeStub(cache)
        runner = make_stub_runner(_paged_attention_runtime=runtime)
        runner._execute_model_state = object()
        runner._request_states["done"] = mr.RequestState(
            token_ids=[1],
            prompt_len=1,
            cache=[],
            sampling_params=SamplingParams(),
            generator=None,
            generated_tokens=0,
        )
        runner._paged_request_seq_lens["done"] = 1

        released_slot = runtime.gdn_state_manager.assign_step_slots(["done"])[0]
        runner._reconcile_request_lifecycle({"done"}, materialize_runtime_state=False)
        reused_slot = runtime.gdn_state_manager.assign_step_slots(["next"])[0]
        assert reused_slot == released_slot

        cache.set_pending_conv_state(
            0, [reused_slot], mx.full((1, 1, 4), 7, dtype=mx.float32)
        )
        cache.set_pending_recurrent_state(
            0,
            [reused_slot],
            mx.full((1, 1, 4, 32), 9, dtype=mx.float32),
        )

        expected_output = object()
        monkeypatch.setattr(
            runner,
            "_sample_paged_batch",
            lambda grammar_output: (mr._ExecutionBatch(), object()),
        )
        monkeypatch.setattr(runner, "_validate_scheduled_outputs", lambda *args: None)
        monkeypatch.setattr(runner, "_build_output", lambda batch: expected_output)

        output = runner.sample_tokens(None)

        assert output is expected_output
        assert not cache.has_pending_conv_state(0)
        assert not cache.has_pending_recurrent_state(0)
        mx.eval(cache.conv_states[0], cache.recurrent_states[0])
        np.testing.assert_array_equal(np.array(cache.conv_states[0][reused_slot]), 7)
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0][reused_slot]),
            9,
        )
        assert runtime.gdn_state_manager.needs_materialize is False


class TestRunnerMlaProperties:
    def _make_runner(self, args: dict) -> mr.MetalModelRunner:
        return make_stub_runner(model_args=args)

    def test_mla_latent_dim_does_not_require_resolve_model_dims(self) -> None:
        runner = self._make_runner(
            {
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "hidden_size": 512,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
            }
        )

        assert runner.mla_latent_dim == 576

    def test_is_mla_true_when_kv_lora_rank_present(self) -> None:
        runner = self._make_runner({"kv_lora_rank": 512})
        assert runner.is_mla is True

    def test_is_mla_false_for_standard_mha(self) -> None:
        runner = self._make_runner(
            {"num_hidden_layers": 32, "num_attention_heads": 32, "hidden_size": 4096}
        )
        assert runner.is_mla is False


class TestMergeVerifyWindows:
    """merge_verify_windows derivation: the runtime decision behind which
    models keep spec-verify windows merged (window mode) vs expanded.
    Window mode is opt-in, so the flag gates everything; with it set the
    model-class exclusions still apply."""

    def test_false_by_default_without_opt_in(self) -> None:
        assert make_stub_runner().merge_verify_windows is False

    def test_true_for_plain_mha_when_opted_in(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_SPEC_VERIFY_WINDOW", "1")
        assert make_stub_runner().merge_verify_windows is True

    def test_false_for_mla_even_when_opted_in(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_SPEC_VERIFY_WINDOW", "1")
        runner = make_stub_runner(model_args={"kv_lora_rank": 512})
        assert runner.merge_verify_windows is False

    def test_false_for_hybrid_even_when_opted_in(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_SPEC_VERIFY_WINDOW", "1")
        runner = make_stub_runner(model_args={"full_attention_interval": 4})
        assert runner.merge_verify_windows is False

    def test_false_past_window_head_bound_even_when_opted_in(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_SPEC_VERIFY_WINDOW", "1")
        runner = make_stub_runner(
            model_config=SimpleNamespace(
                runner_type="generate", get_head_size=lambda: 512
            )
        )
        assert runner.merge_verify_windows is False


class TestLoadModelPipelineSplitOrdering:
    def test_split_runs_before_lora_setup_on_pp_stage(self) -> None:
        # The pipeline split must run adjacent to the (lazy) load and before LoRA
        # setup, so the stage's non-owned layers are pruned before anything
        # materializes them. Pin the order so a future edit cannot move the split
        # back after LoRA and silently reintroduce the full-model peak.
        events: list[str] = []

        class _FakeGroup:
            def rank(self) -> int:
                return 0

            def size(self) -> int:
                return 2

        runner = make_stub_runner(
            pp=PipelineGroup(_FakeGroup()),
            model_config=SimpleNamespace(runner_type="generate", hf_config=None),
            metal_config=SimpleNamespace(use_paged_attention=True),
            scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=1),
            kv_cache_dtype=None,
        )
        runner._model_lifecycle = SimpleNamespace(
            load=lambda: events.append("load"),
            install_decode_dispatch=lambda: events.append("install"),
        )
        runner.apply_pipeline_split = lambda pp: events.append("split")
        runner._lora = SimpleNamespace(setup=lambda **kwargs: events.append("lora"))

        runner.load_model()

        # Decode-dispatch installs wrap model modules in place, so they
        # come after the split prunes non-owned layers, and the split itself
        # stays adjacent to the (lazy) load before LoRA setup.
        assert events == ["load", "split", "install", "lora"]


class _StageDummyRecorder:
    """Stands in for PipelinedModel: records the ids the dummy forward gets."""

    def __init__(self, output: object) -> None:
        self.output = output
        self.seen: object = None

    def dummy_forward(self, input_ids: object) -> object:
        self.seen = input_ids
        return self.output


class _FullPathMustNotRun:
    def __call__(self, input_ids: object) -> object:
        raise AssertionError("full-model dummy path must not run on a PP stage")


class TestDummyForwardOutputsPPRouting:
    class _Group:
        def __init__(self, rank: int, size: int) -> None:
            self._rank, self._size = rank, size

        def rank(self) -> int:
            return self._rank

        def size(self) -> int:
            return self._size

    def _pp_runner(self, rank: int, stage: _StageDummyRecorder) -> mr.MetalModelRunner:
        return make_stub_runner(
            pp=PipelineGroup(self._Group(rank, 2)),
            _pp_model=stage,
            model=_FullPathMustNotRun(),
        )

    def test_non_last_stage_returns_hidden_unextracted(self) -> None:
        # A non-last stage's dummy output is the raw hidden state — exactly what
        # serving evals — so no logits extraction may touch it.
        sentinel = SimpleNamespace(logits="would-be-extracted")
        stage = _StageDummyRecorder(sentinel)
        runner = self._pp_runner(0, stage)
        ids = mx.zeros((1, 3), dtype=mx.int32)

        outs = runner._dummy_forward_outputs(ids)

        assert outs == [sentinel]  # passthrough: extraction did NOT run
        assert stage.seen is ids

    def test_last_stage_extracts_logits(self) -> None:
        stage = _StageDummyRecorder(SimpleNamespace(logits="stage-logits"))
        runner = self._pp_runner(1, stage)

        outs = runner._dummy_forward_outputs(mx.zeros((1, 3), dtype=mx.int32))

        assert outs == ["stage-logits"]  # extraction ran on the last stage

    def test_single_stage_keeps_full_model_path(self) -> None:
        class _RecordingModel:
            seen: object = None

            def __call__(self, input_ids: object) -> object:
                self.seen = input_ids
                return SimpleNamespace(logits="full-logits")

        model = _RecordingModel()
        runner = make_stub_runner(model=model)  # pp/_pp_model default to None
        ids = mx.zeros((1, 3), dtype=mx.int32)

        outs = runner._dummy_forward_outputs(ids)

        assert outs == ["full-logits"]
        assert model.seen is ids

    def test_profile_run_profiles_the_stage_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Caller-level, mirroring TestMetalPoolingProfileWarmup: the mx cache
        # functions are patched so the test cannot clamp the process-wide
        # allocator, and what profile_run evals must be exactly the stage's
        # own hidden output — that is what makes the measured peak stage-shaped.
        hidden = mx.zeros((1, 4, 8), dtype=mx.float16)
        stage = _StageDummyRecorder(hidden)
        runner = make_stub_runner(
            pp=PipelineGroup(self._Group(0, 2)),
            _pp_model=stage,
            model=_FullPathMustNotRun(),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        )
        evaled: list[object] = []
        real_eval = mx.eval
        cache_readings = iter([100, 180])
        limits: list[int] = []

        def eval_and_record(*arrays: object) -> None:
            evaled.extend(arrays)
            real_eval(*arrays)

        monkeypatch.setattr(mr.mx, "clear_cache", lambda: None)
        monkeypatch.setattr(mr.mx, "get_cache_memory", lambda: next(cache_readings))
        monkeypatch.setattr(mr.mx, "set_cache_limit", limits.append)
        monkeypatch.setattr(mr.mx, "eval", eval_and_record)

        overhead = runner.profile_run()

        assert isinstance(stage.seen, mx.array)
        assert stage.seen.shape == (1, 4)
        assert stage.seen.dtype == mx.int32
        assert evaled == [hidden]  # eval saw exactly the stage-shaped output
        assert overhead == 80
        assert limits == [80]


class TestPipelineGateSpecDecodeDerivation:
    """Runner-side gate derivation: spec decode disables the pipeline."""

    @pytest.fixture(autouse=True)
    def _enable_pipeline(self, monkeypatch) -> None:
        # Set explicitly so gate derivation stays deterministic regardless
        # of the flag's default.
        monkeypatch.setenv("VLLM_METAL_DECODE_PIPELINE", "1")

    def test_pipeline_flag_defaults_on(self, monkeypatch) -> None:
        # The kill switch defaults to enabled; "0" is the opt-out.
        monkeypatch.delenv("VLLM_METAL_DECODE_PIPELINE", raising=False)
        assert metal_envs.VLLM_METAL_DECODE_PIPELINE is True

    def _runner(self, drafter: object | None = None) -> mr.MetalModelRunner:
        runner = make_stub_runner()
        runner._drafter = drafter
        runner._paged_attention_runtime = object()
        return runner

    def _scheduler_output(
        self, scheduled_spec_decode_tokens: dict[str, list[int]]
    ) -> SchedulerOutput:
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            num_invalid_spec_tokens=None,
            num_spec_tokens_to_schedule=0,
        )

    def _cached_decode_output(self, req_id: str) -> SchedulerOutput:
        cached = CachedRequestData(
            req_ids=[req_id],
            resumed_req_ids=set(),
            new_token_ids=[[7]],
            all_token_ids={req_id: [1, 7]},
            new_block_ids=[None],
            num_computed_tokens=[1],
            num_output_tokens=[1],
        )
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached,
            num_scheduled_tokens={req_id: 1},
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            num_invalid_spec_tokens=None,
            num_spec_tokens_to_schedule=0,
        )

    def test_prompt_logprobs_request_disables_pipeline(self) -> None:
        # Arrange — a decode-phase request asking for prompt logprobs must
        # keep the synchronous sample path (logprobs need eager logits).
        runner = self._runner(drafter=None)
        runner._request_states = {
            "r0": mr.RequestState(
                token_ids=[1, 7],
                prompt_len=1,
                cache=[],
                sampling_params=SamplingParams(temperature=0.0, prompt_logprobs=5),
                generator=None,
                generated_tokens=1,
            )
        }
        scheduler_output = self._cached_decode_output("r0")

        # Act
        decision = runner._evaluate_pipeline_gate(scheduler_output)

        # Assert
        assert decision.eligible is False
        assert decision.reason == "prompt logprobs requested"

    def test_clean_decode_step_is_eligible(self) -> None:
        # Arrange
        runner = self._runner(drafter=None)
        scheduler_output = self._scheduler_output({})

        # Act
        decision = runner._evaluate_pipeline_gate(scheduler_output)

        # Assert
        assert decision.eligible is True
        assert decision.reason == "eligible"

    @pytest.mark.parametrize("backend", ["mp", "ray", "external_launcher", None])
    def test_non_uniproc_executor_backend_disables_pipeline(self, backend) -> None:
        # Arrange — the pipeline's pending state assumes the in-process
        # uniproc executor; anything else takes the synchronous path.
        runner = self._runner(drafter=None)
        runner.vllm_config = SimpleNamespace(
            speculative_config=None,
            parallel_config=SimpleNamespace(distributed_executor_backend=backend),
        )
        scheduler_output = self._scheduler_output({})

        # Act
        decision = runner._evaluate_pipeline_gate(scheduler_output)

        # Assert
        assert decision.eligible is False
        assert decision.reason == "non-uniproc executor"

    def test_installed_drafter_disables_pipeline_for_whole_serve(self) -> None:
        # A configured drafter turns the pipeline off even on steps that
        # schedule no draft tokens — the honest stated limit of the feature.
        # Arrange
        runner = self._runner(drafter=object())
        scheduler_output = self._scheduler_output({})

        # Act
        decision = runner._evaluate_pipeline_gate(scheduler_output)

        # Assert
        assert decision.eligible is False
        assert decision.reason == "speculative decode"

    def test_active_spec_tokens_disable_pipeline_without_drafter(self) -> None:
        # Arrange — scheduler-driven drafts (e.g. ngram) with no drafter
        runner = self._runner(drafter=None)
        scheduler_output = self._scheduler_output({"req-0": [7, 9]})

        # Act
        decision = runner._evaluate_pipeline_gate(scheduler_output)

        # Assert
        assert decision.eligible is False
        assert decision.reason == "speculative decode"

    def test_configured_spec_decode_disables_pipeline_without_drafter(self) -> None:
        # A speculative_config alone must disable the pipeline — the
        # whole-serve exclusion is structural, not dependent on a drafter
        # having been installed or drafts being scheduled this step.
        # Arrange
        runner = self._runner(drafter=None)
        runner.vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(method="ngram"),
            parallel_config=SimpleNamespace(distributed_executor_backend=None),
        )
        scheduler_output = self._scheduler_output({})

        # Act
        decision = runner._evaluate_pipeline_gate(scheduler_output)

        # Assert
        assert decision.eligible is False
        assert decision.reason == "speculative decode"


class TestDeferredDecodeSampleThreading:
    """Runner-side threading for the deferred (pipelined) sample path."""

    def _decode_state(self, req_id: str) -> mr.RequestState:
        return mr.RequestState(
            token_ids=[3, 9],
            prompt_len=1,
            cache=[],
            sampling_params=SamplingParams(temperature=0.0),
            generator=None,
            generated_tokens=1,
        )

    def _paged_state(
        self, runner: mr.MetalModelRunner, decode_reqs, *, prefill_reqs=()
    ) -> mr._PagedForwardState:
        batch = mr._ExecutionBatch()
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={rid: 1 for rid, _ in decode_reqs},
            total_num_scheduled_tokens=len(decode_reqs),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            num_invalid_spec_tokens=None,
            num_spec_tokens_to_schedule=0,
        )
        return mr._PagedForwardState(
            batch=batch,
            prefill_reqs=list(prefill_reqs),
            decode_reqs=list(decode_reqs),
            scheduler_output=scheduler_output,
            logits=mx.array([[[0.0, 10.0, 0.0, 0.0]] * len(decode_reqs)]),
            target_hidden_states=None,
            pooling_hidden_states=None,
            cu_seqlens=list(range(len(decode_reqs) + 1)),
            decode_segments=[],
            num_decode_tokens=len(decode_reqs),
            mm_prefill_deltas={},
        )

    def test_submit_advances_bookkeeping_and_returns_async_output(self) -> None:
        # Arrange
        runner = make_stub_runner()
        state = self._decode_state("r0")
        runner._paged_request_seq_lens = {"r0": 1}
        runner._execute_model_state = self._paged_state(runner, [("r0", state)])
        runner._decode_pipeline.begin_step(
            mr.PipelineGateDecision(eligible=True, reason="eligible")
        )

        # Act
        output = runner._submit_deferred_decode_sample()

        # Assert — placeholder appended, counts advanced BEFORE resolve,
        # and the async output wrapper is the upstream-facing return type.
        assert isinstance(output, mr.MetalAsyncModelRunnerOutput)
        assert state.token_ids[-1] == mr.PENDING_TOKEN_PLACEHOLDER
        assert state.generated_tokens == 2
        assert runner._paged_request_seq_lens["r0"] == 2

    def test_resolve_backfills_the_submitted_token(self) -> None:
        # Arrange
        runner = make_stub_runner()
        state = self._decode_state("r0")
        runner._paged_request_seq_lens = {"r0": 1}
        runner._execute_model_state = self._paged_state(runner, [("r0", state)])
        runner._decode_pipeline.begin_step(
            mr.PipelineGateDecision(eligible=True, reason="eligible")
        )
        output = runner._submit_deferred_decode_sample()

        # Act — the wrapper's own resolution path fills the placeholder.
        result = output.get_output()

        # Assert
        assert state.token_ids[-1] == 1  # argmax of the logits row
        assert result.sampled_token_ids == [[1]]

    def test_assembled_pending_tokens_reach_the_forward_input(
        self, monkeypatch
    ) -> None:
        # Step k defers its sample; step k+1's forward must consume the lazy
        # sampled tokens as its decode input (device gather) while the host
        # token list still ends in the placeholder.
        # Arrange
        runner = make_stub_runner(model=SimpleNamespace())
        runner.num_layers = 0
        runner._paged_block_size = 4
        state = self._decode_state("r0")
        runner._paged_request_seq_lens = {"r0": 1}
        runner._execute_model_state = self._paged_state(runner, [("r0", state)])
        runner._decode_pipeline.begin_step(
            mr.PipelineGateDecision(eligible=True, reason="eligible")
        )
        runner._submit_deferred_decode_sample()
        runner._decode_pipeline.begin_step(
            mr.PipelineGateDecision(eligible=True, reason="eligible")
        )
        captured: dict[str, object] = {}

        def fake_target_forward(input_ids, *, cache, collect_hidden_states):
            del cache, collect_hidden_states
            captured["input_ids"] = input_ids.tolist()
            return mr.TargetModelForwardOutput(
                logits=mx.zeros((1, 1, 4)), hidden_states=None
            )

        monkeypatch.setattr(mr, "prepare_grouped", lambda *a, **k: None)
        monkeypatch.setattr(runner, "_target_forward", fake_target_forward)
        scheduler_output = self._paged_state(runner, [("r0", state)]).scheduler_output

        # Act
        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[],
            decode_reqs=[("r0", state)],
            scheduler_output=scheduler_output,
        )

        # Assert — argmax of step k's logits row is 1; the forward saw that
        # deferred value while the host list still holds the placeholder.
        assert captured["input_ids"] == [[1]]
        assert state.token_ids[-1] == mr.PENDING_TOKEN_PLACEHOLDER

    def test_pending_tokens_with_prefill_work_raise_gate_desync(
        self, monkeypatch
    ) -> None:
        # Arrange — a pending deferred sample plus prefill work on an
        # (incorrectly) eligible step must fail fast, not mix inputs.
        runner = make_stub_runner(model=SimpleNamespace())
        runner.num_layers = 0
        runner._paged_block_size = 4
        state = self._decode_state("r0")
        runner._paged_request_seq_lens = {"r0": 1}
        runner._execute_model_state = self._paged_state(runner, [("r0", state)])
        runner._decode_pipeline.begin_step(
            mr.PipelineGateDecision(eligible=True, reason="eligible")
        )
        runner._submit_deferred_decode_sample()
        runner._decode_pipeline.begin_step(
            mr.PipelineGateDecision(eligible=True, reason="eligible")
        )
        prefill = mr.PrefillRequest(
            req_id="p0",
            token_ids=[5],
            sampling_params=SamplingParams(),
            block_ids=[[0]],
            generator=None,
            prompt_len=1,
            start_pos=0,
            full_prompt_token_ids=None,
        )
        monkeypatch.setattr(mr, "prepare_grouped", lambda *a, **k: None)
        scheduler_output = self._paged_state(runner, [("r0", state)]).scheduler_output

        # Act / Assert
        with pytest.raises(RuntimeError, match="gate desynced"):
            runner._start_paged_forward(
                mr._ExecutionBatch(),
                prefill_reqs=[prefill],
                decode_reqs=[("r0", state)],
                scheduler_output=scheduler_output,
            )

    def test_reentrant_cached_request_blocks_next_gate(self, monkeypatch) -> None:
        # A cached request that was absent from the deferred step re-enters
        # on the next one; it has no pending row, so the gate must route the
        # step through the synchronous path instead of crashing assembly.
        # Arrange
        monkeypatch.setenv("VLLM_METAL_DECODE_PIPELINE", "1")
        runner = make_stub_runner()
        runner._paged_attention_runtime = SimpleNamespace(
            materialize_pending_state=lambda: None
        )
        state0 = self._decode_state("r0")
        state1 = self._decode_state("r1")
        runner._request_states = {"r0": state0, "r1": state1}
        runner._paged_request_seq_lens = {"r0": 1}
        runner._execute_model_state = self._paged_state(runner, [("r0", state0)])
        runner._decode_pipeline.begin_step(
            mr.PipelineGateDecision(eligible=True, reason="eligible")
        )
        runner._submit_deferred_decode_sample()
        cached = CachedRequestData(
            req_ids=["r0", "r1"],
            resumed_req_ids=set(),
            new_token_ids=[[7], [7]],
            all_token_ids={"r0": [3, 9, 7], "r1": [3, 9, 7]},
            new_block_ids=[None, None],
            num_computed_tokens=[2, 2],
            num_output_tokens=[2, 2],
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached,
            num_scheduled_tokens={"r0": 1, "r1": 1},
            total_num_scheduled_tokens=2,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            num_invalid_spec_tokens=None,
            num_spec_tokens_to_schedule=0,
        )

        # Act
        decision = runner._evaluate_pipeline_gate(scheduler_output)

        # Assert
        assert decision.eligible is False
        assert decision.reason == "cached request without a pending row"

    def test_sample_tokens_routes_eligible_step_to_deferred_submit(self) -> None:
        # The public engine seam: an eligible step's sample_tokens call must
        # return the deferred async output, not a synchronous one.
        # Arrange
        runner = make_stub_runner()
        state = self._decode_state("r0")
        runner._paged_request_seq_lens = {"r0": 1}
        runner._execute_model_state = self._paged_state(runner, [("r0", state)])
        runner._decode_pipeline.begin_step(
            mr.PipelineGateDecision(eligible=True, reason="eligible")
        )

        # Act
        output = runner.sample_tokens(None)

        # Assert
        assert isinstance(output, mr.MetalAsyncModelRunnerOutput)

    def test_sample_tokens_routes_ineligible_step_to_synchronous_path(
        self, monkeypatch
    ) -> None:
        # Arrange
        runner = make_stub_runner()
        state = self._decode_state("r0")
        runner._execute_model_state = self._paged_state(runner, [("r0", state)])
        runner._decode_pipeline.begin_step(
            mr.PipelineGateDecision(eligible=False, reason="prefill-phase requests")
        )
        calls: list[str] = []
        paged_state = runner._execute_model_state

        def fake_sample_paged_batch(grammar_output=None):
            calls.append("sync")
            return mr._ExecutionBatch(), paged_state.scheduler_output

        def unexpected_deferred_submit():
            raise AssertionError("ineligible step must not defer its sample")

        monkeypatch.setattr(runner, "_sample_paged_batch", fake_sample_paged_batch)
        monkeypatch.setattr(
            runner, "_submit_deferred_decode_sample", unexpected_deferred_submit
        )

        # Act
        output = runner.sample_tokens(None)

        # Assert
        assert calls == ["sync"]
        assert isinstance(output, ModelRunnerOutput)

    def test_sample_tokens_rejects_grammar_output_on_eligible_step(self) -> None:
        # Arrange — the gate blocks structured-output steps, so a grammar
        # bitmask arriving on an eligible step is a desync, not a fallback.
        runner = make_stub_runner()
        state = self._decode_state("r0")
        runner._paged_request_seq_lens = {"r0": 1}
        runner._execute_model_state = self._paged_state(runner, [("r0", state)])
        runner._decode_pipeline.begin_step(
            mr.PipelineGateDecision(eligible=True, reason="eligible")
        )
        grammar_output = GrammarOutput(
            structured_output_request_ids=[], grammar_bitmask=None
        )

        # Act / Assert
        with pytest.raises(RuntimeError, match="gate must block structured-output"):
            runner.sample_tokens(grammar_output)

    def test_execute_model_gates_pipeline_before_any_state_mutation(
        self, monkeypatch
    ) -> None:
        # begin_step must run before the first runner-owned mutation
        # (encoder-output free), so an ineligible step drains the pending
        # deferred sample before any state changes underneath it.
        # Arrange
        runner = make_stub_runner()
        events: list[str] = []
        monkeypatch.setattr(
            runner._decode_pipeline,
            "begin_step",
            lambda decision: events.append("begin_step"),
        )

        class _StopError(Exception):
            pass

        def record_free(mm_hashes):
            events.append("free_encoder_outputs")
            raise _StopError

        monkeypatch.setattr(runner, "_free_encoder_outputs", record_free)
        scheduler_output = self._paged_state(runner, []).scheduler_output

        # Act
        with pytest.raises(_StopError):
            runner.execute_model(scheduler_output)

        # Assert
        assert events == ["begin_step", "free_encoder_outputs"]

    def test_finished_request_between_steps_backfills_evicted_state(self) -> None:
        # A request that finishes (and is evicted) between submit and resolve
        # still receives its token through the direct state reference, the
        # next step's assembly gathers only the surviving row, and the
        # runner's dicts are not re-populated for the evicted request.
        # Arrange
        runner = make_stub_runner()
        state0 = self._decode_state("r0")
        state1 = self._decode_state("r1")
        runner._request_states = {"r0": state0, "r1": state1}
        runner._paged_request_seq_lens = {"r0": 1, "r1": 1}
        runner._execute_model_state = self._paged_state(
            runner, [("r0", state0), ("r1", state1)]
        )
        runner._decode_pipeline.begin_step(
            mr.PipelineGateDecision(eligible=True, reason="eligible")
        )
        output = runner._submit_deferred_decode_sample()
        del runner._request_states["r0"]
        del runner._paged_request_seq_lens["r0"]
        surviving_segment = PagedDecodeSegment(
            req_id="r1",
            input_token_ids=(mr.PENDING_TOKEN_PLACEHOLDER,),
            start_row=0,
            num_query_tokens=1,
            draft_token_ids=(),
            cache_start_pos=2,
            block_ids=((0,),),
        )
        runner._decode_pipeline.begin_step(
            mr.PipelineGateDecision(eligible=True, reason="eligible")
        )

        # Act
        input_ids = runner._decode_pipeline.assemble_decode_input_ids(
            [surviving_segment]
        )
        result = output.get_output()

        # Assert
        assert input_ids is not None
        assert input_ids.tolist() == [[1]]  # r1's row only (argmax of its logits)
        assert state0.token_ids[-1] == 1  # evicted state backfilled by reference
        assert state1.token_ids[-1] == 1
        assert result.sampled_token_ids == [[1], [1]]
        assert "r0" not in runner._request_states
        assert "r0" not in runner._paged_request_seq_lens

    def test_prefill_rows_in_deferred_submit_raise(self) -> None:
        # Arrange
        runner = make_stub_runner()
        state = self._decode_state("r0")
        prefill = mr.PrefillRequest(
            req_id="p0",
            token_ids=[5],
            sampling_params=SamplingParams(),
            block_ids=[[0]],
            generator=None,
            prompt_len=1,
            start_pos=0,
            full_prompt_token_ids=None,
        )
        runner._execute_model_state = self._paged_state(
            runner, [("r0", state)], prefill_reqs=[prefill]
        )

        # Act / Assert
        with pytest.raises(RuntimeError, match="pure single-token decode batch"):
            runner._submit_deferred_decode_sample()


def _body_stub(input_ids, cache=None):
    return input_ids


class TestIntermediateBodyOnlyForward:
    """Intermediate-only prefill chunks run the transformer body only."""

    def _intermediate_prefill_request(
        self, req_id: str = "r0", start_pos: int = 0
    ) -> mr.PrefillRequest:
        return mr.PrefillRequest(
            req_id=req_id,
            token_ids=[5, 6],
            sampling_params=SamplingParams(),
            block_ids=[[0]],
            generator=None,
            prompt_len=None,
            start_pos=start_pos,
            full_prompt_token_ids=None,
        )

    def _make_scheduler_output(
        self, num_scheduled_tokens: dict[str, int] | None = None
    ) -> SchedulerOutput:
        scheduled = (
            num_scheduled_tokens if num_scheduled_tokens is not None else {"r0": 2}
        )
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens=scheduled,
            total_num_scheduled_tokens=sum(scheduled.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            num_invalid_spec_tokens=None,
            num_spec_tokens_to_schedule=0,
        )

    def test_intermediate_only_step_runs_body_and_skips_lm_head(
        self, monkeypatch
    ) -> None:
        # Arrange
        captured: dict[str, object] = {}
        hidden = mx.zeros((1, 2, 8))

        def body(input_ids, cache=None):
            captured["body_input_ids"] = input_ids.tolist()
            return hidden

        runner = make_stub_runner(model=SimpleNamespace(model=body))
        runner.num_layers = 0
        runner._paged_block_size = 4

        def unexpected_target_forward(*args, **kwargs):
            raise AssertionError("intermediate-only step must not run lm_head")

        def record_submit(*outputs):
            captured["submitted"] = outputs

        monkeypatch.setattr(mr, "prepare_grouped", lambda *a, **k: None)
        monkeypatch.setattr(runner, "_target_forward", unexpected_target_forward)
        monkeypatch.setattr(runner, "_submit_paged_forward_outputs", record_submit)

        # Act
        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[self._intermediate_prefill_request()],
            decode_reqs=[],
            scheduler_output=self._make_scheduler_output(),
        )

        # Assert — the body ran on the chunk tokens, its hidden states were
        # submitted (forcing the KV/GDN cache writes through the lazy graph),
        # and no logits exist for the sampling step.
        assert captured["body_input_ids"] == [[5, 6]]
        assert any(out is hidden for out in captured["submitted"])
        state = runner._execute_model_state
        assert state is not None
        assert state.logits is None
        assert state.target_hidden_states is None
        assert state.intermediate_only is True
        assert state.cu_seqlens == [0, 2]

    @pytest.mark.parametrize("case", ["final_prefill", "with_decode", "drafter"])
    def test_non_intermediate_only_steps_run_full_forward(
        self, monkeypatch, case
    ) -> None:
        # A final chunk, a decode row, or any installed drafter (its
        # propose() bookkeeping must run every step) keeps the full forward.
        # Arrange
        captured: dict[str, object] = {}

        def body(input_ids, cache=None):
            raise AssertionError("body-only path must not run on this step")

        runner = make_stub_runner(model=SimpleNamespace(model=body))
        runner.num_layers = 0
        runner._paged_block_size = 4
        prefill_reqs = [self._intermediate_prefill_request()]
        decode_reqs: list[tuple[str, mr.RequestState]] = []
        scheduled = {"r0": 2}
        if case == "final_prefill":
            final = mr.PrefillRequest(
                req_id="r1",
                token_ids=[7, 8],
                sampling_params=SamplingParams(),
                block_ids=[[1]],
                generator=None,
                prompt_len=2,
                start_pos=0,
                full_prompt_token_ids=[7, 8],
            )
            prefill_reqs.append(final)
            scheduled["r1"] = 2
        elif case == "with_decode":
            state = mr.RequestState(
                token_ids=[3, 9],
                prompt_len=1,
                cache=[],
                sampling_params=SamplingParams(temperature=0.0),
                generator=None,
                generated_tokens=1,
                block_ids=[[2]],
            )
            decode_reqs = [("d0", state)]
            runner._paged_request_seq_lens = {"d0": 1}
            scheduled["d0"] = 1
        else:
            runner._drafter = SimpleNamespace(
                needs_target_hidden_states=lambda decode_segments, has_final_prefill: (
                    False
                )
            )

        def fake_target_forward(input_ids, *, cache, collect_hidden_states):
            del cache, collect_hidden_states
            captured["full_forward_tokens"] = input_ids.tolist()
            num_tokens = input_ids.shape[1]
            return mr.TargetModelForwardOutput(
                logits=mx.zeros((1, num_tokens, 16)),
                hidden_states=None,
            )

        monkeypatch.setattr(mr, "prepare_grouped", lambda *a, **k: None)
        monkeypatch.setattr(runner, "_target_forward", fake_target_forward)

        # Act
        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=prefill_reqs,
            decode_reqs=decode_reqs,
            scheduler_output=self._make_scheduler_output(scheduled),
        )

        # Assert
        assert "full_forward_tokens" in captured
        state_after = runner._execute_model_state
        assert state_after is not None
        assert state_after.logits is not None

    def test_unsupported_capability_runs_full_forward_silently(
        self, monkeypatch, caplog
    ) -> None:
        # A model the adapter cannot run projection-free keeps the full
        # forward on intermediate steps, with no per-step warning noise.
        # Arrange
        captured: dict[str, object] = {}
        runner = make_stub_runner(model=SimpleNamespace())
        runner.num_layers = 0
        runner._paged_block_size = 4
        runner._intermediate_forward_supported = False

        def fake_target_forward(input_ids, *, cache, collect_hidden_states):
            del cache, collect_hidden_states
            captured["full_forward_tokens"] = input_ids.tolist()
            return mr.TargetModelForwardOutput(
                logits=mx.zeros((1, 2, 16)), hidden_states=None
            )

        monkeypatch.setattr(mr, "prepare_grouped", lambda *a, **k: None)
        monkeypatch.setattr(runner, "_target_forward", fake_target_forward)

        # Act
        with caplog.at_level("WARNING"):
            runner._start_paged_forward(
                mr._ExecutionBatch(),
                prefill_reqs=[self._intermediate_prefill_request()],
                decode_reqs=[],
                scheduler_output=self._make_scheduler_output(),
            )

        # Assert — the full forward ran, the step is still marked
        # intermediate-only (sampling skipped downstream), and no warning.
        assert captured["full_forward_tokens"] == [[5, 6]]
        state = runner._execute_model_state
        assert state is not None
        assert state.intermediate_only is True
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_load_model_resolves_capability_once_from_the_adapter(self) -> None:
        # Arrange — the capability is a load-time fact, not a per-step probe.
        events: list[str] = []
        runner = make_stub_runner(
            model=SimpleNamespace(model=_body_stub),
            model_config=SimpleNamespace(runner_type="generate", hf_config=None),
            metal_config=SimpleNamespace(use_paged_attention=True),
            scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=1),
            kv_cache_dtype=None,
        )
        runner._intermediate_forward_supported = False
        runner._model_lifecycle = SimpleNamespace(
            load=lambda: events.append("load"),
            install_decode_dispatch=lambda: events.append("install"),
        )
        runner._lora = SimpleNamespace(setup=lambda **kwargs: events.append("lora"))

        # Act
        runner.load_model()

        # Assert
        assert runner._intermediate_forward_supported is True

    def test_drafter_needing_hidden_states_forces_full_forward(
        self, monkeypatch
    ) -> None:
        # A drafter that wants target hidden states from this step excludes
        # the body-only path even when every chunk is intermediate — the
        # hidden states only exist on the full forward.
        # Arrange
        captured: dict[str, object] = {}

        def body(input_ids, cache=None):
            raise AssertionError("body-only path must not run for the drafter")

        runner = make_stub_runner(model=SimpleNamespace(model=body))
        runner.num_layers = 0
        runner._paged_block_size = 4
        runner._drafter = SimpleNamespace(
            needs_target_hidden_states=lambda decode_segments, has_final_prefill: True
        )

        def fake_target_forward(input_ids, *, cache, collect_hidden_states):
            del cache
            captured["collect_hidden_states"] = collect_hidden_states
            return mr.TargetModelForwardOutput(
                logits=mx.zeros((1, 2, 16)),
                hidden_states=mx.zeros((1, 2, 8)),
            )

        monkeypatch.setattr(mr, "prepare_grouped", lambda *a, **k: None)
        monkeypatch.setattr(runner, "_target_forward", fake_target_forward)

        # Act
        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[self._intermediate_prefill_request()],
            decode_reqs=[],
            scheduler_output=self._make_scheduler_output(),
        )

        # Assert
        assert captured["collect_hidden_states"] is True
        state = runner._execute_model_state
        assert state is not None
        assert state.target_hidden_states is not None

    def _intermediate_only_state(
        self,
        batch: mr._ExecutionBatch,
        prefill_reqs: list[mr.PrefillRequest],
        decode_reqs: list[tuple[str, mr.RequestState]],
        logits: mx.array | None = None,
    ) -> mr._PagedForwardState:
        return mr._PagedForwardState(
            batch=batch,
            prefill_reqs=prefill_reqs,
            decode_reqs=decode_reqs,
            scheduler_output=self._make_scheduler_output(),
            logits=logits,
            target_hidden_states=None,
            pooling_hidden_states=None,
            cu_seqlens=[0, 2],
            decode_segments=(),
            num_decode_tokens=len(decode_reqs),
            mm_prefill_deltas={},
            intermediate_only=True,
        )

    def test_sample_paged_batch_books_seq_lens_without_logits(self) -> None:
        # Arrange — a continuation chunk (start_pos=3) stashes logits=None;
        # sampling must advance the seq len exactly like the full path
        # (start_pos + chunk length) and leave the pre-filled empty output.
        runner = make_stub_runner(model=SimpleNamespace())
        request = self._intermediate_prefill_request(start_pos=3)
        batch = mr._ExecutionBatch()
        output_idx = batch.add_output("r0", [])
        batch.paged_prefill_entries.append(
            mr._PendingPrefillEntry(
                output_idx=output_idx,
                prefill=request,
                result_mode="intermediate",
            )
        )
        runner._paged_request_seq_lens = {"r0": 3}
        runner._execute_model_state = self._intermediate_only_state(
            batch, [request], []
        )

        # Act
        result_batch, _ = runner._sample_paged_batch()

        # Assert
        assert result_batch.sampled_tokens == [[]]
        assert runner._paged_request_seq_lens["r0"] == 5

    def test_intermediate_only_skips_sampling_even_with_logits(
        self, monkeypatch
    ) -> None:
        # An unsupported adapter ran the full forward, so logits exist —
        # sampling must still be skipped so a seeded request's RNG never
        # advances on a discarded token.
        # Arrange
        runner = make_stub_runner(model=SimpleNamespace())
        generator = torch.Generator()
        generator.manual_seed(7)
        generator_state_before = generator.get_state()
        request = mr.PrefillRequest(
            req_id="r0",
            token_ids=[5, 6],
            sampling_params=SamplingParams(temperature=0.8, seed=7),
            block_ids=[[0]],
            generator=generator,
            prompt_len=None,
            start_pos=0,
            full_prompt_token_ids=None,
        )
        batch = mr._ExecutionBatch()
        output_idx = batch.add_output("r0", [])
        batch.paged_prefill_entries.append(
            mr._PendingPrefillEntry(
                output_idx=output_idx,
                prefill=request,
                result_mode="intermediate",
            )
        )
        runner._paged_request_seq_lens = {"r0": 0}
        runner._execute_model_state = self._intermediate_only_state(
            batch, [request], [], logits=mx.zeros((1, 2, 16))
        )

        def unexpected_sample(*args, **kwargs):
            raise AssertionError("intermediate-only step must not sample")

        monkeypatch.setattr(mr, "sample_prefill_tokens", unexpected_sample)
        monkeypatch.setattr(mr, "sample_decode_tokens", unexpected_sample)

        # Act
        result_batch, _ = runner._sample_paged_batch()

        # Assert
        assert result_batch.sampled_tokens == [[]]
        assert runner._paged_request_seq_lens["r0"] == 2
        assert torch.equal(generator.get_state(), generator_state_before)

    def test_sample_paged_batch_rejects_final_rows_without_logits(self) -> None:
        # Arrange — a final chunk must sample; reaching the no-logits path
        # with one is a routing desync, not a case to book silently.
        runner = make_stub_runner(model=SimpleNamespace())
        request = self._intermediate_prefill_request()
        batch = mr._ExecutionBatch()
        output_idx = batch.add_output("r0", [])
        batch.paged_prefill_entries.append(
            mr._PendingPrefillEntry(
                output_idx=output_idx,
                prefill=request,
                result_mode="new_final",
            )
        )
        runner._execute_model_state = self._intermediate_only_state(
            batch, [request], []
        )

        # Act / Assert
        with pytest.raises(RuntimeError, match="must sample"):
            runner._sample_paged_batch()

    def test_sample_paged_batch_rejects_decode_rows_without_logits(self) -> None:
        # Arrange
        runner = make_stub_runner(model=SimpleNamespace())
        state = mr.RequestState(
            token_ids=[3, 9],
            prompt_len=1,
            cache=[],
            sampling_params=SamplingParams(temperature=0.0),
            generator=None,
            generated_tokens=1,
        )
        runner._execute_model_state = self._intermediate_only_state(
            mr._ExecutionBatch(), [], [("d0", state)]
        )

        # Act / Assert
        with pytest.raises(RuntimeError, match="must sample"):
            runner._sample_paged_batch()


class TestStateBlockIdLifecycle:
    """Track the mamba group rows that the align runtime keys on."""

    def _make_runner(self) -> mr.MetalModelRunner:
        return make_stub_runner(
            _paged_attention_runtime=ForwardOutputRuntimeStub([]),
            _paged_scheduler_group_indices=(3,),
            _paged_state_group_indices=(0, 1, 2),
        )

    def _new_req(self, req_id: str, *, num_computed_tokens: int = 0) -> NewRequestData:
        return NewRequestData(
            req_id=req_id,
            prompt_token_ids=[1, 2, 3, 4, 5, 6],
            mm_features=[],
            sampling_params=SamplingParams(),
            pooling_params=None,
            block_ids=([10, 11], [20, 21], [30, 31], [40, 41]),
            num_computed_tokens=num_computed_tokens,
            lora_request=None,
        )

    def test_admission_tracks_mamba_groups_and_restore_pos(self) -> None:
        runner = self._make_runner()
        batch = mr._ExecutionBatch()

        runner._handle_new_requests(
            batch,
            [self._new_req("hit", num_computed_tokens=2)],  # prefix-hit admission
            SimpleNamespace(num_scheduled_tokens={"hit": 2}),
        )

        assert runner._state_block_ids_by_req["hit"] == [
            [10, 11],
            [20, 21],
            [30, 31],
        ]
        entry = batch.paged_prefill_entries[0]
        assert entry.prefill.start_pos == 2  # restore position, not zero

    def test_cached_append_extends_every_mamba_group(self) -> None:
        runner = self._make_runner()
        runner._state_block_ids_by_req["r"] = [[10], [20], [30]]
        runner._request_states["r"] = mr.RequestState(
            token_ids=[1],
            prompt_len=1,
            cache=[],
            sampling_params=SamplingParams(),
            block_ids=[[40]],
        )

        runner._update_cached_request_blocks(
            CachedRequestData(
                req_ids=["r"],
                resumed_req_ids=set(),
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=[([12], [22], [32], [42])],
                num_computed_tokens=[1],
                num_output_tokens=[0],
            )
        )

        assert runner._state_block_ids_by_req["r"] == [[10, 12], [20, 22], [30, 32]]
        assert runner._request_states["r"].block_ids == [[40, 42]]

    def test_resume_replaces_the_whole_table(self) -> None:
        runner = self._make_runner()
        runner._state_block_ids_by_req["r"] = [[10], [20], [30]]
        runner._request_states["r"] = mr.RequestState(
            token_ids=[1],
            prompt_len=1,
            cache=[],
            sampling_params=SamplingParams(),
            block_ids=[[40]],
        )

        runner._update_cached_request_blocks(
            CachedRequestData(
                req_ids=["r"],
                resumed_req_ids={"r"},
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=[([50], [60], [70], [80])],
                num_computed_tokens=[0],
                num_output_tokens=[0],
            )
        )

        assert runner._state_block_ids_by_req["r"] == [[50], [60], [70]]
        assert runner._request_states["r"].block_ids == [[80]]
