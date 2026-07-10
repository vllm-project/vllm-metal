# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

import vllm_metal.v1.model_runner as mr
from tests.stub_runner import make_stub_runner
from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.attention.runtime.mha import MHAPagedAttentionRuntime
from vllm_metal.attention.state import HybridGDNStateManager
from vllm_metal.distributed.pipeline import PipelineGroup
from vllm_metal.multimodal.qwen3_vl import Qwen3VLMultimodalAdapter
from vllm_metal.v1.gemma4_mtp import Gemma4MTPDraftSeed
from vllm_metal.v1.proposer import Gemma4MTPProposer


class HybridRuntimeStub:
    def __init__(self, state_cache: GDNPagedStateCache) -> None:
        self._gdn_state_manager = HybridGDNStateManager(state_cache)

    def needs_step_context(self) -> bool:
        return True

    @property
    def gdn_state_manager(self) -> HybridGDNStateManager:
        return self._gdn_state_manager

    def populate_step_context(self, *, req_ids: list[str], ctx) -> None:
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

    def populate_step_context(self, *, req_ids: list[str], ctx) -> None:
        del req_ids, ctx

    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        outputs.extend(self._arrays)

    def release_requests(self, req_ids: set[str]) -> None:
        del req_ids

    def materialize_pending_state(self) -> None:
        return None


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


class TestV1MetalModelRunnerGenerate:
    def _make_runner(self) -> mr.MetalModelRunner:
        return make_stub_runner(tokenizer=object())

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


class TestV1MetalModelRunnerSpecDecodeVerification:
    def _make_runner(self) -> mr.MetalModelRunner:
        return make_stub_runner(model_args={"vocab_size": 16})

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
    ) -> SimpleNamespace:
        return SimpleNamespace(
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            num_invalid_spec_tokens=num_invalid_spec_tokens,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
            finished_req_ids=set(),
        )

    def _make_gemma4_mtp_config(self) -> SimpleNamespace:
        return SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="mtp",
                draft_model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(
                        model_type="gemma4_assistant",
                        architectures=["Gemma4AssistantForCausalLM"],
                    )
                ),
            )
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
        runner = self._make_runner()
        runner.vllm_config = self._make_gemma4_mtp_config()
        runner._drafter = Gemma4MTPProposer(runner)
        runner.num_layers = 0
        runner._paged_block_size = 4
        runner._paged_request_seq_lens["r0"] = 1

        captured: dict[str, object] = {}

        def fake_prepare_unified(decode_info, prefill_info, block_size):
            captured["decode_info"] = decode_info
            captured["prefill_info"] = prefill_info
            captured["block_size"] = block_size

        def fake_target_forward(input_ids, *, cache, collect_hidden_states):
            del cache
            captured["input_ids"] = input_ids.tolist()
            captured["collect_hidden_states"] = collect_hidden_states
            return mr.TargetModelForwardOutput(
                logits=mx.zeros((1, 3, 16)),
                hidden_states=mx.ones((3, 4)),
            )

        monkeypatch.setattr(mr, "prepare_unified", fake_prepare_unified)
        monkeypatch.setattr(runner, "_target_forward", fake_target_forward)

        req_state = self._make_state([1, 6])
        req_state.block_ids = [0, 1]
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
        assert captured["decode_info"] == [([0, 1], 1, 3)]
        assert captured["prefill_info"] == []
        assert captured["block_size"] == 4
        assert runner._execute_model_state is not None
        assert runner._execute_model_state.target_hidden_states is not None
        assert runner._execute_model_state.cu_seqlens == [0, 3]

    def test_start_paged_forward_skips_hidden_states_without_drafts(
        self, monkeypatch
    ) -> None:
        runner = self._make_runner()
        runner.num_layers = 0
        runner._paged_block_size = 4
        runner._paged_request_seq_lens["r0"] = 1

        captured: dict[str, object] = {}

        def fake_prepare_unified(decode_info, prefill_info, block_size):
            captured["decode_info"] = decode_info
            captured["prefill_info"] = prefill_info
            captured["block_size"] = block_size

        def fake_target_forward(input_ids, *, cache, collect_hidden_states):
            del cache
            captured["input_ids"] = input_ids.tolist()
            captured["collect_hidden_states"] = collect_hidden_states
            return mr.TargetModelForwardOutput(logits=mx.zeros((1, 1, 16)))

        monkeypatch.setattr(mr, "prepare_unified", fake_prepare_unified)
        monkeypatch.setattr(runner, "_target_forward", fake_target_forward)

        req_state = self._make_state([1, 6])
        req_state.block_ids = [0, 1]
        scheduler_output = self._make_scheduler_output({"r0": 1}, {})

        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[],
            decode_reqs=[("r0", req_state)],
            scheduler_output=scheduler_output,
        )

        assert captured["input_ids"] == [[6]]
        assert captured["collect_hidden_states"] is False
        assert captured["decode_info"] == [([0, 1], 1, 1)]
        assert captured["prefill_info"] == []
        assert captured["block_size"] == 4
        assert runner._execute_model_state is not None
        assert runner._execute_model_state.target_hidden_states is None
        assert runner._execute_model_state.cu_seqlens == [0, 1]

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
        scheduler_output = self._make_scheduler_output({"p0": 1, "p1": 1}, {})
        prefill_reqs = [
            mr.PrefillRequest(
                req_id="p0",
                token_ids=[5],
                sampling_params=SamplingParams(),
                block_ids=[0],
                generator=None,
                prompt_len=1,
                start_pos=0,
                full_prompt_token_ids=None,
            ),
            mr.PrefillRequest(
                req_id="p1",
                token_ids=[6],
                sampling_params=SamplingParams(),
                block_ids=[1],
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

        def fake_prepare_unified(decode_info, prefill_info, block_size):
            captured["decode_info"] = decode_info
            captured["prefill_info"] = prefill_info
            captured["block_size"] = block_size

        def fake_target_forward(input_ids, *, cache, collect_hidden_states):
            del cache
            captured["input_ids"] = input_ids.tolist()
            captured["collect_hidden_states"] = collect_hidden_states
            return mr.TargetModelForwardOutput(
                logits=mx.zeros((1, 1, 16)),
                hidden_states=mx.ones((1, 4)),
            )

        monkeypatch.setattr(mr, "prepare_unified", fake_prepare_unified)
        monkeypatch.setattr(runner, "_target_forward", fake_target_forward)

        req_state = self._make_state([1, 6])
        req_state.block_ids = [0, 1]
        scheduler_output = self._make_scheduler_output({"r0": 1}, {})

        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[],
            decode_reqs=[("r0", req_state)],
            scheduler_output=scheduler_output,
        )

        assert captured["input_ids"] == [[6]]
        assert captured["collect_hidden_states"] is True
        assert captured["decode_info"] == [([0, 1], 1, 1)]
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

        def fake_prepare_unified(decode_info, prefill_info, block_size):
            captured["decode_info"] = decode_info
            captured["prefill_info"] = prefill_info
            captured["block_size"] = block_size

        def fake_target_forward(input_ids, *, cache, collect_hidden_states):
            del cache
            captured["input_ids"] = input_ids.tolist()
            captured["collect_hidden_states"] = collect_hidden_states
            return mr.TargetModelForwardOutput(
                logits=mx.zeros((1, 2, 16)),
                hidden_states=mx.ones((2, 4)),
            )

        monkeypatch.setattr(mr, "prepare_unified", fake_prepare_unified)
        monkeypatch.setattr(runner, "_target_forward", fake_target_forward)

        scheduler_output = self._make_scheduler_output({"r0": 2}, {})

        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[
                mr.PrefillRequest(
                    req_id="r0",
                    token_ids=[5, 6],
                    sampling_params=SamplingParams(),
                    block_ids=[0],
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
        assert captured["prefill_info"] == [([0], 2, 0)]
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

        def fake_prepare_unified(decode_info, prefill_info, block_size):
            captured["decode_info"] = decode_info
            captured["prefill_info"] = prefill_info
            captured["block_size"] = block_size

        def fake_target_forward(input_ids, *, cache, collect_hidden_states):
            del cache
            captured["input_ids"] = input_ids.tolist()
            captured["collect_hidden_states"] = collect_hidden_states
            return mr.TargetModelForwardOutput(logits=mx.zeros((1, 2, 16)))

        monkeypatch.setattr(mr, "prepare_unified", fake_prepare_unified)
        monkeypatch.setattr(runner, "_target_forward", fake_target_forward)

        scheduler_output = self._make_scheduler_output({"r0": 2}, {})

        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[
                mr.PrefillRequest(
                    req_id="r0",
                    token_ids=[5, 6],
                    sampling_params=SamplingParams(),
                    block_ids=[0],
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
        assert captured["prefill_info"] == [([0], 2, 0)]
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
            block_ids=(0,),
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
            block_ids=(0,),
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
            block_ids=(0,),
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
            block_ids=(0,),
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
            block_ids=[0],
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
            block_ids=(0,),
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
            block_ids=(0,),
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
                block_ids=(0,),
            ),
            mr.PagedDecodeSegment(
                req_id="plain",
                input_token_ids=(3,),
                start_row=2,
                num_query_tokens=1,
                draft_token_ids=(),
                cache_start_pos=1,
                block_ids=(1,),
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
                block_ids=(0,),
            ),
            mr.PagedDecodeSegment(
                req_id="plain",
                input_token_ids=(3,),
                start_row=2,
                num_query_tokens=1,
                draft_token_ids=(),
                cache_start_pos=1,
                block_ids=(1,),
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
                block_ids=(0,),
            ),
            mr.PagedDecodeSegment(
                req_id="draft",
                input_token_ids=(3, 7),
                start_row=1,
                num_query_tokens=2,
                draft_token_ids=(7,),
                cache_start_pos=1,
                block_ids=(1,),
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
                block_ids=(0,),
            ),
            mr.PagedDecodeSegment(
                req_id="structured",
                input_token_ids=(3,),
                start_row=2,
                num_query_tokens=1,
                draft_token_ids=(),
                cache_start_pos=1,
                block_ids=(1,),
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
            block_ids=(0,),
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
            block_ids=(0,),
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
        scheduled_new_reqs: list[SimpleNamespace] | None = None,
    ) -> SimpleNamespace:
        req_ids = cached_req_ids or []
        return SimpleNamespace(
            scheduled_new_reqs=scheduled_new_reqs or [],
            scheduled_cached_reqs=SimpleNamespace(
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

    def test_returns_empty_output_directly_for_empty_batch(self) -> None:
        runner = self._make_runner()

        out = runner.execute_model(self._make_scheduler_output())

        assert out is not None
        assert out.req_ids == []
        assert out.req_id_to_index == {}
        assert out.sampled_token_ids == []
        assert runner._pending_output is None

    def test_non_paged_cached_request_without_state_emits_placeholder(self) -> None:
        runner = self._make_runner()

        out = runner.execute_model(self._make_scheduler_output(["req-0"]))

        assert out is None
        pending = runner.sample_tokens(grammar_output=None)
        assert pending is not None
        assert pending.req_ids == ["req-0"]
        assert pending.req_id_to_index == {"req-0": 0}
        assert pending.sampled_token_ids == [[0]]
        assert runner._pending_output is None

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
            scheduled_new_reqs=[SimpleNamespace(req_id="new")],
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
            block_ids=[0],
        )
        runner._request_states["r0"] = req_state
        scheduler_output = self._make_scheduler_output(
            ["r0"],
            scheduled_spec_decode_tokens={"r0": [-1]},
            num_invalid_spec_tokens={"r0": 1},
            scheduled_new_reqs=[SimpleNamespace(req_id="new")],
        )
        scheduler_output.num_scheduled_tokens = {"r0": 2, "new": 1}
        scheduler_output.total_num_scheduled_tokens = 3
        scheduler_output.scheduled_cached_reqs.new_block_ids = [[[99]]]

        with pytest.raises(NotImplementedError, match="scheduler-invalid"):
            runner.execute_model(scheduler_output)

        assert req_state.block_ids == [0]
        assert "new" not in runner._request_states

    def test_gemma4_mtp_async_scheduling_fails_before_request_setup(self) -> None:
        runner = self._make_runner()
        runner.use_async_scheduling = True
        runner.vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="mtp",
                draft_model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(
                        model_type="gemma4_assistant",
                        architectures=["Gemma4AssistantForCausalLM"],
                    )
                ),
            )
        )
        scheduler_output = self._make_scheduler_output(
            scheduled_new_reqs=[SimpleNamespace(req_id="new")]
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
        assert submitted[0][1] is pending_conv
        assert submitted[0][2] is pending_recurrent
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
        runner = make_stub_runner(
            _paged_attention_runtime=runtime,
            _is_pooling=True,
            _paged_block_size=4,
            num_layers=0,
        )
        pooling_hidden_states = mx.array([[[1.0]]], dtype=mx.float32)
        monkeypatch.setattr(
            mr,
            "forward_sequence_hidden_states",
            lambda *args, **kwargs: pooling_hidden_states,
        )
        monkeypatch.setattr(mr.mx, "async_eval", lambda *args: submitted.append(args))

        runner._start_paged_forward(
            mr._ExecutionBatch(),
            prefill_reqs=[
                mr.PrefillRequest(
                    req_id="pool-0",
                    token_ids=[1],
                    sampling_params=SamplingParams(),
                    block_ids=[0],
                    generator=None,
                    prompt_len=1,
                    start_pos=0,
                    full_prompt_token_ids=[1],
                    pooling_params=PoolingParams(),
                )
            ],
            decode_reqs=[],
            scheduler_output=SimpleNamespace(scheduled_spec_decode_tokens={}),
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
                    block_ids=[0],
                    generator=None,
                    prompt_len=1,
                    start_pos=0,
                    full_prompt_token_ids=[1],
                )
            ],
            decode_reqs=[],
            scheduler_output=SimpleNamespace(scheduled_spec_decode_tokens={}),
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
    ) -> SimpleNamespace:
        return SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
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
        decode_state.block_ids = [0]
        prefill = mr.PrefillRequest(
            req_id="prefill-0",
            token_ids=[9],
            sampling_params=SamplingParams(),
            block_ids=[1],
            generator=None,
            prompt_len=1,
            start_pos=0,
            full_prompt_token_ids=[9],
        )
        scheduler_output = SimpleNamespace(scheduled_spec_decode_tokens={})

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
        runner._model_lifecycle = SimpleNamespace(load=lambda: events.append("load"))
        runner.apply_pipeline_split = lambda pp: events.append("split")
        runner._lora = SimpleNamespace(setup=lambda **kwargs: events.append("lora"))

        runner.load_model()

        assert events == ["load", "split", "lora"]
