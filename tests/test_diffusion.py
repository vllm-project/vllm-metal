# SPDX-License-Identifier: Apache-2.0
"""Tests for DiffusionGemma block diffusion on the Metal runner."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import LogprobsLists

import vllm_metal.compat as compat
import vllm_metal.v1.diffusion as diffusion
from vllm_metal.attention.context import get_context
from vllm_metal.platform import MetalPlatform
from vllm_metal.v1.diffusion import (
    DiffusionGemmaRuntime,
    DiffusionSettings,
    canvas_logprobs,
    denoise_update,
    entropy_bound_mask,
    join_canvas_logprobs,
    mask_to_allowed,
    schedule_temperature,
)

_GEN_CONFIG = {
    "confidence_threshold": 0.005,
    "max_denoising_steps": 48,
    "sampler_config": {
        "_cls_name": "EntropyBoundSamplerConfig",
        "entropy_bound": 0.1,
    },
    "stability_threshold": 1,
    "t_max": 0.8,
    "t_min": 0.4,
}
_VOCAB = 16
_CANVAS = 4
_HIDDEN = 8


def _vllm_config(
    *,
    canvas_length: int | None = _CANVAS,
    max_denoising_steps: int | None = None,
    gen_config: dict | None = None,
    model_type: str = "diffusion_gemma",
    long_prefill_token_threshold: int = 0,
) -> SimpleNamespace:
    gen = _GEN_CONFIG if gen_config is None else gen_config
    return SimpleNamespace(
        diffusion_config=SimpleNamespace(
            canvas_length=canvas_length, max_denoising_steps=max_denoising_steps
        ),
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(limit_per_prompt={}),
            try_get_generation_config=lambda: gen,
            get_vocab_size=lambda: _VOCAB,
            hf_config=SimpleNamespace(model_type=model_type),
            logprobs_mode="raw_logprobs",
        ),
        speculative_config=None,
        lora_config=None,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        scheduler_config=SimpleNamespace(
            async_scheduling=True,
            long_prefill_token_threshold=long_prefill_token_threshold,
        ),
    )


def _settings(**overrides) -> DiffusionSettings:
    base = DiffusionSettings.from_vllm_config(_vllm_config())
    return DiffusionSettings(**{**base.__dict__, **overrides})


class TestDiffusionSettings:
    def test_reads_canvas_and_generation_config(self) -> None:
        settings = DiffusionSettings.from_vllm_config(_vllm_config())

        assert settings == DiffusionSettings(
            canvas_length=_CANVAS,
            max_denoising_steps=48,
            t_min=0.4,
            t_max=0.8,
            entropy_bound=0.1,
            confidence_threshold=0.005,
            stability_threshold=1,
        )

    def test_diffusion_config_overrides_max_denoising_steps(self) -> None:
        settings = DiffusionSettings.from_vllm_config(
            _vllm_config(max_denoising_steps=7)
        )

        assert settings.max_denoising_steps == 7

    def test_requires_canvas_length(self) -> None:
        with pytest.raises(ValueError, match="canvas_length"):
            DiffusionSettings.from_vllm_config(_vllm_config(canvas_length=None))

    def test_rejects_non_entropy_bound_sampler(self) -> None:
        gen = {**_GEN_CONFIG, "sampler_config": {"_cls_name": "TopKSamplerConfig"}}

        with pytest.raises(ValueError, match="EntropyBound"):
            DiffusionSettings.from_vllm_config(_vllm_config(gen_config=gen))

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_non_positive_stability_threshold(self, value) -> None:
        # Zero wipes the canvas history every step, so the stability check
        # passes on an empty list and the loop reports convergence at once.
        gen = {**_GEN_CONFIG, "stability_threshold": value}

        with pytest.raises(ValueError, match="stability_threshold"):
            DiffusionSettings.from_vllm_config(_vllm_config(gen_config=gen))

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_non_positive_cli_max_denoising_steps(self, value) -> None:
        # An explicit 0 used to fall back to the default of 48 silently, and a
        # negative value ended every canvas after one step.
        with pytest.raises(ValueError, match="max_denoising_steps"):
            DiffusionSettings.from_vllm_config(_vllm_config(max_denoising_steps=value))

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_non_positive_generation_max_denoising_steps(self, value) -> None:
        gen = {**_GEN_CONFIG, "max_denoising_steps": value}

        with pytest.raises(ValueError, match="max_denoising_steps"):
            DiffusionSettings.from_vllm_config(_vllm_config(gen_config=gen))

    @pytest.mark.parametrize("missing", ["absent", "null"])
    def test_defaults_max_denoising_steps_when_unset(self, missing) -> None:
        gen = {k: v for k, v in _GEN_CONFIG.items() if k != "max_denoising_steps"}
        if missing == "null":
            gen["max_denoising_steps"] = None

        settings = DiffusionSettings.from_vllm_config(_vllm_config(gen_config=gen))

        assert settings.max_denoising_steps == 48


class TestSamplerMath:
    def test_temperature_schedule_runs_from_t_max_to_t_min(self) -> None:
        settings = _settings(max_denoising_steps=4, t_min=0.4, t_max=0.8)

        temps = [schedule_temperature(step, settings) for step in range(4)]

        assert temps == pytest.approx([0.8, 0.7, 0.6, 0.5])

    def test_entropy_bound_accepts_low_entropy_positions_within_budget(self) -> None:
        entropy = mx.array([0.5, 0.01, 0.04, 2.0, 0.03])

        mask = entropy_bound_mask(entropy, 0.1)

        # Ascending: 0.01, 0.03, 0.04, 0.5, 2.0. The entropy before each
        # position (cumsum - cummax) is 0, 0.01, 0.04, 0.08, 0.58, so only
        # the 2.0 position exceeds the 0.1 budget.
        assert mask.tolist() == [True, True, True, False, True]

    def test_entropy_bound_always_accepts_one_position(self) -> None:
        mask = entropy_bound_mask(mx.array([5.0, 4.0, 6.0]), 0.0)

        assert mask.tolist() == [False, True, False]

    def test_converges_when_argmax_is_stable_and_confident(self) -> None:
        settings = _settings()
        peaked = mx.where(
            mx.arange(_VOCAB)[None, :] == mx.array([3, 1, 4, 1])[:, None], 50.0, -50.0
        )
        history: list[mx.array] = []

        first = denoise_update(
            peaked,
            step=0,
            history=history,
            settings=settings,
            vocab_size=_VOCAB,
            max_steps=settings.max_denoising_steps,
        )
        second = denoise_update(
            peaked,
            step=1,
            history=history,
            settings=settings,
            vocab_size=_VOCAB,
            max_steps=settings.max_denoising_steps,
        )

        # No previous canvas to compare against on the first step.
        assert not first.converged
        assert second.converged
        assert second.argmax_canvas.tolist() == [3, 1, 4, 1]
        # Confident positions are all accepted, so nothing is renoised.
        assert first.next_canvas.tolist() == [3, 1, 4, 1]
        assert len(history) == settings.stability_threshold

    def test_converges_on_the_last_step_even_when_uncertain(self) -> None:
        settings = _settings(max_denoising_steps=3)
        flat = mx.zeros((_CANVAS, _VOCAB))

        early = denoise_update(
            flat,
            step=1,
            history=[],
            settings=settings,
            vocab_size=_VOCAB,
            max_steps=settings.max_denoising_steps,
        )
        last = denoise_update(
            flat,
            step=2,
            history=[],
            settings=settings,
            vocab_size=_VOCAB,
            max_steps=settings.max_denoising_steps,
        )

        assert not early.converged
        assert last.converged


class TestStructuredReadMath:
    def test_step_cap_converges_early_on_the_global_temperature_schedule(
        self,
    ) -> None:
        settings = _settings(max_denoising_steps=4, t_max=0.8)
        flat = mx.zeros((_CANVAS, _VOCAB))
        logits = mx.ones((_CANVAS, _VOCAB))

        outcome = denoise_update(
            logits,
            step=0,
            history=[],
            settings=settings,
            vocab_size=_VOCAB,
            max_steps=1,
        )

        assert outcome.converged
        # A one-step read runs at t_max, not at the end of a one-step schedule.
        np.testing.assert_allclose(np.array(outcome.processed_logits), 1 / 0.8)
        assert not denoise_update(
            flat, step=0, history=[], settings=settings, vocab_size=_VOCAB, max_steps=2
        ).converged

    def test_mask_renormalizes_over_the_allowed_ids(self) -> None:
        logits = mx.array([[5.0, 1.0, 2.0, 0.0]])
        allowed = mx.array([False, True, True, False])

        masked = mask_to_allowed(logits, allowed)
        outcome = denoise_update(
            masked,
            step=0,
            history=[],
            settings=_settings(),
            vocab_size=4,
            max_steps=1,
        )

        probs = np.array(mx.softmax(masked, axis=-1))
        np.testing.assert_allclose(probs[0], [0, 1 / (1 + np.e), np.e / (1 + np.e), 0])
        assert outcome.argmax_canvas.tolist() == [2]
        # The finite sentinel keeps the entropy finite, so the step stays defined.
        assert outcome.next_canvas.tolist()[0] in range(4)
        assert not np.isnan(np.array(outcome.processed_logits)).any()


class TestCanvasLogprobs:
    _LOGITS = mx.array([[0.0, 3.0, 1.0, 2.0], [4.0, 0.0, 0.0, 1.0]])

    def _log_softmax(self) -> np.ndarray:
        logits = np.array(self._LOGITS, dtype=np.float64)
        return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))

    def test_reports_the_selected_token_and_the_top_k_per_row(self) -> None:
        rows = canvas_logprobs(
            self._LOGITS,
            mx.array([1, 3]),
            num_logprobs=2,
            token_ids=None,
            logits_mode=False,
        )

        assert rows.logprob_token_ids.tolist() == [[1, 1, 3], [3, 0, 3]]
        expected = np.take_along_axis(
            self._log_softmax(), rows.logprob_token_ids, axis=-1
        )
        np.testing.assert_allclose(rows.logprobs, expected, rtol=1e-5)
        assert rows.sampled_token_ranks.tolist() == [1, 2]

    def test_token_ids_replace_the_top_k(self) -> None:
        rows = canvas_logprobs(
            self._LOGITS,
            mx.array([1, 0]),
            num_logprobs=2,
            token_ids=[2, 0],
            logits_mode=False,
        )

        assert rows.logprob_token_ids.tolist() == [[1, 2, 0], [0, 2, 0]]
        expected = np.take_along_axis(
            self._log_softmax(), rows.logprob_token_ids, axis=-1
        )
        np.testing.assert_allclose(rows.logprobs, expected, rtol=1e-5)

    def test_logits_mode_reports_the_logits(self) -> None:
        rows = canvas_logprobs(
            self._LOGITS,
            mx.array([1, 0]),
            num_logprobs=0,
            token_ids=None,
            logits_mode=True,
        )

        assert rows.logprob_token_ids.tolist() == [[1], [0]]
        assert rows.logprobs.tolist() == [[3.0], [4.0]]

    def test_join_offsets_each_request_and_pads_narrow_rows(self) -> None:
        wide = LogprobsLists(
            np.array([[5, 6, 7], [5, 7, 6]]),
            np.zeros((2, 3), dtype=np.float32),
            np.array([1, 1]),
        )
        narrow = LogprobsLists(
            np.array([[9]]), np.zeros((1, 1), dtype=np.float32), np.array([1])
        )

        joined = join_canvas_logprobs(["a", "idle", "b"], {"a": wide, "b": narrow})

        assert joined is not None
        assert joined.cu_num_generated_tokens == [0, 2, 2]
        assert joined.slice_request(1, 0).logprob_token_ids.shape[0] == 0
        b = joined.slice_request(2, 1)
        assert b.logprob_token_ids.tolist() == [[9, 0, 0]]
        assert b.logprobs.tolist() == [[0.0, float("-inf"), float("-inf")]]
        assert join_canvas_logprobs(["a"], {}) is None


class _StubRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def zero_blocks(self, block_ids) -> None:
        self.calls.append("zero")

    def copy_blocks(self, copies) -> None:
        self.calls.append("copy")

    def materialize_pending_state(self) -> None:
        self.calls.append("materialize")


def _stub_runner() -> SimpleNamespace:
    runner = SimpleNamespace(
        vllm_config=_vllm_config(),
        _paged_attention_runtime=_StubRuntime(),
        _request_states={},
        _draft_token_ids=None,
        _paged_group_block_sizes=(16,),
        tq_prefill_workspace_bytes=0,
        model=None,
    )
    runner.model_config = runner.vllm_config.model_config
    runner._finished_req_ids = lambda so: so.finished_req_ids

    def reconcile(evicted, **_kwargs):
        for req_id in evicted:
            runner._request_states.pop(req_id, None)

    runner._reconcile_request_lifecycle = reconcile
    runner._copy_paged_block_ids = lambda block_ids: [list(g) for g in block_ids]
    runner._update_cached_request_blocks = lambda cached: None
    return runner


def _scheduler_output(
    *,
    new_reqs=(),
    cached: dict[str, int] | None = None,
    scheduled: dict[str, int],
    drafts: dict[str, list[int]] | None = None,
    finished: set[str] | None = None,
) -> SimpleNamespace:
    cached = cached or {}
    return SimpleNamespace(
        scheduled_encoder_inputs={},
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=list(cached),
            num_computed_tokens=list(cached.values()),
            resumed_req_ids=set(),
        ),
        finished_req_ids=finished or set(),
        preempted_req_ids=set(),
        num_scheduled_tokens=scheduled,
        scheduled_spec_decode_tokens=drafts or {},
        new_block_ids_to_zero=None,
        kv_cache_block_copies=None,
    )


def _new_req(
    req_id: str, params: SamplingParams | None = None, prompt=(7, 8)
) -> SimpleNamespace:
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        sampling_params=params,
        block_ids=([0],),
        num_computed_tokens=0,
    )


def _step(rt: DiffusionGemmaRuntime, *req_ids: str, new_reqs=(), finished=()):
    """One engine step: new requests prefill whole, running ones run their drafts."""
    runner = rt._runner
    drafts = runner._draft_token_ids
    pending = (
        {}
        if drafts is None
        else dict(zip(drafts.req_ids, drafts.draft_token_ids, strict=True))
    )
    scheduled = {r.req_id: len(r.prompt_token_ids) for r in new_reqs}
    scheduled.update({r: len(pending[r]) for r in req_ids})
    return rt.execute_model(
        _scheduler_output(
            new_reqs=new_reqs,
            cached={r: len(runner._request_states[r].token_ids) for r in req_ids},
            scheduled=scheduled,
            drafts={r: pending[r] for r in req_ids},
            finished=set(finished),
        )
    )


def _read(**extra_args) -> SamplingParams:
    """A structured read, as structured_server.py sends it."""
    return SamplingParams(
        logprob_token_ids=[3, 4],
        extra_args={"diffusion_read_only": True, **extra_args},
    )


class TestRuntimeStepProtocol:
    """Prefill -> denoise -> commit through the scheduler's draft-token path."""

    @pytest.fixture
    def runtime(self, monkeypatch) -> tuple[DiffusionGemmaRuntime, list]:
        forwards: list[tuple[bool, list[tuple[int, list[int]]]]] = []
        target = mx.array([3, 1, 4, 1])

        def fake_forward(self, segments, *, decoder):
            forwards.append(
                (decoder, [(s.start_pos, list(s.token_ids)) for s in segments])
            )
            total = sum(len(s.token_ids) for s in segments)
            return mx.zeros((1, total, _HIDDEN))

        def peaked_logits(model, hidden):
            return mx.where(
                mx.arange(_VOCAB)[None, :] == target[: hidden.shape[0], None],
                50.0,
                -50.0,
            )

        monkeypatch.setattr(DiffusionGemmaRuntime, "_forward", fake_forward)
        monkeypatch.setattr(diffusion, "canvas_logits", peaked_logits)
        monkeypatch.setattr(
            diffusion,
            "self_conditioning_embeddings",
            lambda model, logits: mx.zeros((logits.shape[0], _HIDDEN)),
        )
        return DiffusionGemmaRuntime(_stub_runner()), forwards

    def test_full_block_cycle(self, runtime) -> None:
        rt, forwards = runtime
        runner = rt._runner
        prompt = [7, 8, 9]
        new_req = SimpleNamespace(
            req_id="r",
            prompt_token_ids=prompt,
            sampling_params=None,
            block_ids=([0],),
            num_computed_tokens=0,
        )

        # Prefill: encoder pass, nothing sampled, a random canvas as drafts.
        out = rt.execute_model(
            _scheduler_output(new_reqs=[new_req], scheduled={"r": 3})
        )
        assert out.sampled_token_ids == [[]]
        canvas = runner._draft_token_ids.draft_token_ids[0]
        assert len(canvas) == _CANVAS
        assert forwards[-1] == (False, [(0, prompt)])

        # Denoise twice: decoder passes over the canvas, nothing sampled.
        for _ in range(2):
            out = rt.execute_model(
                _scheduler_output(
                    cached={"r": 3}, scheduled={"r": _CANVAS}, drafts={"r": canvas}
                )
            )
            assert out.sampled_token_ids == [[]]
            assert forwards[-1] == (True, [(3, canvas)])
            canvas = runner._draft_token_ids.draft_token_ids[0]
        # Converged: the drafts are now the argmax canvas.
        assert canvas == [3, 1, 4, 1]

        # Commit: encoder pass over the argmax canvas, emitted as accepted.
        out = rt.execute_model(
            _scheduler_output(
                cached={"r": 3}, scheduled={"r": _CANVAS}, drafts={"r": canvas}
            )
        )
        assert out.sampled_token_ids == [[3, 1, 4, 1]]
        assert forwards[-1] == (False, [(3, [3, 1, 4, 1])])
        assert runner._request_states["r"].token_ids == prompt + [3, 1, 4, 1]
        # A fresh canvas for the next block.
        assert len(runner._draft_token_ids.draft_token_ids[0]) == _CANVAS

    def test_intermediate_prefill_chunk_hands_back_no_drafts(self, runtime) -> None:
        rt, _ = runtime
        new_req = SimpleNamespace(
            req_id="r",
            prompt_token_ids=[1, 2, 3, 4],
            sampling_params=None,
            block_ids=([0],),
            num_computed_tokens=0,
        )

        rt.execute_model(_scheduler_output(new_reqs=[new_req], scheduled={"r": 2}))

        assert rt._runner._draft_token_ids.draft_token_ids == [[]]

    def test_rejects_drafts_that_differ_from_the_canvas(self, runtime) -> None:
        rt, _ = runtime
        new_req = SimpleNamespace(
            req_id="r",
            prompt_token_ids=[1, 2],
            sampling_params=None,
            block_ids=([0],),
            num_computed_tokens=0,
        )
        rt.execute_model(_scheduler_output(new_reqs=[new_req], scheduled={"r": 2}))

        with pytest.raises(RuntimeError, match="out of sync"):
            rt.execute_model(
                _scheduler_output(
                    cached={"r": 2}, scheduled={"r": _CANVAS}, drafts={"r": [-1] * 4}
                )
            )

    def test_clipped_denoise_step_keeps_the_full_canvas(self, runtime) -> None:
        rt, forwards = runtime
        runner = rt._runner
        new_req = SimpleNamespace(
            req_id="r",
            prompt_token_ids=[1, 2],
            sampling_params=None,
            block_ids=([0],),
            num_computed_tokens=0,
        )
        rt.execute_model(_scheduler_output(new_reqs=[new_req], scheduled={"r": 2}))
        canvas = runner._draft_token_ids.draft_token_ids[0]

        # The token budget clipped the canvas to two rows for this step.
        rt.execute_model(
            _scheduler_output(
                cached={"r": 2}, scheduled={"r": 2}, drafts={"r": canvas[:2]}
            )
        )

        assert forwards[-1] == (True, [(2, canvas[:2])])
        # The unscheduled rows are resampled, so the next step denoises all of them.
        assert len(runner._draft_token_ids.draft_token_ids[0]) == _CANVAS

    def test_clipped_converging_step_commits_only_the_scheduled_rows(
        self, runtime
    ) -> None:
        rt, _ = runtime
        runner = rt._runner
        rt.settings = _settings(max_denoising_steps=1)
        new_req = SimpleNamespace(
            req_id="r",
            prompt_token_ids=[1, 2],
            sampling_params=None,
            block_ids=([0],),
            num_computed_tokens=0,
        )
        rt.execute_model(_scheduler_output(new_reqs=[new_req], scheduled={"r": 2}))
        canvas = runner._draft_token_ids.draft_token_ids[0]

        # The last denoise step converges with only two rows scheduled.
        rt.execute_model(
            _scheduler_output(
                cached={"r": 2}, scheduled={"r": 2}, drafts={"r": canvas[:2]}
            )
        )
        canvas = runner._draft_token_ids.draft_token_ids[0]
        out = rt.execute_model(
            _scheduler_output(cached={"r": 2}, scheduled={"r": 2}, drafts={"r": canvas})
        )

        # The rows the model never saw this step are not committed.
        assert canvas == [3, 1]
        assert out.sampled_token_ids == [[3, 1]]

    def test_logprobs_arrive_with_the_committed_canvas(self, runtime) -> None:
        rt, _ = runtime
        out = _step(rt, new_reqs=[_new_req("r", SamplingParams(logprobs=2))])
        assert out.logprobs is None

        # Two denoise steps converge; nothing is emitted, so no logprobs yet.
        for _ in range(2):
            assert _step(rt, "r").logprobs is None

        out = _step(rt, "r")

        assert out.sampled_token_ids == [[3, 1, 4, 1]]
        rows = out.logprobs.slice_request(0, _CANVAS)
        assert rows.logprob_token_ids[:, 0].tolist() == [3, 1, 4, 1]
        assert rows.logprob_token_ids.shape == (_CANVAS, 3)
        assert rows.sampled_token_ranks.tolist() == [1] * _CANVAS
        np.testing.assert_allclose(rows.logprobs[:, 0], 0.0, atol=1e-5)

    def test_logprobs_follow_their_own_request_in_lockstep(self, runtime) -> None:
        # "a" commits on the step "b" converges. b's logprobs must wait for
        # b's own commit, or the frontend gets fewer rows than tokens.
        rt, _ = runtime
        params = SamplingParams(logprob_token_ids=[4, 5])

        _step(rt, new_reqs=[_new_req("a", params)])
        _step(rt, "a", new_reqs=[_new_req("b", params)])
        _step(rt, "a", "b")  # a converges
        out = _step(rt, "a", "b")  # a commits, b converges

        assert out.sampled_token_ids == [[3, 1, 4, 1], []]
        assert out.logprobs.cu_num_generated_tokens == [0, _CANVAS]
        assert out.logprobs.logprob_token_ids.shape == (_CANVAS, 3)
        a = out.logprobs.slice_request(0, _CANVAS)
        assert a.logprob_token_ids.tolist() == [[t, 4, 5] for t in [3, 1, 4, 1]]

        out = _step(rt, "a", "b")  # b commits

        assert out.sampled_token_ids == [[], [3, 1, 4, 1]]
        assert out.logprobs.cu_num_generated_tokens == [0, 0]
        b = out.logprobs.slice_request(1, _CANVAS)
        assert b.logprob_token_ids.tolist() == [[t, 4, 5] for t in [3, 1, 4, 1]]

    def test_seed_replaces_only_the_first_canvas_after_the_prompt(
        self, runtime, monkeypatch
    ) -> None:
        rt, _ = runtime
        runner = rt._runner
        monkeypatch.setattr(
            diffusion,
            "random_canvas",
            lambda length, vocab_size: mx.full((length,), 15, dtype=mx.int32),
        )
        seeded = SamplingParams(extra_args={"diffusion_seed_canvas": [5, 6, 7, 8]})

        _step(rt, new_reqs=[_new_req("s", seeded), _new_req("u")])

        assert runner._draft_token_ids.draft_token_ids == [[5, 6, 7, 8], [15] * 4]
        for _ in range(3):  # two denoise steps, then the commit
            out = _step(rt, "s", "u")
        assert out.sampled_token_ids == [[3, 1, 4, 1], [3, 1, 4, 1]]
        assert runner._draft_token_ids.draft_token_ids == [[15] * 4, [15] * 4]

    def test_pinned_rows_hold_the_seed_and_drop_self_conditioning(
        self, runtime, monkeypatch
    ) -> None:
        rt, _ = runtime
        monkeypatch.setattr(
            diffusion,
            "self_conditioning_embeddings",
            lambda model, logits: mx.ones((logits.shape[0], _HIDDEN)),
        )
        params = SamplingParams(
            extra_args={
                "diffusion_seed_canvas": [5, 6, 7, 8],
                "diffusion_pinned": [0, 2],
            }
        )
        _step(rt, new_reqs=[_new_req("r", params)])

        _step(rt, "r")

        # The model predicts [3, 1, 4, 1]; the pinned rows keep the seed.
        assert rt._runner._draft_token_ids.draft_token_ids == [[5, 1, 7, 1]]
        soft = rt._soft_embeddings(rt._requests["r"], _CANVAS)
        assert soft.sum(axis=-1).tolist() == [0, _HIDDEN, 0, _HIDDEN]

    def test_step_cap_is_per_request(self, runtime) -> None:
        rt, _ = runtime
        capped = SamplingParams(extra_args={"diffusion_max_steps": 1})
        _step(rt, new_reqs=[_new_req("capped", capped), _new_req("free")])

        _step(rt, "capped", "free")  # only the capped canvas converges
        out = _step(rt, "capped", "free")

        assert out.sampled_token_ids == [[3, 1, 4, 1], []]
        # A cap past the model's budget is the budget.
        huge = SamplingParams(extra_args={"diffusion_max_steps": 1000})
        assert rt._new_request(huge).max_steps == rt.settings.max_denoising_steps

    def test_read_emits_on_convergence_while_generation_commits(self, runtime) -> None:
        rt, forwards = runtime
        gen = _new_req("gen", SamplingParams(logprobs=1))
        read = _new_req(
            "read",
            _read(diffusion_seed_canvas=[9] * _CANVAS, diffusion_max_steps=2),
            prompt=(1, 2, 3),
        )
        _step(rt, new_reqs=[gen])
        _step(rt, "gen", new_reqs=[read])
        _step(rt, "gen", "read")  # gen converges, the read takes its first step

        out = _step(rt, "gen", "read")  # gen commits, the read converges

        assert out.sampled_token_ids == [[3, 1, 4, 1], [3, 1, 4, 1]]
        # The commit ran the encoder over gen's canvas; the read only denoised.
        assert forwards[-2] == (False, [(2, [3, 1, 4, 1])])
        assert forwards[-1] == (True, [(3, [3, 1, 4, 1])])
        assert out.logprobs.cu_num_generated_tokens == [0, _CANVAS]
        gen_rows = out.logprobs.slice_request(0, _CANVAS)
        read_rows = out.logprobs.slice_request(1, _CANVAS)
        assert gen_rows.logprob_token_ids[:, :2].tolist() == [
            [t, t] for t in [3, 1, 4, 1]
        ]
        assert read_rows.logprob_token_ids.tolist() == [[t, 3, 4] for t in [3, 1, 4, 1]]
        # Temperature 1: logits of +-50 put an off-target id at -100, where
        # the step's 0.8 schedule would put it at -125.
        np.testing.assert_allclose(read_rows.logprobs[0, 2], -100.0, rtol=1e-4)
        state = rt._runner._request_states["read"]
        assert state.token_ids == [1, 2, 3, 3, 1, 4, 1]

        # Reaching max_tokens ends the read; it never runs a commit forward.
        _step(rt, "gen", finished={"read"})
        assert "read" not in rt._requests
        assert all(start != 3 for _, segs in forwards[-1:] for start, _ in segs)

    def test_clipped_read_emits_its_rows_and_converges_again(self, runtime) -> None:
        rt, _ = runtime
        runner = rt._runner
        _step(rt, new_reqs=[_new_req("r", _read(diffusion_max_steps=1))])
        canvas = runner._draft_token_ids.draft_token_ids[0]

        # The token budget clipped the read's only step to two rows.
        out = rt.execute_model(
            _scheduler_output(
                cached={"r": 2}, scheduled={"r": 2}, drafts={"r": canvas[:2]}
            )
        )
        assert out.sampled_token_ids == [[3, 1]]
        assert out.logprobs.logprob_token_ids.shape == (2, 3)

        # Past its cap, the read converges at once on what is left.
        out = _step(rt, "r")
        assert out.sampled_token_ids == [[3, 1, 4, 1]]

    def test_constrained_read_stays_inside_its_labels(self, runtime) -> None:
        rt, _ = runtime
        constrained = _read(diffusion_max_steps=1, diffusion_constrained=True)
        free = _read(diffusion_max_steps=1)
        _step(rt, new_reqs=[_new_req("c", constrained), _new_req("f", free)])

        out = _step(rt, "c", "f")

        # Only ids 3 and 4 are allowed: the +-50 logits tie them at -50
        # wherever the target is neither, and argmax takes the first.
        assert out.sampled_token_ids == [[3, 3, 4, 3], [3, 1, 4, 1]]
        c = out.logprobs.slice_request(0, _CANVAS)
        np.testing.assert_allclose(
            np.exp(c.logprobs[:, 1:]).sum(axis=1), 1.0, rtol=1e-5
        )
        f = out.logprobs.slice_request(1, _CANVAS)
        np.testing.assert_allclose(f.logprobs[1, 1:], -100.0, rtol=1e-4)

    def test_finished_requests_drop_their_canvas(self, runtime) -> None:
        rt, _ = runtime
        new_req = SimpleNamespace(
            req_id="r",
            prompt_token_ids=[1, 2],
            sampling_params=None,
            block_ids=([0],),
            num_computed_tokens=0,
        )
        rt.execute_model(_scheduler_output(new_reqs=[new_req], scheduled={"r": 2}))

        rt.execute_model(_scheduler_output(scheduled={}, finished={"r"}))

        assert rt._requests == {}
        assert "r" not in rt._runner._request_states


class TestEncoderHiddenStates:
    def test_embeds_through_the_public_input_embeddings_api(self) -> None:
        seen = {}

        def get_input_embeddings(input_ids):
            seen["input_ids"] = input_ids.tolist()
            return SimpleNamespace(inputs_embeds=mx.ones((1, 2, _HIDDEN)))

        def layer(h, mask, cache, *, layer_scalar):
            return h * layer_scalar

        decoder = SimpleNamespace(layers=[layer, layer], norm=lambda h: h + 1)
        encoder_layers = [SimpleNamespace(layer_scalar=s) for s in (2.0, 3.0)]
        model = SimpleNamespace(
            get_input_embeddings=get_input_embeddings,
            model=SimpleNamespace(
                decoder=decoder,
                encoder=SimpleNamespace(
                    language_model=SimpleNamespace(layers=encoder_layers)
                ),
            ),
        )

        hidden = diffusion.encoder_hidden_states(model, mx.array([[5, 6]]))

        assert seen["input_ids"] == [[5, 6]]
        # Decoder layers scaled by the encoder's layer_scalars, then the norm.
        assert mx.array_equal(hidden, mx.full((1, 2, _HIDDEN), 7.0))


class TestDecoderForwardContext:
    def test_decoder_marks_each_canvas_bidirectional_on_every_layer(
        self, monkeypatch
    ) -> None:
        seen = {}

        def capture(model, input_ids, soft_embeddings):
            ctx = get_context()
            seen["ranges"] = ctx.segment_bidi_ranges
            seen["kinds"] = ctx.bidi_layer_kinds
            seen["anchored"] = ctx.bidi_window_at_block_start
            seen["soft"] = soft_embeddings.shape
            return mx.zeros((1, input_ids.shape[1], _HIDDEN))

        monkeypatch.setattr(diffusion, "decoder_hidden_states", capture)
        runner = _stub_runner()
        decoder = SimpleNamespace(
            config=SimpleNamespace(hidden_size=_HIDDEN),
            embed_tokens=nn.Embedding(_VOCAB, _HIDDEN),
        )
        runner.model = SimpleNamespace(model=SimpleNamespace(decoder=decoder))
        rt = DiffusionGemmaRuntime(runner)
        for req_id in ("a", "b"):
            rt._requests[req_id] = rt._new_request(SamplingParams())
            rt._requests[req_id].phase = "denoise"
        segments = [
            diffusion._Segment("a", [1, 2, 3, 4], 10, [[0]]),
            diffusion._Segment("b", [5, 6, 7, 8], 20, [[1, 2]]),
        ]

        rt._forward(segments, decoder=True)

        assert seen["ranges"] == [[(10, 14)], [(20, 24)]]
        assert seen["kinds"] == frozenset({"sliding", "full"})
        assert seen["anchored"] is True
        assert seen["soft"] == (1, 8, _HIDDEN)
        assert get_context() is None


class TestV1RunnerGuardPatch:
    def test_drops_the_diffusion_entry_for_supported_models_only(
        self, monkeypatch
    ) -> None:
        from vllm.config import VllmConfig

        monkeypatch.setattr(
            VllmConfig,
            "_get_v1_model_runner_unsupported_features",
            lambda self: ["diffusion models", "other"],
        )
        compat.ensure_vllm_v1_diffusion_guard_patch()
        unsupported = VllmConfig._get_v1_model_runner_unsupported_features

        def config(model_type):
            hf_config = SimpleNamespace(model_type=model_type)
            return SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config))

        assert unsupported(config("diffusion_gemma")) == ["other"]
        assert unsupported(config("llada")) == ["diffusion models", "other"]

    def test_shares_one_wrapper_with_the_dspark_bridge(self, monkeypatch) -> None:
        from vllm.config import VllmConfig

        from vllm_metal.patches.dspark_config import enable_dspark_for_metal_runner

        def upstream(self):
            return ["diffusion models", "dspark speculative decoding", "other"]

        monkeypatch.setattr(
            VllmConfig, "_get_v1_model_runner_unsupported_features", upstream
        )
        compat.ensure_vllm_v1_diffusion_guard_patch()
        enable_dspark_for_metal_runner()
        unsupported = VllmConfig._get_v1_model_runner_unsupported_features

        assert unsupported.__wrapped__ is upstream
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="diffusion_gemma")
            ),
            parallel_config=SimpleNamespace(
                worker_cls="vllm_metal.v1.worker.MetalWorker"
            ),
            speculative_config=SimpleNamespace(method="dspark"),
        )
        assert unsupported(config) == ["other"]


class TestPlatformDiffusionConfig:
    @pytest.fixture(autouse=True)
    def guard_patch_calls(self, monkeypatch) -> list[bool]:
        calls: list[bool] = []
        monkeypatch.setattr(
            compat, "ensure_vllm_v1_diffusion_guard_patch", lambda: calls.append(True)
        )
        return calls

    def test_disables_async_scheduling_and_lifts_the_v1_guard(
        self, guard_patch_calls
    ) -> None:
        vllm_config = _vllm_config()

        MetalPlatform._check_diffusion_config(vllm_config)

        assert vllm_config.scheduler_config.async_scheduling is False
        assert guard_patch_calls == [True]

    def test_disables_multimodal_inputs(self) -> None:
        vllm_config = _vllm_config()

        MetalPlatform._check_diffusion_config(vllm_config)

        # Image/video requests fail validation instead of reaching the runner.
        limits = vllm_config.model_config.multimodal_config.limit_per_prompt
        assert {m: o.count for m, o in limits.items()} == {
            "image": 0,
            "video": 0,
            "audio": 0,
        }

    def test_requires_canvas_length(self) -> None:
        with pytest.raises(ValueError, match="canvas_length"):
            MetalPlatform._check_diffusion_config(_vllm_config(canvas_length=None))

    @pytest.mark.parametrize(
        "params",
        [
            {"logprobs": 2},
            {"logprob_token_ids": [1, 2]},
            {
                "logprob_token_ids": [1, 2],
                "extra_args": {
                    "diffusion_canvas_length": _CANVAS,
                    "diffusion_seed_canvas": [1] * _CANVAS,
                    "diffusion_pinned": [0],
                    "diffusion_max_steps": 1,
                    "diffusion_read_only": True,
                    "diffusion_constrained": True,
                },
            },
        ],
    )
    def test_accepts_logprobs_and_reads_while_serving_diffusion(
        self, monkeypatch, params
    ) -> None:
        monkeypatch.setattr(MetalPlatform, "_serves_diffusion", True)

        MetalPlatform.validate_request(None, SamplingParams(**params))

    @pytest.mark.parametrize(
        ("params", "parameter"),
        [
            ({"logprobs": -1}, "logprobs"),
            ({"prompt_logprobs": 1}, "prompt_logprobs"),
            ({"prompt_logprob_token_ids": [1]}, "prompt_logprob_token_ids"),
        ],
    )
    def test_rejects_prompt_and_full_vocab_logprobs_while_serving_diffusion(
        self, monkeypatch, params, parameter
    ) -> None:
        from vllm.exceptions import VLLMValidationError

        monkeypatch.setattr(MetalPlatform, "_serves_diffusion", True)

        with pytest.raises(VLLMValidationError, match="not supported") as exc_info:
            MetalPlatform.validate_request(None, SamplingParams(**params))
        assert exc_info.value.parameter == parameter

    @pytest.mark.parametrize(
        ("params", "parameter"),
        [({"top_k": 5}, "top_k"), ({"top_p": 0.9}, "top_p")],
    )
    def test_rejects_top_k_and_top_p_while_serving_diffusion(
        self, monkeypatch, params, parameter
    ) -> None:
        from vllm.exceptions import VLLMValidationError

        monkeypatch.setattr(MetalPlatform, "_serves_diffusion", True)

        # Disabled values (the defaults, or top_k=-1) are accepted.
        MetalPlatform.validate_request(None, SamplingParams(top_k=-1, top_p=1.0))
        with pytest.raises(VLLMValidationError, match=parameter) as exc_info:
            MetalPlatform.validate_request(None, SamplingParams(**params))
        assert exc_info.value.parameter == parameter

    def test_rejects_turboquant(self, monkeypatch) -> None:
        import vllm_metal.platform as platform_mod

        monkeypatch.setattr(
            platform_mod, "get_config", lambda: SimpleNamespace(turboquant=True)
        )
        with pytest.raises(NotImplementedError, match="TurboQuant"):
            MetalPlatform._check_diffusion_config(_vllm_config())

    def test_rejects_long_prefill_threshold_below_the_canvas(self) -> None:
        MetalPlatform._check_diffusion_config(
            _vllm_config(long_prefill_token_threshold=_CANVAS)
        )
        with pytest.raises(NotImplementedError, match="long-prefill-token-threshold"):
            MetalPlatform._check_diffusion_config(
                _vllm_config(long_prefill_token_threshold=_CANVAS - 1)
            )

    def test_rejects_unsupported_diffusion_model_types(self) -> None:
        with pytest.raises(NotImplementedError, match="llada"):
            MetalPlatform._check_diffusion_config(_vllm_config(model_type="llada"))
