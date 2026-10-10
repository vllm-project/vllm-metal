# SPDX-License-Identifier: Apache-2.0
"""DiffusionGemma block diffusion on the paged Metal path.

vLLM schedules discrete-diffusion LMs through the speculative-decoding data
path (``DiffusionConfig``): ``num_speculative_tokens == canvas_length`` and
``num_sampled_tokens_per_step == 0``, so the canvas travels as draft tokens
and every step either rejects all of them (rolling ``num_computed_tokens``
back) or accepts all of them. Each engine step runs one phase per request:

* ``prefill`` — prompt chunks in encoder mode. Emits nothing; after the final
  chunk a random canvas goes back to the scheduler as the draft tokens.
* ``denoise`` — the canvas is scheduled as draft tokens and runs in decoder
  mode. Emits nothing, so the scheduler rejects every draft and the canvas
  slots are reused next step; the sampler's updated canvas is the next draft.
* ``commit`` — once converged, the argmax canvas is scheduled, run in encoder
  mode to write its KV, and emitted as accepted tokens. Requested logprobs
  come from the converging denoise step's logits and travel with this
  emission only, so a request never receives them on another's step.

Structured reads (vllm#57250) set ``extra_args``: a seed canvas replaces the
first random one, pinned rows keep their seed through every denoise step and
feed no self-conditioning, a step cap ends the canvas early, and a read-only
request emits its argmax canvas on the converging step, with temperature-1
logprobs, instead of committing it. A constrained request's logits are
masked to its ``logprob_token_ids``.

Encoder and decoder mode share every weight (mlx_vlm ``diffusion_gemma``):
the encoder uses plain token embeddings, causal attention and its own
per-layer ``layer_scalar``; the decoder feeds the embeddings through the
self-conditioning MLP, attends bidirectionally over the canvas plus the
committed KV, and uses the decoder ``layer_scalar``. On the paged path the
decoder's bidirectional canvas is one ``segment_bidi_ranges`` block on every
layer kind, with the sliding window anchored at the canvas start
(``bidi_window_at_block_start``): every canvas row sees the last
``sliding_window - 1`` committed tokens and the whole canvas, as in mlx_vlm's
decoder masks. Its K/V lands in the speculative slots the scheduler rolls back.

The sampler mirrors upstream ``DiffusionSampler`` / mlx_vlm's entropy-bound
loop: linear temperature schedule ``t_max -> t_min``, categorical sample,
entropy-bound acceptance with random renoise of the rest, and convergence
once the argmax canvas is stable and confident (or the step budget is spent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import DraftTokenIds, LogprobsLists, ModelRunnerOutput

from vllm_metal.attention.context import clear_context, get_context, prepare_grouped
from vllm_metal.v1.prompt_logprobs import _LOGITS_LOGPROBS_MODES

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_metal.v1.model_runner import MetalModelRunner

logger = init_logger(__name__)

# mlx_vlm model types whose structure ``DiffusionGemmaRuntime`` drives.
SUPPORTED_DIFFUSION_MODEL_TYPES = frozenset({"diffusion_gemma"})

# The decoder's canvas block applies on both attention kinds.
_ALL_LAYER_KINDS = frozenset({"sliding", "full"})


@dataclass(frozen=True)
class DiffusionSettings:
    """Canvas size from ``--diffusion-config``; the rest from generation_config."""

    canvas_length: int
    max_denoising_steps: int
    t_min: float
    t_max: float
    entropy_bound: float
    confidence_threshold: float
    # Previous argmax canvases the current one must match (HF semantics).
    stability_threshold: int

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> DiffusionSettings:
        diffusion_config = vllm_config.diffusion_config
        if diffusion_config is None or diffusion_config.canvas_length is None:
            raise ValueError(
                "Diffusion models on Metal require --diffusion-config with a "
                "canvas_length, e.g. --diffusion-config '{\"canvas_length\": 32}'."
            )
        gen = vllm_config.model_config.try_get_generation_config() or {}
        sampler_config = gen.get("sampler_config") or {}
        if "EntropyBound" not in sampler_config.get("_cls_name", ""):
            raise ValueError(
                "DiffusionGemma requires an EntropyBound sampler_config in "
                f"generation_config.json (got {sampler_config!r})."
            )
        entropy_bound = float(sampler_config.get("entropy_bound") or 0.0)
        if entropy_bound <= 0:
            raise ValueError(
                f"entropy_bound must be a positive float (got {entropy_bound})."
            )
        stability_threshold = int(gen.get("stability_threshold", 1))
        if stability_threshold <= 0:
            raise ValueError(
                "stability_threshold must be a positive integer "
                f"(got {stability_threshold})."
            )
        max_denoising_steps = diffusion_config.max_denoising_steps
        if max_denoising_steps is None:
            max_denoising_steps = gen.get("max_denoising_steps")
        if max_denoising_steps is None:
            max_denoising_steps = 48
        max_denoising_steps = int(max_denoising_steps)
        if max_denoising_steps <= 0:
            raise ValueError(
                "max_denoising_steps must be a positive integer "
                f"(got {max_denoising_steps})."
            )
        return cls(
            canvas_length=int(diffusion_config.canvas_length),
            max_denoising_steps=max_denoising_steps,
            t_min=float(gen.get("t_min", 0.4)),
            t_max=float(gen.get("t_max", 0.8)),
            entropy_bound=entropy_bound,
            confidence_threshold=float(gen.get("confidence_threshold", 0.005)),
            stability_threshold=stability_threshold,
        )


# ---------------------------------------------------------------------------
# Sampler math (pure functions over one request's canvas)
# ---------------------------------------------------------------------------


def schedule_temperature(step: int, settings: DiffusionSettings) -> float:
    """Linear ``t_max -> t_min`` schedule; ``step`` counts denoise steps done."""
    remaining = max(settings.max_denoising_steps - step, 1)
    return settings.t_min + (settings.t_max - settings.t_min) * (
        remaining / settings.max_denoising_steps
    )


def token_entropy(log_probs: mx.array) -> mx.array:
    """Per-position entropy of ``(..., vocab)`` log-probabilities."""
    return -mx.sum(mx.exp(log_probs) * log_probs, axis=-1)


def entropy_bound_mask(entropy: mx.array, entropy_bound: float) -> mx.array:
    """Accept the lowest-entropy positions while their entropy budget holds.

    Positions are taken in ascending entropy; position ``i`` is accepted when
    the summed entropy of the ones before it stays within ``entropy_bound``
    (``cumsum - cummax`` of the sorted entropies), so at least one position is
    always accepted.
    """
    order = mx.argsort(entropy, axis=-1)
    sorted_entropy = mx.take_along_axis(entropy, order, axis=-1)
    sorted_mask = (
        mx.cumsum(sorted_entropy, axis=-1) - mx.cummax(sorted_entropy, axis=-1)
    ) <= entropy_bound
    return mx.put_along_axis(mx.zeros_like(sorted_mask), order, sorted_mask, axis=-1)


def random_canvas(length: int, vocab_size: int) -> mx.array:
    return mx.random.randint(0, vocab_size, (length,), dtype=mx.int32)


@dataclass
class DenoiseOutcome:
    next_canvas: mx.array  # accepted samples, renoised elsewhere
    argmax_canvas: mx.array
    converged: bool
    processed_logits: mx.array  # temperature-scaled, for self-conditioning


def denoise_update(
    logits: mx.array,
    *,
    step: int,
    history: list[mx.array],
    settings: DiffusionSettings,
    vocab_size: int,
    max_steps: int,
) -> DenoiseOutcome:
    """One accept/renoise step over a ``(canvas, vocab)`` logits block.

    ``step`` is the number of denoise steps already run on this canvas;
    ``history`` holds the previous argmax canvases and is updated in place.
    The canvas converges at the latest on step ``max_steps``, while the
    temperature schedule keeps following ``settings.max_denoising_steps``.
    """
    processed = logits.astype(mx.float32) / schedule_temperature(step, settings)
    log_probs = processed - mx.logsumexp(processed, axis=-1, keepdims=True)
    entropy = token_entropy(log_probs)
    argmax_canvas = mx.argmax(processed, axis=-1).astype(mx.int32)
    sampled = mx.random.categorical(processed).astype(mx.int32)
    accept = entropy_bound_mask(entropy, settings.entropy_bound)
    next_canvas = mx.where(
        accept, sampled, random_canvas(argmax_canvas.shape[0], vocab_size)
    )

    stable = len(history) == settings.stability_threshold and all(
        mx.array_equal(previous, argmax_canvas) for previous in history
    )
    confident = mx.mean(entropy) < settings.confidence_threshold
    mx.eval(next_canvas, argmax_canvas, confident)
    history.append(argmax_canvas)
    del history[: -settings.stability_threshold or None]

    converged = (bool(stable) and bool(confident.item())) or step + 1 >= max_steps
    return DenoiseOutcome(next_canvas, argmax_canvas, converged, processed)


# Large and finite rather than -inf: the entropy multiplies probabilities by
# log-probabilities, and 0 * -inf is NaN (upstream ``_MASKED_LOGIT``).
_MASKED_LOGIT = -1e20


def mask_to_allowed(logits: mx.array, allowed: mx.array) -> mx.array:
    """Restrict ``(rows, vocab)`` logits to a boolean vocabulary mask.

    The softmax of a masked row is the distribution renormalized over the
    allowed ids, so argmax, sampling, entropy and self-conditioning all stay
    inside the set (``diffusion_constrained``).
    """
    return mx.where(allowed, logits, _MASKED_LOGIT)


def canvas_logprobs(
    logits: mx.array,
    selected: mx.array,
    *,
    num_logprobs: int,
    token_ids: list[int] | None,
    logits_mode: bool,
) -> LogprobsLists:
    """Sample logprobs for every row of a ``(canvas, vocab)`` logits block.

    Column 0 is ``selected``; the rest are ``token_ids`` when given, else the
    top-``num_logprobs`` ids. ``logits_mode`` reports the logits themselves
    instead of their log-softmax. Ranks are 1-based and count ties, as in
    vLLM's ``Sampler.gather_logprobs``.
    """
    scores = (
        logits if logits_mode else logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    )
    rows = scores.shape[0]
    columns = [selected[:, None]]
    if token_ids:
        columns.append(
            mx.broadcast_to(mx.array(token_ids, dtype=mx.int32), (rows, len(token_ids)))
        )
    elif num_logprobs > 0:
        top = mx.argpartition(-scores, num_logprobs - 1, axis=-1)[:, :num_logprobs]
        order = mx.argsort(-mx.take_along_axis(scores, top, axis=-1), axis=-1)
        columns.append(mx.take_along_axis(top, order, axis=-1).astype(mx.int32))
    ids = mx.concatenate(columns, axis=1)
    values = mx.take_along_axis(scores, ids, axis=-1)
    ranks = mx.sum(scores >= values[:, :1], axis=-1)
    mx.eval(ids, values, ranks)
    return LogprobsLists(
        np.array(ids, dtype=np.int32),
        np.array(values, dtype=np.float32),
        np.array(ranks, dtype=np.int32),
    )


def join_canvas_logprobs(
    req_ids: list[str], logprobs: dict[str, LogprobsLists]
) -> LogprobsLists | None:
    """Concatenate per-request logprob rows in ``req_ids`` order.

    ``cu_num_generated_tokens`` holds each request's first row, so a request
    without rows this step takes none. Narrower stashes are padded with id 0
    and ``-inf``.
    """
    if not logprobs:
        return None
    width = max(rows.logprob_token_ids.shape[1] for rows in logprobs.values())
    ids, values, ranks = [], [], []
    starts: list[int] = []
    offset = 0
    for req_id in req_ids:
        starts.append(offset)
        rows = logprobs.get(req_id)
        if rows is None:
            continue
        pad = ((0, 0), (0, width - rows.logprob_token_ids.shape[1]))
        ids.append(np.pad(rows.logprob_token_ids, pad, constant_values=0))
        values.append(np.pad(rows.logprobs, pad, constant_values=float("-inf")))
        ranks.append(rows.sampled_token_ranks)
        offset += rows.logprob_token_ids.shape[0]
    return LogprobsLists(
        np.concatenate(ids), np.concatenate(values), np.concatenate(ranks), starts
    )


# ---------------------------------------------------------------------------
# Model forward (mlx_vlm DiffusionGemma, attention routed by the paged context)
# ---------------------------------------------------------------------------


def _backbone(model: Any) -> Any:
    return model.model


def validate_diffusion_model(model: Any) -> None:
    """Fail fast unless ``model`` has the mlx_vlm DiffusionGemma layout."""
    backbone = getattr(model, "model", None)
    decoder = getattr(backbone, "decoder", None)
    encoder = getattr(backbone, "encoder", None)
    if (
        getattr(model, "model_type", None) not in SUPPORTED_DIFFUSION_MODEL_TYPES
        or decoder is None
        or encoder is None
        or not hasattr(decoder, "self_conditioning")
    ):
        raise NotImplementedError(
            "Diffusion on Metal supports the mlx_vlm DiffusionGemma model only "
            f"(got model_type={getattr(model, 'model_type', None)!r})."
        )


def encoder_hidden_states(model: Any, input_ids: mx.array) -> mx.array:
    """Causal encoder pass: plain embeddings, encoder ``layer_scalar``s."""
    backbone = _backbone(model)
    text = backbone.decoder
    h = model.get_input_embeddings(input_ids).inputs_embeds
    scalars = [layer.layer_scalar for layer in backbone.encoder.language_model.layers]
    for layer, scalar in zip(text.layers, scalars, strict=True):
        h = layer(h, None, None, layer_scalar=scalar)
    return text.norm(h)


def decoder_hidden_states(
    model: Any, canvas_ids: mx.array, soft_embeddings: mx.array
) -> mx.array:
    """Bidirectional canvas pass: self-conditioned embeddings, decoder scalars.

    ``soft_embeddings`` matches ``canvas_ids`` row for row; zeros for a canvas
    without a previous step (mlx_vlm ``DecoderModel._embed_canvas``).
    """
    text = _backbone(model).decoder
    h = text.embed_tokens(canvas_ids) * text.embed_scale
    h = text.self_conditioning(h, soft_embeddings.astype(h.dtype))
    for layer in text.layers:
        h = layer(h, None, None)
    return text.norm(h)


def canvas_logits(model: Any, hidden_states: mx.array) -> mx.array:
    """Tied LM head plus the fp32 final-logit softcap."""
    logits = _backbone(model).decoder.embed_tokens.as_linear(hidden_states)
    cap = float(model.final_logit_softcapping)
    return mx.tanh(logits.astype(mx.float32) / cap) * cap


def _embedding_dtype(embed: nn.Module) -> mx.Dtype:
    if isinstance(embed, nn.QuantizedEmbedding):
        return embed.scales.dtype
    return embed.weight.dtype


def self_conditioning_embeddings(model: Any, processed_logits: mx.array) -> mx.array:
    """``softmax(logits) @ embed_tokens.weight * embed_scale`` for the next step."""
    text = _backbone(model).decoder
    embed = text.embed_tokens
    dtype = _embedding_dtype(embed)
    probs = mx.softmax(processed_logits, axis=-1, precise=True).astype(dtype)
    if isinstance(embed, nn.QuantizedEmbedding):
        soft = mx.quantized_matmul(
            probs,
            embed.weight,
            embed.scales,
            embed.biases,
            transpose=False,
            group_size=embed.group_size,
            bits=embed.bits,
            mode=getattr(embed, "mode", "affine"),
        )
    else:
        soft = probs @ embed.weight
    return soft.astype(dtype) * text.embed_scale


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


def _warn_ignored_sampling_params(params: SamplingParams) -> None:
    """The canvas sampler only follows the model's temperature schedule.

    Upstream ignores penalties the same way; top_k/top_p, which it applies,
    are refused in ``MetalPlatform.validate_request``.
    """
    ignored = [
        name
        for name, is_set in (
            ("presence_penalty", params.presence_penalty != 0.0),
            ("frequency_penalty", params.frequency_penalty != 0.0),
            ("repetition_penalty", params.repetition_penalty != 1.0),
        )
        if is_set
    ]
    if ignored:
        logger.warning_once(
            "DiffusionGemma on Metal ignores these sampling parameters: %s.",
            ", ".join(ignored),
        )


@dataclass
class _DiffusionRequest:
    # Denoise steps after which a canvas converges at the latest.
    max_steps: int
    phase: Literal["prefill", "denoise", "commit"] = "prefill"
    # Tokens the next denoise/commit step runs (and the scheduler's drafts).
    canvas: mx.array | None = None
    step: int = 0
    history: list[mx.array] = field(default_factory=list)
    soft_embeddings: mx.array | None = None
    # Taken on the converging step, delivered with the tokens it emits.
    logprobs: LogprobsLists | None = None
    # Structured-read options (vllm#57250), from the request's extra_args.
    seed_canvas: mx.array | None = None
    pinned: mx.array | None = None  # bool, (canvas,)
    read_only: bool = False
    allowed: mx.array | None = None  # bool, (vocab,)


@dataclass
class _Segment:
    req_id: str
    token_ids: list[int]
    start_pos: int
    block_ids: list[list[int]]


class DiffusionGemmaRuntime:
    """Runs DiffusionGemma engine steps for ``MetalModelRunner``."""

    def __init__(self, runner: MetalModelRunner) -> None:
        self._runner = runner
        self.settings = DiffusionSettings.from_vllm_config(runner.vllm_config)
        self._vocab_size = runner.model_config.get_vocab_size()
        self._logits_mode = runner.model_config.logprobs_mode in _LOGITS_LOGPROBS_MODES
        self._requests: dict[str, _DiffusionRequest] = {}

    def _new_request(self, params: SamplingParams) -> _DiffusionRequest:
        """Read the structured-read extra_args.

        vLLM's ``validate_diffusion_sampling_params`` has checked them by
        now: the seed spans the canvas, pins come with a seed and stay inside
        it, and ``diffusion_constrained`` comes with ``logprob_token_ids``.
        """
        args = params.extra_args or {}
        settings = self.settings
        request = _DiffusionRequest(
            max_steps=min(
                int(args.get("diffusion_max_steps", settings.max_denoising_steps)),
                settings.max_denoising_steps,
            ),
            read_only=bool(args.get("diffusion_read_only", False)),
        )
        seed = args.get("diffusion_seed_canvas")
        if seed is not None:
            request.seed_canvas = mx.array(seed, dtype=mx.int32)
            pinned = np.zeros(settings.canvas_length, dtype=np.bool_)
            pinned[args.get("diffusion_pinned") or []] = True
            if pinned.any():
                request.pinned = mx.array(pinned)
        if args.get("diffusion_constrained"):
            allowed = np.zeros(self._vocab_size, dtype=np.bool_)
            allowed[params.logprob_token_ids] = True
            request.allowed = mx.array(allowed)
        return request

    def _new_canvas(self, request: _DiffusionRequest, *, seeded: bool) -> None:
        request.phase = "denoise"
        if seeded and request.seed_canvas is not None:
            request.canvas = request.seed_canvas
        else:
            request.canvas = random_canvas(
                self.settings.canvas_length, self._vocab_size
            )
        request.step = 0
        request.history.clear()
        request.soft_embeddings = None

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        runner = self._runner
        if runner._paged_attention_runtime is None:
            raise RuntimeError("Paged attention runtime is not initialized.")
        if scheduler_output.scheduled_encoder_inputs:
            raise NotImplementedError(
                "Image inputs are not supported for DiffusionGemma on Metal yet."
            )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        evicted = runner._finished_req_ids(scheduler_output)
        runner._reconcile_request_lifecycle(
            evicted,
            preempted_req_ids=scheduler_output.preempted_req_ids,
            resumed_req_ids=cached_reqs.resumed_req_ids,
        )
        # A preempted request re-prefills its prompt and committed tokens.
        for req_id in (
            evicted
            | (scheduler_output.preempted_req_ids or set())
            | cached_reqs.resumed_req_ids
        ):
            self._requests.pop(req_id, None)

        from vllm_metal.v1.model_runner import RequestState

        num_computed: dict[str, int] = {}
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            sampling_params = new_req.sampling_params or SamplingParams()
            _warn_ignored_sampling_params(sampling_params)
            runner._request_states[new_req.req_id] = RequestState(
                token_ids=token_ids,
                prompt_len=len(token_ids),
                sampling_params=sampling_params,
                block_ids=runner._copy_paged_block_ids(new_req.block_ids),
                num_computed_tokens=new_req.num_computed_tokens,
            )
            num_computed[new_req.req_id] = new_req.num_computed_tokens
        runner._update_cached_request_blocks(cached_reqs)
        num_computed.update(
            zip(cached_reqs.req_ids, cached_reqs.num_computed_tokens, strict=True)
        )

        encoder_segments: list[_Segment] = []
        decoder_segments: list[_Segment] = []
        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            state = runner._request_states[req_id]
            request = self._requests.get(req_id)
            if request is None:
                request = self._new_request(state.sampling_params)
                self._requests[req_id] = request
            start = num_computed[req_id]
            drafts = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if request.phase == "prefill":
                if drafts:
                    raise RuntimeError(
                        f"Diffusion request {req_id!r} got draft tokens while "
                        "prefilling; scheduler and runner are out of sync."
                    )
                token_ids = state.token_ids[start : start + num_tokens]
            else:
                assert request.canvas is not None
                # The scheduler can clip the canvas (token budget,
                # long_prefill_token_threshold, max_model_len); the step runs
                # the scheduled prefix and the canvas keeps its full length.
                token_ids = request.canvas[:num_tokens].tolist()
                if drafts is None or list(drafts) != token_ids:
                    raise RuntimeError(
                        f"Diffusion request {req_id!r} was scheduled with drafts "
                        "that differ from its canvas; scheduler and runner are "
                        "out of sync."
                    )
            segment = _Segment(req_id, token_ids, start, state.block_ids)
            if request.phase == "denoise":
                decoder_segments.append(segment)
            else:
                encoder_segments.append(segment)

        runtime = runner._paged_attention_runtime
        if scheduler_output.new_block_ids_to_zero:
            runtime.zero_blocks(scheduler_output.new_block_ids_to_zero)
        if scheduler_output.kv_cache_block_copies:
            runtime.copy_blocks(scheduler_output.kv_cache_block_copies)
        sampled: dict[str, list[int]] = {}
        logprobs: dict[str, LogprobsLists] = {}
        if encoder_segments:
            self._run_encoder(encoder_segments, sampled, logprobs)
        if decoder_segments:
            self._run_decoder(decoder_segments, sampled, logprobs)

        req_ids = list(scheduler_output.num_scheduled_tokens)
        draft_token_ids: list[list[int]] = []
        for req_id in req_ids:
            request = self._requests[req_id]
            canvas = request.canvas if request.phase != "prefill" else None
            draft_token_ids.append([] if canvas is None else canvas.tolist())
        runner._draft_token_ids = DraftTokenIds(req_ids, draft_token_ids)
        runtime.materialize_pending_state()
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[sampled.get(req_id, []) for req_id in req_ids],
            logprobs=join_canvas_logprobs(req_ids, logprobs),
        )

    def _forward(self, segments: list[_Segment], *, decoder: bool) -> mx.array:
        runner = self._runner
        prepare_grouped(
            [],
            [(s.block_ids, len(s.token_ids), s.start_pos) for s in segments],
            runner._paged_group_block_sizes,
            tq_prefill_workspace_bytes=runner.tq_prefill_workspace_bytes,
        )
        try:
            ctx = get_context()
            assert ctx is not None
            input_ids = mx.array(
                [[t for s in segments for t in s.token_ids]], dtype=mx.int32
            )
            if not decoder:
                hidden = encoder_hidden_states(runner.model, input_ids)
            else:
                ctx.bidi_layer_kinds = _ALL_LAYER_KINDS
                ctx.bidi_window_at_block_start = True
                ctx.segment_bidi_ranges = [
                    [(s.start_pos, s.start_pos + len(s.token_ids))] for s in segments
                ]
                soft = [
                    self._soft_embeddings(self._requests[s.req_id], len(s.token_ids))
                    for s in segments
                ]
                hidden = decoder_hidden_states(
                    runner.model, input_ids, mx.concatenate(soft, axis=0)[None]
                )
            # Forces the paged KV writes, which run inside the forward.
            mx.eval(hidden)
        finally:
            clear_context()
        return hidden

    def _soft_embeddings(self, request: _DiffusionRequest, length: int) -> mx.array:
        if request.soft_embeddings is not None:
            return request.soft_embeddings[:length]
        text = _backbone(self._runner.model).decoder
        return mx.zeros(
            (length, text.config.hidden_size), dtype=_embedding_dtype(text.embed_tokens)
        )

    def _run_encoder(
        self,
        segments: list[_Segment],
        sampled: dict[str, list[int]],
        logprobs: dict[str, LogprobsLists],
    ) -> None:
        self._forward(segments, decoder=False)
        runner = self._runner
        for segment in segments:
            request = self._requests[segment.req_id]
            state = runner._request_states[segment.req_id]
            if request.phase == "commit":
                self._emit(segment.req_id, segment.token_ids, sampled, logprobs)
                self._new_canvas(request, seeded=False)
            elif segment.start_pos + len(segment.token_ids) >= len(state.token_ids):
                # The seed replaces only the first canvas after the prompt.
                self._new_canvas(
                    request, seeded=len(state.token_ids) == state.prompt_len
                )

    def _emit(
        self,
        req_id: str,
        token_ids: list[int],
        sampled: dict[str, list[int]],
        logprobs: dict[str, LogprobsLists],
    ) -> None:
        request = self._requests[req_id]
        state = self._runner._request_states[req_id]
        sampled[req_id] = token_ids
        if request.logprobs is not None:
            logprobs[req_id] = request.logprobs.slice_request(0, len(token_ids))
            request.logprobs = None
        state.token_ids.extend(token_ids)
        state.generated_tokens += len(token_ids)

    def _run_decoder(
        self,
        segments: list[_Segment],
        sampled: dict[str, list[int]],
        logprobs: dict[str, LogprobsLists],
    ) -> None:
        hidden = self._forward(segments, decoder=True)
        model = self._runner.model
        offset = 0
        for segment in segments:
            request = self._requests[segment.req_id]
            length = len(segment.token_ids)
            # One request's logits at a time bounds the (canvas, vocab) fp32
            # transient regardless of batch size.
            logits = canvas_logits(model, hidden[0, offset : offset + length])
            offset += length
            if request.allowed is not None:
                logits = mask_to_allowed(logits, request.allowed)
            # Like upstream, pad a clipped canvas with uniform (zero) logits:
            # the unscheduled rows are resampled at random and, at maximum
            # entropy, cannot make the canvas converge on their own.
            padding = self.settings.canvas_length - length
            if padding > 0:
                logits = mx.concatenate(
                    [logits, mx.zeros((padding, logits.shape[1]), dtype=logits.dtype)]
                )
            outcome = denoise_update(
                logits,
                step=request.step,
                history=request.history,
                settings=self.settings,
                vocab_size=self._vocab_size,
                max_steps=request.max_steps,
            )
            request.step += 1
            if not outcome.converged:
                request.canvas = outcome.next_canvas
                soft = self_conditioning_embeddings(model, outcome.processed_logits)
                if request.pinned is not None:
                    # A pinned row holds its seed token, so the model's own
                    # prediction there must not reach the next step either.
                    request.canvas = mx.where(
                        request.pinned, request.seed_canvas, request.canvas
                    )
                    soft = mx.where(request.pinned[:, None], 0, soft)
                request.soft_embeddings = soft
                mx.eval(request.canvas, request.soft_embeddings)
                continue

            # Padded rows only converge on the last step; emit only the rows
            # the model saw.
            emitted = outcome.argmax_canvas[:length]
            request.soft_embeddings = None
            params = self._runner._request_states[segment.req_id].sampling_params
            if params.num_logprobs is not None:
                # As upstream: reads report temperature-1 logprobs, generation
                # the schedule-tempered logits of this step.
                source = logits if request.read_only else outcome.processed_logits
                request.logprobs = canvas_logprobs(
                    source[:length],
                    emitted,
                    num_logprobs=params.logprobs or 0,
                    token_ids=params.logprob_token_ids,
                    logits_mode=self._logits_mode,
                )
            if request.read_only:
                # A read emits now and skips the commit forward. Its canvas KV
                # is the decoder's, which is harmless: emitting the canvas
                # reaches max_tokens, so the request ends. A clipped read
                # keeps its argmax canvas and emits again when it next
                # converges, as upstream does.
                self._emit(segment.req_id, emitted.tolist(), sampled, logprobs)
                request.canvas = outcome.argmax_canvas
            else:
                request.phase = "commit"
                request.canvas = emitted
