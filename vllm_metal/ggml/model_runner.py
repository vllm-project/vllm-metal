# SPDX-License-Identifier: Apache-2.0
"""vLLM v1 model runner backed by the Rust/ggml engine.

Scheduling, request bookkeeping and sampling stay in Python (vLLM's torch
sampler on CPU); the forward pass, paged KV cache and recurrent state live
in the native engine. Each ``execute_model`` call packs the scheduled tokens
(decodes first, so they form one attention group), runs one engine step and
stashes the logits rows for ``sample_tokens``.

KV cache contract with vLLM: the runner reports a single synthetic
``FullAttentionSpec`` whose page size equals the engine's per-block bytes
across *all* attention layers, so vLLM hands out one block table per request
that the engine applies to every layer. Recurrent (linear-attention) state is
per-sequence and managed here via a slot pool, so prefix caching is disabled
for such models at config time (see ``vllm_metal.ggml.policy``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, LogprobsLists, ModelRunnerOutput
from vllm.v1.sample.sampler import Sampler

import vllm_metal.envs as envs
from vllm_metal.ggml import load_extension
from vllm_metal.v1.sampling_batch import SamplingBatch, create_request_generator

logger = init_logger(__name__)

# Name of the single synthetic KV-cache "layer" reported to vLLM.
KV_LAYER_NAME = "ggml_kv"
# Headroom kept free on top of the measured activation buffer.
_ACTIVATION_MARGIN_BYTES = 256 * 1024**2


@dataclass
class _Request:
    token_ids: list[int]
    prompt_len: int
    sampling_params: SamplingParams
    generator: torch.Generator | None
    block_ids: list[int]
    state_slot: int
    num_computed: int = 0
    # Recurrent state must be zeroed before the next forward.
    reset_state: bool = True


@dataclass
class _StepState:
    req_ids: list[str]
    sample_req_ids: list[str]
    logits: np.ndarray  # [len(sample_req_ids), vocab]


class GGMLModelRunner:
    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.engine: Any = None
        self.info: dict[str, Any] = {}
        self.requests: dict[str, _Request] = {}
        self.sampler = Sampler()
        self._pending: _StepState | None = None
        self._free_slots: list[int] = []
        self._logits_buf = np.empty((0, 0), np.float32)
        self.block_size: int | None = None
        self.num_blocks = 0
        self.model = None  # WorkerBase.get_model contract; no torch module here
        self._t_forward = 0.0

    # ------------------------------------------------------------------ setup

    def _resolve_model_dir(self) -> str:
        from pathlib import Path

        from vllm_metal.utils import get_model_download_path

        path = get_model_download_path(
            self.model_config.model, revision=self.model_config.revision
        )
        if Path(path).is_dir():
            return str(path)
        from huggingface_hub import snapshot_download

        return snapshot_download(
            path,
            revision=self.model_config.revision,
            allow_patterns=["*.json", "*.safetensors"],
        )

    def load_model(self) -> None:
        path = self._resolve_model_dir()
        device = envs.VLLM_METAL_GGML_DEVICE
        ext = load_extension()
        self._attach_engine(ext.Engine(path, device))

    def _attach_engine(self, engine: Any) -> None:
        self.engine = engine
        self.info = dict(self.engine.info())
        self.attn_budget = envs.VLLM_METAL_GGML_ATTN_BUDGET_MB * 1024**2
        if hasattr(engine, "set_attn_budget"):
            engine.set_attn_budget(self.attn_budget)
        vocab = self.model_config.get_vocab_size()
        if vocab != self.info["vocab_size"]:
            raise ValueError(
                f"vocab size mismatch: vLLM {vocab} vs ggml engine {self.info['vocab_size']}"
            )
        self.vocab_size = vocab
        self.has_state = bool(self.info["state_layers"])
        self.max_num_seqs = self.scheduler_config.max_num_seqs
        logger.info(
            "ggml engine loaded %s on %s: %.2f GiB weights, %d KV layers, %d state layers",
            self.info["arch"],
            self.info["backend"],
            self.info["weight_bytes"] / 2**30,
            len(self.info["kv_layers"]),
            len(self.info["state_layers"]),
        )

    def _kv_head_size(self) -> int:
        return sum(kv["head_dim"] * kv["num_kv_heads"] for kv in self.info["kv_layers"])

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return {
            KV_LAYER_NAME: FullAttentionSpec(
                block_size=self.cache_config.block_size,
                num_kv_heads=1,
                head_size=self._kv_head_size(),
                dtype=torch.float16,
            )
        }

    def get_cache_block_size_bytes(self) -> int:
        return self.cache_config.block_size * self.info["kv_bytes_per_token"]

    def _state_slots(self) -> int:
        return self.max_num_seqs + 1 if self.has_state else 1

    def profile_run(self) -> int:
        """Run a worst-case prefill to size the activation buffer.

        Returns the device bytes that must stay free for activations.
        """
        bs = self.cache_config.block_size
        n = min(
            self.scheduler_config.max_num_batched_tokens,
            self.model_config.max_model_len,
        )
        nblocks = (n + bs - 1) // bs
        self.engine.init_cache(nblocks + 1, bs, 1)
        free0, _ = self.engine.memory()
        ids = np.zeros(n, np.int32)
        one = lambda v: np.array([v], np.int32)  # noqa: E731
        bt = np.arange(1, nblocks + 1, dtype=np.int32)[None]
        out = np.empty((1, self.vocab_size), np.float32)
        self.engine.forward(ids, one(n), one(n), bt, one(0), one(1), one(1), out)
        free1, _ = self.engine.memory()
        used = max(0, free0 - free1)
        logger.info(
            "ggml profile run (%d tokens): %.1f MiB activations", n, used / 2**20
        )
        return used

    def determine_available_memory(self) -> int:
        _, total = self.engine.memory()
        activations = self.profile_run() + self.attn_budget + _ACTIVATION_MARGIN_BYTES
        state = self._state_slots() * self.info["state_bytes_per_slot"]
        budget = int(total * self.cache_config.gpu_memory_utilization)
        available = budget - self.info["weight_bytes"] - state - activations
        logger.info(
            "ggml memory: device %.1f GiB x %.2f, weights %.2f GiB, state %.2f GiB, "
            "activations %.2f GiB -> KV budget %.2f GiB",
            total / 2**30,
            self.cache_config.gpu_memory_utilization,
            self.info["weight_bytes"] / 2**30,
            state / 2**30,
            activations / 2**30,
            available / 2**30,
        )
        if available <= 0:
            raise RuntimeError(
                "Not enough device memory for the KV cache with the ggml backend; "
                "lower --max-num-seqs / --max-num-batched-tokens or raise "
                "--gpu-memory-utilization."
            )
        return available

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1 or groups[0].layer_names != [KV_LAYER_NAME]:
            raise RuntimeError(f"unexpected KV cache groups for ggml backend: {groups}")
        self.block_size = groups[0].kv_cache_spec.block_size
        self.num_blocks = kv_cache_config.num_blocks
        slots = self._state_slots()
        nbytes = self.engine.init_cache(self.num_blocks, self.block_size, slots)
        # Slot 0 is reserved (never handed out) so a stray zero is harmless.
        self._free_slots = list(range(slots - 1, 0, -1)) if self.has_state else []
        logger.info(
            "ggml cache: %d blocks x %d tokens, %d state slots (%.2f GiB)",
            self.num_blocks,
            self.block_size,
            slots,
            nbytes / 2**30,
        )

    def warm_up(self) -> None:
        """Compile Metal pipelines for prefill- and decode-shaped steps.

        Runs before any request exists, using state slot 0 (never handed out)
        and low block ids whose contents are overwritten before first use.
        """
        if self.engine is None or self.block_size is None or self.num_blocks < 3:
            return
        a = lambda x: np.asarray(x, np.int32)  # noqa: E731
        out = np.empty((2, self.vocab_size), np.float32)
        bt = np.array([[1], [2]], np.int32)
        prompt = [1, 2, 3]
        # small prefill, then single and batched decode
        self.engine.forward(
            a(prompt), a([3]), a([3]), bt[:1], a([0]), a([1]), a([1]), out
        )
        self.engine.forward(a([1]), a([1]), a([4]), bt[:1], a([0]), a([0]), a([1]), out)
        self.engine.forward(
            a([1, 1]), a([1, 1]), a([5, 1]), bt, a([0, 0]), a([0, 1]), a([1, 1]), out
        )

    # ------------------------------------------------------------ bookkeeping

    def _alloc_slot(self) -> int:
        if not self.has_state:
            return 0
        if not self._free_slots:
            raise RuntimeError("ggml runner: out of recurrent-state slots")
        return self._free_slots.pop()

    def _free_slot(self, req: _Request) -> None:
        if self.has_state and req.state_slot > 0:
            self._free_slots.append(req.state_slot)
        req.state_slot = -1

    def _release(self, req_id: str) -> None:
        req = self.requests.pop(req_id, None)
        if req is not None:
            self._free_slot(req)

    def _update_states(self, so: SchedulerOutput) -> None:
        for req_id in so.finished_req_ids:
            self._release(req_id)
        # Preempted requests keep their token history but give back their
        # recurrent-state slot; they recompute from scratch when resumed.
        for req_id in so.preempted_req_ids or ():
            req = self.requests.get(req_id)
            if req is not None:
                self._free_slot(req)
                req.reset_state = True

        for nr in so.scheduled_new_reqs:
            if nr.req_id in self.requests:  # aborted + resubmitted under same id
                self._release(nr.req_id)
            if nr.prompt_embeds is not None:
                raise NotImplementedError(
                    "prompt_embeds are not supported by the ggml backend"
                )
            sp = nr.sampling_params or SamplingParams()
            tokens = list(nr.prompt_token_ids or [])
            self.requests[nr.req_id] = _Request(
                token_ids=tokens,
                prompt_len=len(tokens),
                sampling_params=sp,
                generator=create_request_generator(sp),
                block_ids=list(nr.block_ids[0]),
                state_slot=self._alloc_slot(),
                num_computed=nr.num_computed_tokens,
            )

        cd = so.scheduled_cached_reqs
        for i, req_id in enumerate(cd.req_ids):
            req = self.requests.get(req_id)
            if req is None:
                raise RuntimeError(f"ggml runner: unknown cached request {req_id!r}")
            new_blocks = cd.new_block_ids[i]
            if req_id in cd.resumed_req_ids:
                # Preempted and rescheduled: new block table, recompute state.
                # The runner's own token history stays authoritative.
                req.block_ids = list(new_blocks[0]) if new_blocks is not None else []
                req.reset_state = True
            elif new_blocks is not None:
                req.block_ids.extend(new_blocks[0])
            if req.state_slot < 0:
                req.state_slot = self._alloc_slot()
                req.reset_state = True
            # The scheduler's count includes in-flight placeholders, so it only
            # ever lags behind when tokens were discarded upstream.
            num_out = cd.num_output_tokens[i] if cd.num_output_tokens else None
            if num_out is not None and len(req.token_ids) > req.prompt_len + num_out:
                del req.token_ids[req.prompt_len + num_out :]
            req.num_computed = cd.num_computed_tokens[i]

    # -------------------------------------------------------------- execution

    def execute_model(self, so: SchedulerOutput) -> ModelRunnerOutput | None:
        self._update_states(so)
        if so.total_num_scheduled_tokens == 0:
            return EMPTY_MODEL_RUNNER_OUTPUT
        if so.scheduled_spec_decode_tokens:
            raise NotImplementedError(
                "speculative decoding is not supported by the ggml backend"
            )

        # Decodes (1 token) first so they share one attention/state group.
        req_ids = sorted(
            so.num_scheduled_tokens, key=lambda r: so.num_scheduled_tokens[r] != 1
        )
        tokens: list[int] = []
        q_lens, ctx_lens, slots, resets, n_logits = [], [], [], [], []
        sample_req_ids: list[str] = []
        max_blocks = 1
        for req_id in req_ids:
            req = self.requests[req_id]
            n = so.num_scheduled_tokens[req_id]
            start = req.num_computed
            end = start + n
            if end > len(req.token_ids):
                raise RuntimeError(
                    f"ggml runner: request {req_id!r} scheduled past its known tokens "
                    f"({end} > {len(req.token_ids)})"
                )
            if start == 0:
                req.reset_state = True
            tokens.extend(req.token_ids[start:end])
            q_lens.append(n)
            ctx_lens.append(end)
            slots.append(req.state_slot)
            resets.append(1 if req.reset_state else 0)
            req.reset_state = False
            wants = end == len(req.token_ids)
            n_logits.append(1 if wants else 0)
            if wants:
                sample_req_ids.append(req_id)
            max_blocks = max(max_blocks, len(req.block_ids))

        bt = np.zeros((len(req_ids), max_blocks), np.int32)
        for i, req_id in enumerate(req_ids):
            b = self.requests[req_id].block_ids
            bt[i, : len(b)] = b

        rows = len(sample_req_ids)
        if (
            self._logits_buf.shape[0] < max(rows, 1)
            or self._logits_buf.shape[1] != self.vocab_size
        ):
            self._logits_buf = np.empty((max(rows, 1), self.vocab_size), np.float32)
        a = lambda x: np.asarray(x, np.int32)  # noqa: E731
        t0 = time.perf_counter()
        got = self.engine.forward(
            a(tokens),
            a(q_lens),
            a(ctx_lens),
            bt,
            a(slots),
            a(resets),
            a(n_logits),
            self._logits_buf,
        )
        self._t_forward = time.perf_counter() - t0
        assert got == rows
        for req_id, n in zip(req_ids, q_lens, strict=True):
            self.requests[req_id].num_computed += n

        # Keep vLLM's scheduled order for the output.
        self._pending = _StepState(
            req_ids=list(so.num_scheduled_tokens),
            sample_req_ids=sample_req_ids,
            logits=self._logits_buf[:rows],
        )
        return None

    def _apply_grammar(
        self, st: _StepState, logits: torch.Tensor, grammar: GrammarOutput
    ) -> None:
        import xgrammar as xgr

        row_of = {r: i for i, r in enumerate(st.sample_req_ids)}
        pairs = [
            (row_of[r], bi)
            for bi, r in enumerate(grammar.structured_output_request_ids)
            if r in row_of
        ]
        if not pairs:
            return
        rows = [p[0] for p in pairs]
        sub = logits[rows].contiguous()
        xgr.apply_token_bitmask_inplace(
            sub, torch.from_numpy(grammar.grammar_bitmask[[p[1] for p in pairs]])
        )
        logits[rows] = sub

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | None:
        st = self._pending
        self._pending = None
        if st is None:
            return None

        t0 = time.perf_counter()
        sampled: dict[str, int] = {}
        logprobs = None
        if st.sample_req_ids:
            logits = torch.from_numpy(st.logits.copy())
            if grammar_output is not None:
                self._apply_grammar(st, logits, grammar_output)
            reqs = [self.requests[r] for r in st.sample_req_ids]
            batch = SamplingBatch(
                [r.sampling_params for r in reqs],
                [r.token_ids[: r.prompt_len] for r in reqs],
                [r.token_ids[r.prompt_len :] for r in reqs],
                vocab_size=self.vocab_size,
                generators={
                    i: r.generator
                    for i, r in enumerate(reqs)
                    if r.generator is not None
                },
            )
            out = self.sampler(logits, batch.make_sampling_metadata(logits))
            toks = out.sampled_token_ids[:, 0].tolist()
            for req_id, req, tok in zip(st.sample_req_ids, reqs, toks, strict=True):
                req.token_ids.append(tok)
                sampled[req_id] = tok
            if out.logprobs_tensors is not None:
                logprobs = self._scatter_logprobs(st, out.logprobs_tensors.tolists())
        if envs.VLLM_METAL_GGML_PROFILE:
            logger.info(
                "ggml step: %d reqs, %d sampled | forward %.2f ms, sample %.2f ms",
                len(st.req_ids),
                len(st.sample_req_ids),
                self._t_forward * 1e3,
                (time.perf_counter() - t0) * 1e3,
            )

        return ModelRunnerOutput(
            req_ids=st.req_ids,
            req_id_to_index={r: i for i, r in enumerate(st.req_ids)},
            sampled_token_ids=[
                [sampled[r]] if r in sampled else [] for r in st.req_ids
            ],
            logprobs=logprobs,
            prompt_logprobs_dict={},
            pooler_output=None,
        )

    @staticmethod
    def _scatter_logprobs(st: _StepState, lp: LogprobsLists) -> LogprobsLists:
        """Expand sampled-row logprobs to one row per output request."""
        n, width = len(st.req_ids), lp.logprob_token_ids.shape[1]
        ids = np.zeros((n, width), np.int32)
        vals = np.full((n, width), float("-inf"), np.float32)
        ranks = np.zeros((n,), np.int32)
        index = {r: i for i, r in enumerate(st.req_ids)}
        for row, req_id in enumerate(st.sample_req_ids):
            j = index[req_id]
            ids[j] = lp.logprob_token_ids[row]
            vals[j] = lp.logprobs[row]
            ranks[j] = lp.sampled_token_ranks[row]
        return LogprobsLists(
            logprob_token_ids=ids,
            logprobs=vals,
            sampled_token_ranks=ranks,
            cu_num_generated_tokens=None,
        )

    # ------------------------------------------------------------- misc API

    def take_draft_token_ids(self):
        return None

    def supported_worker_tasks(self) -> tuple[str, ...]:
        return ("generate",)

    def reset_mm_cache(self) -> None:
        pass

    def reset_encoder_cache(self) -> None:
        pass
