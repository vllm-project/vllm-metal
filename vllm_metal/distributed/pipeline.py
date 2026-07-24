# SPDX-License-Identifier: Apache-2.0
"""Pipeline-parallel primitives for vLLM Metal (Phase 0: single node, N stages).

Convention (N-general, validated at 2 stages):
  - rank 0           = FIRST stage  (embeds the input tokens)
  - rank ``size - 1`` = LAST stage   (runs the final norm + head, samples)

Each stage owns a contiguous slice of the backbone's transformer layers. The
hidden state is piped point-to-point: stage ``r`` receives from ``r - 1`` and
sends to ``r + 1``. Only the last stage produces logits.

This module stays light: it must NOT import vllm_metal runner/adapter code (so the
worker can create the group before the model runner exists). It does reuse vLLM's
pure ``get_pp_indices`` helper for the layer split rather than reinventing it.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from functools import cached_property
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.core.distributed import Group
from mlx.utils import tree_flatten
from vllm.distributed.utils import get_pp_indices

import vllm_metal.envs as envs

logger = logging.getLogger(__name__)


def _ring_hosts(peer_ips: list[str]) -> list[list[str]]:
    """Build the MLX ring hostfile body from per-rank node IPs.

    ``peer_ips[r]`` is the IP of the node hosting rank ``r`` — identical on a
    single node, distinct across Macs. The ring backend wants a bare list of
    ``["ip:port"]`` lists, one per rank in rank order. Rank ``r`` listens on
    ``VLLM_METAL_RING_BASE_PORT + r`` (default base 32323, the mlx.launch
    ``starting_port``), so co-located ranks still get distinct ports; set the
    same base on every node to move the ring off a busy port. The env var is
    read per call, like every other vllm_metal env knob.
    """
    base_port = envs.VLLM_METAL_RING_BASE_PORT
    if not (1024 <= base_port <= 65535):
        raise ValueError(
            "VLLM_METAL_RING_BASE_PORT must be in the user-port range "
            f"[1024, 65535], got {base_port}"
        )
    max_port = base_port + len(peer_ips) - 1
    if max_port > 65535:
        raise ValueError(
            "VLLM_METAL_RING_BASE_PORT is too high for the pipeline size: "
            f"base {base_port} with {len(peer_ips)} ranks would use port {max_port}"
        )
    return [[f"{ip}:{base_port + r}"] for r, ip in enumerate(peer_ips)]


class PipelineGroup:
    """Thin wrapper over an ``mx.distributed`` group with stage helpers.

    ``rank``/``size`` come straight from the underlying MLX group. The first
    stage is rank 0; the last stage is rank ``size - 1``. A plain-python
    process (no launcher) yields a singleton size-1 group, so a single process
    is BOTH the first and the last stage (straight-through, no comms).
    """

    def __init__(self, group: Group) -> None:
        self.group = group
        self.rank: int = group.rank()
        self.size: int = group.size()
        self.is_first: bool = self.rank == 0
        self.is_last: bool = self.rank == self.size - 1

    @classmethod
    def bootstrap_ring(cls, rank: int, peer_ips: list[str]) -> PipelineGroup:
        """Form the MLX ring across ``peer_ips`` (one per rank), then init.

        The vLLM engine spawns one worker process per pipeline stage (the
        ``mp``/``ray`` executor), so — unlike the standalone ``mlx.launch``
        harness — no launcher injects the ring environment. The compiled ring
        backend reads two variables: ``MLX_RANK`` and ``MLX_HOSTFILE`` (a path
        to a JSON file whose root is a bare list of ``["ip:port"]`` lists, one
        per rank in rank order). ``peer_ips[r]`` is the IP of the node hosting
        rank ``r`` — identical on a single node, distinct across Macs — so the
        same code path serves single- and multi-node. The caller discovers
        these IPs (e.g. an ``all_gather`` of each worker's address over the
        control-plane group). Each worker writes its own copy of the (identical)
        hostfile and points ``MLX_HOSTFILE`` at it; the backend selects this
        worker's slot via ``MLX_RANK``.

        Fails loud if the formed group's size does not match ``len(peer_ips)``
        (e.g. ``MLX_RANK``/``MLX_HOSTFILE`` never reached the backend, or a node
        IP in the hostfile was unreachable).
        """
        world_size = len(peer_ips)
        hosts = _ring_hosts(peer_ips)
        fd, path = tempfile.mkstemp(prefix=f"mlx_ring_rank{rank}_", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(hosts, f)
        os.environ["MLX_RANK"] = str(rank)
        os.environ["MLX_HOSTFILE"] = path
        logger.info(
            "MLX ring bootstrap: rank=%d/%d hosts=%s hostfile=%s",
            rank,
            world_size,
            [h[0] for h in hosts],
            path,
        )

        # Form the real ring group from the MLX_RANK / MLX_HOSTFILE env set
        # above. backend="ring" pins the transport: the default "any" takes the
        # first backend that initializes (e.g. a singleton MPI group when
        # libmpi is present), which would fail the size check below with a
        # misleading hint.
        pp = cls(mx.distributed.init(backend="ring"))
        # init() consumed the hostfile synchronously; drop the temp file now
        # (before the size check, so the raise path is cleaned up too).
        Path(path).unlink(missing_ok=True)
        if pp.size != world_size:
            raise RuntimeError(
                "MLX ring did not form the expected pipeline group: "
                f"got size {pp.size}, expected {world_size}. Check that "
                "MLX_RANK/MLX_HOSTFILE reached the ring backend and that every "
                "node IP in the hostfile is reachable."
            )
        return pp


def is_non_last_stage(pp: PipelineGroup | None) -> bool:
    """Whether the caller runs a non-last stage of a *multi-stage* pipeline.

    Stage role lives on :class:`PipelineGroup` (``rank``/``is_first``/``is_last``),
    mirroring how upstream vLLM exposes ``get_pp_group().is_last_rank`` on the
    group rather than on the model runner. The runner holds an *optional* group
    (``None`` on the single-stage path, where no group is created), so fold that
    case in here: no group -> the only stage -> the last stage -> not a non-last
    stage. ``not is_last`` also already excludes a size-1 group, so no size check
    is needed. The one canonical "is there a stage downstream of me?" test.
    """
    return pp is not None and not pp.is_last


def apply_pipeline_split(model: nn.Module, pp: PipelineGroup) -> tuple[int, int] | None:
    """Slice ``model`` in place to the stage owned by ``pp``.

    Phase 0 = load-then-slice: every rank loads the full weights, then drops the
    layers (and, off the last stage, the final norm) it does not own.

    - Fails LOUD if ``model.model.layers`` is not a sliceable ``list``.
    - Slices ``backbone.layers`` to this rank's contiguous range.
    - On non-last stages replaces ``backbone.norm`` with ``nn.Identity`` so the
      final norm runs only on the last stage (the middle stages emit raw
      hidden states).

    Returns the ``(start, end)`` slice, or ``None`` for a singleton (size-1)
    group, where the whole model stays on the single stage untouched.
    """
    if pp.size == 1:
        return None

    backbone = model.model
    layers = backbone.layers
    if not isinstance(layers, list):
        raise TypeError(
            f"apply_pipeline_split requires backbone.layers to be a sliceable "
            f"list, got {type(layers).__name__}"
        )

    start, end = get_pp_indices(len(layers), pp.rank, pp.size)
    if start >= end:
        raise NotImplementedError(
            f"Pipeline parallelism with {pp.size} stages over a model with "
            f"{len(layers)} layers leaves stage {pp.rank} with no layers "
            f"(get_pp_indices returned [{start}, {end})); use at most "
            f"{len(layers)} pipeline stages."
        )
    backbone.layers = layers[start:end]

    if not pp.is_last:
        backbone.norm = nn.Identity()

    return start, end


def pipeline_recv(
    pp: PipelineGroup,
    n_tokens: int,
    hidden: int,
    dtype: mx.Dtype,
) -> mx.array:
    """Receive the hidden state from the previous stage as ``(1, n_tokens, hidden)``.

    The wire carries the compact 2D ``(n_tokens, hidden)`` activation; this adds
    the batch axis back so the result feeds straight into the model forward as
    ``input_embeddings``. Lazy: the transfer only happens on ``mx.eval`` of the
    returned array. The declared ``(shape, dtype)`` MUST exactly match the
    sender's wire array — a mismatch deadlocks (the ring backend does not
    validate across peers).
    """
    wire = mx.distributed.recv((n_tokens, hidden), dtype, pp.rank - 1, group=pp.group)
    return wire[None]


def pipeline_send(h: mx.array, pp: PipelineGroup) -> mx.array:
    """Send the model hidden state ``h`` to the next stage.

    ``h`` is the model-native ``(1, n_tokens, hidden)`` activation; the batch
    axis (always 1 under Phase 0 single-sequence batching) is dropped so the
    wire carries the compact 2D ``(n_tokens, hidden)`` that :func:`pipeline_recv`
    restores. Lazy: returns an array that performs the send when evaluated, so
    the caller MUST ``mx.eval`` (or otherwise materialize) it or the bytes never
    move.
    """
    if h.ndim != 3 or h.shape[0] != 1:
        raise ValueError(
            f"pipeline_send expects a (1, n_tokens, hidden) hidden state, "
            f"got shape {tuple(h.shape)}"
        )
    return mx.distributed.send(h[0], pp.rank + 1, group=pp.group)


class PipelinedModel:
    """PP-aware forward wrapper: runs an mlx_lm model as a single pipeline stage.

    Owns the stage *forward*: the first stage embeds ``input_ids``; every later
    stage receives the upstream hidden state and feeds it in as
    ``input_embeddings``; the local (already-sliced) layers run, and the last
    stage applies the final norm + head. Returns this stage's raw model output
    on the last rank (the caller extracts logits), otherwise the raw hidden
    state for the caller to send downstream.

    vLLM's own models carry this contract internally (``make_layers`` +
    ``forward(intermediate_tensors=...)``); here we wrap instead, because the
    mlx_lm model files stay untouched. The matching *send* is owned by the
    caller (the runner), so the lazy transfer is evaluated alongside its other
    forward evals.
    """

    def __init__(self, model: Any, pp: PipelineGroup) -> None:
        self._model = model
        self._pp = pp

    # Wire descriptor: (hidden, dtype) of the activation that crosses stages, read
    # from state every stage owns (config width + a stage-owned compute parameter),
    # not by probing ``embed_tokens`` which a streaming non-first stage does not
    # own. Lazy: a singleton stage crosses no wire, so it never reads either half.
    @cached_property
    def _wire_hidden(self) -> int:
        return self._model.args.hidden_size

    @cached_property
    def _wire_dtype(self) -> mx.Dtype:
        # First floating-point parameter of the first owned layer: norm weights and
        # quant scales are floating, a packed quantized weight is uint32. tree_flatten
        # is typed list|dict, so pin the list branch for mypy.
        first_layer = self._model.model.layers[0]
        flat: list[tuple[str, Any]] = tree_flatten(first_layer.parameters())
        return next(
            param.dtype for _, param in flat if mx.issubdtype(param.dtype, mx.floating)
        )

    def __call__(self, input_ids: mx.array, *, cache: Any = None) -> mx.array:
        h_in: mx.array | None = None
        if not self._pp.is_first:
            h_in = pipeline_recv(
                self._pp, input_ids.shape[1], self._wire_hidden, self._wire_dtype
            )
        return self._stage_forward(input_ids, h_in, cache=cache)

    def dummy_forward(self, input_ids: mx.array) -> mx.array:
        """Run this stage's forward locally for profiling/warm-up: no ring I/O.

        Non-first stages take a zeros hidden state at the wire descriptor in
        place of ``pipeline_recv`` (mirroring upstream vLLM's ``_dummy_run`` +
        ``make_empty_intermediate_tensors``), so profiling never materializes
        the embedding a non-first stage does not use; the stage body — and its
        wire-dtype fail-fast — is shared with ``__call__``.
        """
        h_in: mx.array | None = None
        if not self._pp.is_first:
            h_in = mx.zeros(
                (input_ids.shape[0], input_ids.shape[1], self._wire_hidden),
                dtype=self._wire_dtype,
            )
        return self._stage_forward(input_ids, h_in)

    def _stage_forward(
        self, input_ids: mx.array, h_in: mx.array | None, *, cache: Any = None
    ) -> mx.array:
        if self._pp.is_last:
            # full backbone + final norm + tied/explicit head -> model output
            return self._model(input_ids, cache=cache, input_embeddings=h_in)
        # backbone only (norm is nn.Identity on non-last) -> raw hidden state.
        # The downstream stage receives at the wire dtype (pipeline_recv); a
        # mismatch deadlocks the ring, which does not validate across peers.
        h_out = self._model.model(input_ids, cache=cache, input_embeddings=h_in)
        if h_out.dtype != self._wire_dtype:
            raise TypeError(
                f"PP stage produced hidden {h_out.dtype}, but the wire dtype "
                f"(stage compute dtype) is {self._wire_dtype}; mismatch "
                f"deadlocks the ring."
            )
        return h_out
