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
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.core.distributed import Group
from vllm.distributed.utils import get_pp_indices

logger = logging.getLogger(__name__)

# Base TCP port for the MLX ring data plane (matches the mlx.launch default
# ``starting_port``). Rank ``r`` listens on ``_MLX_RING_BASE_PORT + r`` so
# co-located ranks (multiple stages on one Mac) still get distinct ports.
_MLX_RING_BASE_PORT = 32323


def _ring_hosts(peer_ips: list[str]) -> list[list[str]]:
    """Build the MLX ring hostfile body from per-rank node IPs.

    ``peer_ips[r]`` is the IP of the node hosting rank ``r`` — identical on a
    single node, distinct across Macs. The ring backend wants a bare list of
    ``["ip:port"]`` lists, one per rank in rank order.
    """
    return [[f"{ip}:{_MLX_RING_BASE_PORT + r}"] for r, ip in enumerate(peer_ips)]


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

        # mx.distributed.init() forms the real ring group from the MLX_RANK /
        # MLX_HOSTFILE env set above (a singleton size-1 group in plain python).
        pp = cls(mx.distributed.init())
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

    def __call__(self, input_ids: mx.array, *, cache: Any = None) -> mx.array:
        pp = self._pp
        h_in: mx.array | None = None
        if not pp.is_first:
            embed = self._model.model.embed_tokens.weight
            h_in = pipeline_recv(pp, input_ids.shape[1], embed.shape[-1], embed.dtype)
        if pp.is_last:
            # full backbone + final norm + tied/explicit head -> model output
            return self._model(input_ids, cache=cache, input_embeddings=h_in)
        # backbone only (norm is nn.Identity on non-last) -> raw hidden state.
        # The downstream stage receives at its embed-weight dtype (pipeline_recv);
        # a mismatch deadlocks the ring, which does not validate across peers.
        h_out = self._model.model(input_ids, cache=cache, input_embeddings=h_in)
        embed_dtype = self._model.model.embed_tokens.weight.dtype
        if h_out.dtype != embed_dtype:
            raise TypeError(
                f"PP stage produced hidden {h_out.dtype}, but the wire dtype is "
                f"the embed-weight dtype {embed_dtype}; mismatch deadlocks the ring."
            )
        return h_out
