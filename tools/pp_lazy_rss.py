#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure per-stage peak memory for pipeline-parallel lazy loading and profiling.

Weight-materialization modes (evidence for the lazy PP load): a PP stage loads
the generic mlx_lm weights lazily and prunes its non-owned layers
(``apply_pipeline_split``) before the first ``mx.eval``, so per-stage peak stays
owned-only instead of full-model. This mirrors the load -> prune -> eval
mechanism ``ModelLifecycle`` uses on the ``pp.size > 1`` path (calling
``mlx_lm.utils.load_model`` directly and skipping the metadata-only lifecycle
install steps, which do not materialize weights).

    full        eager load, everything materializes (Phase-0 baseline)
    lazy        lazy load + prune, then materialize what remains

Dummy-forward modes (evidence for stage-shaped profiling): materialization is
forward-driven, so what profiling evals decides what a stage pays for. These
modes eval ONLY the dummy-forward outputs — never ``model.parameters()``:

    dummy-full   the old profile shape: full top-level model(input_ids) call,
                 embedding + logits on every stage
    dummy-stage  PipelinedModel.dummy_forward: zeros hidden in place of the
                 ring recv (no ring I/O, so a mid stage is measurable offline)

Run each mode in its OWN process so the high-water peak is isolated:

    python tools/pp_lazy_rss.py mlx-community/Qwen3-8B-4bit full
    python tools/pp_lazy_rss.py mlx-community/Qwen3-8B-4bit lazy 2 1
    python tools/pp_lazy_rss.py mlx-community/Qwen3-8B-4bit dummy-full 3 1
    python tools/pp_lazy_rss.py mlx-community/Qwen3-8B-4bit dummy-stage 3 1

Reports ``mx.get_peak_memory()`` (MLX allocator high-water) and ``ru_maxrss``
(process RSS).
"""

from __future__ import annotations

import resource
import sys
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm.utils import load_model

from vllm_metal.distributed.pipeline import (
    PipelinedModel,
    PipelineGroup,
    apply_pipeline_split,
)

# Any small nonzero length works: the dummy profiles the forward SHAPE (which
# modules materialize), not throughput, so the token count is arbitrary.
N_DUMMY_TOKENS = 8


class _OfflineStage:
    """Minimal PipelineGroup backing for an offline (rank, size) split."""

    def __init__(self, rank: int, size: int) -> None:
        self._rank, self._size = rank, size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _rss_gb() -> float:
    # macOS ru_maxrss is in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(
            "usage: pp_lazy_rss.py <hf_model_id> "
            "<full|lazy|dummy-full|dummy-stage> [pp_size] [rank]"
        )
    repo, mode = sys.argv[1], sys.argv[2]
    pp_size = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    rank = int(sys.argv[4]) if len(sys.argv) > 4 else pp_size - 1
    path = Path(snapshot_download(repo, local_files_only=True))

    if mode == "full":
        model, _ = load_model(path, lazy=False)
        total = owned = len(model.model.layers)
        mx.eval(model.parameters())
    elif mode in ("lazy", "dummy-full", "dummy-stage"):
        model, _ = load_model(path, lazy=True, strict=False)
        total = len(model.model.layers)
        pp = PipelineGroup(_OfflineStage(rank, pp_size))
        apply_pipeline_split(model, pp)
        owned = len(model.model.layers)
        if mode == "lazy":
            mx.eval(model.parameters())
        else:
            ids = mx.zeros((1, N_DUMMY_TOKENS), dtype=mx.int32)
            if mode == "dummy-full":
                mx.eval(model(ids))
            else:
                mx.eval(PipelinedModel(model, pp).dummy_forward(ids))
    else:
        sys.exit(f"unknown mode {mode!r}; use full|lazy|dummy-full|dummy-stage")

    print(
        f"{repo}  mode={mode}  rank={rank}/{pp_size}  owned_layers={owned}/{total}  "
        f"MLX_peak={mx.get_peak_memory() / 1e9:.3f}GB  "
        f"ru_maxrss={_rss_gb():.3f}GB"
    )


if __name__ == "__main__":
    main()
