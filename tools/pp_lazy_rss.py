#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure per-stage peak memory for a pipeline-parallel lazy load vs a full load.

Evidence for the lazy pipeline-parallel load: a PP stage loads the generic mlx_lm
weights lazily and prunes its non-owned layers (``apply_pipeline_split``) before the
first ``mx.eval``, so per-stage peak stays owned-only instead of full-model. This
mirrors the load -> prune -> eval mechanism ``ModelLifecycle`` uses on the
``pp.size > 1`` path (calling ``mlx_lm.utils.load_model`` directly and skipping the
metadata-only lifecycle install steps, which do not materialize weights).

Run each mode in its OWN process so the peak is isolated:

    python tools/pp_lazy_rss.py mlx-community/Qwen3-8B-4bit full
    python tools/pp_lazy_rss.py mlx-community/Qwen3-8B-4bit lazy   # last of pp_size=2

Reports ``mx.get_peak_memory()`` (MLX allocator high-water) and ``ru_maxrss``
(process RSS). If anything materialized the full model before the split, the lazy
peak would match the full peak.
"""

from __future__ import annotations

import resource
import sys
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm.utils import load_model

from vllm_metal.distributed.pipeline import PipelineGroup, apply_pipeline_split


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
        sys.exit("usage: pp_lazy_rss.py <hf_model_id> <full|lazy> [pp_size] [rank]")
    repo, mode = sys.argv[1], sys.argv[2]
    pp_size = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    rank = int(sys.argv[4]) if len(sys.argv) > 4 else pp_size - 1
    path = Path(snapshot_download(repo, local_files_only=True))

    if mode == "full":
        model, _ = load_model(path, lazy=False)
        total = owned = len(model.model.layers)
    elif mode == "lazy":
        model, _ = load_model(path, lazy=True, strict=False)
        total = len(model.model.layers)
        apply_pipeline_split(model, PipelineGroup(_OfflineStage(rank, pp_size)))
        owned = len(model.model.layers)
    else:
        sys.exit(f"unknown mode {mode!r}; use full or lazy")

    mx.eval(model.parameters())
    print(
        f"{repo}  mode={mode}  owned_layers={owned}/{total}  "
        f"MLX_peak={mx.get_peak_memory() / 1e9:.3f}GB  "
        f"ru_maxrss={_rss_gb():.3f}GB"
    )


if __name__ == "__main__":
    main()
