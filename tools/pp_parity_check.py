#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pipeline-parallel parity harness for vLLM Metal (Phase 0).

Run modes (selected by the MLX group size, i.e. by the launcher):

  size == 1 (plain python)        -> REFERENCE producer.
      Computes full-model logits for a fixed prompt (plain mlx_lm forward, no
      paged attention) and saves them to /tmp/pp_ref_<safe_model>.npy.

  size  > 1 (mlx.launch -n N ring) -> PIPELINE consumer.
      Splits the model across stages, runs the per-stage recv -> layers -> send
      chain, and on the LAST rank loads the saved reference and asserts
      bit-close parity (tol 1e-4). Prints a single PARITY PASS / PARITY FAIL.

Usage:
    python tools/pp_parity_check.py [HF_MODEL_ID]            # reference
    mlx.launch -n 2 --backend ring tools/pp_parity_check.py [HF_MODEL_ID]
"""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load

from vllm_metal.distributed.pipeline import (
    PipelineGroup,
    apply_pipeline_split,
    pipeline_recv,
    pipeline_send,
)

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"
PARITY_TOL = 1e-4


def _ref_path(model_id: str) -> str:
    safe = model_id.replace("/", "__")
    return f"/tmp/pp_ref_{safe}.npy"


def _token_ids(tokenizer, prompt: str) -> list[int]:
    return tokenizer.encode(prompt)


def _hidden_size(model) -> int:
    return int(model.args.hidden_size)


def run_reference(model, tokenizer, model_id: str) -> None:
    """Full-model forward; save logits for the last token position."""
    ids = _token_ids(tokenizer, PROMPT)
    input_ids = mx.array([ids])
    logits = model(input_ids, cache=None)
    mx.eval(logits)
    logits_last = np.array(logits[0, -1].astype(mx.float32))

    path = _ref_path(model_id)
    np.save(path, logits_last)
    print(f"[ref] model={model_id} n_tokens={len(ids)} hidden={_hidden_size(model)}")
    print(
        f"[ref] logits shape={logits_last.shape} "
        f"min={logits_last.min():.4f} max={logits_last.max():.4f} "
        f"argmax={int(logits_last.argmax())}"
    )
    print(f"[ref] saved -> {path}")


def run_pipeline(model, tokenizer, model_id: str, pp: PipelineGroup) -> None:
    """Per-stage forward (recv -> local layers -> send); last rank checks parity."""
    span = apply_pipeline_split(model, pp)
    ids = _token_ids(tokenizer, PROMPT)
    input_ids = mx.array([ids])
    n_tokens = len(ids)
    hidden = _hidden_size(model)
    dtype = model.model.embed_tokens.weight.dtype

    print(
        f"[rank {pp.rank}/{pp.size}] is_first={pp.is_first} is_last={pp.is_last} "
        f"layers={span} n_tokens={n_tokens} hidden={hidden} dtype={dtype}"
    )

    # pipeline_send/recv own the batch-axis squeeze/unsqueeze, so the model
    # forward always sees the batched (1, n_tokens, hidden) while the wire
    # carries the compact 2D (n_tokens, hidden).
    if pp.is_first:
        h_in = None
    else:
        h_in = pipeline_recv(pp, n_tokens, hidden, dtype)

    if pp.is_last:
        # Full backbone (norm intact) + tied/explicit head -> logits.
        logits = model(input_ids, cache=None, input_embeddings=h_in)
        mx.eval(logits)
        logits_last = np.array(logits[0, -1].astype(mx.float32))

        ref = np.load(_ref_path(model_id))
        max_abs_diff = float(np.max(np.abs(logits_last - ref)))
        ref_argmax = int(ref.argmax())
        got_argmax = int(logits_last.argmax())
        print(
            f"[rank {pp.rank}] argmax got={got_argmax} ref={ref_argmax} "
            f"max_abs_diff={max_abs_diff:.3e}"
        )
        if max_abs_diff <= PARITY_TOL and got_argmax == ref_argmax:
            print(f"PARITY PASS max_abs_diff={max_abs_diff:.3e}")
        else:
            print(f"PARITY FAIL max_abs_diff={max_abs_diff:.3e}")
    else:
        # Raw hidden (norm == Identity on non-last) -> send downstream. The
        # wrapper drops the batch axis for the compact 2D wire.
        h_out = model.model(input_ids, cache=None, input_embeddings=h_in)
        sent = pipeline_send(h_out, pp)
        mx.eval(sent)
        print(f"[rank {pp.rank}] sent hidden {h_out.shape} -> rank {pp.rank + 1}")


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    group = mx.distributed.init()
    model, tokenizer = load(model_id)

    if group.size() == 1:
        run_reference(model, tokenizer, model_id)
    else:
        pp = PipelineGroup(group)
        run_pipeline(model, tokenizer, model_id, pp)


if __name__ == "__main__":
    main()
