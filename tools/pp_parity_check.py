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
    PipelinedModel,
    PipelineGroup,
    apply_pipeline_split,
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

    print(
        f"[rank {pp.rank}/{pp.size}] is_first={pp.is_first} is_last={pp.is_last} "
        f"layers={span} n_tokens={len(ids)} hidden={_hidden_size(model)}"
    )

    # Drive the real PP wrapper the engine runner uses: it owns the stage forward
    # (recv -> local layers -> final norm + head on the last stage). The caller
    # owns the send, exactly as the runner does.
    stage_output = PipelinedModel(model, pp)(input_ids, cache=None)

    if pp.is_last:
        mx.eval(stage_output)
        logits_last = np.array(stage_output[0, -1].astype(mx.float32))

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
        # Runner owns the send: push the raw hidden state to the next stage.
        sent = pipeline_send(stage_output, pp)
        mx.eval(sent)
        print(
            f"[rank {pp.rank}] sent hidden {stage_output.shape} -> rank {pp.rank + 1}"
        )


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
