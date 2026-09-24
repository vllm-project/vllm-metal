# SPDX-License-Identifier: Apache-2.0
"""Self-consistency tests for the Rust/ggml engine on real weights.

Skipped unless the extension is built (``python -m vllm_metal.ggml.build``)
and the checkpoint is already in the local HF cache. HF-reference parity is
covered by ``tools/ggml_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow

BLOCK = 16
MODEL = "Qwen/Qwen3.5-0.8B"
PROMPT = [
    760,
    6511,
    314,
    9338,
    369,
    12095,
    13,
    576,
    6511,
    314,
    9500,
    369,
    11,
    32,
    7,
    99,
]


def _load_engine():
    try:
        from vllm_metal.ggml import load_extension

        ext = load_extension()
    except ImportError as e:
        pytest.skip(f"ggml engine not built: {e}")
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip(f"{MODEL} not in the local HF cache")
    eng = ext.Engine(path)
    eng.init_cache(64, BLOCK, 8)
    return eng


@pytest.fixture(scope="module")
def engine():
    return _load_engine()


def _a(x) -> np.ndarray:
    return np.asarray(x, np.int32)


def _run(eng, seqs):
    """seqs: list of (tokens, ctx_len, blocks, slot, reset, n_logits)."""
    vocab = eng.info()["vocab_size"]
    width = max(len(s[2]) for s in seqs)
    bt = np.zeros((len(seqs), width), np.int32)
    for i, s in enumerate(seqs):
        bt[i, : len(s[2])] = s[2]
    rows = sum(s[5] for s in seqs)
    out = np.empty((max(rows, 1), vocab), np.float32)
    toks = [t for s in seqs for t in s[0]]
    n = eng.forward(
        _a(toks),
        _a([len(s[0]) for s in seqs]),
        _a([s[1] for s in seqs]),
        bt,
        _a([s[3] for s in seqs]),
        _a([int(s[4]) for s in seqs]),
        _a([s[5] for s in seqs]),
        out,
    )
    assert n == rows
    return out[:rows]


def _kl(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    def lsm(x):
        x = x.astype(np.float64)
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))

    lp, lq = lsm(p_logits), lsm(q_logits)
    return float((np.exp(lp) * (lp - lq)).sum(-1).max())


def test_info(engine) -> None:
    info = engine.info()
    assert info["arch"] == "qwen3_5"
    assert len(info["kv_layers"]) == 6 and len(info["state_layers"]) == 18
    assert info["backend"].startswith("MTL")


def test_chunked_prefill_and_decode_match_full_prefill(engine) -> None:
    n = len(PROMPT)
    full = _run(engine, [(PROMPT, n, [1], 1, True, n)])
    rows = [_run(engine, [(PROMPT[:5], 5, [2], 2, True, 5)])]
    rows.append(_run(engine, [(PROMPT[5:9], 9, [2], 2, False, 4)]))
    for i in range(9, n):
        rows.append(_run(engine, [(PROMPT[i : i + 1], i + 1, [2], 2, False, 1)]))
    chunked = np.concatenate(rows)
    assert (full.argmax(-1) == chunked.argmax(-1)).all()
    assert _kl(full, chunked) < 1e-2


def test_batched_sequences_are_independent(engine) -> None:
    n = len(PROMPT)
    single = _run(engine, [(PROMPT, n, [3], 3, True, 1)])
    other = [5, 6, 7, 8, 9, 10, 11]
    batch = _run(
        engine,
        [
            (other, len(other), [4], 4, True, 1),
            (PROMPT, n, [5], 5, True, 1),
            (PROMPT[:10], 10, [6], 6, True, 1),
        ],
    )
    assert batch[1].argmax() == single[0].argmax()
    assert _kl(single, batch[1:2]) < 1e-3


def test_reset_state_reuses_slot_cleanly(engine) -> None:
    n = len(PROMPT)
    first = _run(engine, [(PROMPT, n, [7], 7, True, 1)])
    # pollute slot 7 with a different sequence, then reuse it with reset
    _run(engine, [([1, 2, 3], 3, [7], 7, True, 1)])
    again = _run(engine, [(PROMPT, n, [7], 7, True, 1)])
    np.testing.assert_allclose(first, again, rtol=0, atol=1e-4)


def test_invalid_block_table_is_rejected(engine) -> None:
    with pytest.raises(RuntimeError, match="block"):
        _run(engine, [(PROMPT, len(PROMPT), [10_000], 1, True, 1)])
