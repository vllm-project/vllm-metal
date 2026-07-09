# SPDX-License-Identifier: Apache-2.0
"""Numeric parity tests for the GLM-4.7-Flash native MTP head MLX model.

A self-contained numpy reference implements the upstream nextn math (enorm /
hnorm / eh_proj concat, position-0 embedding zeroing, absorbed-MLA attention with
traditional RoPE, noaux_tc sigmoid gating with e_score_correction_bias, shared
expert add, shared_head norm + head readout). The MLX head must match it within
fp32 tolerance, incremental slab forwards must equal a single multi-row forward
and a recompute-from-scratch, and ``convert_nextn_weights`` must produce the exact
module tree the model loads under a strict ``load_weights``.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_unflatten
from mlx_lm.models.cache import KVCache

from vllm_metal.v1.mtp_heads.glm4_moe_lite_mtp import (
    Glm4MoeLiteMTPArgs,
    Glm4MoeLiteMTPModel,
    convert_nextn_weights,
)

# Tiny dims (plan): hidden 64, kv_lora_rank 16, qk_rope 8, qk_nope 16, v_head 16,
# 4 experts top-2, n_group 1, topk_group 1, vocab 32. q_lora_rank set so the
# q_a_proj/q_a_layernorm/q_b_proj path (as in the real GLM-4.7-Flash) is exercised.
H = 64
NH = 4
QL = 32
KVL = 16
ROPE = 8
NOPE = 16
VH = 16
NROUT = 4
NSHR = 1
TOPK = 2
MOEI = 32
VOCAB = 32
SCALE_F = 1.5
EPS = 1e-5
THETA = 1_000_000.0
QHD = NOPE + ROPE


def _args() -> Glm4MoeLiteMTPArgs:
    return Glm4MoeLiteMTPArgs(
        vocab_size=VOCAB,
        hidden_size=H,
        intermediate_size=128,
        moe_intermediate_size=MOEI,
        num_attention_heads=NH,
        num_key_value_heads=NH,
        n_shared_experts=NSHR,
        n_routed_experts=NROUT,
        kv_lora_rank=KVL,
        q_lora_rank=QL,
        qk_rope_head_dim=ROPE,
        qk_nope_head_dim=NOPE,
        v_head_dim=VH,
        num_experts_per_tok=TOPK,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        routed_scaling_factor=SCALE_F,
        rms_norm_eps=EPS,
        rope_theta=THETA,
        norm_topk_prob=True,
    )


def _random_flat_weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.1).astype(np.float32)

    def norm(*shape: int) -> np.ndarray:
        return (1.0 + rnd(*shape)).astype(np.float32)

    weights: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": rnd(VOCAB, H),
        "model.enorm.weight": norm(H),
        "model.hnorm.weight": norm(H),
        "model.eh_proj.weight": rnd(H, 2 * H),
        "model.mtp_block.input_layernorm.weight": norm(H),
        "model.mtp_block.post_attention_layernorm.weight": norm(H),
        "model.mtp_block.self_attn.q_a_proj.weight": rnd(QL, H),
        "model.mtp_block.self_attn.q_a_layernorm.weight": norm(QL),
        "model.mtp_block.self_attn.q_b_proj.weight": rnd(NH * QHD, QL),
        "model.mtp_block.self_attn.kv_a_proj_with_mqa.weight": rnd(KVL + ROPE, H),
        "model.mtp_block.self_attn.kv_a_layernorm.weight": norm(KVL),
        "model.mtp_block.self_attn.kv_b_proj.weight": rnd(NH * (NOPE + VH), KVL),
        "model.mtp_block.self_attn.o_proj.weight": rnd(H, NH * VH),
        "model.mtp_block.mlp.gate.weight": rnd(NROUT, H),
        "model.mtp_block.mlp.gate.e_score_correction_bias": rnd(NROUT),
        "model.mtp_block.mlp.shared_experts.gate_proj.weight": rnd(MOEI, H),
        "model.mtp_block.mlp.shared_experts.up_proj.weight": rnd(MOEI, H),
        "model.mtp_block.mlp.shared_experts.down_proj.weight": rnd(H, MOEI),
        "model.shared_head_norm.weight": norm(H),
        "lm_head.weight": rnd(VOCAB, H),
    }
    for e in range(NROUT):
        weights[f"model.mtp_block.mlp.experts.{e}.gate_proj.weight"] = rnd(MOEI, H)
        weights[f"model.mtp_block.mlp.experts.{e}.up_proj.weight"] = rnd(MOEI, H)
        weights[f"model.mtp_block.mlp.experts.{e}.down_proj.weight"] = rnd(H, MOEI)
    return weights


def _rms(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (x / np.sqrt((x**2).mean(-1, keepdims=True) + EPS)) * w


def _rope_traditional(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    # x: (T, num_heads, ROPE) — interleaved-pair rotation (mlx traditional=True).
    d = x.shape[-1]
    half = d // 2
    inv = THETA ** (-(2 * np.arange(half)) / d)
    ang = np.outer(pos, inv)
    cos = np.cos(ang)[:, None, :]
    sin = np.sin(ang)[:, None, :]
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _reference_forward(
    weights: dict[str, np.ndarray],
    token_ids: np.ndarray,
    hidden_rows: np.ndarray,
    first_position: int,
) -> tuple[np.ndarray, np.ndarray]:
    def w(name: str) -> np.ndarray:
        return weights[name]

    num_slots = len(token_ids)
    positions = first_position + np.arange(num_slots)

    emb = w("model.embed_tokens.weight")[token_ids]
    emb = emb * (positions != 0)[:, None]
    fused = np.concatenate(
        [
            _rms(emb, w("model.enorm.weight")),
            _rms(hidden_rows, w("model.hnorm.weight")),
        ],
        axis=-1,
    )
    x = fused @ w("model.eh_proj.weight").T

    # --- absorbed MLA attention (un-absorbed reference form) ---
    residual = x
    xin = _rms(x, w("model.mtp_block.input_layernorm.weight"))
    qa = _rms(
        xin @ w("model.mtp_block.self_attn.q_a_proj.weight").T,
        w("model.mtp_block.self_attn.q_a_layernorm.weight"),
    )
    q = (qa @ w("model.mtp_block.self_attn.q_b_proj.weight").T).reshape(
        num_slots, NH, QHD
    )
    q_nope, q_pe = q[..., :NOPE], q[..., NOPE:]
    compressed = xin @ w("model.mtp_block.self_attn.kv_a_proj_with_mqa.weight").T
    kv_c, k_pe = compressed[:, :KVL], compressed[:, KVL:]
    kv_latent = _rms(kv_c, w("model.mtp_block.self_attn.kv_a_layernorm.weight"))

    q_pe = _rope_traditional(q_pe, positions)
    k_pe = _rope_traditional(k_pe[:, None, :], positions)[:, 0, :]

    kv_b = (kv_latent @ w("model.mtp_block.self_attn.kv_b_proj.weight").T).reshape(
        num_slots, NH, NOPE + VH
    )
    k_nope, value = kv_b[..., :NOPE], kv_b[..., NOPE:]
    k_pe_heads = np.broadcast_to(k_pe[:, None, :], (num_slots, NH, ROPE))
    keys = np.concatenate([k_nope, k_pe_heads], axis=-1)
    queries = np.concatenate([q_nope, q_pe], axis=-1)

    scale = QHD**-0.5
    causal = np.triu(np.ones((num_slots, num_slots), dtype=bool), 1)
    head_outs = []
    for h in range(NH):
        scores = (queries[:, h, :] @ keys[:, h, :].T) * scale
        scores = np.where(causal, -np.inf, scores)
        scores = scores - scores.max(-1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(-1, keepdims=True)
        head_outs.append(probs @ value[:, h, :])
    attn = np.concatenate(head_outs, axis=-1)
    attn_out = attn @ w("model.mtp_block.self_attn.o_proj.weight").T
    hidden = residual + attn_out

    # --- MoE (noaux_tc sigmoid gating + shared expert) ---
    residual = hidden
    xmlp = _rms(hidden, w("model.mtp_block.post_attention_layernorm.weight"))
    gates = xmlp @ w("model.mtp_block.mlp.gate.weight").T
    sig = 1.0 / (1.0 + np.exp(-gates))
    biased = sig + w("model.mtp_block.mlp.gate.e_score_correction_bias")

    routed = np.zeros((num_slots, H), dtype=np.float32)
    for t in range(num_slots):
        idx = np.argsort(-biased[t])[:TOPK]
        weight = sig[t, idx].copy()
        weight = weight / (weight.sum() + 1e-20)
        weight = weight * SCALE_F
        acc = np.zeros(H, dtype=np.float32)
        for k, e in enumerate(idx):
            gate = xmlp[t] @ w(f"model.mtp_block.mlp.experts.{e}.gate_proj.weight").T
            up = xmlp[t] @ w(f"model.mtp_block.mlp.experts.{e}.up_proj.weight").T
            expert = (_silu(gate) * up) @ w(
                f"model.mtp_block.mlp.experts.{e}.down_proj.weight"
            ).T
            acc += weight[k] * expert
        routed[t] = acc

    shared_gate = xmlp @ w("model.mtp_block.mlp.shared_experts.gate_proj.weight").T
    shared_up = xmlp @ w("model.mtp_block.mlp.shared_experts.up_proj.weight").T
    shared = (_silu(shared_gate) * shared_up) @ w(
        "model.mtp_block.mlp.shared_experts.down_proj.weight"
    ).T
    hidden = residual + routed + shared

    out_hidden = _rms(hidden, w("model.shared_head_norm.weight"))
    logits = out_hidden @ w("lm_head.weight").T
    return out_hidden, logits


def _build_model(
    weights: dict[str, np.ndarray],
) -> Glm4MoeLiteMTPModel:
    model = Glm4MoeLiteMTPModel(_args())
    converted = convert_nextn_weights(
        {k: mx.array(v) for k, v in weights.items()},
        model.args,
    )
    model.update(tree_unflatten(list(converted.items())))
    mx.eval(model.parameters())
    return model


@pytest.fixture()
def fixture():  # noqa: ANN201
    rng = np.random.default_rng(1234)
    weights = _random_flat_weights(rng)
    model = _build_model(weights)
    token_ids = rng.integers(0, VOCAB, size=6).astype(np.int32)
    hidden_rows = (rng.standard_normal((6, H)) * 0.5).astype(np.float32)
    return model, weights, token_ids, hidden_rows


def test_forward_matches_numpy_reference(fixture) -> None:  # noqa: ANN001
    model, weights, token_ids, hidden_rows = fixture

    cache = KVCache()
    x = model.build_slot_inputs(mx.array(token_ids), mx.array(hidden_rows), 0)
    hidden = model.forward_slots(x, cache)
    logits = model.compute_logits(hidden)
    mx.eval(hidden, logits)

    ref_hidden, ref_logits = _reference_forward(weights, token_ids, hidden_rows, 0)

    assert np.allclose(np.array(hidden), ref_hidden, atol=2e-4, rtol=2e-4)
    assert np.allclose(np.array(logits), ref_logits, atol=2e-4, rtol=2e-4)
    assert (
        np.array(mx.argmax(logits, axis=-1)).tolist() == ref_logits.argmax(-1).tolist()
    )


def test_position_zero_embedding_is_masked(fixture) -> None:  # noqa: ANN001
    model, weights, token_ids, hidden_rows = fixture

    # Row at absolute position 0 must ignore its token embedding: swapping the
    # position-0 token cannot change the slot input.
    x0 = model.build_slot_inputs(mx.array(token_ids), mx.array(hidden_rows), 0)
    swapped = token_ids.copy()
    swapped[0] = (int(swapped[0]) + 7) % VOCAB
    x1 = model.build_slot_inputs(mx.array(swapped), mx.array(hidden_rows), 0)
    mx.eval(x0, x1)
    assert np.allclose(np.array(x0[0]), np.array(x1[0]), atol=1e-6)

    # But with a non-zero first_position, row 0 is a real position and does use
    # its embedding.
    x2 = model.build_slot_inputs(mx.array(token_ids), mx.array(hidden_rows), 3)
    x3 = model.build_slot_inputs(mx.array(swapped), mx.array(hidden_rows), 3)
    mx.eval(x2, x3)
    assert not np.allclose(np.array(x2[0]), np.array(x3[0]), atol=1e-6)


def test_incremental_slots_equal_full_forward(fixture) -> None:  # noqa: ANN001
    model, _weights, token_ids, hidden_rows = fixture
    num_slots = len(token_ids)

    full_cache = KVCache()
    x_full = model.build_slot_inputs(mx.array(token_ids), mx.array(hidden_rows), 0)
    full = model.forward_slots(x_full, full_cache, expected_offset=0)
    mx.eval(full)
    full_np = np.array(full)

    inc_cache = KVCache()
    rows = []
    for i in range(num_slots):
        xi = model.build_slot_inputs(
            mx.array(token_ids[i : i + 1]),
            mx.array(hidden_rows[i : i + 1]),
            i,
        )
        # The cache offset and the caller's first_position must agree per slot.
        hi = model.forward_slots(xi, inc_cache, expected_offset=i)
        mx.eval(hi)
        rows.append(np.array(hi))
    incremental = np.concatenate(rows, axis=0)

    assert incremental.shape == full_np.shape
    assert np.allclose(incremental, full_np, atol=2e-4, rtol=2e-4)
    assert inc_cache.offset == num_slots


def test_split_prefill_then_decode_equals_full(fixture) -> None:  # noqa: ANN001
    model, _weights, token_ids, hidden_rows = fixture
    num_slots = len(token_ids)

    x_full = model.build_slot_inputs(mx.array(token_ids), mx.array(hidden_rows), 0)
    full = model.forward_slots(x_full, KVCache())
    mx.eval(full)
    full_np = np.array(full)

    # Prefill the first `split` rows in one shot, then feed the remainder one at
    # a time, mirroring a decode-after-prefill sequence.
    split = 4
    cache = KVCache()
    x_prefill = model.build_slot_inputs(
        mx.array(token_ids[:split]), mx.array(hidden_rows[:split]), 0
    )
    prefill = model.forward_slots(x_prefill, cache, expected_offset=0)
    mx.eval(prefill)
    rows = [np.array(prefill)]
    for i in range(split, num_slots):
        xi = model.build_slot_inputs(
            mx.array(token_ids[i : i + 1]),
            mx.array(hidden_rows[i : i + 1]),
            i,
        )
        hi = model.forward_slots(xi, cache, expected_offset=i)
        mx.eval(hi)
        rows.append(np.array(hi))
    combined = np.concatenate(rows, axis=0)

    assert np.allclose(combined, full_np, atol=2e-4, rtol=2e-4)


def test_convert_nextn_weights_round_trips_into_model() -> None:
    rng = np.random.default_rng(7)
    weights = _random_flat_weights(rng)
    model = Glm4MoeLiteMTPModel(_args())

    converted = convert_nextn_weights(
        {k: mx.array(v) for k, v in weights.items()},
        model.args,
    )

    # The transformed weights must be exactly the module tree — strict load
    # rejects any missing or extra key.
    model.load_weights(list(converted.items()), strict=True)

    # kv_b_proj is consumed; embed_q / unembed_out are produced with the right
    # absorbed shapes; experts are stacked into switch_mlp.
    assert "model.mtp_block.self_attn.kv_b_proj.weight" not in converted
    assert converted["model.mtp_block.self_attn.embed_q.weight"].shape == (
        NH,
        KVL,
        NOPE,
    )
    assert converted["model.mtp_block.self_attn.unembed_out.weight"].shape == (
        NH,
        VH,
        KVL,
    )
    assert converted["model.mtp_block.mlp.switch_mlp.gate_proj.weight"].shape == (
        NROUT,
        MOEI,
        H,
    )
    assert "model.mtp_block.mlp.experts.0.gate_proj.weight" not in converted


def test_convert_nextn_weights_is_idempotent() -> None:
    rng = np.random.default_rng(9)
    weights = _random_flat_weights(rng)
    args = _args()

    once = convert_nextn_weights({k: mx.array(v) for k, v in weights.items()}, args)
    twice = convert_nextn_weights(once, args)

    assert set(once) == set(twice)
    for key in once:
        assert np.array_equal(np.array(once[key]), np.array(twice[key]))


def test_convert_nextn_weights_reports_missing_expert() -> None:
    # A checkpoint with fewer experts than n_routed_experts must fail with a
    # descriptive error (naming the first missing expert + the extraction tool),
    # not a bare KeyError leaking from the stacking pop.
    rng = np.random.default_rng(3)
    weights = _random_flat_weights(rng)
    args = _args()
    dropped = f"model.mtp_block.mlp.experts.{NROUT - 1}.gate_proj.weight"
    del weights[dropped]

    with pytest.raises(ValueError) as excinfo:
        convert_nextn_weights({k: mx.array(v) for k, v in weights.items()}, args)

    message = str(excinfo.value)
    assert f"missing expert {NROUT - 1}" in message
    assert dropped in message
    assert f"n_routed_experts={NROUT}" in message
    assert "extract_glm47_mtp_head.py" in message


def test_forward_slots_expected_offset_guards_against_desync(fixture) -> None:  # noqa: ANN001
    model, _weights, token_ids, hidden_rows = fixture

    cache = KVCache()
    x = model.build_slot_inputs(mx.array(token_ids), mx.array(hidden_rows), 0)
    # Fresh cache is at offset 0 — matching expected_offset passes.
    first = model.forward_slots(x, cache, expected_offset=0)
    mx.eval(first)
    assert cache.offset == len(token_ids)

    # Re-using the now-advanced cache while still claiming offset 0 is a
    # position anchor desync and must fail loud.
    x2 = model.build_slot_inputs(mx.array(token_ids[:1]), mx.array(hidden_rows[:1]), 0)
    with pytest.raises(ValueError, match="does not match expected_offset"):
        model.forward_slots(x2, cache, expected_offset=0)


# ---------------------------------------------------------------------------
# Quantized weight transform (convert_nextn_weights on a 4-bit checkpoint)
# ---------------------------------------------------------------------------

# mx.quantize requires group_size in {32, 64, 128} and the quantized last dim
# divisible by it, so the quantized fixture uses coarser dims than the fp32
# tests above.
QH = 32  # hidden / head dims (all multiples of the group size)
QNH = 2  # attention heads
QKVL = 32  # kv_lora_rank
QNOPE = 32  # qk_nope_head_dim
QVH = 32  # v_head_dim
QNROUT = 2  # routed experts
QMOEI = 32  # moe intermediate
QBITS = 4
QGROUP = 32


def _quant_args() -> Glm4MoeLiteMTPArgs:
    return Glm4MoeLiteMTPArgs(
        vocab_size=64,
        hidden_size=QH,
        intermediate_size=64,
        moe_intermediate_size=QMOEI,
        num_attention_heads=QNH,
        num_key_value_heads=QNH,
        n_shared_experts=1,
        n_routed_experts=QNROUT,
        kv_lora_rank=QKVL,
        q_lora_rank=QH,
        qk_rope_head_dim=QGROUP,
        qk_nope_head_dim=QNOPE,
        v_head_dim=QVH,
        num_experts_per_tok=1,
        first_k_dense_replace=1,
        moe_layer_freq=1,
    )


def _quantize(w: np.ndarray) -> tuple[mx.array, mx.array, mx.array]:
    q, s, b = mx.quantize(mx.array(w), bits=QBITS, group_size=QGROUP)
    mx.eval(q, s, b)
    return q, s, b


def _dequant(q: mx.array, s: mx.array, b: mx.array) -> np.ndarray:
    d = mx.dequantize(q, s, b, bits=QBITS, group_size=QGROUP)
    mx.eval(d)
    return np.array(d)


def test_convert_nextn_weights_quantized_kv_b_proj_round_trips() -> None:
    rng = np.random.default_rng(11)
    prefix = "model.mtp_block.self_attn"
    # kv_b_proj: (num_heads * (qk_nope + v_head), kv_lora_rank).
    kv_b_fp32 = (rng.standard_normal((QNH * (QNOPE + QVH), QKVL)) * 0.1).astype(
        np.float32
    )
    q, s, b = _quantize(kv_b_fp32)
    weights = {
        f"{prefix}.kv_b_proj.weight": q,
        f"{prefix}.kv_b_proj.scales": s,
        f"{prefix}.kv_b_proj.biases": b,
    }

    out = convert_nextn_weights(weights, _quant_args())

    # Fused kv_b_proj is consumed; absorbed embed_q / unembed_out appear with
    # quantized triplets.
    assert f"{prefix}.kv_b_proj.weight" not in out
    for name in ("embed_q", "unembed_out"):
        for suffix in ("weight", "scales", "biases"):
            assert f"{prefix}.{name}.{suffix}" in out
    assert out[f"{prefix}.embed_q.weight"].shape[0] == QNH
    assert out[f"{prefix}.unembed_out.weight"].shape[0] == QNH

    # Numeric round-trip: reproduce the reference by splitting the *unquantized*
    # weights and quantizing, then compare dequantized values (4-bit tolerance).
    v = kv_b_fp32.reshape(QNH, QNOPE + QVH, QKVL)
    wk_ref = np.ascontiguousarray(v[:, :QNOPE, :].swapaxes(-1, -2))
    wv_ref = np.ascontiguousarray(v[:, QNOPE:, :])
    ek = _dequant(*_quantize(wk_ref))
    ev = _dequant(*_quantize(wv_ref))
    got_ek = _dequant(
        out[f"{prefix}.embed_q.weight"],
        out[f"{prefix}.embed_q.scales"],
        out[f"{prefix}.embed_q.biases"],
    )
    got_ev = _dequant(
        out[f"{prefix}.unembed_out.weight"],
        out[f"{prefix}.unembed_out.scales"],
        out[f"{prefix}.unembed_out.biases"],
    )
    # The convert path quantizes -> dequantizes -> splits -> re-quantizes (two
    # 4-bit rounds) vs the reference's single round on the exact fp32 split, so
    # compare dequantized values within a 4-bit-appropriate tolerance.
    assert np.allclose(got_ek, ek, atol=0.08)
    assert np.allclose(got_ev, ev, atol=0.08)


def test_convert_nextn_weights_stacks_quantized_experts() -> None:
    rng = np.random.default_rng(13)
    prefix = "model.mtp_block.mlp"
    per_expert: dict[str, mx.array] = {}
    expected: dict[str, list[tuple[mx.array, mx.array, mx.array]]] = {
        "gate_proj": [],
        "up_proj": [],
        "down_proj": [],
    }
    shapes = {
        "gate_proj": (QMOEI, QH),
        "up_proj": (QMOEI, QH),
        "down_proj": (QH, QMOEI),
    }
    for e in range(QNROUT):
        for proj, shape in shapes.items():
            w = (rng.standard_normal(shape) * 0.1).astype(np.float32)
            q, s, b = _quantize(w)
            per_expert[f"{prefix}.experts.{e}.{proj}.weight"] = q
            per_expert[f"{prefix}.experts.{e}.{proj}.scales"] = s
            per_expert[f"{prefix}.experts.{e}.{proj}.biases"] = b
            expected[proj].append((q, s, b))

    out = convert_nextn_weights(per_expert, _quant_args())

    for proj in ("gate_proj", "up_proj", "down_proj"):
        # Per-expert keys are consumed.
        assert f"{prefix}.experts.0.{proj}.weight" not in out
        for si, suffix in enumerate(("weight", "scales", "biases")):
            stacked = out[f"{prefix}.switch_mlp.{proj}.{suffix}"]
            assert stacked.shape[0] == QNROUT
            # Stacking preserves each expert's quantized tensor exactly (no
            # dequant): row e equals expert e's tensor.
            for e in range(QNROUT):
                np.testing.assert_array_equal(
                    np.array(stacked[e]), np.array(expected[proj][e][si])
                )
