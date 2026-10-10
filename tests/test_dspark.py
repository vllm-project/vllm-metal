# SPDX-License-Identifier: Apache-2.0
"""DSpark block alignment and checkpoint heads, independent of serving."""

import json
from dataclasses import asdict, replace

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from mlx.utils import tree_flatten

from tests.test_dflash import _config, _target_config, _torch_forward
from vllm_metal.v1.draft_checkpoint import COMMON_DRAFT_OPTIONS, load_draft_weights
from vllm_metal.v1.dspark import DSparkConfig, DSparkModel, load_dspark


def config(**kwargs):
    return DSparkConfig(
        backbone=replace(_config(), block_size=7, target_layer_ids=(0, 2, 3)),
        markov_rank=4,
        enable_confidence_head=kwargs.get("confidence", True),
        confidence_head_with_markov=kwargs.get("with_markov", True),
    )


def raw_config(cfg):
    raw = asdict(cfg.backbone)
    raw.update(
        architectures=["Qwen3DSparkModel"],
        model_type="qwen3",
        markov_head_type="vanilla",
        markov_rank=cfg.markov_rank,
        enable_confidence_head=cfg.enable_confidence_head,
        confidence_head_with_markov=cfg.confidence_head_with_markov,
    )
    return raw


def array(x):
    return np.array(x.astype(mx.float32))


@pytest.mark.parametrize(
    "dtype,tolerance", [(mx.float32, 2e-5), (mx.float16, 0.006), (mx.bfloat16, 0.04)]
)
@pytest.mark.parametrize("batch,length,width", [(1, 1, 1), (2, 17, 3), (2, 33, 7)])
def test_backbone_uses_anchor_slot_and_matches_independent_attention(
    dtype, tolerance, batch, length, width
):
    model = DSparkModel(config())
    model.set_dtype(dtype)
    features = [mx.random.normal((batch, length, 32)).astype(dtype) for _ in range(3)]
    anchors = mx.arange(batch, dtype=mx.int64) + 3
    ids = mx.concatenate(
        [anchors[:, None], mx.full((batch, width - 1), 63, dtype=mx.int64)], axis=1
    )
    expected = _torch_forward(model.backbone, model.embed_tokens(ids), features)
    hidden = model.block_hidden(anchors, features, num_draft_tokens=width)
    np.testing.assert_allclose(array(hidden), expected, atol=tolerance, rtol=tolerance)
    tokens, logits, confidence = model.draft(anchors, features, num_draft_tokens=width)
    assert tokens.shape == (batch, width)
    assert logits.shape == (batch, width, 64)
    assert confidence.shape == (batch, width)
    assert confidence.dtype == mx.float32
    # Causal block attention is a negative control, not the trained mask.
    if width > 1:
        causal = _torch_forward(
            model.backbone, model.embed_tokens(ids), features, causal=True
        )
        assert not np.allclose(array(hidden), causal, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    "confidence,with_markov", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("draft_topk", [None, 1, 8, 64])
def test_greedy_heads_match_explicit_torch_math_and_compiled_replay(
    confidence, with_markov, draft_topk
):
    model = DSparkModel(config(confidence=confidence, with_markov=with_markov))
    weights = {
        name: torch.from_numpy(array(value))
        for name, value in tree_flatten(model.parameters())
    }
    compiled = mx.compile(
        lambda hidden, anchors: model.greedy_proposal(
            hidden, anchors, draft_topk=draft_topk
        )
    )
    for offset in (0, 11):
        hidden = mx.random.normal((2, 7, 32))
        anchors = mx.array([1 + offset, 4 + offset])
        base = functional.linear(
            torch.from_numpy(array(hidden)), weights["lm_head.weight"]
        )
        previous = torch.from_numpy(np.array(anchors).astype(np.int64))
        expected_logits, expected_tokens, predecessors = [], [], []
        for i in range(7):
            predecessors.append(previous)
            correction = functional.linear(
                weights["markov_head.markov_w1.weight"][previous],
                weights["markov_head.markov_w2.weight"],
            )
            step = base[:, i] + correction
            if draft_topk is not None:
                # Independent dense correction followed by candidate masking.
                selected = base[:, i].topk(draft_topk, dim=-1).indices
                masked = torch.full_like(step, -torch.inf)
                step = masked.scatter(-1, selected, step.gather(-1, selected))
            previous = step.argmax(-1)
            expected_logits.append(step)
            expected_tokens.append(previous)
        for result in (
            model.greedy_proposal(hidden, anchors, draft_topk=draft_topk),
            compiled(hidden, anchors),
        ):
            tokens, logits, actual_confidence = result
            np.testing.assert_array_equal(
                np.array(tokens), torch.stack(expected_tokens, 1).numpy()
            )
            np.testing.assert_allclose(
                array(logits),
                torch.stack(expected_logits, 1).numpy(),
                atol=1e-5,
                rtol=1e-5,
            )
            if confidence:
                inputs = torch.from_numpy(array(hidden))
                if with_markov:
                    inputs = torch.cat(
                        [
                            inputs,
                            weights["markov_head.markov_w1.weight"][
                                torch.stack(predecessors, 1)
                            ],
                        ],
                        -1,
                    )
                expected = functional.linear(
                    inputs,
                    weights["confidence_head.proj.weight"],
                    weights["confidence_head.proj.bias"],
                ).squeeze(-1)
                np.testing.assert_allclose(
                    array(actual_confidence), expected.numpy(), atol=1e-5, rtol=1e-5
                )
            else:
                assert actual_confidence is None


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_full_vocabulary_topk_preserves_default_outputs(dtype):
    model = DSparkModel(config())
    model.set_dtype(dtype)
    hidden = mx.random.normal((2, 7, 32)).astype(dtype)
    anchors = mx.array([3, 5])
    default = model.greedy_proposal(hidden, anchors)
    full = model.greedy_proposal(hidden, anchors, draft_topk=64)
    for actual, expected in zip(full, default, strict=True):
        np.testing.assert_array_equal(array(actual), array(expected))


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_candidate_limit_excludes_outside_winner_and_preserves_token_id_ties(dtype):
    model = DSparkModel(config())
    base = mx.full((64, 32), -10.0)
    base[2, 0], base[5, 0] = 5, 4
    model.lm_head.weight = base
    embedding = mx.zeros((64, 4))
    embedding[3, 0], embedding[5, 0], embedding[2, 1] = 1, -1, 1
    model.markov_head.markov_w1.weight = embedding
    correction = mx.zeros((64, 4))
    correction[5, :2] = mx.array([2, -2])
    correction[7, :2] = 100
    model.markov_head.markov_w2.weight = correction
    model.confidence_head.proj.weight = mx.zeros((1, 36))
    model.confidence_head.proj.weight[0, 32:34] = mx.array([1, 2])
    model.confidence_head.proj.bias = mx.zeros((1,))
    model.set_dtype(dtype)
    hidden = mx.zeros((1, 3, 32), dtype=dtype)
    hidden[..., 0] = 1
    anchor = mx.array([3])
    tokens, logits, confidence = model.greedy_proposal(hidden, anchor, draft_topk=2)
    assert tokens.tolist() == [[5, 2, 2]]
    assert confidence.tolist() == [[1, -1, 2]]
    assert mx.isneginf(logits[..., 7]).all().item()
    assert model.greedy_proposal(hidden, anchor)[0][0, 0].item() == 7
    assert model.greedy_proposal(hidden, anchor, draft_topk=1)[0].tolist() == [[2] * 3]
    # Corrected candidates 2 and 5 tie: choose vocabulary ID 2, not the
    # candidate's position in an unspecified partition ordering.
    model.markov_head.markov_w2.weight[5, 0] = 1
    assert model.greedy_proposal(hidden, anchor, draft_topk=2)[0][0, 0].item() == 2


def test_topk_draft_skips_dense_corrected_logits_when_disabled(monkeypatch):
    model = DSparkModel(config())
    model.set_dtype(mx.float32)
    hidden = mx.random.normal((1, 7, 32))
    anchors = mx.array([3])
    calls = 0
    real_full_like = mx.full_like

    def counting_full_like(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_full_like(*args, **kwargs)

    monkeypatch.setattr(mx, "full_like", counting_full_like)

    tokens, logits, _ = model.greedy_proposal(
        hidden, anchors, draft_topk=2, corrected_logits=False
    )
    assert logits is None
    assert calls == 0

    dense_tokens, dense_logits, _ = model.greedy_proposal(hidden, anchors, draft_topk=2)
    assert dense_logits is not None
    assert calls == 7
    assert mx.array_equal(tokens, dense_tokens).item()


@pytest.mark.parametrize(
    "draft_topk", [np.int32(1), np.int64(8), np.uint64(8), np.int64(64)]
)
def test_candidate_limit_accepts_integral_types(draft_topk):
    model = DSparkModel(config())
    hidden = mx.random.normal((2, 7, 32))
    anchors = mx.array([3, 5])
    expected = model.greedy_proposal(hidden, anchors, draft_topk=int(draft_topk))
    compiled = mx.compile(
        lambda hidden, anchors: model.greedy_proposal(
            hidden, anchors, draft_topk=draft_topk
        )
    )
    for result in (
        model.greedy_proposal(hidden, anchors, draft_topk=draft_topk),
        compiled(hidden, anchors),
    ):
        np.testing.assert_array_equal(np.array(result[0]), np.array(expected[0]))
        for actual, reference in zip(result[1:], expected[1:], strict=True):
            np.testing.assert_allclose(
                array(actual), array(reference), atol=1e-5, rtol=1e-5
            )


@pytest.mark.parametrize(
    "draft_topk", [0, -1, 65, True, False, 1.5, "8", np.int64(65), np.bool_(True)]
)
@pytest.mark.parametrize("vocab_size", [32, 64])
def test_invalid_candidate_limit_is_rejected(draft_topk, vocab_size):
    cfg = config()
    cfg = replace(
        cfg,
        backbone=replace(
            cfg.backbone, vocab_size=vocab_size, mask_token_id=vocab_size - 1
        ),
    )
    with pytest.raises(ValueError, match="draft_topk") as error:
        DSparkModel(cfg).greedy_proposal(
            mx.zeros((1, 7, 32)), mx.array([3]), draft_topk=draft_topk
        )
    assert f"[1, {vocab_size}]" in str(error.value)
    assert f"got {draft_topk!r}" in str(error.value)


def test_markov_and_confidence_follow_previous_prediction_not_anchor_or_current_token():
    cfg = replace(
        config(),
        backbone=replace(config().backbone, vocab_size=8, mask_token_id=7),
        markov_rank=2,
    )
    model = DSparkModel(cfg)
    model.lm_head.weight = mx.zeros((8, 32))
    first = mx.zeros((8, 2))
    first[mx.array([3, 5, 2, 6])] = mx.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    model.markov_head.markov_w1.weight = first
    second = mx.zeros((8, 2))
    second[mx.array([5, 2, 6, 3])] = mx.array([[10, 0], [0, 10], [-10, 0], [0, -10]])
    model.markov_head.markov_w2.weight = second
    model.confidence_head.proj.weight = mx.concatenate(
        [mx.zeros((1, 32)), mx.array([[1, 2]])], -1
    )
    model.confidence_head.proj.bias = mx.zeros((1,))
    tokens, _, confidence = model.greedy_proposal(
        mx.zeros((2, 3, 32)), mx.array([3, 5])
    )
    assert tokens.tolist() == [[5, 2, 6], [2, 6, 3]]
    assert confidence.tolist() == [[1, 2, -1], [2, -1, -2]]


@pytest.mark.parametrize("width", [0, 8, True, 1.5])
def test_width_bounds_are_checked_before_embedding(width):
    with pytest.raises(ValueError, match="num_draft_tokens"):
        DSparkModel(config()).block_hidden(mx.array([1]), [], num_draft_tokens=width)


@pytest.mark.parametrize("ids", [[-1], [64], [2**32 + 1]])
def test_external_anchor_validation_does_not_narrow_int64(ids):
    with pytest.raises(ValueError):
        DSparkModel(config()).validate_anchors(mx.array(ids, dtype=mx.int64))


@pytest.mark.parametrize(
    "field,value",
    [
        ("architectures", ["DFlashDraftModel"]),
        ("model_type", "gemma4_text"),
        ("markov_rank", 0),
        ("markov_rank", True),
        ("markov_head_type", "rnn"),
        ("markov_head_type", "gated"),
        ("enable_confidence_head", 1),
        ("confidence_head_with_markov", "false"),
        ("target_layer_ids", [2, 0]),
        ("target_layer_ids", [-1, 1]),
        ("target_layer_ids", [1, 1]),
        ("attention_bias", True),
        ("hidden_act", "gelu"),
        ("tie_word_embeddings", True),
        ("rope_parameters", {"rope_type": "yarn"}),
        ("rope_parameters", []),
        ("rope_parameters", 0),
        ("rope_parameters", {"partial_rotary_factor": 0.5}),
        ("rope_scaling", {"factor": 2}),
        ("partial_rotary_factor", 0.5),
        ("use_sliding_window", True),
        ("layer_types", ["full_attention"]),
        ("quantization_config", {"bits": 4}),
        ("dflash_query_causal", True),
        ("log_snr_conditioning", True),
        ("enable_qwen35_gated_q_proj", True),
        ("input_embedding_scale", 2),
        ("output_multiplier", 2),
        ("final_logit_softcapping", 30),
    ],
)
def test_incompatible_checkpoint_semantics_fail(field, value):
    with pytest.raises(ValueError):
        DSparkConfig.from_dict({**raw_config(config()), field: value})


def test_default_rope_aliases_and_required_fields():
    raw = raw_config(config())
    theta = raw.pop("rope_theta")
    raw["rope_parameters"] = {"rope_type": "default", "rope_theta": theta}
    assert DSparkConfig.from_dict(raw) == config()
    with pytest.raises(ValueError, match="Conflicting"):
        DSparkConfig.from_dict({**raw, "rope_theta": theta * 2})
    for field in (
        "markov_head_type",
        "mask_token_id",
        "markov_rank",
        "enable_confidence_head",
        "confidence_head_with_markov",
    ):
        invalid = dict(raw)
        del invalid[field]
        with pytest.raises(ValueError):
            DSparkConfig.from_dict(invalid)


def checkpoint(path, *, dtype=mx.float32, damage=None, confidence=True):
    cfg = config(confidence=confidence)
    model = DSparkModel(cfg)
    model.set_dtype(dtype)
    weights = {
        name.removeprefix("backbone."): value
        for name, value in tree_flatten(model.parameters())
    }
    if damage is not None:
        damage(weights)
    (path / "config.json").write_text(json.dumps(raw_config(cfg)))
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    return cfg, weights


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("confidence", [False, True])
def test_checkpoint_round_trip_preserves_all_heads_and_precision(
    tmp_path, dtype, confidence
):
    cfg, weights = checkpoint(tmp_path, dtype=dtype, confidence=confidence)
    model = load_dspark(tmp_path, target_config=_target_config(cfg.backbone))
    assert not model.training
    for name, value in tree_flatten(model.parameters()):
        expected = weights[name.removeprefix("backbone.")]
        assert value.dtype == dtype
        np.testing.assert_array_equal(array(value), array(expected))


@pytest.mark.parametrize("sample_from_anchor", [None, True, False])
def test_checkpoint_anchor_layout_matches_dspark(tmp_path, sample_from_anchor):
    cfg, _ = checkpoint(tmp_path)
    path = tmp_path / "config.json"
    raw = json.loads(path.read_text())
    if sample_from_anchor is not None:
        raw["sample_from_anchor"] = sample_from_anchor
        path.write_text(json.dumps(raw))
    if sample_from_anchor is False:
        with pytest.raises(ValueError, match="sample_from_anchor"):
            load_dspark(tmp_path, target_config=_target_config(cfg.backbone))
    else:
        model = load_dspark(tmp_path, target_config=_target_config(cfg.backbone))
        assert model.config == cfg


def test_checkpoint_rename_rejects_colliding_model_parameters(tmp_path):
    cfg, _ = checkpoint(tmp_path)
    model = DSparkModel(cfg)
    # A top-level fc would alias backbone.fc in the checkpoint namespace.
    model.fc = model.backbone.fc
    with pytest.raises(ValueError, match="collide"):
        load_draft_weights(
            model, tmp_path, model_name="DSpark", strip_prefix="backbone."
        )


@pytest.mark.parametrize(
    "damage",
    [
        lambda w: w.pop("markov_head.markov_w1.weight"),
        lambda w: w.update(extra=mx.zeros((1,))),
        lambda w: w.update({"confidence_head.proj.bias": mx.zeros((2,))}),
        lambda w: w.update(
            {
                "markov_head.markov_w2.weight": w[
                    "markov_head.markov_w2.weight"
                ].astype(mx.float16)
            }
        ),
        lambda w: w.update(
            {"norm.weight": mx.full(w["norm.weight"].shape, float("nan"))}
        ),
        lambda w: w.update(
            {"norm.weight": mx.full(w["norm.weight"].shape, float("inf"))}
        ),
    ],
)
def test_bad_checkpoint_weights_fail(tmp_path, damage):
    cfg, _ = checkpoint(tmp_path, damage=damage)
    with pytest.raises(ValueError):
        load_dspark(tmp_path, target_config=_target_config(cfg.backbone))


def test_wrong_target_rejected_before_loading_weights(tmp_path, monkeypatch):
    cfg, _ = checkpoint(tmp_path)

    def fail_load(*args, **kwargs):
        raise AssertionError("Must validate target before loading weights")

    monkeypatch.setattr(mx, "load", fail_load)
    with pytest.raises(ValueError, match="target"):
        load_dspark(
            tmp_path, target_config={**_target_config(cfg.backbone), "vocab_size": 10}
        )


@pytest.mark.parametrize("width", [1, 3, 7])
@pytest.mark.parametrize("confidence", [False, True])
def test_compiled_full_forward_replays_with_new_features_and_anchors(width, confidence):
    model = DSparkModel(config(confidence=confidence))
    compiled = mx.compile(lambda a, f: model.draft(a, f, num_draft_tokens=width))
    for length, anchors in [(17, [1, 2]), (17, [12, 13]), (33, [21, 22])]:
        features = [mx.random.normal((2, length, 32)) for _ in range(3)]
        anchors = mx.array(anchors)
        actual = compiled(anchors, features)
        expected = model.draft(anchors, features, num_draft_tokens=width)
        np.testing.assert_array_equal(np.array(actual[0]), np.array(expected[0]))
        for a, b in zip(actual[1:], expected[1:], strict=True):
            if b is None:
                assert a is None
            else:
                np.testing.assert_allclose(array(a), array(b), atol=1e-5, rtol=1e-5)


def test_incompatible_feature_shapes_precision_and_context_limits():
    model = DSparkModel(config())
    for features in (
        [],
        [mx.zeros((1, 1, 32))],
        [mx.zeros((1, 1, 31))] * 3,
        [mx.zeros((1, 128, 32))] * 3,
        [mx.zeros((1, 1, 32), dtype=mx.float16)] * 3,
    ):
        with pytest.raises(ValueError):
            model.block_hidden(mx.array([1]), features, num_draft_tokens=7)
    with pytest.raises(ValueError, match="precision"):
        model.greedy_proposal(mx.zeros((1, 1, 32), dtype=mx.float16), mx.array([1]))
    with pytest.raises(ValueError, match="anchor per block"):
        model.greedy_proposal(mx.zeros((2, 1, 32)), mx.array([1]))


@pytest.mark.parametrize(
    "extra_file", ["part-2.safetensors", "model.safetensors.index.json"]
)
def test_sharded_checkpoint_rejected(tmp_path, extra_file):
    cfg, _ = checkpoint(tmp_path)
    (tmp_path / extra_file).write_text("{}")
    with pytest.raises(ValueError, match="unsharded"):
        load_dspark(tmp_path, target_config=_target_config(cfg.backbone))


@pytest.mark.parametrize("name", list(COMMON_DRAFT_OPTIONS))
def test_shared_unsupported_semantics_rejected_by_both_drafters(name):
    from tests.test_dflash import _raw
    from vllm_metal.v1.dflash import DFlashConfig

    expected = COMMON_DRAFT_OPTIONS[name]
    invalid = (
        not expected
        if isinstance(expected, bool)
        else (
            "unsupported"
            if expected is None or isinstance(expected, str)
            else expected + 1
        )
    )
    for parse, raw in (
        (DFlashConfig.from_dict, _raw()),
        (DSparkConfig.from_dict, raw_config(config())),
    ):
        with pytest.raises(ValueError, match=name):
            parse({**raw, name: invalid})


def test_confidence_reuses_each_markov_embedding(monkeypatch):
    import mlx.nn as nn

    model = DSparkModel(config())
    original = nn.Embedding.__call__
    calls = []

    def counted(self, ids):
        if self is model.markov_head.markov_w1:
            calls.append(ids.shape)
        return original(self, ids)

    monkeypatch.setattr(nn.Embedding, "__call__", counted)
    result = model.greedy_proposal(mx.ones((2, 7, 32)), mx.array([3, 5]))
    mx.eval(result)
    assert calls == [(2,)] * 7
