# SPDX-License-Identifier: Apache-2.0
"""Tests for the reusable hidden-state layer tap (DSpark prep)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.qwen3 import Qwen3Model

pytest.importorskip("vllm", reason="vllm not installed")

from vllm_metal.v1.hidden_state_tap import run_backbone_with_capture


class _Id:
    """Callable identity (stands in for norm)."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


class _Embed:
    """Stand-in for embed_tokens: [B, T] ids -> [B, T, 1] (hidden dim of 1)."""

    def __call__(self, ids: mx.array) -> mx.array:
        return ids[..., None]


class _AddLayer:
    """Fake transformer layer: adds its index to the residual stream."""

    def __init__(self, i: int) -> None:
        self.i = i

    def __call__(self, h: mx.array, mask, cache) -> mx.array:  # noqa: ANN001
        return h + self.i


def _toy_backbone(num_layers: int) -> Qwen3Model:
    # Drive the real native Qwen3 body with easily checked layer arithmetic.
    body = Qwen3Model.__new__(Qwen3Model)
    nn.Module.__init__(body)
    body.embed_tokens = _Embed()
    body.layers = [_AddLayer(i) for i in range(num_layers)]
    body.norm = _Id()
    return body


# --- fused capture from run_backbone_with_capture ---


def test_capture_requires_at_least_one_layer() -> None:
    backbone = _toy_backbone(3)
    with pytest.raises(ValueError):
        run_backbone_with_capture(
            backbone, mx.array([[1.0, 2.0, 3.0]]), cache=[None] * 3, layer_ids=[]
        )


def test_capture_fuses_requested_layers_in_order() -> None:
    # embed: ids -> ids[...,None]; layer i adds i to the running residual, so
    # after layer i the residual is ids + sum(0..i). Capturing [0, 2] fuses the
    # residual after layer 0 (ids+0) and after layer 2 (ids+0+1+2 = ids+3).
    backbone = _toy_backbone(3)
    ids = mx.array([[1.0, 2.0, 3.0]])  # [1, 3]
    _, out = run_backbone_with_capture(
        backbone, ids, cache=[None] * 3, layer_ids=[0, 2]
    )
    mx.eval(out)
    assert out.shape == (1, 3, 2)  # [batch, tokens, len(layer_ids) * hidden(=1)]
    expected = mx.concatenate([ids[..., None], (ids + 3)[..., None]], axis=-1)
    assert mx.array_equal(out, expected)


def test_capture_single_last_layer_matches_full_residual() -> None:
    # Capturing only the last layer reproduces the body's own final residual
    # (the pre-norm stream) — the fidelity invariant the drafter relies on.
    backbone = _toy_backbone(4)
    ids = mx.array([[1.0, 2.0, 3.0]])
    _, out = run_backbone_with_capture(backbone, ids, cache=[None] * 4, layer_ids=[3])
    final = (ids + 0 + 1 + 2 + 3)[..., None]  # residual after all 4 layers
    mx.eval(out)
    assert out.shape == (1, 3, 1)
    assert mx.array_equal(out, final)


def test_capture_rejects_out_of_range_layer() -> None:
    backbone = _toy_backbone(3)
    with pytest.raises(IndexError):
        run_backbone_with_capture(
            backbone, mx.array([[1.0, 2.0]]), cache=[None] * 3, layer_ids=[5]
        )


# --- run_backbone_with_capture (final + fused from one traversal) ---


def test_run_backbone_returns_final_post_norm_and_fused() -> None:
    # One traversal must yield BOTH the post-norm final hidden (for target logits)
    # and the fused intermediate captures (for the drafter) — the integrated path
    # can't afford a second forward.
    backbone = _toy_backbone(3)
    ids = mx.array([[1.0, 2.0, 3.0]])
    final, fused = run_backbone_with_capture(
        backbone, ids, cache=[None] * 3, layer_ids=[0, 2]
    )
    mx.eval(final, fused)
    # final = post-norm residual after ALL layers = ids + 0+1+2 = ids+3
    assert mx.array_equal(final, (ids + 3)[..., None])
    # fused = residuals after layer 0 (ids+0) and layer 2 (ids+3)
    assert mx.array_equal(
        fused, mx.concatenate([ids[..., None], (ids + 3)[..., None]], axis=-1)
    )


# --- target_forward wiring (capture_layer_ids routing) ---


def test_target_forward_capture_routes_through_helper(monkeypatch) -> None:
    """capture_layer_ids=[...] routes through run_backbone_with_capture and
    returns its fused output as hidden_states plus logits from
    _compute_target_logits (computed from the same forward's final hidden)."""
    import vllm_metal.v1.model_adapter as ma

    adapter = ma.DefaultModelAdapter()
    toy = _toy_backbone(3)
    monkeypatch.setattr(adapter, "_target_backbone", lambda model: toy)
    sentinel_logits = mx.full((1, 2, 3), 7.0)
    monkeypatch.setattr(
        adapter, "_compute_target_logits", lambda model, h: sentinel_logits
    )

    seen: dict = {}
    fused_sentinel = mx.full((1, 2, 4), 9.0)  # [1, tokens, len(layer_ids) * hidden]

    def _spy(backbone, ids, *, cache, layer_ids):
        seen["layer_ids"] = list(layer_ids)
        seen["backbone_is_toy"] = backbone is toy
        return mx.zeros((1, 2, 1)), fused_sentinel  # final, fused

    monkeypatch.setattr(ma, "run_backbone_with_capture", _spy)

    out = adapter.target_forward(
        object(), mx.array([[1.0, 2.0]]), cache=[None] * 3, capture_layer_ids=[0, 2]
    )

    assert seen["layer_ids"] == [0, 2]
    assert seen["backbone_is_toy"]
    assert mx.array_equal(out.logits, sentinel_logits)
    assert out.hidden_states.shape == (2, 4)  # fused [1,2,4] flattened
    assert mx.array_equal(out.hidden_states, mx.full((2, 4), 9.0))


def test_target_forward_none_path_does_not_capture(monkeypatch) -> None:
    """capture_layer_ids=None must take the existing forward path: the helper is
    never called, and _forward_target_hidden_states runs as before."""
    import vllm_metal.v1.model_adapter as ma

    adapter = ma.DefaultModelAdapter()
    capture_calls = {"n": 0}
    monkeypatch.setattr(
        ma,
        "run_backbone_with_capture",
        lambda *a, **k: capture_calls.__setitem__("n", capture_calls["n"] + 1),
    )
    fwd_calls = {"n": 0}

    def _fwd_spy(self, model, input_ids, *, cache):
        fwd_calls["n"] += 1
        return mx.zeros((1, 2, 1))

    monkeypatch.setattr(
        ma.DefaultModelAdapter, "_forward_target_hidden_states", _fwd_spy
    )
    monkeypatch.setattr(
        adapter, "_compute_target_logits", lambda model, h: mx.zeros((1, 2, 3))
    )

    adapter.target_forward(
        object(),
        mx.array([[1.0, 2.0]]),
        cache=[None] * 3,
        collect_hidden_states=True,
        capture_layer_ids=None,
    )

    assert capture_calls["n"] == 0
    assert fwd_calls["n"] == 1


@pytest.mark.parametrize("layer_ids", [[2, 0], [1, 1]])
def test_capture_rejects_reordered_or_duplicate_features(layer_ids):
    with pytest.raises(ValueError, match="strictly increasing"):
        run_backbone_with_capture(
            _toy_backbone(3), mx.array([[1]]), cache=None, layer_ids=layer_ids
        )


def test_capture_rejects_unqualified_body_and_incomplete_cache():
    with pytest.raises(NotImplementedError, match="Qwen3Model"):
        run_backbone_with_capture(object(), mx.array([[1]]), cache=None, layer_ids=[0])
    with pytest.raises(ValueError, match="one entry per decoder layer"):
        run_backbone_with_capture(
            _toy_backbone(3), mx.array([[1]]), cache=[None], layer_ids=[0]
        )


def test_capture_keeps_native_body_and_live_layers_on_success_and_failure():
    body = _toy_backbone(3)
    layers = body.layers
    # Native final normalization must still run, independently of fused taps.
    body.norm = lambda h: h * 2
    ids = mx.array([[2.0, 4.0]])
    final, fused = run_backbone_with_capture(body, ids, cache=None, layer_ids=[0, 2])
    assert mx.array_equal(final, body(ids))
    assert mx.array_equal(fused[..., 1:], final / 2)
    assert body.layers is layers

    def fail(*args, **kwargs):
        raise RuntimeError("layer failed")

    layers[1] = fail
    with pytest.raises(RuntimeError, match="layer failed"):
        run_backbone_with_capture(body, ids, cache=None, layer_ids=[0, 2])
    assert body.layers is layers
    assert isinstance(body.layers[0], _AddLayer)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("select", [False, True])
@pytest.mark.parametrize("collect_final", [False, True])
def test_all_capture_hidden_and_logits_layout_combinations(
    capture, select, collect_final
):
    from types import SimpleNamespace

    from vllm_metal.v1.model_adapter import DefaultModelAdapter

    body = _toy_backbone(3)
    body.norm = lambda h: h * 2
    model = SimpleNamespace(model=body, lm_head=lambda h: h + 1)

    # The no-capture/no-hidden/full-logits route calls the model directly.
    class Target:
        def __init__(self):
            self.model = model.model
            self.lm_head = model.lm_head

        def __call__(self, ids, cache=None):
            return self.lm_head(self.model(ids, cache=cache))

    target = Target()
    ids = mx.arange(12, dtype=mx.float32)[None]
    selection = mx.array([0, 1, 2, 6, 11]) if select else None
    output = DefaultModelAdapter().target_forward(
        target,
        ids,
        collect_hidden_states=collect_final,
        logits_indices=selection,
        capture_layer_ids=[0, 2] if capture else None,
    )
    expected_logits = target(ids)
    if select:
        expected_logits = expected_logits[:, selection]
    assert mx.array_equal(output.logits, expected_logits)
    if capture:
        expected_features = mx.stack([ids[0], ids[0] + 3], axis=-1)
        assert mx.array_equal(output.hidden_states, expected_features)
    elif collect_final:
        assert mx.array_equal(output.hidden_states, body(ids)[0])
    else:
        assert output.hidden_states is None
