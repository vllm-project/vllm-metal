# SPDX-License-Identifier: Apache-2.0
"""Laya encoder parity and token-classification contracts without downloads."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch import nn
from transformers import ModernBertConfig, ModernBertModel
from vllm.pooling_params import PoolingParams

from tests.test_v1_pooling import _new_req, _scheduler_output
from vllm_metal.v1.pooling.backends.encoder.models.laya import (
    LayaPoolingBackend,
    _create_model,
    clamp_temperature,
    register,
    temperature_for,
)
from vllm_metal.v1.pooling.contract import EncoderPoolingRequest
from vllm_metal.v1.pooling.validation import PoolingConfigView


def models():
    torch.manual_seed(71)
    ec = ModernBertConfig(
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=3,
        num_attention_heads=2,
        vocab_size=512,
        pad_token_id=0,
        cls_token_id=2,
        sep_token_id=3,
        local_attention=128,
        reference_compile=False,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        mlp_dropout=0.0,
    )
    ec._attn_implementation = "eager"
    ac = {
        "head_layers": 2,
        "act_costs": {"escalate": 0.5},
        "mask_token_id": 7,
        "qtype_token_ids": [10, 11, 12],
        "temperature": [1.2, 1.4, 1.6],
        "temperature_by_options": {"choice:2": 1.9, "score:3-5": 1.3, "noul:2": 2.0},
    }
    reference = nn.Module()
    reference.encoder = ModernBertModel(ec)
    reference.type_emb = nn.Embedding(3, 128)
    reference.head = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            128, 2, 512, dropout=0.0, batch_first=True, norm_first=True
        ),
        2,
        enable_nested_tensor=False,
    )
    reference.scorer = nn.Sequential(
        nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1)
    )
    reference.act_head = nn.Sequential(
        nn.Linear(132, 256), nn.GELU(), nn.Linear(256, 2)
    )
    reference.register_buffer("temperature", torch.ones(3))
    reference.eval()
    model = _create_model(ec.to_dict(), ac)
    weights = model.sanitize(
        {k: mx.array(v.detach().numpy()) for k, v in reference.state_dict().items()}
    )
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    hf = SimpleNamespace(laya_config=ac)
    pooler_config = SimpleNamespace(
        task="token_classify", tok_pooling_type="ALL", enable_chunked_processing=False
    )
    view = PoolingConfigView(
        SimpleNamespace(
            hf_config=hf,
            pooler_config=pooler_config,
            multimodal_config=None,
            runner_type="pooling",
            served_model_name="laya-test",
            model="laya-test",
        )
    )
    return model, reference, LayaPoolingBackend(view, model, ac), ac


def pool_request(backend, request):
    scheduled = _new_req(
        request.req_id, list(request.token_ids), pooling_params=request.pooling_params
    )
    outputs = backend.pool_scheduler_output(
        _scheduler_output(new_reqs=[scheduled]), backend.config.model_config
    )
    return outputs[0].pooler_output


@pytest.mark.parametrize(
    "qtype,count", [(0, 1), (0, 2), (0, 6), (0, 11), (1, 3), (2, 2)]
)
@pytest.mark.parametrize(
    "torch_dtype,mlx_dtype,atol",
    [
        (torch.float32, mx.float32, 2e-5),
        (torch.float16, mx.float16, 1e-3),
        (torch.bfloat16, mx.bfloat16, 1e-2),
    ],
)
def test_torch_parity(qtype, count, torch_dtype, mlx_dtype, atol):
    model, reference, backend, ac = models()
    model.set_dtype(mlx_dtype)
    reference.to(torch_dtype)
    # Upstream combines FP32 action features with the cast action-head weights.
    reference.act_head.float()
    ids = torch.randint(20, 512, (1, 150))
    ids[0, 0], ids[0, 1] = 2, ac["qtype_token_ids"][qtype]
    marker_positions = torch.arange(5, 5 + count * 3, 3)
    ids[0, marker_positions] = 7
    mask = torch.ones_like(ids)
    with torch.no_grad():
        h = reference.encoder(ids, attention_mask=mask).last_hidden_state
        h = reference.head(h + reference.type_emb(torch.tensor([qtype]))[:, None, :])
        logits = reference.scorer(h[:, marker_positions]).squeeze(-1).float()
        p = logits.softmax(-1)
        top = torch.nn.functional.pad(p, (0, 1)).topk(2, -1).values
        k = torch.tensor(float(max(count, 2)))
        ent = -(p * p.clamp_min(1e-9).log()).sum(-1) / k.log()
        features = torch.stack(
            [top[:, 0], top[:, 0] - top[:, 1], ent, (k / 255).expand(1)], -1
        )
        assert h.dtype == torch_dtype
        actions = reference.act_head(torch.cat([h[:, 0].float(), features], -1))
    for activated in (False, True):
        req = EncoderPoolingRequest(
            "r",
            tuple(ids[0].tolist()),
            PoolingParams(task="token_classify", use_activation=activated),
        )
        actual = pool_request(backend, req).numpy()
        expected_p = (
            (logits / temperature_for(ac, qtype, count)).softmax(-1)
            if activated
            else logits
        )
        expected_a = actions.softmax(-1) if activated else actions
        expected = torch.cat(
            [expected_p[:, :, None], expected_a[:, None, :].expand(1, count, 2)], -1
        )[0]
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected.numpy(), atol=atol, rtol=atol)


def test_padding_and_profiling():
    model, _, backend, _ = models()
    ids = mx.array([[2, 11, 7, 21, 7, 22]])
    h = backend.forward_padded(ids, mx.ones(ids.shape, dtype=mx.int32))
    padded = mx.concatenate([ids, mx.zeros((1, 7), dtype=mx.int32)], -1)
    mask = mx.concatenate([mx.ones_like(ids), mx.zeros((1, 7), dtype=mx.int32)], -1)
    hp = backend.forward_padded(padded, mask)
    np.testing.assert_allclose(np.array(h), np.array(hp[:, :6]), atol=2e-5, rtol=2e-5)
    for length in (1, 150):
        dummy = backend.profile_forward(mx.zeros((1, length), dtype=mx.int32))
        assert bool(mx.all(mx.isfinite(dummy)).item())
    with pytest.raises(NotImplementedError, match="dimensions"):
        backend.validate_params(PoolingParams(task="token_classify", dimensions=2))
    with pytest.raises(NotImplementedError, match="token_classify"):
        backend.validate_params(PoolingParams(task="embed"))


def test_no_markers_returns_empty_without_model_forward(monkeypatch):
    model, _, backend, _ = models()

    def unexpected_forward(*args):
        pytest.fail("A marker-free request must not run the decision model")

    monkeypatch.setattr(type(model), "__call__", unexpected_forward)
    req = EncoderPoolingRequest("empty", (0,), PoolingParams(task="token_classify"))
    assert pool_request(backend, req).shape == (0, 3)


def test_scheduler_preserves_laya_request_order_and_rejects_partial_prompts():
    _, _, backend, _ = models()
    params = PoolingParams(task="token_classify")
    requests = [
        _new_req("two", [2, 10, 7, 21, 7, 22], pooling_params=params),
        _new_req("one", [2, 12, 7, 21], pooling_params=params),
    ]
    config = backend.config.model_config
    result = backend.pool_scheduler_output(_scheduler_output(new_reqs=requests), config)
    assert [out.req_id for out in result] == ["two", "one"]
    assert [out.pooler_output.shape for out in result] == [(2, 3), (1, 3)]
    with pytest.raises(NotImplementedError, match="full-prompt"):
        backend.pool_scheduler_output(
            _scheduler_output(
                new_reqs=requests, num_scheduled_tokens={"two": 1, "one": 4}
            ),
            config,
        )


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, 1.0),
        (None, 1.0),
        ("bad", 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (0.1, 0.5),
        (9, 5.0),
        (1.5, 1.5),
    ],
)
def test_temperature(value, expected):
    assert clamp_temperature(value) == expected


def test_registry_metadata():
    from vllm.model_executor.models import ModelRegistry

    register()
    register()
    info = ModelRegistry.models["LayaForDecision"].inspect_model_cls()
    assert info.is_pooling_model
    assert info.attn_type == "encoder_only"
    assert info.default_tok_pooling_type == "ALL"


def checkpoint_config(reference, ac, path, **overrides):
    config = reference.encoder.config
    config.architectures = ["LayaForDecision"]
    config.laya_config = ac
    values = {
        "model": str(path),
        "tokenizer": "laya-tokenizer",
        "tokenizer_revision": "pinned-tokenizer-revision",
        "trust_remote_code": False,
        "hf_config": config,
        "dtype": torch.float32,
        "quantization": None,
        "runner_type": "pooling",
        "multimodal_config": None,
        "pooler_config": SimpleNamespace(
            task="token_classify",
            tok_pooling_type="ALL",
            enable_chunked_processing=False,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "torch_dtype,mlx_dtype",
    [
        (torch.float32, mx.float32),
        (torch.float16, mx.float16),
        (torch.bfloat16, mx.bfloat16),
    ],
)
def test_checkpoint_loads_through_factory(
    tmp_path, monkeypatch, torch_dtype, mlx_dtype
):
    from safetensors.torch import save_file

    from vllm_metal.v1.pooling.backends.encoder.factory import (
        load_encoder_pooling_backend,
        supports_encoder_pooling_backend,
    )
    from vllm_metal.v1.pooling.backends.encoder.models import laya

    model, reference, _, ac = models()
    # Exercise stored FP16 weights, as shipped by the real checkpoint, rather
    # than testing only the in-memory FP32 mapping used by the math test.
    state = {k: v.half().contiguous() for k, v in reference.state_dict().items()}
    save_file(state, str(tmp_path / "model.safetensors"))
    expected = model.sanitize(
        {k: mx.array(v.float().numpy()).astype(mlx_dtype) for k, v in state.items()}
    )
    model.load_weights(list(expected.items()), strict=True)
    calls = []

    class Tokenizer:
        mask_token_id = 7

        def __call__(self, text, **kwargs):
            return {
                "input_ids": [{"choice": 10, "score": 11, "noul": 12}[text.split()[0]]]
            }

    tokenizer = Tokenizer()

    def load_tokenizer(config):
        calls.append(config)
        return tokenizer

    monkeypatch.setattr(laya, "cached_tokenizer_from_config", load_tokenizer)
    config = checkpoint_config(reference, ac, tmp_path, dtype=torch_dtype)
    config.hf_config.laya_config = {
        key: value
        for key, value in ac.items()
        if key not in ("mask_token_id", "qtype_token_ids")
    }
    assert supports_encoder_pooling_backend(config)
    loaded = load_encoder_pooling_backend(config)
    assert {value.dtype for _, value in tree_flatten(loaded.model.parameters())} == {
        mlx_dtype
    }
    ids = mx.array([[2, 10, 7, 21, 7, 22]], dtype=mx.int32)
    hidden = loaded.pooling_backend.forward_padded(ids, mx.ones_like(ids))
    assert hidden.dtype == mlx_dtype
    assert loaded.tokenizer is tokenizer
    assert calls == [config]
    assert loaded.model_args["laya_config"]["qtype_token_ids"] == [10, 11, 12]
    assert "qtype_token_ids" not in config.hf_config.laya_config
    request = EncoderPoolingRequest(
        "loaded",
        (2, 10, 7, 21, 7, 22),
        PoolingParams(task="token_classify", use_activation=False),
    )
    expected_backend = LayaPoolingBackend(PoolingConfigView(config), model, ac)
    np.testing.assert_allclose(
        pool_request(loaded.pooling_backend, request).numpy(),
        pool_request(expected_backend, request).numpy(),
        atol=1e-6,
    )
    assert loaded.pooling_backend.supported_tasks() == ("token_classify",)
    assert not supports_encoder_pooling_backend(
        checkpoint_config(reference, ac, tmp_path, runner_type="generate")
    )
    assert not supports_encoder_pooling_backend(
        checkpoint_config(reference, ac, tmp_path, multimodal_config=object())
    )


@pytest.mark.parametrize("mask_token,types", [(None, [10, 11, 12]), (7, [10, 10, 12])])
def test_invalid_tokenizer_is_rejected_before_weights(
    tmp_path, monkeypatch, mask_token, types
):
    from vllm_metal.v1.pooling.backends.encoder.models import laya

    _, reference, _, ac = models()

    class Tokenizer:
        mask_token_id = mask_token

        def __call__(self, text, **kwargs):
            return {
                "input_ids": [types[("choice", "score", "noul").index(text.split()[0])]]
            }

    monkeypatch.setattr(
        laya, "cached_tokenizer_from_config", lambda config: Tokenizer()
    )

    def unexpected_weights(*args):
        pytest.fail("Invalid tokenizer must be rejected before loading weights")

    monkeypatch.setattr(laya, "encoder_model_path", unexpected_weights)
    with pytest.raises(ValueError, match="MASK token and three distinct"):
        laya.load_laya_backend(checkpoint_config(reference, ac, tmp_path))


@pytest.mark.parametrize(
    "overrides,config_field,value,error,message",
    [
        ({"quantization": "awq"}, None, None, NotImplementedError, "quantization"),
        ({}, "quantization_config", {"bits": 4}, NotImplementedError, "quantization"),
        ({}, "hidden_activation", "relu", NotImplementedError, "GELU"),
        ({}, "model_type", "bert", NotImplementedError, "ModernBERT"),
        (
            {},
            "rope_parameters",
            {"full_attention": {"rope_type": "linear"}},
            NotImplementedError,
            "RoPE",
        ),
    ],
)
def test_loader_rejects_unsupported_before_io(
    tmp_path, monkeypatch, overrides, config_field, value, error, message
):
    from vllm_metal.v1.pooling.backends.encoder.models import laya

    _, reference, _, ac = models()
    config = checkpoint_config(reference, ac, tmp_path / "not-downloaded", **overrides)
    if config_field:
        values = {**config.hf_config.to_dict(), config_field: value}
        monkeypatch.setattr(config.hf_config, "to_dict", lambda: values)

    def unexpected_io(*args, **kwargs):
        pytest.fail("Rejected model must not download or load weights/tokenizer")

    monkeypatch.setattr(laya, "encoder_model_path", unexpected_io)
    monkeypatch.setattr(laya, "cached_tokenizer_from_config", unexpected_io)
    with pytest.raises(error, match=message):
        laya.load_laya_backend(config)


@pytest.mark.parametrize("act_costs", [{}, {"escalate": 0.5, "delegate": 0.2}])
def test_loader_rejects_other_action_counts_before_io(tmp_path, monkeypatch, act_costs):
    from vllm_metal.v1.pooling.backends.encoder.models import laya

    _, reference, _, ac = models()
    config = checkpoint_config(reference, {**ac, "act_costs": act_costs}, tmp_path)

    def unexpected_io(*args):
        pytest.fail("Unsupported action heads must be rejected before loading weights")

    monkeypatch.setattr(laya, "encoder_model_path", unexpected_io)
    with pytest.raises(NotImplementedError, match="two action classes"):
        laya.load_laya_backend(config)


@pytest.mark.parametrize("preexisting", [False, True])
def test_registry_preserves_existing_architecture(monkeypatch, preexisting):
    import vllm.model_executor.models as vllm_models

    sentinel = object()
    entries = {"LayaForDecision": sentinel} if preexisting else {}
    registry = SimpleNamespace(
        get_supported_archs=lambda: list(entries),
        register_model=lambda name, model: entries.update({name: model}),
    )
    monkeypatch.setattr(vllm_models, "ModelRegistry", registry)
    register()
    register()
    assert entries["LayaForDecision"] == (
        sentinel
        if preexisting
        else "vllm_metal.v1.pooling.backends.encoder.models.laya:LayaForDecision"
    )
