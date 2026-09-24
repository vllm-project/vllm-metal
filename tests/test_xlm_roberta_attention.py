# SPDX-License-Identifier: Apache-2.0
"""Numerical coverage for bidirectional encoder attention on Metal."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from transformers import XLMRobertaConfig
from transformers import XLMRobertaModel as TorchXLMRobertaModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaSelfAttention as TorchXLMRobertaSelfAttention,
)

from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx
from vllm_metal.v1.pooling.backends.encoder.models.xlm_roberta import (
    XLMRobertaArgs,
    XLMRobertaModel,
    XLMRobertaSelfAttention,
)

_DTYPES = [
    pytest.param(torch.float32, mx.float32, 1e-5, id="float32"),
    pytest.param(torch.float16, mx.float16, 1e-3, id="float16"),
    pytest.param(torch.bfloat16, mx.bfloat16, 1e-2, id="bfloat16"),
]


@pytest.mark.parametrize("torch_dtype,mlx_dtype,atol", _DTYPES)
@pytest.mark.parametrize("head_dim", [4, 64])
@pytest.mark.parametrize("seq_len", [1, 37, 129])
def test_encoder_attention_matches_transformers(
    torch_dtype, mlx_dtype, atol, head_dim, seq_len
):
    config = XLMRobertaConfig(
        hidden_size=2 * head_dim,
        num_attention_heads=2,
        attention_probs_dropout_prob=0.0,
    )
    config._attn_implementation = "eager"
    reference = TorchXLMRobertaSelfAttention(config).to(torch_dtype).eval()
    attention = XLMRobertaSelfAttention(XLMRobertaArgs.from_config(config.to_dict()))
    attention.load_weights(
        [(name, torch_to_mlx(value)) for name, value in reference.state_dict().items()]
    )
    hidden = torch.randn(2, seq_len, config.hidden_size).to(torch_dtype)
    # Different left/right padding lengths also exercise mask broadcasting
    # across heads and query positions. A single token remains unmasked.
    mask = torch.zeros(2, 1, 1, seq_len)
    if seq_len > 1:
        mask[0, :, :, -(seq_len // 3) :] = -1e4
        mask[1, :, :, : seq_len // 4] = -1e4
    with torch.no_grad():
        expected = reference(hidden, attention_mask=mask.to(torch_dtype))[0].float()

    actual = attention(torch_to_mlx(hidden), torch_to_mlx(mask))
    assert actual.dtype == mlx_dtype
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), expected.numpy(), atol=atol, rtol=atol
    )


@pytest.mark.parametrize("torch_dtype,mlx_dtype,atol", _DTYPES)
def test_encoder_padding_preserves_hidden_states_and_pooling(
    torch_dtype, mlx_dtype, atol
):
    config = XLMRobertaConfig(
        vocab_size=32,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=1,
    )
    reference = TorchXLMRobertaModel(config).to(torch_dtype).eval()
    model = XLMRobertaModel(XLMRobertaArgs.from_config(config.to_dict()))
    model.load_weights(
        list(
            model.sanitize(
                {
                    name: torch_to_mlx(value)
                    for name, value in reference.state_dict().items()
                }
            ).items()
        ),
        strict=True,
    )
    ids = torch.tensor([[0, 5, 6, 7, 2, 1, 1], [1, 1, 0, 8, 9, 10, 2]])
    mask = (ids != config.pad_token_id).long()
    with torch.no_grad():
        expected = reference(ids, attention_mask=mask).last_hidden_state.float().numpy()
    actual_mx = model(torch_to_mlx(ids.int()), torch_to_mlx(mask.int()))
    assert actual_mx.dtype == mlx_dtype
    actual = np.array(actual_mx.astype(mx.float32))
    np.testing.assert_allclose(actual, expected, atol=atol * 4, rtol=atol * 4)

    # Padding must not alter valid tokens or their normalized MEAN embedding.
    for row in range(2):
        valid = mask[row].bool().numpy()
        unpadded = model(mx.array(ids[row, valid].numpy()[None], dtype=mx.int32))
        unpadded = np.array(unpadded.astype(mx.float32))[0]
        np.testing.assert_allclose(actual[row, valid], unpadded, atol=atol, rtol=atol)
        pooled = actual[row, valid].mean(axis=0)
        expected_pooled = expected[row, valid].mean(axis=0)
        np.testing.assert_allclose(
            pooled / np.linalg.norm(pooled),
            expected_pooled / np.linalg.norm(expected_pooled),
            atol=atol,
            rtol=atol,
        )
