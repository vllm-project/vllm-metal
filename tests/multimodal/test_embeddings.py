# SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal embedding splice helpers."""

from __future__ import annotations

import mlx.core as mx
import pytest

from vllm_metal.multimodal import merge_multimodal_embeddings


def test_merge_no_multimodal_returns_input() -> None:
    inputs = mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=mx.float32)
    mask = mx.array([False, False])

    output = merge_multimodal_embeddings(inputs, [], mask)

    assert output.tolist() == inputs.tolist()


def test_merge_single_image_block() -> None:
    inputs = mx.array(
        [[[1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [7.0, 8.0]]],
        dtype=mx.float32,
    )
    image_embeddings = mx.array(
        [[3.0, 4.0], [5.0, 6.0]],
        dtype=mx.float32,
    )
    is_multimodal = mx.array([False, True, True, False])

    output = merge_multimodal_embeddings(
        inputs,
        image_embeddings,
        is_multimodal,
    )

    assert output.tolist() == [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]


def test_merge_count_mismatch_raises() -> None:
    inputs = mx.zeros((1, 4, 2), dtype=mx.float32)
    mm_embeds = mx.zeros((3, 2), dtype=mx.float32)
    mask = mx.array([True, True, True, True])

    with pytest.raises(
        ValueError,
        match="Attempted to assign 3 multimodal tokens to 4 placeholders",
    ):
        merge_multimodal_embeddings(inputs, mm_embeds, mask)


def test_merge_rejects_ambiguous_1d_mask_for_multi_batch() -> None:
    inputs = mx.zeros((2, 2, 2), dtype=mx.float32)
    mm_embeds = mx.ones((1, 2), dtype=mx.float32)
    mask = mx.array([False, True])

    with pytest.raises(
        ValueError,
        match="1D multimodal mask is only supported for packed batch size 1",
    ):
        merge_multimodal_embeddings(inputs, mm_embeds, mask)


def test_merge_rejects_mask_shape_mismatch() -> None:
    inputs = mx.zeros((1, 2, 2), dtype=mx.float32)
    mm_embeds = mx.ones((1, 2), dtype=mx.float32)
    mask = mx.array([[False, True, False]])

    with pytest.raises(ValueError, match="Multimodal mask shape"):
        merge_multimodal_embeddings(inputs, mm_embeds, mask)


def test_merge_dtype_preserved() -> None:
    inputs = mx.zeros((1, 2, 2), dtype=mx.bfloat16)
    mm_embeds = mx.ones((1, 2), dtype=mx.float32)
    mask = mx.array([False, True])

    output = merge_multimodal_embeddings(inputs, mm_embeds, mask)

    assert output.dtype == mx.bfloat16


def test_merge_does_not_mutate_input() -> None:
    inputs = mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=mx.float32)
    original = inputs.tolist()
    mm_embeds = mx.array([[9.0, 10.0]], dtype=mx.float32)
    mask = mx.array([False, True, False])

    output = merge_multimodal_embeddings(inputs, mm_embeds, mask)

    assert output.tolist() == [[[1.0, 2.0], [9.0, 10.0], [5.0, 6.0]]]
    assert inputs.tolist() == original
