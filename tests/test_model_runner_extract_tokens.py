# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

import vllm_metal.model_runner as mr  # noqa: E402


class DummySeqData:
    def __init__(self, token_ids) -> None:
        self._token_ids = token_ids

    def get_token_ids(self):
        return self._token_ids


def test_extract_tokens_none_returns_empty() -> None:
    assert mr.MetalModelRunner._extract_tokens(DummySeqData(None)) == []


def test_extract_tokens_non_iterable_raises() -> None:
    with pytest.raises(ValueError, match="Sequence data lacks token ids"):
        mr.MetalModelRunner._extract_tokens(DummySeqData(123))
