# SPDX-License-Identifier: Apache-2.0
"""Tests for the draft-model speculative decoding proposer."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from vllm_metal.v1.draft_model_proposer import DraftModelProposer
from vllm_metal.v1.spec_decode import SpeculativeDecodeController


def _speculative_config(*, use_heterogeneous_vocab: bool) -> SimpleNamespace:
    return SimpleNamespace(
        method="draft_model",
        draft_model_config=SimpleNamespace(model="unused/draft-model"),
        use_heterogeneous_vocab=use_heterogeneous_vocab,
    )


class TestDraftModelProposerBuild:
    def test_rejects_heterogeneous_draft_vocabulary(self) -> None:
        speculative_config = _speculative_config(use_heterogeneous_vocab=True)

        with pytest.raises(NotImplementedError, match="heterogeneous draft"):
            DraftModelProposer.build(
                speculative_config=speculative_config,  # type: ignore[arg-type]
                controller=SpeculativeDecodeController(),
                extract_logits=lambda output: output,
                num_blocks=1,
                block_size=16,
                dtype=mx.float16,
            )

    def test_rejects_missing_draft_model_config(self) -> None:
        speculative_config = _speculative_config(use_heterogeneous_vocab=False)
        speculative_config.draft_model_config = None

        with pytest.raises(ValueError, match="requires a draft_model_config"):
            DraftModelProposer.build(
                speculative_config=speculative_config,  # type: ignore[arg-type]
                controller=SpeculativeDecodeController(),
                extract_logits=lambda output: output,
                num_blocks=1,
                block_size=16,
                dtype=mx.float16,
            )
