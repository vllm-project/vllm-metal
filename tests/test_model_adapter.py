# SPDX-License-Identifier: Apache-2.0
"""Tests for model adapter behavior."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

import vllm_metal.envs as envs
from vllm_metal.config import reset_config
from vllm_metal.multimodal.paddleocr_vl import PaddleOCRVLMultimodalAdapter
from vllm_metal.multimodal.qwen3_vl import Qwen3VLMultimodalAdapter
from vllm_metal.v1.model_adapter import (
    DefaultModelAdapter,
    IntermediateForwardOutput,
    TargetModelForwardOutput,
)


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch):
    for var in envs.environment_variables:
        monkeypatch.delenv(var, raising=False)
    reset_config()
    yield
    reset_config()


class TestShouldForceTextBackbone:
    """Tests for should_force_text_backbone()."""

    def test_gemma4_model_type_is_overridden(self) -> None:
        hf_config = SimpleNamespace(model_type="gemma4")
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is True

    def test_qwen35_fp8_conditional_generation_uses_auto_override(self) -> None:
        hf_config = SimpleNamespace(
            model_type="qwen3_5",
            architectures=["Qwen3_5ForConditionalGeneration"],
            quantization_config={"quant_method": "fp8"},
        )
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is True

    def test_qwen36_fp8_conditional_generation_uses_auto_override(self) -> None:
        hf_config = SimpleNamespace(
            model_type="qwen3_6",
            architectures=["Qwen3_6ForConditionalGeneration"],
            quantization_config={"quant_method": "fp8"},
        )
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is True

    @pytest.mark.parametrize("bits", [4, 8])
    @pytest.mark.parametrize(
        ("model_type", "architecture"),
        [
            # Dense and MoE MLX 4-bit wrappers from issues #580 and #571.
            ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
            ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
            ("qwen3_6", "Qwen3_6ForConditionalGeneration"),
        ],
    )
    def test_mlx_quant_conditional_generation_uses_auto_override(
        self, model_type: str, architecture: str, bits: int
    ) -> None:
        hf_config = SimpleNamespace(
            model_type=model_type,
            architectures=[architecture],
            quantization={"group_size": 64, "bits": bits, "mode": "affine"},
        )
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is True

    @pytest.mark.parametrize(
        ("model_type", "architecture", "vision_config"),
        [
            # The flagship native-multimodal checkpoint family
            # (mlx-community Qwen3-VL MLX conversions carry the same
            # top-level quantization dict) must keep the native path.
            (
                "qwen3_vl",
                "Qwen3VLForConditionalGeneration",
                SimpleNamespace(spatial_merge_size=2),
            ),
            # An arbitrary non-allowlisted multimodal model.
            ("llava", "LlavaForConditionalGeneration", None),
        ],
    )
    def test_mlx_quant_natively_adapted_arch_skips_auto_override(
        self, model_type: str, architecture: str, vision_config: object | None
    ) -> None:
        hf_config = SimpleNamespace(
            model_type=model_type,
            architectures=[architecture],
            quantization={"group_size": 64, "bits": 4, "mode": "affine"},
            vision_config=vision_config,
        )
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is False

    def test_mlx_quant_qwen35_wrapper_uses_auto_override_with_vision_config(
        self,
    ) -> None:
        hf_config = SimpleNamespace(
            model_type="qwen3_5",
            architectures=["Qwen3_5ForConditionalGeneration"],
            quantization={"group_size": 64, "bits": 4, "mode": "affine"},
            vision_config=SimpleNamespace(spatial_merge_size=2),
            text_config=SimpleNamespace(model_type="qwen3_5_text"),
        )
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is True

    def test_multimodal_native_mode_disables_mlx_quant_override(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "multimodal-native")
        reset_config()

        hf_config = SimpleNamespace(
            model_type="qwen3_6",
            architectures=["Qwen3_6MoeForConditionalGeneration"],
            quantization={"group_size": 64, "bits": 4, "mode": "affine"},
        )
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is False

    def test_omitted_quantization_key_skips_auto_override(self) -> None:
        # The common dense-checkpoint case: no `quantization` key at all.
        hf_config = SimpleNamespace(
            model_type="qwen3_6",
            architectures=["Qwen3_6ForConditionalGeneration"],
        )
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is False

    def test_non_dict_quantization_value_skips_auto_override(self) -> None:
        hf_config = SimpleNamespace(
            model_type="qwen3_6",
            architectures=["Qwen3_6ForConditionalGeneration"],
            quantization="int4",
        )
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is False

    def test_quantization_dict_without_bits_skips_auto_override(self) -> None:
        hf_config = SimpleNamespace(
            model_type="qwen3_6",
            architectures=["Qwen3_6ForConditionalGeneration"],
            quantization={"group_size": 64},
        )
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is False

    def test_qwen35_non_fp8_conditional_generation_skips_auto_override(self) -> None:
        hf_config = SimpleNamespace(
            model_type="qwen3_5",
            architectures=["Qwen3_5ForConditionalGeneration"],
            quantization_config={"quant_method": "mxfp4"},
        )
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is False

    def test_non_overridden_model_type_is_not_forced_in_auto_mode(self) -> None:
        hf_config = SimpleNamespace(model_type="qwen3_5", architectures=[])
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is False

    def test_multimodal_native_mode_disables_auto_override(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "multimodal-native")
        reset_config()

        hf_config = SimpleNamespace(
            model_type="qwen3_5",
            architectures=["Qwen3_5ForConditionalGeneration"],
            quantization_config={"quant_method": "fp8"},
        )
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(hf_config)
        assert result is False

    def test_none_hf_config_is_not_forced_in_auto_mode(self) -> None:
        adapter = DefaultModelAdapter()
        result = adapter.should_force_text_backbone(None)
        assert result is False


class TestTargetForward:
    def test_plain_forward_extracts_logits_without_hidden_states(self) -> None:
        class Model:
            def __call__(self, input_ids, *, cache=None):
                del input_ids, cache
                return mx.array([[[1.0, 2.0]]])

        output = DefaultModelAdapter().target_forward(
            Model(),
            mx.array([[1]]),
            cache=[],
            collect_hidden_states=False,
        )

        assert isinstance(output, TargetModelForwardOutput)
        assert output.logits.tolist() == [[[1.0, 2.0]]]
        assert output.hidden_states is None

    def test_collects_hidden_states_and_recomputes_logits(self) -> None:
        class Backbone:
            def __call__(self, input_ids, *, cache=None):
                del input_ids, cache
                return mx.array([[[-2.0, 0.0, 2.0]]])

        class Head:
            def __call__(self, hidden_states):
                return hidden_states * 3.0

        model = SimpleNamespace(
            model=Backbone(),
            lm_head=Head(),
            final_logit_softcapping=2.0,
        )

        output = DefaultModelAdapter().target_forward(
            model,
            mx.array([[1]]),
            cache=[],
            collect_hidden_states=True,
        )

        expected_logits = mx.tanh(mx.array([[[-2.0, 0.0, 2.0]]]) * 3.0 / 2.0) * 2.0
        mx.eval(output.logits, expected_logits)
        assert output.hidden_states.tolist() == [[-2.0, 0.0, 2.0]]
        assert mx.allclose(output.logits, expected_logits).item()

    def test_collects_hidden_states_with_model_owned_compute_logits(self) -> None:
        class Backbone:
            def __call__(self, input_ids, *, cache=None):
                del input_ids, cache
                return mx.array([[[1.0, 2.0]]])

        class Model:
            def __init__(self) -> None:
                self.model = Backbone()
                self.final_logit_softcapping = 2.0

            def compute_logits(self, hidden_states):
                return hidden_states * 3.0

        output = DefaultModelAdapter().target_forward(
            Model(),
            mx.array([[1]]),
            cache=[],
            collect_hidden_states=True,
        )

        assert output.hidden_states.tolist() == [[1.0, 2.0]]
        assert output.logits.tolist() == [[[3.0, 6.0]]]

    def test_collects_hidden_states_with_gemma4_tied_embedding_logits(self) -> None:
        class Embedding:
            def as_linear(self, hidden_states):
                return hidden_states + 5.0

        class Backbone:
            def __init__(self) -> None:
                self.embed_tokens = Embedding()

            def __call__(self, input_ids, *, cache=None):
                del input_ids, cache
                return mx.array([[[1.0, 2.0]]])

        model = SimpleNamespace(model=Backbone(), final_logit_softcapping=2.0)

        output = DefaultModelAdapter().target_forward(
            model,
            mx.array([[1]]),
            cache=[],
            collect_hidden_states=True,
        )

        expected_logits = mx.tanh((mx.array([[[1.0, 2.0]]]) + 5.0) / 2.0) * 2.0
        mx.eval(output.logits, expected_logits)
        assert output.hidden_states.tolist() == [[1.0, 2.0]]
        assert mx.allclose(output.logits, expected_logits).item()

    def test_collects_hidden_states_from_language_model_wrapper(self) -> None:
        class Embedding:
            def as_linear(self, hidden_states):
                return hidden_states + 4.0

        class Backbone:
            def __init__(self) -> None:
                self.embed_tokens = Embedding()

            def __call__(self, input_ids, *, cache=None):
                del input_ids, cache
                return mx.array([[[2.0, 3.0]]])

        language_model = SimpleNamespace(
            model=Backbone(),
            tie_word_embeddings=True,
            final_logit_softcapping=None,
        )
        model = SimpleNamespace(language_model=language_model)

        output = DefaultModelAdapter().target_forward(
            model,
            mx.array([[1]]),
            cache=[],
            collect_hidden_states=True,
        )

        assert output.hidden_states.tolist() == [[2.0, 3.0]]
        assert output.logits.tolist() == [[[6.0, 7.0]]]

    def test_selective_logits_match_the_full_projection_rows(self) -> None:
        # The head must see only the requested rows, and the result must equal
        # those rows of the full projection.
        class Backbone:
            def __call__(self, input_ids, *, cache=None):
                del input_ids, cache
                return mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])

        class Head:
            def __init__(self) -> None:
                self.seen_rows: int | None = None

            def __call__(self, hidden_states):
                self.seen_rows = hidden_states.shape[-2]
                return hidden_states * 3.0

        head = Head()
        model = SimpleNamespace(
            model=Backbone(), lm_head=head, final_logit_softcapping=None
        )
        adapter = DefaultModelAdapter()

        full = adapter.target_forward(
            model, mx.array([[1, 2, 3, 4]]), cache=[], collect_hidden_states=True
        )
        selected = adapter.target_forward(
            model,
            mx.array([[1, 2, 3, 4]]),
            cache=[],
            logits_indices=mx.array([1, 3], dtype=mx.int32),
        )

        assert head.seen_rows == 2
        assert selected.logits.tolist() == [
            [full.logits[0, 1].tolist(), full.logits[0, 3].tolist()]
        ]

    def test_selective_logits_keep_full_hidden_states_for_mtp(self) -> None:
        # Gemma4 MTP seeds index target_hidden_row in the packed layout, so the
        # hidden states must stay row-major over every input row.
        class Backbone:
            def __call__(self, input_ids, *, cache=None):
                del input_ids, cache
                return mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

        model = SimpleNamespace(
            model=Backbone(),
            lm_head=lambda hidden_states: hidden_states,
            final_logit_softcapping=None,
        )

        output = DefaultModelAdapter().target_forward(
            model,
            mx.array([[1, 2, 3]]),
            cache=[],
            collect_hidden_states=True,
            logits_indices=mx.array([2], dtype=mx.int32),
        )

        assert output.hidden_states.shape == (3, 2)
        assert output.logits.shape == (1, 1, 2)

    def test_selective_logits_apply_softcapping_to_selected_rows(self) -> None:
        # Post-projection scaling must not be skipped by the selective path.
        class Backbone:
            def __call__(self, input_ids, *, cache=None):
                del input_ids, cache
                return mx.array([[[-2.0, 0.0], [2.0, 4.0]]])

        model = SimpleNamespace(
            model=Backbone(),
            lm_head=lambda hidden_states: hidden_states,
            final_logit_softcapping=2.0,
        )

        output = DefaultModelAdapter().target_forward(
            model,
            mx.array([[1, 2]]),
            cache=[],
            logits_indices=mx.array([1], dtype=mx.int32),
        )

        expected = mx.tanh(mx.array([[[2.0, 4.0]]]) / 2.0) * 2.0
        mx.eval(output.logits, expected)
        assert mx.allclose(output.logits, expected).item()


class TestSupportsSelectiveLogits:
    """The load-time probe that decides whether the split backbone/head path
    reproduces a model's own output head.

    A wrapper may scale inside its own ``__call__``, which the split path would
    not reproduce, so the probe compares a one-row forward both ways and only
    accepts an exact match."""

    def test_accepts_a_head_reproduced_by_the_split_path(self) -> None:
        class Model:
            def __init__(self) -> None:
                self.model = lambda input_ids, cache=None: mx.array([[[1.0, 2.0]]])
                self.final_logit_softcapping = None

            def __call__(self, input_ids, cache=None):
                return self.compute_logits(self.model(input_ids, cache=cache))

            def compute_logits(self, hidden_states):
                return hidden_states * 3.0

        assert DefaultModelAdapter().supports_selective_logits(Model()) is True

    def test_rejects_a_head_with_scaling_hidden_in_the_forward(self) -> None:
        # The wrapper multiplies after the head, the way Cohere's logit_scale or
        # Granite's logits_scaling do.  The split path cannot see that, so the
        # probe must decline rather than return wrong logits.
        class Model:
            def __init__(self) -> None:
                self.model = lambda input_ids, cache=None: mx.array([[[1.0, 2.0]]])
                self.final_logit_softcapping = None

            def __call__(self, input_ids, cache=None):
                return self.compute_logits(self.model(input_ids, cache=cache)) * 0.5

            def compute_logits(self, hidden_states):
                return hidden_states * 3.0

        assert DefaultModelAdapter().supports_selective_logits(Model()) is False

    def test_rejects_a_model_the_split_path_cannot_drive(self) -> None:
        # No `.model` backbone: the split path raises, so the probe declines
        # instead of propagating out of load_model.
        class Model:
            def __call__(self, input_ids, cache=None):
                del input_ids, cache
                return mx.array([[[1.0, 2.0]]])

        assert DefaultModelAdapter().supports_selective_logits(Model()) is False


class TestTargetForwardEmbeddings:
    def test_target_input_embeddings_use_target_embed_scale(self) -> None:
        class Embedding:
            def __call__(self, input_ids):
                return mx.ones((*input_ids.shape, 2))

        model = SimpleNamespace(
            model=SimpleNamespace(
                embed_tokens=Embedding(),
                embed_scale=3.0,
            )
        )

        embeddings = DefaultModelAdapter().target_input_embeddings(
            model,
            mx.array([[1, 2]], dtype=mx.int32),
        )

        assert embeddings.tolist() == [[[3.0, 3.0], [3.0, 3.0]]]

    def test_target_input_embeddings_use_language_model_wrapper(self) -> None:
        class Embedding:
            def __call__(self, input_ids):
                return mx.ones((*input_ids.shape, 2))

        language_model = SimpleNamespace(
            model=SimpleNamespace(
                embed_tokens=Embedding(),
                embed_scale=4.0,
            )
        )
        model = SimpleNamespace(language_model=language_model)

        embeddings = DefaultModelAdapter().target_input_embeddings(
            model,
            mx.array([[1, 2]], dtype=mx.int32),
        )

        assert embeddings.tolist() == [[[4.0, 4.0], [4.0, 4.0]]]

    def test_rejects_batched_target_hidden_states(self) -> None:
        class Backbone:
            def __call__(self, input_ids, *, cache=None):
                del input_ids, cache
                return mx.zeros((2, 1, 4))

        class Head:
            def __call__(self, hidden_states):
                return hidden_states

        model = SimpleNamespace(model=Backbone(), lm_head=Head())

        with pytest.raises(ValueError, match="row-major"):
            DefaultModelAdapter().target_forward(
                model,
                mx.array([[1]]),
                cache=[],
                collect_hidden_states=True,
            )


class TestNormalizeModelConfig:
    """Tests for normalize_model_config()."""

    def test_clears_multimodal_config_for_gemma4(self) -> None:
        model_config = SimpleNamespace(
            multimodal_config=SimpleNamespace(language_model_only=False),
            hf_config=SimpleNamespace(model_type="gemma4"),
        )

        DefaultModelAdapter().normalize_model_config(model_config)

        assert model_config.multimodal_config is None

    def test_clears_multimodal_config_for_qwen35_fp8_in_auto_mode(self) -> None:
        model_config = SimpleNamespace(
            multimodal_config=SimpleNamespace(language_model_only=False),
            hf_config=SimpleNamespace(
                model_type="qwen3_5",
                architectures=["Qwen3_5ForConditionalGeneration"],
                quantization_config={"quant_method": "fp8"},
            ),
        )

        DefaultModelAdapter().normalize_model_config(model_config)

        assert model_config.multimodal_config is None

    @pytest.mark.parametrize(
        ("model_type", "architecture"),
        [
            ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
            ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
            ("qwen3_6", "Qwen3_6ForConditionalGeneration"),
        ],
    )
    def test_clears_multimodal_config_for_mlx_quant_wrapper_in_auto_mode(
        self, model_type: str, architecture: str
    ) -> None:
        model_config = SimpleNamespace(
            multimodal_config=SimpleNamespace(language_model_only=False),
            hf_config=SimpleNamespace(
                model_type=model_type,
                architectures=[architecture],
                quantization={"group_size": 64, "bits": 4, "mode": "affine"},
            ),
        )

        DefaultModelAdapter().normalize_model_config(model_config)

        assert model_config.multimodal_config is None

    def test_preserves_multimodal_config_for_other_models(self) -> None:
        sentinel = SimpleNamespace(language_model_only=False)
        model_config = SimpleNamespace(
            multimodal_config=sentinel,
            hf_config=SimpleNamespace(model_type="qwen3_vl", architectures=[]),
        )

        DefaultModelAdapter().normalize_model_config(model_config)

        assert model_config.multimodal_config is sentinel

    def test_multimodal_native_mode_preserves_qwen35_fp8_multimodal_config(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "multimodal-native")
        reset_config()

        sentinel = SimpleNamespace(language_model_only=False)
        model_config = SimpleNamespace(
            multimodal_config=sentinel,
            hf_config=SimpleNamespace(
                model_type="qwen3_5",
                architectures=["Qwen3_5ForConditionalGeneration"],
                quantization_config={"quant_method": "fp8"},
            ),
        )

        DefaultModelAdapter().normalize_model_config(model_config)

        assert model_config.multimodal_config is sentinel

    def test_noop_when_multimodal_config_already_none(self) -> None:
        model_config = SimpleNamespace(
            multimodal_config=None,
            hf_config=SimpleNamespace(model_type="gemma4"),
        )

        DefaultModelAdapter().normalize_model_config(model_config)

        assert model_config.multimodal_config is None


class TestTextModel:
    def test_returns_language_model_when_present(self) -> None:
        language_model = object()
        vlm = SimpleNamespace(language_model=language_model)
        adapter = DefaultModelAdapter()
        assert adapter.text_model(vlm) is language_model

    def test_returns_model_when_no_language_model(self) -> None:
        model = object()
        adapter = DefaultModelAdapter()
        assert adapter.text_model(model) is model


def _body_stub(input_ids, cache=None):
    return mx.array([[7.0]])


class TestIntermediateForward:
    """Adapter-owned projection-free forward for intermediate chunks."""

    def test_supports_text_model_body(self) -> None:
        adapter = DefaultModelAdapter()
        assert adapter.supports_intermediate_forward(SimpleNamespace(model=_body_stub))

    def test_supports_wrapped_language_model_body(self) -> None:
        adapter = DefaultModelAdapter()
        model = SimpleNamespace(language_model=SimpleNamespace(model=_body_stub))
        assert adapter.supports_intermediate_forward(model)

    def test_unresolvable_structure_is_unsupported(self) -> None:
        adapter = DefaultModelAdapter()
        assert not adapter.supports_intermediate_forward(SimpleNamespace())

    def test_runs_body_and_returns_typed_hidden_states(self) -> None:
        # Arrange
        captured: dict[str, object] = {}
        hidden = mx.zeros((1, 2, 8))
        sentinel_cache = object()

        def body(input_ids, cache=None):
            captured["input_ids"] = input_ids.tolist()
            captured["cache"] = cache
            return hidden

        adapter = DefaultModelAdapter()
        model = SimpleNamespace(model=body)

        # Act
        output = adapter.intermediate_forward(
            model, mx.array([[5, 6]]), cache=sentinel_cache
        )

        # Assert
        assert isinstance(output, IntermediateForwardOutput)
        assert output.hidden_states is hidden
        assert captured["input_ids"] == [[5, 6]]
        assert captured["cache"] is sentinel_cache

    def test_unsupported_model_raises(self) -> None:
        adapter = DefaultModelAdapter()
        with pytest.raises(ValueError, match="supports_intermediate_forward"):
            adapter.intermediate_forward(SimpleNamespace(), mx.array([[5]]))


class _Qwen35LanguageModelStub:
    """Mirrors mlx_vlm 0.4.x ``LanguageModel.__call__`` so signature sniffing works.

    Also exposes ``self.model.embed_tokens`` to mirror the bottom-level
    embedding callable the real LM holds — ``from_loaded_model`` resolves
    it at load time.
    """

    def __init__(self) -> None:
        self.model = SimpleNamespace(embed_tokens=lambda input_ids: input_ids)

    def __call__(
        self,
        inputs: object,
        inputs_embeds: object | None = None,
        cache: object | None = None,
        position_ids: object | None = None,
        mask: object | None = None,
    ) -> None:
        return None


class _PaddleOCRLanguageModelStub:
    def __init__(self) -> None:
        self.model = SimpleNamespace(embed_tokens=lambda input_ids: input_ids)

    def __call__(
        self,
        inputs: object,
        inputs_embeds: object | None = None,
        cache: object | None = None,
        position_ids: object | None = None,
        mask: object | None = None,
    ) -> None:
        return None


class TestBuildMultimodalAdapter:
    def test_builds_qwen35_adapter_from_loaded_vlm(self) -> None:
        vision_tower = object()
        language_model = _Qwen35LanguageModelStub()
        model = SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(spatial_merge_size=2),
            ),
            vision_tower=vision_tower,
            language_model=language_model,
        )
        hf_config = SimpleNamespace(model_type="qwen3_5")

        adapter = DefaultModelAdapter().build_multimodal_adapter(model, hf_config)

        assert isinstance(adapter, Qwen3VLMultimodalAdapter)
        assert adapter.text_model() is language_model

    def test_builds_qwen3_vl_adapter_from_architecture(self) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(spatial_merge_size=2),
            ),
            vision_tower=object(),
            language_model=_Qwen35LanguageModelStub(),
        )
        hf_config = SimpleNamespace(
            model_type="custom",
            architectures=["Qwen3VLForConditionalGeneration"],
        )

        adapter = DefaultModelAdapter().build_multimodal_adapter(model, hf_config)

        assert isinstance(adapter, Qwen3VLMultimodalAdapter)

    def test_builds_paddleocr_vl_adapter_from_model_type(self) -> None:
        language_model = _PaddleOCRLanguageModelStub()
        model = SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(spatial_merge_size=2),
            ),
            visual=object(),
            language_model=language_model,
        )
        hf_config = SimpleNamespace(model_type="paddleocr_vl")

        adapter = DefaultModelAdapter().build_multimodal_adapter(model, hf_config)

        assert isinstance(adapter, PaddleOCRVLMultimodalAdapter)
        assert adapter.text_model() is language_model

    def test_builds_paddleocr_vl_adapter_from_architecture(self) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(spatial_merge_size=2),
            ),
            visual=object(),
            language_model=_PaddleOCRLanguageModelStub(),
        )
        hf_config = SimpleNamespace(
            model_type="custom",
            architectures=["PaddleOCRVLForConditionalGeneration"],
        )

        adapter = DefaultModelAdapter().build_multimodal_adapter(model, hf_config)

        assert isinstance(adapter, PaddleOCRVLMultimodalAdapter)

    def test_generic_vlm_has_no_model_owned_adapter(self) -> None:
        model = SimpleNamespace()
        hf_config = SimpleNamespace(model_type="phi3_v")

        adapter = DefaultModelAdapter().build_multimodal_adapter(model, hf_config)

        assert adapter is None


class TestResolveMaxHeadDim:
    """Tests for resolve_max_head_dim()."""

    def test_returns_global_when_larger(self) -> None:
        args = {"global_head_dim": 512}
        head_dim = 256
        adapter = DefaultModelAdapter()
        result = adapter.resolve_max_head_dim(args, head_dim)
        assert result == 512

    def test_returns_head_dim_when_larger(self) -> None:
        args = {"global_head_dim": 128}
        head_dim = 256
        adapter = DefaultModelAdapter()
        result = adapter.resolve_max_head_dim(args, head_dim)
        assert result == 256

    def test_returns_head_dim_when_global_missing(self) -> None:
        args = {}
        head_dim = 128
        adapter = DefaultModelAdapter()
        result = adapter.resolve_max_head_dim(args, head_dim)
        assert result == 128

    def test_returns_none_when_head_dim_none(self) -> None:
        args = {"global_head_dim": 512}
        head_dim = None
        adapter = DefaultModelAdapter()
        result = adapter.resolve_max_head_dim(args, head_dim)
        assert result is None


class TestYocoCacheIntegration:
    """Integration tests for YOCO KV cache sharing.

    Verifies reduced cache-byte accounting, backend wiring, and
    shared layers reusing the remapped cache slot.
    """

    # Gemma4-like config: 8 layers, 3 shared, alternating types
    _LAYER_TYPES = [
        "sliding",
        "sliding",
        "full",
        "sliding",
        "full",
        "sliding",
        "full",
        "sliding",
    ]
    _NUM_HIDDEN = len(_LAYER_TYPES)
    _NUM_SHARED = 3
    _NUM_UNIQUE = _NUM_HIDDEN - _NUM_SHARED
    _NUM_KV_HEADS = 2
    _HEAD_DIM = 4
    _BLOCK_SIZE = 16
    _VOCAB_SIZE = 100

    def _gemma4_args(self) -> dict:
        return {
            "vocab_size": self._VOCAB_SIZE,
            "num_hidden_layers": self._NUM_HIDDEN,
            "num_kv_shared_layers": self._NUM_SHARED,
            "layer_types": list(self._LAYER_TYPES),
            "num_key_value_heads": self._NUM_KV_HEADS,
            "num_attention_heads": self._NUM_KV_HEADS,
            "head_dim": self._HEAD_DIM,
        }

    def test_num_kv_cache_layers_reduced(self) -> None:
        """Runner's num_kv_cache_layers uses unique count, not total."""
        from tests.stub_runner import make_stub_runner

        args = self._gemma4_args()
        runner = make_stub_runner(model_args=args)
        runner.num_layers = self._NUM_HIDDEN
        runner._model_adapter = DefaultModelAdapter()

        yoco = runner._model_adapter.build_yoco_cache_mapping(args)
        assert yoco is not None
        num_unique, _ = yoco
        assert num_unique == self._NUM_UNIQUE

    def test_cache_block_bytes_uses_unique_layers(self) -> None:
        """get_cache_block_size_bytes should use num_kv_cache_layers, not num_layers."""
        import mlx.core as mx

        from tests.stub_runner import make_stub_runner

        args = self._gemma4_args()
        runner = make_stub_runner(model_args=args)
        runner.num_layers = self._NUM_HIDDEN
        runner.num_kv_cache_layers = self._NUM_UNIQUE
        runner.num_kv_heads = self._NUM_KV_HEADS
        runner.head_dim = self._HEAD_DIM
        runner.kv_cache_dtype = mx.float16
        runner.cache_config = SimpleNamespace(block_size=self._BLOCK_SIZE)

        block_bytes = runner.get_cache_block_size_bytes()

        # 2 (K+V) * num_unique * block_size * kv_heads * head_dim * dtype
        dtype_size = mx.float16.size
        expected = (
            2
            * self._NUM_UNIQUE
            * self._BLOCK_SIZE
            * self._NUM_KV_HEADS
            * self._HEAD_DIM
            * dtype_size
        )
        assert block_bytes == expected

    def test_backend_uses_compact_layer_count(self) -> None:
        """Runner backend factory should create MHA backend with reduced num_layers."""
        import mlx.core as mx

        from tests.stub_runner import make_stub_runner

        adapter = DefaultModelAdapter()
        args = self._gemma4_args()
        yoco = adapter.build_yoco_cache_mapping(args)

        runner = make_stub_runner(
            model_args=args,
            num_layers=self._NUM_HIDDEN,
            num_kv_heads=self._NUM_KV_HEADS,
            head_dim=self._HEAD_DIM,
            kv_cache_dtype=mx.float16,
            _model_adapter=adapter,
            _yoco_cache_mapping=yoco,
            num_kv_cache_layers=yoco[0],
        )

        backend = runner.build_paged_attention_runtime(block_size=self._BLOCK_SIZE)

        assert backend._num_layers == self._NUM_UNIQUE
        assert backend._cache_idx_map is not None
        # Shared layers map to a unique layer of same type
        for i in range(self._NUM_UNIQUE, self._NUM_HIDDEN):
            ref = backend._cache_idx_map[i]
            assert ref < self._NUM_UNIQUE
            assert self._LAYER_TYPES[ref] == self._LAYER_TYPES[i]

    def test_shared_layer_reuses_cache_slot(self) -> None:
        """Shared layers should get same cache_idx as their reference layer."""
        adapter = DefaultModelAdapter()
        args = self._gemma4_args()
        result = adapter.build_yoco_cache_mapping(args)
        assert result is not None
        _, mapping = result

        # Layer 5 (sliding, shared) should map to the same cache_idx as
        # the last unique sliding layer in 0..4
        shared_sliding = 5
        ref = mapping[shared_sliding]
        assert self._LAYER_TYPES[ref] == "sliding"
        assert ref < self._NUM_UNIQUE

        # Layer 6 (full, shared) → same cache_idx as last unique full layer
        shared_full = 6
        ref_full = mapping[shared_full]
        assert self._LAYER_TYPES[ref_full] == "full"
        assert ref_full < self._NUM_UNIQUE


class TestRequireUniformKvHeads:
    """Tests for require_uniform_kv_heads().

    The method still raises on mismatched KV head counts under the uniform
    cache path.  For models whose adapter populates ``kv_heads_per_layer``
    via :meth:`build_per_layer_kv_shapes` (Gemma4 26B/31B), the uniform
    guard is gated out at the call-site in ``ModelCachePolicy`` — see
    :class:`tests.test_model_lifecycle.TestResolveModelDims` for that path.
    """

    def test_allows_uniform_heads(self) -> None:
        args = {"num_global_key_value_heads": 8}
        num_kv_heads = 8
        adapter = DefaultModelAdapter()
        adapter.require_uniform_kv_heads(args, num_kv_heads)

    def test_allows_missing_global(self) -> None:
        args: dict[str, int] = {}
        num_kv_heads = 8
        adapter = DefaultModelAdapter()
        adapter.require_uniform_kv_heads(args, num_kv_heads)

    def test_rejects_gemma4_31b_config(self) -> None:
        args = {"num_global_key_value_heads": 4}
        num_kv_heads = 16
        adapter = DefaultModelAdapter()
        with pytest.raises(ValueError, match="variable KV head count"):
            adapter.require_uniform_kv_heads(args, num_kv_heads)

    def test_rejects_gemma4_26b_config(self) -> None:
        args = {"num_global_key_value_heads": 2}
        num_kv_heads = 8
        adapter = DefaultModelAdapter()
        with pytest.raises(ValueError, match="VLLM_METAL_USE_PAGED_ATTENTION=0"):
            adapter.require_uniform_kv_heads(args, num_kv_heads)


class TestBuildPerLayerKVShapes:
    """Tests for :meth:`DefaultModelAdapter.build_per_layer_kv_shapes`."""

    def test_returns_none_for_uniform_model(self) -> None:
        adapter = DefaultModelAdapter()
        result = adapter.build_per_layer_kv_shapes(
            args={},
            num_layers=4,
            num_kv_heads=8,
            head_dim=128,
        )
        assert result is None

    def test_returns_none_when_layer_types_length_mismatches(self) -> None:
        adapter = DefaultModelAdapter()
        args = {
            "layer_types": ["sliding_attention", "full_attention"],
            "global_head_dim": 512,
            "num_global_key_value_heads": 4,
        }
        result = adapter.build_per_layer_kv_shapes(
            args=args,
            num_layers=4,
            num_kv_heads=16,
            head_dim=256,
        )
        assert result is None

    def test_returns_none_without_global_head_dim(self) -> None:
        """No ``global_head_dim`` → uniform path, even with layer_types present."""
        adapter = DefaultModelAdapter()
        args = {"layer_types": ["full_attention", "full_attention"]}
        result = adapter.build_per_layer_kv_shapes(
            args=args,
            num_layers=2,
            num_kv_heads=8,
            head_dim=128,
        )
        assert result is None

    def test_gemma4_31b_style_full_override(self) -> None:
        """31B-style configs with distinct full-attention KV heads and head_dim."""
        adapter = DefaultModelAdapter()
        args = {
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            "global_head_dim": 512,
            "num_global_key_value_heads": 4,
        }
        result = adapter.build_per_layer_kv_shapes(
            args=args,
            num_layers=4,
            num_kv_heads=16,
            head_dim=256,
        )

        assert result == ([16, 4, 16, 4], [256, 512, 256, 512])

    def test_gemma4_e2b_style_head_dim_only_override(self) -> None:
        """E2B-style configs omit ``num_global_key_value_heads`` entirely.

        Full-attention layers must fall back to the sliding-layer KV-head
        count instead of collapsing to a uniform shape, which would cause
        full-attention writes to overflow the allocated cache head_dim.
        """
        adapter = DefaultModelAdapter()
        args = {
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            "global_head_dim": 512,
        }
        result = adapter.build_per_layer_kv_shapes(
            args=args,
            num_layers=4,
            num_kv_heads=1,
            head_dim=256,
        )

        assert result == ([1, 1, 1, 1], [256, 512, 256, 512])

    def test_unknown_layer_type_raises(self) -> None:
        """Unknown layer_type surfaces as a ValueError pinpointing the index
        rather than silently collapsing to full-attention shapes."""
        adapter = DefaultModelAdapter()
        args = {
            "layer_types": ["sliding_attention", "linear_attention"],
            "global_head_dim": 512,
            "num_global_key_value_heads": 4,
        }
        with pytest.raises(
            ValueError,
            match=r"Unsupported Gemma4 layer_type at index 1: 'linear_attention'",
        ):
            adapter.build_per_layer_kv_shapes(
                args=args,
                num_layers=2,
                num_kv_heads=16,
                head_dim=256,
            )


class TestBuildSlidingWindowPerLayer:
    """Tests for build_sliding_window_per_layer()."""

    def test_returns_none_for_model_without_layer_types(self) -> None:
        adapter = DefaultModelAdapter()
        result = adapter.build_sliding_window_per_layer({}, num_layers=4)
        assert result is None

    def test_returns_none_without_sliding_window_config(self) -> None:
        args = {"layer_types": ["sliding_attention", "full_attention"]}
        adapter = DefaultModelAdapter()
        result = adapter.build_sliding_window_per_layer(args, num_layers=2)
        assert result is None

    def test_gemma4_sliding_layers_get_window_full_layers_disabled(self) -> None:
        args = {
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
            ],
            "sliding_window": 1024,
        }
        adapter = DefaultModelAdapter()
        result = adapter.build_sliding_window_per_layer(args, num_layers=4)
        assert result == [1024, 1024, -1, 1024]

    def test_length_mismatch_returns_none(self) -> None:
        args = {
            "layer_types": ["sliding_attention"],
            "sliding_window": 1024,
        }
        adapter = DefaultModelAdapter()
        result = adapter.build_sliding_window_per_layer(args, num_layers=4)
        assert result is None
