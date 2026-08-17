# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the compiled stateless-MLP decode dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.qwen3_next import (
    ModelArgs,
    Qwen3NextMLP,
    Qwen3NextSparseMoeBlock,
)
from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

import vllm_metal.envs as envs
from tests.stub_runner import make_stub_runner
from vllm_metal.compiled_mlp import (
    CompiledMLPBlock,
    CompiledMLPBlocks,
    CompiledTargetVerifyMLPBlock,
)
from vllm_metal.v1.model_lifecycle import ModelLifecycle

DIM = 64
HIDDEN = 128


def _require_metal() -> None:
    try:
        available = mx.metal.is_available()
    except AttributeError:
        available = False
    if not available:
        pytest.skip("MLX Metal is not available")


@pytest.fixture(autouse=True)
def _clear_flag(monkeypatch):
    monkeypatch.delenv("VLLM_METAL_COMPILED_MLP", raising=False)
    yield


class _Host(nn.Module):
    """Minimal module tree hosting one MLP block (the install target)."""

    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp


def _dense_mlp() -> Qwen3NextMLP:
    mx.random.seed(11)
    mlp = Qwen3NextMLP(DIM, HIDDEN)
    mlp.eval()  # mirror serve state: mlx_lm load puts models in eval mode
    mx.eval(mlp.parameters())
    return mlp


def _moe_block() -> Qwen3NextSparseMoeBlock:
    mx.random.seed(12)
    args = ModelArgs(
        model_type="qwen3_next",
        hidden_size=DIM,
        num_hidden_layers=1,
        intermediate_size=HIDDEN,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=HIDDEN,
        mlp_only_layers=[],
        moe_intermediate_size=HIDDEN,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        rope_theta=10000.0,
        partial_rotary_factor=0.25,
        max_position_embeddings=2048,
        head_dim=16,
    )
    block = Qwen3NextSparseMoeBlock(args)
    block.eval()
    mx.eval(block.parameters())
    return block


def _install(monkeypatch, model: nn.Module) -> int:
    monkeypatch.setenv("VLLM_METAL_COMPILED_MLP", "1")
    return CompiledMLPBlocks.install(model)


def _spy_compiled(wrapper: CompiledMLPBlock) -> dict[str, int]:
    """Reach spy: count dispatches through the compiled trace."""
    calls = {"n": 0}
    orig = wrapper._compiled

    def spy(x):
        calls["n"] += 1
        return orig(x)

    wrapper._compiled = spy
    return calls


class TestInstall:
    def test_flag_defaults_off(self):
        # Assert — opt-in while the dispatch gathers serve mileage.
        assert envs.VLLM_METAL_COMPILED_MLP is False

    def test_kill_switch_leaves_modules_untouched(self, monkeypatch):
        # Arrange
        _require_metal()
        mlp = _dense_mlp()
        host = _Host(mlp)
        monkeypatch.setenv("VLLM_METAL_COMPILED_MLP", "0")

        # Act
        wrapped = CompiledMLPBlocks.install(host)

        # Assert
        assert wrapped == 0
        assert host.mlp is mlp

    def test_install_wraps_exact_types_only(self, monkeypatch):
        # Arrange
        _require_metal()

        class NotQuiteMLP(Qwen3NextMLP):
            pass

        host = _Host(_dense_mlp())
        sub = NotQuiteMLP(DIM, HIDDEN)
        sub_host = _Host(sub)

        # Act
        wrapped = _install(monkeypatch, host)
        wrapped_sub = _install(monkeypatch, sub_host)

        # Assert: exact type wraps; a subclass is not silently claimed.
        assert wrapped == 1
        assert isinstance(host.mlp, CompiledMLPBlock)
        assert wrapped_sub == 0
        assert sub_host.mlp is sub

    def test_install_counts_every_target_in_a_nested_tree(self, monkeypatch):
        # Arrange — dense and MoE targets at different depths.
        _require_metal()

        class _Deep(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = [_Host(_dense_mlp()), _Host(_moe_block())]
                self.head = _Host(_dense_mlp())

        deep = _Deep()

        # Act
        wrapped = _install(monkeypatch, deep)

        # Assert — the MoE block counts once; its inner shared expert is
        # part of the compiled trace, not a second install target.
        assert wrapped == 3
        assert all(isinstance(h.mlp, CompiledMLPBlock) for h in deep.layers)
        assert isinstance(deep.head.mlp, CompiledMLPBlock)

    def test_install_is_idempotent(self, monkeypatch):
        # Arrange — the MoE block nests another target (its shared-expert
        # MLP); a re-install must not wrap anything beneath the existing
        # wrapper.
        _require_metal()
        host = _Host(_moe_block())

        # Act
        first = _install(monkeypatch, host)
        wrapper = host.mlp
        second = _install(monkeypatch, host)

        # Assert
        assert first == 1
        assert second == 0
        assert host.mlp is wrapper
        assert not isinstance(wrapper.inner.shared_expert, CompiledMLPBlock)

    def test_vlm_moe_family_is_registered(self, monkeypatch):
        # Arrange — the qwen3_5_moe package's blocks use the target_verify
        # convention; the dense variant is cheap to construct, the sparse
        # block's registration and wrapper class are pinned via the policy.
        _require_metal()
        from mlx_vlm.models.qwen3_5_moe.language import (
            Qwen3_5MoeMLP,
            Qwen3_5MoeSparseMoeBlock,
        )

        policies = CompiledMLPBlocks._target_policies()
        assert policies[Qwen3_5MoeSparseMoeBlock] is CompiledTargetVerifyMLPBlock
        mx.random.seed(15)
        inner = Qwen3_5MoeMLP(DIM, HIDDEN)
        inner.eval()
        mx.eval(inner.parameters())
        host = _Host(inner)
        x = mx.random.normal((1, 2, DIM)).astype(mx.float16)
        reference = inner(x)

        # Act
        assert _install(monkeypatch, host) == 1
        calls = _spy_compiled(host.mlp)
        out = host.mlp(x)
        out_verify = host.mlp(x, True)

        # Assert
        assert isinstance(host.mlp, CompiledTargetVerifyMLPBlock)
        assert calls["n"] == 1  # plain engaged; target_verify=True eager
        mx.eval(reference, out, out_verify)
        np.testing.assert_array_equal(
            np.array(reference.astype(mx.float32)),
            np.array(out.astype(mx.float32)),
        )

    def test_non_module_install_target_fails_fast(self, monkeypatch):
        # Arrange / Act / Assert
        monkeypatch.setenv("VLLM_METAL_COMPILED_MLP", "1")
        with pytest.raises(TypeError, match="expects an mlx nn.Module"):
            CompiledMLPBlocks.install(object())

    def test_weights_stay_in_the_parameter_tree(self, monkeypatch):
        # Arrange — the wrapper registers the inner module normally.
        _require_metal()
        host = _Host(_dense_mlp())
        from mlx.utils import tree_flatten

        before = {k for k, _ in tree_flatten(host.parameters())}

        # Act
        _install(monkeypatch, host)
        after = {k for k, _ in tree_flatten(host.parameters())}

        # Assert — same parameter set, re-rooted under the wrapper.
        assert {k.replace("mlp.", "mlp.inner.") for k in before} == after


class TestDispatch:
    @pytest.mark.parametrize("block", ["dense", "moe"])
    def test_decode_shaped_call_engages_and_is_bitwise_identical(
        self, monkeypatch, block
    ):
        # Arrange — quantized blocks, mirroring serve checkpoints: the
        # quantized path is bitwise-stable under compile, while unquantized
        # fp16 fusion may reorder accumulation at the 1e-9 ulp level.
        _require_metal()
        if block == "dense":
            inner = _dense_mlp()
            nn.quantize(inner, group_size=64, bits=4)
        else:
            inner = _moe_block()
            nn.quantize(inner.switch_mlp, group_size=64, bits=4)
            nn.quantize(inner.shared_expert, group_size=64, bits=4)
        mx.eval(inner.parameters())
        host = _Host(inner)
        x = mx.random.normal((1, 3, DIM)).astype(mx.float16)
        reference = inner(x)
        assert _install(monkeypatch, host) == 1
        calls = _spy_compiled(host.mlp)

        # Act
        out = host.mlp(x)

        # Assert — the compiled trace really ran, bitwise-equal output.
        assert calls["n"] == 1
        mx.eval(reference, out)
        np.testing.assert_array_equal(
            np.array(reference.astype(mx.float32)),
            np.array(out.astype(mx.float32)),
        )

    def test_prefill_sized_call_keeps_the_eager_path(self, monkeypatch):
        # Arrange — one past the token cap stays eager, at the cap compiles.
        _require_metal()
        host = _Host(_dense_mlp())
        assert _install(monkeypatch, host) == 1
        calls = _spy_compiled(host.mlp)
        cap = CompiledMLPBlocks.MAX_COMPILED_TOKENS
        at_cap = mx.random.normal((1, cap, DIM)).astype(mx.float16)
        over_cap = mx.random.normal((1, cap + 1, DIM)).astype(mx.float16)

        # Act
        host.mlp(at_cap)
        engaged_at_cap = calls["n"]
        host.mlp(over_cap)

        # Assert
        assert engaged_at_cap == 1
        assert calls["n"] == 1  # over-cap call stayed eager

    def test_vlm_mlp_wraps_and_extra_args_stay_eager(self, monkeypatch):
        # Arrange — mlx_vlm's dense MLP takes a target_verify kwarg; the
        # plain call compiles, the kwarg-carrying call stays eager.
        _require_metal()
        mx.random.seed(13)
        inner = Qwen3_5MLP(DIM, HIDDEN)
        inner.eval()
        mx.eval(inner.parameters())
        host = _Host(inner)
        x = mx.random.normal((1, 2, DIM)).astype(mx.float16)
        reference = inner(x)
        assert _install(monkeypatch, host) == 1
        calls = _spy_compiled(host.mlp)

        # Act — default-off target_verify (positional or named) is the
        # plain call and engages; a truthy flag stays eager, and an invalid
        # duplicate raises at the wrapper exactly like the unwrapped module.
        assert isinstance(host.mlp, CompiledTargetVerifyMLPBlock)
        out_plain = host.mlp(x)
        out_default = host.mlp(x, target_verify=False)
        out_verify = host.mlp(x, True)
        with pytest.raises(TypeError):
            host.mlp(x, False, target_verify=False)

        # Assert
        assert calls["n"] == 2  # plain + default-off engaged; others eager
        mx.eval(reference, out_plain, out_default, out_verify)
        np.testing.assert_array_equal(
            np.array(reference.astype(mx.float32)),
            np.array(out_plain.astype(mx.float32)),
        )

    def test_mlx_lm_target_rejects_stray_positional_flag(self, monkeypatch):
        # Arrange — the mlx_lm blocks take no extra arguments, so a stray
        # positional False must NOT be treated as the plain call; it
        # delegates to the inner block, which raises like the unwrapped
        # module would.
        _require_metal()
        host = _Host(_dense_mlp())
        assert _install(monkeypatch, host) == 1
        calls = _spy_compiled(host.mlp)
        x = mx.random.normal((1, 2, DIM)).astype(mx.float16)

        # Act / Assert
        with pytest.raises(TypeError):
            host.mlp(x, False)
        assert calls["n"] == 0

    def test_lora_serve_skips_the_install(self, monkeypatch):
        # Arrange — LoRA setup rebinds projections inside the blocks after
        # install; the lifecycle must keep LoRA serves on the eager path.
        def unexpected_install(model):
            raise AssertionError("LoRA serve must not install compiled MLPs")

        monkeypatch.setattr(CompiledMLPBlocks, "install", unexpected_install)
        runner = make_stub_runner(model=SimpleNamespace())
        runner.vllm_config.lora_config = object()
        lifecycle = ModelLifecycle(runner, runner._model_adapter)

        # Act / Assert (no raise)
        lifecycle.install_decode_dispatch()

    def test_train_mode_keeps_the_eager_path(self, monkeypatch):
        # Arrange — train()/eval() propagate through the registered inner.
        _require_metal()
        host = _Host(_dense_mlp())
        assert _install(monkeypatch, host) == 1
        calls = _spy_compiled(host.mlp)
        x = mx.random.normal((1, 2, DIM)).astype(mx.float16)

        # Act
        host.train()
        host.mlp(x)
        host.eval()
        host.mlp(x)

        # Assert
        assert calls["n"] == 1  # only the eval-mode call engaged


class TestLifecycleInstall:
    def test_install_decode_dispatch_threads_the_forward_model(self, monkeypatch):
        # Arrange — the lifecycle hook must hand the runner's forward model
        # (not some other object) to the installer.
        received: list[object] = []
        monkeypatch.setattr(
            CompiledMLPBlocks, "install", lambda model: received.append(model)
        )
        runner = make_stub_runner(model=SimpleNamespace())
        lifecycle = ModelLifecycle(runner, runner._model_adapter)

        # Act
        lifecycle.install_decode_dispatch()

        # Assert
        assert received == [runner._forward_model]

    def test_pooling_runner_skips_decode_dispatch_install(self, monkeypatch):
        # Arrange — pooling runners never decode, and their backends may
        # wrap the model in non-nn.Module shims install must not walk.
        def unexpected_install(model):
            raise AssertionError("pooling runner must not install decode dispatch")

        monkeypatch.setattr(CompiledMLPBlocks, "install", unexpected_install)
        runner = make_stub_runner(
            model=SimpleNamespace(),
            model_config=SimpleNamespace(
                runner_type="pooling", get_head_size=lambda: 128, max_model_len=64
            ),
            _is_pooling=True,
            _pooling_backend=None,
        )
        lifecycle = ModelLifecycle(runner, runner._model_adapter)

        # Act / Assert (no raise)
        lifecycle.install_decode_dispatch()
