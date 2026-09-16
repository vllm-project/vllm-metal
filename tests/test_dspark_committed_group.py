# SPDX-License-Identifier: Apache-2.0
"""The DSpark drafter's committed context as a scheduler-owned KV-cache group.

These pin the runner/policy seam that a serve run exercises but a unit test on the
proposer alone cannot: the specs the policy registers for the drafter's layers, how the
proposer-local scratch tail is sized, and that the group handoff goes to any drafter
that consumes a committed group -- not only to a ``DraftModelProposer``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

from tests.stub_runner import make_stub_runner
from tests.test_dspark_proposer import _proposer
from vllm_metal.v1.draft_model_proposer import DraftDims

DSPARK = DraftDims(num_layers=5, num_kv_heads=8, head_dim=128, lookahead_positions=7)
NAMES = tuple(f"draft_layers.{i}.self_attn" for i in range(5))


def _runner(dims=DSPARK, *, max_num_seqs=16):
    runner = make_stub_runner()
    runner._draft_dims = dims
    # The stub carries no scheduler config on the runner itself; the policy reads
    # ``max_num_seqs`` from there, so give it exactly that.
    runner.scheduler_config = SimpleNamespace(max_num_seqs=max_num_seqs)
    return runner


class TestTheDrafterLayersAreRegisteredLikeTheDraftModels:
    def test_one_full_attention_spec_per_drafter_layer(self):
        policy = _runner()._cache_policy
        specs = policy._draft_layer_specs(block_size=16, torch_dtype=torch.float16)
        assert tuple(specs) == NAMES
        for spec in specs.values():
            assert isinstance(spec, FullAttentionSpec)
            assert (spec.block_size, spec.num_kv_heads, spec.head_size) == (16, 8, 128)

    def test_no_drafter_registers_nothing(self):
        policy = _runner(dims=None)._cache_policy
        assert policy._draft_layer_specs(block_size=16, torch_dtype=torch.float16) == {}


class TestTheScratchTailIsSizedByTheDrafterWidth:
    """The tail covers every active request drafting its whole block at once.

    The draft model looks ahead ``num_speculative_tokens``; a block drafter writes its
    block. ``DraftDims.lookahead_positions`` carries whichever applies, so this sizing
    does not silently assume the spec's K for a drafter that writes seven.
    """

    @pytest.mark.parametrize(("width", "seqs"), [(7, 16), (17, 16), (7, 3), (33, 2)])
    def test_max_num_seqs_times_pages_for_the_width(self, width, seqs):
        dims = DraftDims(
            num_layers=5, num_kv_heads=8, head_dim=128, lookahead_positions=width
        )
        policy = _runner(dims, max_num_seqs=seqs)._cache_policy
        assert policy.draft_scratch_reserve_blocks() == seqs * cdiv(width, 16)

    def test_zero_without_a_drafter(self):
        assert _runner(dims=None)._cache_policy.draft_scratch_reserve_blocks() == 0


class TestTheGroupHandoffAsksForTheCapability:
    """`_adopt_draft_scheduler_group` serves any `CommittedKVGroupConsumer`."""

    @staticmethod
    def _config(group_names=NAMES):
        spec = FullAttentionSpec(
            block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16
        )
        return KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["model.layers.0.self_attn"], kv_cache_spec=spec
                ),
                KVCacheGroupSpec(layer_names=list(group_names), kv_cache_spec=spec),
            ],
        )

    def test_a_dspark_proposer_is_told_its_group(self):
        runner = _runner()
        runner._drafter = _proposer()
        runner._cache_policy._adopt_draft_scheduler_group(self._config())
        assert runner._drafter._committed_group_index == 1

    def test_a_drafter_that_holds_no_group_is_refused_loudly(self):
        class Ngramish:
            capture_layer_ids = None

            def propose(self, ctx):
                return None

            def release_requests(self, req_ids):
                pass

        runner = _runner()
        runner._drafter = Ngramish()
        with pytest.raises(RuntimeError, match="does not consume a committed KV group"):
            runner._cache_policy._adopt_draft_scheduler_group(self._config())

    def test_layers_split_across_groups_are_refused(self):
        runner = _runner()
        runner._drafter = _proposer()
        spec = FullAttentionSpec(
            block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16
        )
        split = KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=list(NAMES[:2]), kv_cache_spec=spec),
                KVCacheGroupSpec(layer_names=list(NAMES[2:]), kv_cache_spec=spec),
            ],
        )
        with pytest.raises(NotImplementedError, match="one scheduler KV cache group"):
            runner._cache_policy._adopt_draft_scheduler_group(split)


class TestPrefixCachingIsRefusedWithAScheduledContext:
    """What the docs promise: a loud refusal at startup, not stale draft KV later."""

    def test_refused_when_on(self):
        from vllm_metal.v1.dspark.paging import require_prefix_caching_off

        with pytest.raises(ValueError, match="no-enable-prefix-caching"):
            require_prefix_caching_off(True)

    def test_allowed_when_off(self):
        from vllm_metal.v1.dspark.paging import require_prefix_caching_off

        require_prefix_caching_off(False)
