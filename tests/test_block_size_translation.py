# SPDX-License-Identifier: Apache-2.0
"""Unit tests for hybrid block-size translation in Metal paged attention.

Verifies that pick_kernel_block_size and build_block_tables correctly
translate large vLLM block sizes (e.g. 544 for hybrid models) into
kernel-compatible block sizes (8, 16, 32).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from vllm_metal.attention.block_tables import build_block_tables, pick_kernel_block_size
from vllm_metal.attention.context import (
    clear_context,
    get_context,
    prepare_grouped,
    prepare_unified,
)
from vllm_metal.attention.impls.sdpa import _kernel_metadata
from vllm_metal.metal.constants import KERNEL_BLOCK_SIZES, MLA_KERNEL_BLOCK_SIZES


class TestPickKernelBlockSize:
    """Tests for pick_kernel_block_size."""

    def test_returns_exact_match(self):
        for bs in KERNEL_BLOCK_SIZES:
            assert pick_kernel_block_size(bs) == bs

    def test_picks_largest_divisor(self):
        # 544 % 32 == 0, so should pick 32 (not 16 or 8)
        assert pick_kernel_block_size(544) == 32

    def test_picks_16_when_32_does_not_divide(self):
        # 48 % 32 != 0, but 48 % 16 == 0
        assert pick_kernel_block_size(48) == 16

    def test_picks_8_as_fallback(self):
        # 24 % 32 != 0, 24 % 16 != 0, but 24 % 8 == 0
        assert pick_kernel_block_size(24) == 8

    def test_raises_on_indivisible(self):
        with pytest.raises(ValueError, match="not divisible"):
            pick_kernel_block_size(7)


class TestBuildBlockTables:
    """Tests for build_block_tables."""

    def test_no_translation_for_supported_sizes(self):
        bt, kbs = build_block_tables([[0, 1], [2]], 16)
        assert kbs == 16
        assert bt.tolist() == [[0, 1], [2, 0]]

    def test_translation_single_block(self):
        # 544 -> 32, ratio=17
        bt, kbs = build_block_tables([[0], [1]], 544)
        assert kbs == 32
        ratio = 544 // 32  # 17
        # block 0 -> [0, 1, ..., 16]
        assert bt[0].tolist() == list(range(0, ratio))
        # block 1 -> [17, 18, ..., 33]
        assert bt[1].tolist() == list(range(ratio, 2 * ratio))

    def test_translation_multi_block(self):
        bt, kbs = build_block_tables([[0, 2]], 544)
        ratio = 544 // 32
        expected = list(range(0, ratio)) + list(range(2 * ratio, 3 * ratio))
        assert bt[0].tolist() == expected

    def test_translation_with_padding(self):
        # Unequal block table lengths — shorter rows are zero-padded before
        # expansion, so padding block_id=0 expands to [0, 1, …, ratio-1].
        # The kernel never reads these entries (bounded by context_len).
        bt, kbs = build_block_tables([[0, 1], [2]], 544)
        ratio = 544 // 32
        assert bt.shape[0] == 2
        assert bt.shape[1] == 2 * ratio
        # Second row: block 2 expanded, then padded block 0 expanded
        row1 = bt[1].tolist()
        assert row1[:ratio] == list(range(2 * ratio, 3 * ratio))
        assert row1[ratio:] == list(range(0, ratio))

    def test_output_shape(self):
        bt, kbs = build_block_tables([[0, 1, 2]], 544)
        ratio = 544 // 32
        assert bt.shape == (1, 3 * ratio)

    def test_empty_block_tables(self):
        bt, kbs = build_block_tables([], 16)
        assert bt.shape == (0, 0)
        assert kbs == 16

    def test_empty_block_tables_hybrid(self):
        bt, kbs = build_block_tables([], 544)
        assert bt.shape == (0, 0)
        assert kbs == 544


class TestKernelMetadataMemo:
    """Per-forward memo of kernel-format metadata (_kernel_metadata).

    The paged context lives for exactly one forward pass and its list
    metadata is identical for every layer of a KV group, so the converted
    mx arrays are built once per (group, cache block size) and reused by
    the remaining layers instead of being rebuilt per layer.
    """

    @pytest.fixture(autouse=True)
    def _clean_context(self):
        yield
        clear_context()

    def _fresh_ctx(self, block_ids, seq_len, num_tokens):
        prepare_unified([(block_ids, seq_len, num_tokens)], [], 16)
        return get_context()

    def test_returns_same_objects_within_one_context(self):
        ctx = self._fresh_ctx([0, 1, 2], 40, 4)
        first = _kernel_metadata(ctx, 0, ctx.slot_mapping, ctx.block_tables, 16)
        second = _kernel_metadata(ctx, 0, ctx.slot_mapping, ctx.block_tables, 16)
        assert second is first

    def test_matches_direct_conversion(self):
        ctx = self._fresh_ctx([0, 1, 2], 40, 4)
        meta = _kernel_metadata(ctx, 0, ctx.slot_mapping, ctx.block_tables, 16)
        direct_bt, direct_bs = build_block_tables(ctx.block_tables, 16)
        assert meta.block_tables.tolist() == direct_bt.tolist()
        assert meta.block_size == direct_bs
        assert meta.slot_mapping.tolist() == ctx.slot_mapping
        assert meta.seq_lens.tolist() == ctx.context_lens
        assert meta.cu_seqlens_q.tolist() == ctx.cu_seqlens
        assert meta.max_seq_len == max(ctx.context_lens)

    def test_new_context_is_not_served_stale_arrays(self):
        ctx1 = self._fresh_ctx([0, 1, 2], 40, 4)
        stale = _kernel_metadata(ctx1, 0, ctx1.slot_mapping, ctx1.block_tables, 16)
        ctx2 = self._fresh_ctx([7, 8], 16, 1)
        fresh = _kernel_metadata(ctx2, 0, ctx2.slot_mapping, ctx2.block_tables, 16)
        assert fresh is not stale
        assert fresh.block_tables.tolist() == [[7, 8]]

    def test_cache_keys_by_block_size(self):
        ctx = self._fresh_ctx([0, 1, 2], 32, 2)
        m16 = _kernel_metadata(ctx, 0, ctx.slot_mapping, ctx.block_tables, 16)
        m32 = _kernel_metadata(ctx, 0, ctx.slot_mapping, ctx.block_tables, 32)
        assert m32 is not m16
        assert m32.block_size == 32
        assert _kernel_metadata(ctx, 0, ctx.slot_mapping, ctx.block_tables, 16) is m16

    def test_cache_keys_by_group_with_equal_block_size(self):
        # Two scheduler groups with the SAME cache block size: only the
        # group index separates their memo entries.  Guards the key against
        # ever being reduced to cache_block_size alone, which would serve
        # group 0's tables to group 1 on hybrid models.
        prepare_grouped([(([10, 11], [77, 78]), 24)], [], (16, 16))
        ctx = get_context()
        g0 = ctx.kv_groups[0]
        g1 = ctx.kv_groups[1]
        m0 = _kernel_metadata(ctx, 0, g0.slot_mapping, g0.block_tables, 16)
        m1 = _kernel_metadata(ctx, 1, g1.slot_mapping, g1.block_tables, 16)
        assert m1 is not m0
        assert m0.block_tables.tolist() == [[10, 11]]
        assert m1.block_tables.tolist() == [[77, 78]]


class TestMLAKernelBlockSizes:
    """MLA_KERNEL_BLOCK_SIZES is the Python-side copy of the block sizes the
    mla.metal single-pass kernel is instantiated for — drift between them
    means the admission check admits a block size with no compiled kernel
    (or rejects one that exists).  Parse the instantiate_mla calls and keep
    the two in lockstep."""

    _MLA_METAL = (
        Path(__file__).resolve().parent.parent
        / "vllm_metal"
        / "metal"
        / "kernels_v2"
        / "mla.metal"
    )
    _PAGED_OPS = (
        Path(__file__).resolve().parent.parent
        / "vllm_metal"
        / "metal"
        / "paged_ops.cpp"
    )

    def _instantiations(self) -> set[tuple[str, int, int, int, int, int, int]]:
        """The full (type, kvr, pe, bs, g, nt, ps) tuple per call site.

        ``instantiate_mla(type, kv_lora_rank, qk_rope_head_dim, block_size,
        heads_per_tg, num_threads, partition_size)`` — a call at the start
        of a line; the #define and comments are excluded by that anchor.
        """
        src = self._MLA_METAL.read_text()
        rows = {
            (m.group(1), *map(int, m.groups()[1:]))
            for m in re.finditer(
                r"^instantiate_mla\(\s*(\w+),\s*(\d+),\s*(\d+),\s*(\d+),"
                r"\s*(\d+),\s*(\d+),\s*(\d+)\s*\)",
                src,
                re.M,
            )
        }
        assert rows, "no instantiate_mla call sites found in mla.metal"
        return rows

    @staticmethod
    def _fn_body(src: str, signature: str) -> str:
        """One top-level C++ function body: signature to its column-0 ``}``."""
        start = src.index(signature)
        return src[start : src.index("\n}\n", start)]

    def _gate_rows(self) -> set[tuple[int, int, int, int, int, int]]:
        """The (kvr, pe, bs, g, nt, ps) rows of the shared spec table.

        ``kMlaKernelSpecs`` in paged_ops.cpp is the single source of truth
        the dispatch gate validates against — one ``{kvr, pe, bs, g, nt,
        ps}`` row per instantiated shape.
        """
        cpp = self._PAGED_OPS.read_text()
        table = cpp[cpp.index("kMlaKernelSpecs") :]
        rows = {
            tuple(int(v) for v in m.groups())
            for m in re.finditer(
                r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,"
                r"\s*(\d+)\s*,\s*(\d+)\s*\}",
                table[: table.index("};")],
            )
        }
        assert rows, "could not parse the kMlaKernelSpecs table"
        return rows

    def test_matches_mla_metal_instantiations(self):
        instantiated = {bs for _, _, _, bs, _, _, _ in self._instantiations()}
        assert instantiated == set(MLA_KERNEL_BLOCK_SIZES)

    def test_instantiations_match_the_cpp_dispatch_gate(self):
        """Every instantiated specialization must be dispatchable.

        ``dispatch_mla_paged_attention`` admits only the
        (kv_lora_rank, qk_rope_head_dim, block_size, heads_per_tg,
        num_threads, partition_size) rows in ``kMlaKernelSpecs``; an
        instantiation beyond the table compiles dead binary and a table
        row without an instantiation throws at runtime. The call sites
        must be exactly the table's admitted space.
        """
        cpp = self._PAGED_OPS.read_text()

        # mla_validate_t_dtypes admits only fp16/bf16 for the T buffers; map
        # them through dtype_to_metal to the instantiated type names.
        dtype_map = dict(
            re.findall(
                r'case (\w+):\s*return "(\w+)";',
                self._fn_body(cpp, "static std::string dtype_to_metal"),
            )
        )
        gate_dtypes = {
            dtype_map[d]
            for d in re.findall(
                r"expected != (\w+)",
                self._fn_body(cpp, "static void mla_validate_t_dtypes"),
            )
        }
        assert gate_dtypes, "could not parse the C++ dtype gate"

        expected = {(t, *row) for t in gate_dtypes for row in self._gate_rows()}
        assert self._instantiations() == expected

    def test_python_g_picker_stays_inside_the_gate(self):
        """_pick_heads_per_tg may only return G values the gate maps."""
        from vllm_metal.attention.impls.mla import MLAPagedAttentionWrapper

        admitted = {g for _, _, _, g, _, _ in self._gate_rows()}
        picked = {
            MLAPagedAttentionWrapper._pick_heads_per_tg(num_heads, batch)
            for num_heads in range(1, 65)
            for batch in (1, 2, 8, 32)
        }
        assert picked <= admitted


class TestNaxKernelInstantiations:
    """The NAX prefill gate and its kernel instantiations cannot drift.

    ``nax_eligible`` in paged_ops.cpp admits a (dtype, head_size, block_size)
    space; ``instantiate_paged_attention_nax_all`` in
    pagedattention_nax.metal compiles one specialization per admitted row.
    An instantiation beyond the gate compiles dead binary; a gate value
    without an instantiation dispatches a kernel name that does not exist.
    """

    _NAX_METAL = (
        Path(__file__).resolve().parent.parent
        / "vllm_metal"
        / "metal"
        / "kernels_v2"
        / "pagedattention_nax.metal"
    )
    _PAGED_OPS = TestMLAKernelBlockSizes._PAGED_OPS

    def _instantiations(self) -> set[tuple[str, int, int]]:
        """The (metal-type, head_size, block_size) rows the library compiles.

        Rows live in the ``instantiate_paged_attention_nax_all`` macro body
        (its ``type`` parameter is bound by the ``..._all(<dtype>)`` call
        sites below it).
        """
        src = self._NAX_METAL.read_text()
        body_start = src.index("#define instantiate_paged_attention_nax_all")
        body_end = src.index("\n\n", body_start)
        shapes = {
            (int(hs), int(bs))
            for hs, bs in re.findall(
                r"instantiate_paged_attention_nax\(\s*type\s*,\s*(\d+)\s*,"
                r"\s*(\d+)\s*\)",
                src[body_start:body_end],
            )
        }
        assert shapes, "no instantiate_paged_attention_nax rows found"

        dtypes = set(
            re.findall(r"^instantiate_paged_attention_nax_all\((\w+)\);", src, re.M)
        )
        assert dtypes, "no instantiate_paged_attention_nax_all call sites found"
        return {(t, hs, bs) for t in dtypes for hs, bs in shapes}

    def test_instantiations_match_the_cpp_gate(self) -> None:
        cpp = self._PAGED_OPS.read_text()
        gate = TestMLAKernelBlockSizes._fn_body(cpp, "static bool nax_eligible")

        head_sizes = {int(v) for v in re.findall(r"head_size == (\d+)", gate)}
        block_sizes = {int(v) for v in re.findall(r"block_size == (\d+)", gate)}
        dtype_map = dict(
            re.findall(
                r'case (\w+):\s*return "(\w+)";',
                TestMLAKernelBlockSizes._fn_body(
                    cpp, "static std::string dtype_to_metal"
                ),
            )
        )
        gate_dtypes = {dtype_map[d] for d in re.findall(r"dtype == (\w+)", gate)}
        for name, parsed in (
            ("head_size", head_sizes),
            ("block_size", block_sizes),
            ("dtype", gate_dtypes),
        ):
            assert parsed, f"could not parse the C++ NAX {name} gate"

        expected = {
            (t, hs, bs) for t in gate_dtypes for hs in head_sizes for bs in block_sizes
        }
        assert self._instantiations() == expected
