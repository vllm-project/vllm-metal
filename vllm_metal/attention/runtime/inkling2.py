# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned KV and short-convolution histories for Inkling 2 visits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import torch
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.inkling2.language import l2_clip
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.storage import KVCacheStorage, gather_cache_rows
from vllm_metal.attention.context import get_context
from vllm_metal.attention.impls.sdpa import _kernel_metadata
from vllm_metal.attention.runtime.base import PagedAttentionRuntimeBase
from vllm_metal.metal import get_ops
from vllm_metal.metal.constants import KERNEL_BLOCK_SIZES


@dataclass(frozen=True)
class Inkling2Plan:
    """Physical weights may repeat; every execution visit owns its history."""

    layer_indices: tuple[int, ...]
    windows: tuple[int, ...]
    kv_heads: tuple[int, ...]
    head_dims: tuple[int, ...]
    hidden_size: int
    conv_kernel_size: int

    @classmethod
    def from_config(cls, config: Any) -> Inkling2Plan:
        indices = config.layer_indices
        sliding = tuple(config.layer_is_sliding(i) for i in indices)
        return cls(
            indices,
            tuple(config.sliding_window_size if local else -1 for local in sliding),
            tuple(
                config.swa_num_key_value_heads if local else config.num_key_value_heads
                for local in sliding
            ),
            tuple(
                config.swa_head_dim if local else config.head_dim for local in sliding
            ),
            config.hidden_size,
            config.sconv_kernel_size,
        )

    @staticmethod
    def attention_name(visit: int) -> str:
        return f"visits.{visit}.self_attn"

    @staticmethod
    def conv_name(visit: int, stream: int) -> str:
        return f"visits.{visit}.conv.{stream}"

    def cache_specs(self, block_size: int, dtype: torch.dtype) -> dict[str, Any]:
        if block_size not in KERNEL_BLOCK_SIZES:
            # Splitting a padded scheduler page with reshape can copy the pool.
            raise ValueError(
                f"Inkling requires a native kernel block size {KERNEL_BLOCK_SIZES}; "
                f"got {block_size}"
            )
        specs = {}
        for visit, window in enumerate(self.windows):
            attention = {
                "block_size": block_size,
                "num_kv_heads": self.kv_heads[visit],
                "head_size": self.head_dims[visit],
                "dtype": dtype,
            }
            specs[self.attention_name(visit)] = (
                SlidingWindowSpec(**attention, sliding_window=window)
                if window > 0
                else FullAttentionSpec(**attention)
            )
            kv_width = self.kv_heads[visit] * self.head_dims[visit]
            for stream, width in enumerate(
                (kv_width, kv_width, self.hidden_size, self.hidden_size)
            ):
                # Keep the input history, so every cached token is a valid
                # convolution restart point, including shared prefix blocks.
                # Inputs arrive in the model dtype and upstream widens them to
                # FP32 exactly, so a wider history dtype would only add bytes.
                specs[self.conv_name(visit, stream)] = SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=width,
                    head_size_v=0,
                    dtype=dtype,
                    sliding_window=self.conv_kernel_size,
                )
        return specs


@dataclass(frozen=True)
class _ConvContext:
    visit: int

    def advance(self, count: int) -> None:
        # Absolute positions come from the scheduler, once per forward.
        pass


# Match the upstream FP32 dot and rounding without constructing a Q-by-K mask.
_relative_bias_kernel: Any = mx.fast.metal_kernel(
    name="inkling_distance_bias",
    input_names=["rel", "proj"],
    output_names=["out"],
    source="""
        const uint distance = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        if (distance >= EXTENT || row >= rel_shape[0] * rel_shape[1]) return;
        float acc = 0.0f;
        for (uint d = 0; d < D_REL; ++d)
            acc += float(rel[row * D_REL + d]) * float(proj[d * EXTENT + distance]);
        out[row * EXTENT + distance] = T(acc);
    """,
)


def _relative_bias(rel: mx.array, proj: mx.array) -> mx.array:
    rows, heads, width = rel.shape
    extent = proj.shape[1]
    return _relative_bias_kernel(
        inputs=[rel, proj],
        template=[("T", rel.dtype), ("D_REL", width), ("EXTENT", extent)],
        grid=(extent, rows * heads, 1),
        threadgroup=(32, 4, 1),
        output_shapes=[(rows, heads, extent)],
        output_dtypes=[rel.dtype],
    )[0]


class _Attention(nn.Module):
    def __init__(self, inner, runtime):
        super().__init__()
        self.inner = inner
        object.__setattr__(self, "runtime", runtime)

    def __call__(self, x, key_input=None, value_input=None, cache=None, conv_mask=None):
        if get_context() is None:
            return self.inner(x, key_input, value_input, cache, conv_mask)
        runtime, attn = self.runtime, self.inner
        cfg, ctx = attn.config, runtime.context
        visit, conv = cache[1].visit, cache[1]
        length = x.shape[1]
        key_input = x if key_input is None else key_input
        value_input = x if value_input is None else value_input
        k, v = attn.k_proj(key_input), attn.v_proj(value_input)
        if attn.is_sliding:
            k, v = attn.k_sconv(k, conv), attn.v_sconv(v, conv)
        else:
            k = attn.k_sconv(k, key_input, conv)
            v = attn.v_sconv(v, value_input, conv)
        q = l2_clip(
            attn.q_proj(x).reshape(length, attn.n_heads, attn.head_dim),
            cfg.qk_l2_clip_threshold,
            attn.q_scale,
        )
        k = l2_clip(
            k.reshape(length, attn.n_kv, attn.head_dim), cfg.qk_l2_clip_threshold
        )
        v = v.reshape(length, attn.n_kv, attn.head_dim)
        if cfg.use_value_norm_no_gamma:
            v = mx.fast.rms_norm(v.astype(mx.float32), None, cfg.rms_norm_eps).astype(
                x.dtype
            )
        r = l2_clip(
            attn.r_proj(x).reshape(length, attn.n_heads, cfg.d_rel),
            cfg.rel_l2_clip_threshold,
            attn.rel_scale,
        )
        if not attn.is_sliding and cfg.log_scaling_n_floor is not None:
            positions = mx.concatenate(
                [
                    mx.arange(hi - lo) + ctx.offsets[request] + 1
                    for request, (lo, hi) in enumerate(runtime.segments())
                ]
            )
            tau = 1 + cfg.log_scaling_alpha * mx.log(
                mx.maximum(positions / cfg.log_scaling_n_floor, 1)
            )
            q = (q.astype(mx.float32) * tau[:, None, None]).astype(x.dtype)
            r = (r.astype(mx.float32) * tau[:, None, None]).astype(x.dtype)
        group_index = runtime._groups[runtime.plan.attention_name(visit)]
        group = ctx.kv_groups[group_index]
        meta = _kernel_metadata(
            ctx, group_index, group.slot_mapping, group.block_tables, group.block_size
        )
        kv = runtime.kv_cache
        keys, values = get_ops().reshape_and_cache(
            mx.contiguous(k.astype(kv.dtype)),
            mx.contiguous(v.astype(kv.dtype)),
            kv.key_caches[visit],
            kv.value_caches[visit],
            meta.slot_mapping,
        )
        kv.key_caches[visit], kv.value_caches[visit] = keys, values
        out = (
            get_ops()
            .paged_attention_primitive(
                mx.contiguous(q.astype(kv.dtype)),
                keys,
                values,
                attn.n_kv,
                attn.head_dim**-0.5,
                0.0,
                meta.block_tables,
                meta.seq_lens,
                meta.cu_seqlens_q,
                meta.block_size,
                meta.max_seq_len,
                runtime.plan.windows[visit],
                num_decode_requests=ctx.num_decode_requests,
                num_decode_tokens=ctx.num_decode_tokens,
                max_decode_context_len=ctx.max_decode_context_len,
                sinks=attn.sink_proj(x)[0].astype(mx.float32),
                relative_bias=_relative_bias(r, attn.rel_proj.astype(x.dtype)).astype(
                    kv.dtype
                ),
            )
            .astype(x.dtype)
            .reshape(1, length, -1)
        )
        out = (
            out.astype(mx.float32) * mx.sigmoid(attn.gate_proj(x).astype(mx.float32))
        ).astype(x.dtype)
        return attn.o_proj(out)


class _Convolution(nn.Module):
    def __init__(self, inner, runtime):
        super().__init__()
        self.inner = inner
        object.__setattr__(self, "runtime", runtime)

    def __call__(self, x, *args, cache=None, mask=None):
        dynamic = args[:1] if hasattr(self.inner, "weight_proj") else ()
        positional = args[len(dynamic) :]
        if positional:
            cache = positional[0]
        if len(positional) > 1:
            mask = positional[1]
        if get_context() is None:
            return self.inner(x, *dynamic, cache=cache, mask=mask)
        runtime, stream = self.runtime, self.inner.conv_idx
        name = runtime.plan.conv_name(cache.visit, stream)
        index = runtime.conv_indices[name]
        views = runtime.conv_views
        outputs = []
        for request, (lo, hi) in enumerate(runtime.segments()):
            offset = runtime.context.offsets[request]
            tail = self.inner.kernel_size - 1
            history = runtime.slots(name, request, max(0, offset - tail), offset)
            state = gather_cache_rows(
                views[index], history, views.descriptors[index][2][0]
            ).astype(mx.float32)
            if offset < tail:
                state = mx.concatenate(
                    [mx.zeros((tail - offset, x.shape[-1]), mx.float32), state]
                )
            native_cache = ArraysCache(4)
            native_cache[stream] = state[None]
            inputs = x[:, lo:hi]
            slots = runtime.slots(name, request, offset, offset + hi - lo)
            views[index] = get_ops().gdn_state_scatter(
                views[index], inputs[0].astype(views[index].dtype), slots, paged=True
            )
            outputs.append(
                self.inner(
                    inputs,
                    *(arg[:, lo:hi] for arg in dynamic),
                    cache=native_cache,
                )
            )
        return mx.concatenate(outputs, axis=1)


class _Backbone(nn.Module):
    def __init__(self, inner, plan):
        super().__init__()
        self.inner = inner
        object.__setattr__(
            self,
            "_visit_caches",
            tuple(
                (None, _ConvContext(visit)) for visit in range(len(plan.layer_indices))
            ),
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "inner" not in self:
                raise
            return getattr(self["inner"], name)

    def __call__(self, *args, cache=None, **kwargs):
        if get_context() is not None:
            cache = self._visit_caches
        return self.inner(*args, cache=cache, **kwargs)


class Inkling2Runtime(PagedAttentionRuntimeBase):
    """Keep upstream loop/math and adapt temporal operators to packed requests."""

    def __init__(self, plan: Inkling2Plan, dtype: mx.Dtype):
        self.plan, self.dtype = plan, dtype

    def initialize_from_config(self, config):
        self.storage = KVCacheStorage(config)
        names = tuple(
            self.plan.attention_name(i) for i in range(len(self.plan.windows))
        )
        self._cache = MetalPagedKVCache.from_upstream(
            self.storage, names, dtype=self.dtype
        )
        self._groups = {
            name: i
            for i, group in enumerate(config.kv_cache_groups)
            for name in group.layer_names
        }
        conv_names = tuple(
            self.plan.conv_name(visit, stream)
            for visit in range(len(names))
            for stream in range(4)
        )
        self.conv_indices = {name: i for i, name in enumerate(conv_names)}
        self.conv_views = self.storage.views(
            [self.storage.tensors[name][:, 0] for name in conv_names]
        )

    @property
    def context(self):
        ctx = get_context()
        if ctx is None or ctx.kv_groups is None:
            raise RuntimeError(
                "Inkling paged execution requires grouped cache metadata"
            )
        return ctx

    @property
    def kv_cache(self):
        return self._require_initialized("kv_cache")

    def segments(self):
        lengths = self.context.cu_seqlens
        return zip(lengths[:-1], lengths[1:], strict=True)

    def slots(self, name: str, request: int, start: int, end: int):
        group = self.context.kv_groups[self._groups[name]]
        block_size = group.block_size
        first = start // block_size
        table = group.block_tables[request][
            first : (end + block_size - 1) // block_size
        ]
        positions = mx.arange(start, end, dtype=mx.int32)
        if start == end:
            return positions
        blocks = mx.array(table, dtype=mx.int32)
        return (
            blocks[positions // block_size - first] * block_size
            + positions % block_size
        )

    def patch_model(self, model):
        text = getattr(model, "language_model", model)
        for layer in text.model.layers:
            attn = layer.self_attn
            attn.k_sconv = _Convolution(attn.k_sconv, self)
            attn.v_sconv = _Convolution(attn.v_sconv, self)
            layer.self_attn = _Attention(attn, self)
            layer.attn_sconv = _Convolution(layer.attn_sconv, self)
            layer.mlp_sconv = _Convolution(layer.mlp_sconv, self)
        text.model = _Backbone(text.model, self.plan)
        return len(self.plan.layer_indices)

    def kv_scheduler_group_indices(self):
        return tuple(range(len(self.storage.config.kv_cache_groups)))

    def kv_group_block_sizes(self):
        return tuple(
            g.kv_cache_spec.block_size for g in self.storage.config.kv_cache_groups
        )

    def extend_forward_eval_outputs(self, outputs):
        self.storage.depend(outputs)
        outputs.extend(self.storage.buffers)

    def release_kv_memory(self):
        self.storage.release()

    def restore_kv_memory(self):
        self.storage.reallocate()
