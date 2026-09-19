# Expert Parallelism (EP2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third distributed mode — experts parallelism (TP2 + expert-count split for GPT-OSS) — to vllm-metal, and expose it as "Experts" in the inference-ui parallelism selector.

**Architecture:** Attention shards exactly as tensor mode (heads halved, `o_proj` all_sum); the 128 routed experts per layer split by count (default 64/64, uneven via `VLLM_METAL_EXPERT_PARTITION`) by axis-0 slicing of the stacked expert tensors. A patched `MLPBlock.__call__` remaps the router's global top-k to local indices and zeroes non-local weights; the existing `sharding_group` `all_sum` combines partials across ranks. No all-to-all, no mlx-lm edits.

**Tech Stack:** Python 3.12, mlx 0.32.1 (exact pin), mlx-lm 0.32.0 (pinned git, never edited), vllm 0.29.0, unittest-style pytest, Gradio (inference-ui, Python 3.12).

**Spec:** `docs/superpowers/specs/2026-09-18-expert-parallelism-design.md`

## Global Constraints

- mlx is an exact pin (`mlx==0.32.1`) — never edit anything under `.venv/lib/python3.12/site-packages/mlx*`.
- vllm-metal: ruff line length 88, target py312; commits need DCO (`git commit -s`); branch `feat/expert-parallelism`; baseline suite must stay green (`pytest -m "not slow" tests/` = 2225 passed / 5 skipped / 24 deselected before this work).
- vllm-metal unit tests run on CPU: wrap model work in `with mx.stream(mx.cpu):`; no GPU, downloads, or distributed processes.
- inference-ui is NOT a git repo — no commits there, edits are additive and minimal.
- Expert axis is 0 for every `SwitchLinear`/`QuantizedSwitchLinear` parameter: `weight (E, out, in)`, `scales (E, out, groups)`, `bias (E, out)`, `biases` (quant offsets, may be None).
- Do not call mlx-lm `Model.shard()` under EP — it width-slices experts. Attention sharding is mirrored in vllm-metal instead.
- inference-ui tests: `uv run pytest` (asyncio_mode=auto), `uv run ruff check`.

---

### Task 1: Partition env var + parser

**Files:**
- Modify: `vllm_metal/envs.py` (TYPE_CHECKING block ~line 35, `environment_variables` dict ~line 37)
- Create: `vllm_metal/distributed/experts.py`
- Test: `tests/test_ep_experts.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `vllm_metal.distributed.experts.expert_partition(world_size: int, num_experts: int) -> list[int]` — per-rank expert counts, validated; env var `VLLM_METAL_EXPERT_PARTITION` (e.g. `"56,72"`), default even split.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ep_experts.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel sharding: partition parsing, weight slicing, masked routing."""

import mlx.core as mx
import pytest

from vllm_metal.distributed.experts import expert_partition


def test_partition_defaults_to_even_and_validates_overrides(monkeypatch):
    assert expert_partition(2, 128) == [64, 64]
    monkeypatch.setenv("VLLM_METAL_EXPERT_PARTITION", "56,72")
    assert expert_partition(2, 128) == [56, 72]
    for bad in ("64", "63,65", "0,128", "64,64,0", "a,b", ""):
        monkeypatch.setenv("VLLM_METAL_EXPERT_PARTITION", bad)
        with pytest.raises(ValueError, match="VLLM_METAL_EXPERT_PARTITION"):
            expert_partition(2, 128)


def test_partition_rejects_uneven_default(monkeypatch):
    monkeypatch.delenv("VLLM_METAL_EXPERT_PARTITION", raising=False)
    with pytest.raises(ValueError, match="divide evenly"):
        expert_partition(2, 127)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ep_experts.py -v`
Expected: FAIL — `No module named 'vllm_metal.distributed.experts'`

- [ ] **Step 3: Implement**

In `vllm_metal/envs.py`, add to the TYPE_CHECKING block:

```python
    VLLM_METAL_EXPERT_PARTITION: str | None = None
```

and to `environment_variables`:

```python
    # Expert-parallel partition across ranks, e.g. "56,72" for two Macs with
    # 128 routed experts. Default: even split (validated to divide evenly).
    "VLLM_METAL_EXPERT_PARTITION": lambda: os.getenv("VLLM_METAL_EXPERT_PARTITION"),
```

Create `vllm_metal/distributed/experts.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS expert parallelism: count-split experts, replicated routing."""

from __future__ import annotations

import os


def expert_partition(world_size: int, num_experts: int) -> list[int]:
    """Per-rank expert counts from VLLM_METAL_EXPERT_PARTITION or an even split."""
    raw = os.getenv("VLLM_METAL_EXPERT_PARTITION")
    if raw is None:
        if num_experts % world_size:
            raise ValueError(
                f"{num_experts} experts do not divide evenly across {world_size} "
                "ranks; set VLLM_METAL_EXPERT_PARTITION."
            )
        return [num_experts // world_size] * world_size
    try:
        counts = [int(part) for part in raw.split(",")]
    except ValueError:
        counts = []
    if (
        len(counts) != world_size
        or sum(counts) != num_experts
        or any(count <= 0 for count in counts)
    ):
        raise ValueError(
            f"VLLM_METAL_EXPERT_PARTITION must be {world_size} positive integers "
            f"summing to {num_experts}; got {raw!r}."
        )
    return counts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ep_experts.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add vllm_metal/envs.py vllm_metal/distributed/experts.py tests/test_ep_experts.py
git commit -s -m "Add VLLM_METAL_EXPERT_PARTITION parsing for expert parallelism"
```

---

### Task 2: Admit EP in the tensor config validator

**Files:**
- Modify: `vllm_metal/distributed/tensor.py:24-37` (the `unsupported` tuple in `validate_tensor_config`)
- Test: `tests/test_ep_experts.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `validate_tensor_config` accepts `parallel_config.enable_expert_parallel=True` alongside the existing TP2 whitelist (TP=2, PP=1, DP=1, ray, gpt_oss, sync, explicit JACCL `tensor_transport`). Everything else still rejected identically.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ep_experts.py` (helper mirrors `tests/test_tp_tensor.py::config`):

```python
def _config(**overrides):
    from types import SimpleNamespace

    values = {
        "parallel_config": SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend="ray",
            enable_expert_parallel=False,
        ),
        "model_config": SimpleNamespace(
            hf_config=SimpleNamespace(model_type="gpt_oss"),
            quantization=None,
            multimodal_config=None,
            runner_type="generate",
        ),
        "scheduler_config": SimpleNamespace(async_scheduling=False),
        "speculative_config": None,
        "lora_config": None,
        "additional_config": {
            "tensor_transport": {
                "backend": "jaccl",
                "device_matrix": [
                    [None, ["rdma_en1", "rdma_en2"]],
                    [["rdma_en2", "rdma_en1"], None],
                ],
            }
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_expert_parallel_is_admitted_on_the_tp2_whitelist():
    from vllm_metal.distributed.tensor import validate_tensor_config

    cfg = _config()
    cfg.parallel_config.enable_expert_parallel = True
    validate_tensor_config(cfg)  # admitted
    cfg = _config()
    cfg.parallel_config.enable_expert_parallel = True
    cfg.scheduler_config.async_scheduling = True
    with pytest.raises(NotImplementedError):
        validate_tensor_config(cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ep_experts.py::test_expert_parallel_is_admitted_on_the_tp2_whitelist -v`
Expected: FAIL with NotImplementedError (EP still in the unsupported tuple)

- [ ] **Step 3: Implement**

In `vllm_metal/distributed/tensor.py`, delete the line

```python
        or getattr(parallel, "enable_expert_parallel", False)
```

from the `unsupported` tuple in `validate_tensor_config` (line 29). No other change — the remaining whitelist entries constrain EP exactly as TP.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ep_experts.py tests/test_tp_tensor.py -v`
Expected: all PASS (test_tp_tensor's `enable_expert_parallel=False` fixture is unaffected)

- [ ] **Step 5: Commit**

```bash
git add vllm_metal/distributed/tensor.py tests/test_ep_experts.py
git commit -s -m "Admit enable_expert_parallel on the TP2 GPT-OSS whitelist"
```

---

### Task 3: `apply_expert_shard` — expert-count slicing + attention mirror

**Files:**
- Modify: `vllm_metal/distributed/experts.py`
- Test: `tests/test_ep_experts.py`

**Interfaces:**
- Consumes: `expert_partition` (Task 1); a group object with `.rank()`, `.size()`, usable by `shard_linear` (same contract `apply_tensor_shard` relies on).
- Produces: `apply_expert_shard(model: Any, tp: Any) -> None` — requires gpt-oss + size 2; per layer: attention mirrored from mlx-lm `shard()`; experts axis-0 sliced; sets `layer.mlp.sharding_group = tp.group` and `layer.mlp.expert_partition = (start, end)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ep_experts.py` (fixtures mirror `tests/test_pp_gpt_oss.py::_Group` / `::_model`; note EP fixture uses 4 experts / top-2 so masking is exercised):

```python
class _Group:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def _ep_model(*, quantized: bool = False):
    from mlx_lm.models import gpt_oss

    model = gpt_oss.Model(
        gpt_oss.ModelArgs(
            num_hidden_layers=4,
            num_local_experts=4,
            num_experts_per_tok=2,
            vocab_size=64,
            hidden_size=64,
            intermediate_size=64,
            head_dim=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention"] * 2,
        )
    )
    for layer in model.layers:
        layer.self_attn.sinks = mx.linspace(-1, 1, 8)
    if quantized:
        import mlx.nn as nn

        model.set_dtype(mx.float16)

        def quantization(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            if ".experts." in path:
                return {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            return {"group_size": 32, "bits": 8, "mode": "affine"}

        nn.quantize(model, class_predicate=quantization)
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("rank,partition,expected", [(0, "2,2", 2), (1, "2,2", 2), (1, "1,3", 3)])
@pytest.mark.parametrize("quantized", [False, True], ids=["float32", "mxfp4-q8"])
def test_expert_shard_slices_experts_by_count_not_width(
    monkeypatch, rank, partition, expected, quantized
):
    from vllm_metal.distributed.experts import apply_expert_shard

    monkeypatch.setenv("VLLM_METAL_EXPERT_PARTITION", partition)
    with mx.stream(mx.cpu):
        model = _ep_model(quantized=quantized)
        apply_expert_shard(model, _Group(rank, 2))
        start = 0 if rank == 0 else int(partition.split(",")[0])
        for layer in model.layers:
            attn = layer.self_attn
            assert attn.num_attention_heads == 4
            assert attn.num_key_value_heads == 2
            assert attn.sinks.shape == (4,)
            assert attn.q_proj.weight.shape[0] == 32
            experts = layer.mlp.experts
            for proj in (experts.gate_proj, experts.up_proj, experts.down_proj):
                assert proj.weight.shape[0] == expected
                assert proj.scales.shape[0] == expected
                assert proj.weight.shape[1] == 64  # width NOT halved
            assert layer.mlp.router.weight.shape[0] == 4  # router stays full
            assert layer.mlp.sharding_group is not None
            assert layer.mlp.expert_partition == (start, start + expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ep_experts.py -k expert_shard -v`
Expected: FAIL — `ImportError: cannot import name 'apply_expert_shard'`

- [ ] **Step 3: Implement**

Append to `vllm_metal/distributed/experts.py`:

```python
def apply_expert_shard(model, tp) -> None:
    """Shard GPT-OSS for expert parallelism: attention as TP2, experts by count.

    Mirrors mlx_lm gpt_oss.Model.shard() for attention only — shard() itself
    width-slices experts, which expert parallelism must not do. Slicing runs
    before weight evaluation, on the same lazy lifecycle as apply_tensor_shard.
    """
    import mlx.core as mx
    from mlx.nn.layers.distributed import shard_linear

    if getattr(model, "model_type", None) != "gpt_oss" or tp.size != 2:
        raise NotImplementedError("Expert sharding supports GPT-OSS on two Macs.")
    args = model.args
    if args.num_attention_heads % tp.size or args.num_key_value_heads % tp.size:
        raise ValueError("GPT-OSS attention heads must divide evenly across ranks.")
    counts = expert_partition(tp.size, args.num_local_experts)
    start = sum(counts[: tp.rank()])
    end = start + counts[tp.rank()]

    for layer in model.layers:
        attn = layer.self_attn
        attn.q_proj = shard_linear(attn.q_proj, sharding="all-to-sharded", group=tp.group)
        attn.k_proj = shard_linear(attn.k_proj, sharding="all-to-sharded", group=tp.group)
        attn.v_proj = shard_linear(attn.v_proj, sharding="all-to-sharded", group=tp.group)
        attn.o_proj = shard_linear(attn.o_proj, sharding="sharded-to-all", group=tp.group)
        attn.num_attention_heads //= tp.size
        attn.num_key_value_heads //= tp.size
        attn.num_key_value_groups = attn.num_attention_heads // attn.num_key_value_heads
        attn.sinks = attn.sinks[
            attn.num_attention_heads * tp.rank() : attn.num_attention_heads * (tp.rank() + 1)
        ]

        experts = layer.mlp.experts
        for proj in (experts.gate_proj, experts.up_proj, experts.down_proj):
            proj.weight = proj.weight[start:end]
            proj.scales = proj.scales[start:end]
            if "bias" in proj:
                proj.bias = proj.bias[start:end]
            quant_biases = proj.get("biases")
            if quant_biases is not None:
                proj.biases = quant_biases[start:end]
        layer.mlp.sharding_group = tp.group
        layer.mlp.expert_partition = (start, end)
```

(`"bias" in proj` / `proj.get("biases")` follow `SwitchLinear.__call__`'s own membership checks in `mlx_lm/models/switch_layers.py:88-89` — mlx `nn.Module` supports both.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ep_experts.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add vllm_metal/distributed/experts.py tests/test_ep_experts.py
git commit -s -m "Add apply_expert_shard: expert-count slicing with TP2 attention"
```

---

### Task 4: Masked routing — exact EP forward

**Files:**
- Modify: `vllm_metal/distributed/experts.py`
- Test: `tests/test_ep_experts.py`

**Interfaces:**
- Consumes: `layer.mlp.expert_partition` and `layer.mlp.sharding_group` (Task 3).
- Produces: `install_expert_routing() -> None` — idempotent patch of `mlx_lm.models.gpt_oss.MLPBlock.__call__`; unpatched instances (no `expert_partition` attribute) keep the original forward byte-for-byte. Called from `apply_expert_shard` (Task 3 already shipped without it — this task adds the call).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ep_experts.py`:

```python
def _run_ep_forward(monkeypatch, model, tokens):
    """One forward with a fake all_sum that returns the running total of all
    partials seen under this monkeypatch — so calling it for shard0 then
    shard1 makes the second call's return value the combined output."""
    partials = []

    def fake_all_sum(value, *, group=None, stream=None):
        import mlx.core as mx

        partials.append(value)
        total = partials[0]
        for extra in partials[1:]:
            total = total + extra
        mx.eval(total)
        return total

    import mlx.core as mx

    monkeypatch.setattr(mx.distributed, "all_sum", fake_all_sum)
    with mx.stream(mx.cpu):
        output = model(mx.array(tokens, dtype=mx.int32))
        mx.eval(output)
    return output


@pytest.mark.parametrize("quantized", [False, True], ids=["float32", "mxfp4-q8"])
def test_expert_forward_matches_unsplit_reference(monkeypatch, quantized):
    """Both ranks' masked partials, summed by all_sum, equal the unsplit MoE."""
    from vllm_metal.distributed.experts import apply_expert_shard

    tokens = [[1, 7, 11, 23, 42, 3, 9, 17, 5, 8, 2, 13]]  # 12 tokens x top-2 = 24 pairs
    with mx.stream(mx.cpu):
        reference = _ep_model(quantized=quantized)
        expected = reference(mx.array(tokens, dtype=mx.int32))
        mx.eval(expected)
        from copy import deepcopy

        shard0 = deepcopy(reference)
        shard1 = deepcopy(reference)
    monkeypatch.setenv("VLLM_METAL_EXPERT_PARTITION", "2,2")
    apply_expert_shard(shard0, _Group(0, 2))
    apply_expert_shard(shard1, _Group(1, 2))
    _run_ep_forward(monkeypatch, shard0, tokens)  # rank 0 partial
    combined = _run_ep_forward(monkeypatch, shard1, tokens)  # rank 1 sees the sum
    tolerance = 1e-4 if not quantized else 5e-2
    assert float(mx.max(mx.abs(combined - expected)).item()) <= tolerance


def test_expert_routing_sorted_path_and_full_coverage(monkeypatch):
    """Indices.size >= 64 exercises SwitchGLU's sorted path; a partition that
    covers all experts on one rank must reproduce the reference exactly."""
    from vllm_metal.distributed import experts as experts_mod
    from vllm_metal.distributed.experts import apply_expert_shard

    tokens = [list(range(1, 33))]  # 32 tokens x top-2 = 64 pairs -> do_sort fires
    with mx.stream(mx.cpu):
        reference = _ep_model()
        expected = reference(mx.array(tokens, dtype=mx.int32))
        mx.eval(expected)
        from copy import deepcopy

        full = deepcopy(reference)
    monkeypatch.setattr(experts_mod, "expert_partition", lambda ws, ne: [4, 0])
    apply_expert_shard(full, _Group(0, 2))
    got = _run_ep_forward(monkeypatch, full, tokens)
    assert float(mx.max(mx.abs(got - expected)).item()) <= 1e-4
```

The invariant under test: `forward(shard0)` + `forward(shard1)` partials, added, equal the unsplit reference within tolerance; the fake `all_sum` returns the running total so the second model's output IS the combined result.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ep_experts.py -k expert_ -v`
Expected: FAIL — masked routing not installed; `expert_partition` attribute is set but `MLPBlock.__call__` ignores it, so the EP forward uses global indices against sliced weights (shape/garbage mismatch or wrong logits)

- [ ] **Step 3: Implement**

Append to `vllm_metal/distributed/experts.py`:

```python
def install_expert_routing() -> None:
    """Route the replicated GPT-OSS router through rank-local experts.

    Both ranks see identical activations (post-o_proj all_sum), so the global
    top-k and softmax weights are identical; each rank zeroes the slots it
    does not own and points them at a dummy local expert, and the existing
    sharding_group all_sum sums the complementary partials — exact in exact
    arithmetic. Dummy slots keep shapes static; their zero weight cancels
    both expert output and bias.
    """
    import mlx.core as mx
    from mlx_lm.models import gpt_oss as mlx_gpt_oss

    if getattr(mlx_gpt_oss.MLPBlock, "_ep_routing_installed", False):
        return
    original = mlx_gpt_oss.MLPBlock.__call__

    def ep_call(self, x):
        if getattr(self, "expert_partition", None) is None:
            return original(self, x)
        start, end = self.expert_partition
        g = self.router(x)
        scores, indices = mlx_gpt_oss.mlx_topk(g, k=self.num_experts_per_tok, axis=-1)
        expert_weights = mx.softmax(scores, axis=-1, precise=True)
        mask = (indices < start) | (indices >= end)
        local = mx.where(mask, mx.zeros_like(indices), indices - start)
        expert_weights = mx.where(mask, mx.zeros_like(expert_weights), expert_weights)
        out = self.experts(x, local)
        out = out * mx.expand_dims(expert_weights, axis=-1)
        y = out.sum(axis=-2)
        return mx.distributed.all_sum(y, group=self.sharding_group)

    mlx_gpt_oss.MLPBlock.__call__ = ep_call
    mlx_gpt_oss.MLPBlock._ep_routing_installed = True
```

and add `install_expert_routing()` as the last line of `apply_expert_shard` (after the layer loop).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ep_experts.py -v`
Expected: all PASS (including the Task 3 shape tests, which now also install routing — no attribute on unpatched instances, original path intact)

Also run the untouched-path guard:

```bash
.venv/bin/python -m pytest tests/test_pp_gpt_oss.py tests/test_tp_tensor.py -v
```
Expected: all PASS (non-EP paths byte-identical — `expert_partition` unset there)

- [ ] **Step 5: Commit**

```bash
git add vllm_metal/distributed/experts.py tests/test_ep_experts.py
git commit -s -m "Add masked expert routing with exact cross-rank combine"
```

---

### Task 5: Worker + runner wiring

**Files:**
- Modify: `vllm_metal/distributed/tensor.py:61-63` (`TensorGroup.bootstrap`) and `:56-59` (`__init__`)
- Modify: `vllm_metal/v1/model_runner.py:600-615` (the `if self.tp is not None:` block in `load_model`)
- Test: `tests/test_ep_experts.py`

**Interfaces:**
- Consumes: `apply_expert_shard` (Task 3).
- Produces: `TensorGroup.expert_parallel: bool` (default False; `bootstrap` records `parallel_config.enable_expert_parallel`); `load_model` selects `apply_expert_shard` when `self.tp.expert_parallel` else `apply_tensor_shard`. KV-head halving stays in the shared path (attention is TP2 either way).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ep_experts.py`:

```python
def test_bootstrap_records_expert_parallel(monkeypatch):
    from vllm_metal.distributed.tensor import TensorGroup
    from vllm_metal.distributed.transport import PipelineTransportConfig

    monkeypatch.setattr(
        PipelineTransportConfig,
        "bootstrap_jaccl",
        classmethod(lambda cls, rank, peer_ips: _Group(rank, 2)),
    )
    cfg = _config()
    assert TensorGroup.bootstrap(0, ["a", "b"], cfg).expert_parallel is False
    cfg = _config()
    cfg.parallel_config.enable_expert_parallel = True
    assert TensorGroup.bootstrap(0, ["a", "b"], cfg).expert_parallel is True


def test_runner_applies_expert_shard_when_flagged(monkeypatch):
    import vllm_metal.v1.model_runner as runner_mod

    calls = []

    class _FakeExperts:
        def apply_expert_shard(self, model, tp):
            calls.append("ep")

    class _FakeTensor:
        def apply_tensor_shard(self, model, tp):
            calls.append("tp")

    import sys
    import types

    fake_experts = _FakeExperts()
    fake_tensor = _FakeTensor()
    runner_mod_experts = types.ModuleType("vllm_metal.distributed.experts")
    runner_mod_experts.apply_expert_shard = fake_experts.apply_expert_shard
    runner_mod_tensor = types.ModuleType("vllm_metal.distributed.tensor")
    runner_mod_tensor.apply_tensor_shard = fake_tensor.apply_tensor_shard
    monkeypatch.setitem(sys.modules, "vllm_metal.distributed.experts", runner_mod_experts)
    monkeypatch.setitem(sys.modules, "vllm_metal.distributed.tensor", runner_mod_tensor)

    tp = types.SimpleNamespace(rank=0, size=2, expert_parallel=True)
    runner = types.SimpleNamespace(
        tp=tp,
        model=object(),
        num_kv_heads=8,
        num_layers=4,
        kv_heads_per_layer=[8, 8],
    )
    # Extracted branch under test (see Step 3): a module-level helper keeps
    # this testable without building a full MetalModelRunner.
    runner_mod._apply_tensor_parallel_shards(runner)
    assert calls == ["ep"]
    runner.tp.expert_parallel = False
    runner_mod._apply_tensor_parallel_shards(runner)
    assert calls == ["ep", "tp"]
    assert runner.num_kv_heads == 2 and runner.kv_heads_per_layer == [2, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ep_experts.py -k "bootstrap_records or runner_applies" -v`
Expected: FAIL — `TensorGroup` has no `expert_parallel`; `runner_mod._apply_tensor_parallel_shards` missing

- [ ] **Step 3: Implement**

In `vllm_metal/distributed/tensor.py`:

```python
    def __init__(self, group: Any):
        self.group = group
        self.rank = group.rank()
        self.size = group.size()
        self.expert_parallel = False

    @classmethod
    def bootstrap(cls, rank: int, peer_ips: list[str], config: Any) -> TensorGroup:
        self = cls(tensor_transport(config).bootstrap_jaccl(rank, peer_ips))
        self.expert_parallel = bool(
            getattr(config.parallel_config, "enable_expert_parallel", False)
        )
        return self
```

In `vllm_metal/v1/model_runner.py`, replace the body of the `if self.tp is not None:` block in `load_model` (lines 600-615) with a call to a new module-level helper defined just above `class MetalModelRunner` (or at module bottom — keep it near the other module helpers):

```python
def _apply_tensor_parallel_shards(runner: MetalModelRunner) -> None:
    """Shard weights per TP mode; both modes halve attention, so KV follows."""
    if getattr(runner.tp, "expert_parallel", False):
        from vllm_metal.distributed.experts import apply_expert_shard

        apply_expert_shard(runner.model, runner.tp)
        mode = "expert"
    else:
        from vllm_metal.distributed.tensor import apply_tensor_shard

        apply_tensor_shard(runner.model, runner.tp)
        mode = "tensor"
    runner.num_kv_heads //= runner.tp.size
    if runner.kv_heads_per_layer is not None:
        runner.kv_heads_per_layer = [n // runner.tp.size for n in runner.kv_heads_per_layer]
    logging.getLogger(__name__).info(
        "%s shard rank=%d/%d layers=%d local_kv_heads=%d",
        mode.capitalize(),
        runner.tp.rank,
        runner.tp.size,
        runner.num_layers,
        runner.num_kv_heads,
    )
```

and in `load_model`:

```python
        if self.tp is not None:
            _apply_tensor_parallel_shards(self)
```

(Use the module's existing `logger` instead of `logging.getLogger(__name__)` if one is already defined at module scope — check and reuse.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ep_experts.py tests/test_v1_worker.py tests/test_tp_tensor.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add vllm_metal/distributed/tensor.py vllm_metal/v1/model_runner.py tests/test_ep_experts.py
git commit -s -m "Wire expert-parallel selection through TensorGroup and the runner"
```

---

### Task 6: Two-Mac EP smoke mode

**Files:**
- Modify: `tools/jaccl_pp_smoke.py`
- Test: manual import check only (hardware tool)

**Interfaces:**
- Consumes: `apply_expert_shard`, `install_expert_routing` (Tasks 3-4), `TensorGroup`, `PipelineTransportConfig`.
- Produces: `--expert-parallel` flag: both ranks run the full tiny GPT-OSS with experts split, compare logits against the unsplit reference, and report via the existing summary/agree_check machinery.

- [ ] **Step 1: Add the flag and check function**

In `parse_args` (tools/jaccl_pp_smoke.py:39-44), add:

```python
    parser.add_argument(
        "--expert-parallel",
        action="store_true",
        help="split routed experts across ranks instead of pipeline stages",
    )
```

Add the check (modeled on `check_model_parity`; both ranks hold all layers and compare directly — no stage handoffs):

```python
def check_expert_parity(
    peer_ips: list[str], config: PipelineTransportConfig, rank: int, summary: dict
) -> None:
    """Tiny GPT-OSS under expert parallelism: both ranks compare to reference."""
    from gpt_oss_smoke_model import parity_batches, tiny_model

    group = config.bootstrap_jaccl(rank, peer_ips)
    from vllm_metal.distributed.tensor import TensorGroup

    tg = TensorGroup(group)
    batches = parity_batches()
    reference = []
    full = tiny_model()
    from mlx_lm.models.cache import make_prompt_cache

    ref_cache = make_prompt_cache(full)
    for _, batch in batches:
        logits = full(batch, cache=ref_cache)
        mx.eval(logits)
        reference.append(logits)
    model = tiny_model()
    from vllm_metal.distributed.experts import apply_expert_shard

    apply_expert_shard(model, tg)
    cache = make_prompt_cache(model)
    record = {"steps": [], "tolerance": PARITY_TOLERANCE}
    summary["tiny_gpt_oss_expert"] = record
    context_tokens = 0
    for step, (label, batch) in enumerate(batches):
        output = model(batch, cache=cache)
        context_tokens += batch.shape[1]
        mx.eval(output)
        difference = float(mx.max(mx.abs(output - reference[step])).item())
        argmax_equal = bool(
            mx.all(mx.argmax(output, axis=-1) == mx.argmax(reference[step], axis=-1)).item()
        )
        error = None if (difference <= PARITY_TOLERANCE and argmax_equal) else (
            f"logits differ: max_abs_diff={difference}, {argmax_equal=}"
        )
        agree_check(group, error, label)
        summary["checks_passed"] += 1
        record["steps"].append(
            {"phase": label, "max_abs_diff": difference, "argmax_equal": argmax_equal}
        )
```

Before wiring: read `agree_check` (tools/jaccl_pp_smoke.py:69-84) and pass whatever it needs — if it uses `PipelineGroup`-only attributes (`is_first`/`is_last`), wrap `group` in a 4-line adapter exposing those; if it only uses `rank()`/`size()`, `group` passes as-is. Wire the flag in `main()`: when `args.expert_parallel`, require `--model gpt-oss`, run `check_all_sum` + `check_expert_parity` (passing the `peer_ips` that `load_config` already parses) and skip the pipeline checks.

- [ ] **Step 2: Verify it imports and the CLI parses**

```bash
.venv/bin/python -c "import ast; ast.parse(open('tools/jaccl_pp_smoke.py').read())"
.venv/bin/python tools/jaccl_pp_smoke.py --help
```
Expected: help text shows `--expert-parallel`

- [ ] **Step 3: Full unit suite still green**

```bash
.venv/bin/python -m pytest -m "not slow" tests/ -q
```
Expected: 2225 + new EP tests passed, 5 skipped

- [ ] **Step 4: Commit**

```bash
git add tools/jaccl_pp_smoke.py
git commit -s -m "Add --expert-parallel mode to the JACCL two-Mac smoke"
```

---

### Task 7: inference-ui — `experts` mode in ClusterConfig

**Files:**
- Modify: `../inference-ui/cluster_config.py` (`__post_init__`:36-38, `environment()`:67-80, `server_command()`:111-151, `parse_args()`:164-171)
- Test: `../inference-ui/tests/test_cluster_config.py`

**Interfaces:**
- Consumes: vllm-metal EP admission (Task 2): `--tensor-parallel-size 2 --enable-expert-parallel` + `tensor_transport` additional-config.
- Produces: `ClusterConfig(parallelism="experts")` valid; `server_command()` emits PP=1/TP=2/`--enable-expert-parallel`/0.93 memory fraction; `environment()` passes through `VLLM_METAL_EXPERT_PARTITION` when set and mode is experts.

- [ ] **Step 1: Write the failing tests**

In `../inference-ui/tests/test_cluster_config.py`, extend the existing exact-argv test class with (follow the file's existing construction style — read it first):

```python
def test_experts_mode_maps_to_tp2_expert_parallel():
    config = ClusterConfig(parallelism="experts")
    argv = config.server_command()
    assert argv[argv.index("--pipeline-parallel-size") + 1] == "1"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert "--enable-expert-parallel" in argv
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.93"
    additional = json.loads(argv[argv.index("--additional-config") + 1])
    assert additional["tensor_transport"]["backend"] == "jaccl"


def test_experts_mode_rejects_unknown_and_passes_partition(monkeypatch):
    with pytest.raises(ValueError):
        ClusterConfig(parallelism="nope")
    monkeypatch.setenv("VLLM_METAL_EXPERT_PARTITION", "56,72")
    config = ClusterConfig(parallelism="experts")
    assert config.environment()["VLLM_METAL_EXPERT_PARTITION"] == "56,72"
    # Leak prevention: the partition only rides along for the experts mode.
    pipeline = ClusterConfig(parallelism="pipeline")
    assert "VLLM_METAL_EXPERT_PARTITION" not in pipeline.environment()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ../inference-ui && uv run pytest tests/test_cluster_config.py -v`
Expected: FAIL — `parallelism must be pipeline or tensor`

- [ ] **Step 3: Implement**

In `cluster_config.py`:

`__post_init__` (line 36-38):

```python
    def __post_init__(self):
        if self.parallelism not in ("pipeline", "tensor", "experts"):
            raise ValueError("parallelism must be pipeline, tensor, or experts")
```

`environment()` (after the `VLLM_PP_LAYER_PARTITION` block, line 78-80):

```python
        if self.parallelism == "experts":
            partition = os.environ.get("VLLM_METAL_EXPERT_PARTITION")
            if partition:
                environment["VLLM_METAL_EXPERT_PARTITION"] = partition
```

(add `import os` at the top).

`server_command()` — replace lines 115-116 and 129-132, 139-142:

```python
        if self.parallelism in ("tensor", "experts"):
            transport = {"tensor_transport": transport["pipeline_transport"]}
        experts_flags = (
            ["--enable-expert-parallel"] if self.parallelism == "experts" else []
        )
```

then PP size stays `"2" if self.parallelism == "pipeline" else "1"`, TP stays `"1" if self.parallelism == "pipeline" else "2"`, insert `*experts_flags` after the TP-size pair, and the memory fraction becomes:

```python
            str(
                0.93
                if self.parallelism in ("tensor", "experts")
                else self.memory_fraction
            ),
```

`parse_args()` (line 168-170): `choices=("pipeline", "tensor", "experts")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ../inference-ui && uv run pytest tests/test_cluster_config.py -v && uv run ruff check`
Expected: all PASS, ruff clean

- [ ] **Step 5: No commit (inference-ui is not a git repo)**

---

### Task 8: inference-ui — switch whitelists, scrub lists, dropdown, README

**Files:**
- Modify: `../inference-ui/cluster.py:86-88` (local env pops), `:96-99` (remote `env -u` list), `:241-243` (switch whitelist)
- Modify: `../inference-ui/runtime_control.py:112-113` (switch validation + message)
- Modify: `../inference-ui/parallelism_ui.py:18` (dropdown choices)
- Modify: `../inference-ui/README.md:34-59` (selector docs)
- Test: `../inference-ui/tests/` (extend only if a test enumerates modes)

**Interfaces:**
- Consumes: Task 7's valid `experts` mode.
- Produces: the UI offers and can switch to Experts; expert partition env cannot leak across mode switches.

- [ ] **Step 1: Extend tests if they enumerate modes**

Read `tests/test_parallelism_switch.py` and `tests/test_runtime_control.py` — they use `'tensor'` as a literal target, so they pass unchanged. Add one case to whichever file tests unknown-mode rejection, asserting `"experts"` is now accepted by `RuntimeController.switch` validation (mirror the existing unknown-mode test shape with `mode="experts"` — it should NOT raise the "Select Pipeline or Tensor." error).

- [ ] **Step 2: Implement**

`cluster.py` local scrub (line 86-88) — add:

```python
        env.pop("VLLM_METAL_EXPERT_PARTITION", None)
```

`cluster.py` remote guard (line 96-99) — extend the `env -u` list:

```python
                "-u",
                "VLLM_METAL_EXPERT_PARTITION",
```

`cluster.py` switch whitelist (line 241-243):

```python
        if mode not in ("pipeline", "tensor", "experts"):
            finish(False, "Unknown parallelism mode")
            return
```

`runtime_control.py` (line 112-113):

```python
        if mode not in ("pipeline", "tensor", "experts"):
            raise ValueError("Select Pipeline, Tensor, or Experts.")
```

`parallelism_ui.py` (line 18):

```python
                choices=[
                    ("Pipeline", "pipeline"),
                    ("Tensor", "tensor"),
                    ("Experts", "experts"),
                ],
```

`README.md`: add an "Experts" bullet beside the Tensor one (splits the 128 routed experts across the Macs, `VLLM_METAL_EXPERT_PARTITION="56,72"` for uneven splits, same 93% cache budget as Tensor to start).

- [ ] **Step 3: Run the suite**

Run: `cd ../inference-ui && uv run pytest && uv run ruff check`
Expected: all PASS, ruff clean

- [ ] **Step 4: No commit (not a git repo)**

---

### Task 9: Docs + full validation

**Files:**
- Modify: `docs/distributed.md` (new "GPT-OSS expert parallelism" section after the tensor-parallelism section)
- Modify: `docs/configuration.md` (env-var table: `VLLM_METAL_EXPERT_PARTITION`)

**Interfaces:**
- Consumes: everything above.
- Produces: documented mode.

- [ ] **Step 1: Write the docs**

`docs/distributed.md` — new section (match the file's voice: requirements, exact serve command with `--enable-expert-parallel`, how masking works in two sentences, the partition env var with the 56/72 example, the dummy-GEMM cost note, and the validation status line "validated on tiny models + the two-Mac smoke; 120B serving pending" until Task 10 fills it in). `docs/configuration.md` — one row for `VLLM_METAL_EXPERT_PARTITION`.

- [ ] **Step 2: Full suites**

```bash
.venv/bin/python -m pytest -m "not slow" tests/ -q
.venv/bin/python tools/check_parity.py --help
cd ../inference-ui && uv run pytest && uv run ruff check
```
Expected: vllm-metal green (2225 + EP tests), parity tool help renders, inference-ui green.

- [ ] **Step 3: Commit**

```bash
git add docs/distributed.md docs/configuration.md
git commit -s -m "Document GPT-OSS expert parallelism and its partition env var"
```

---

### Task 10: Hardware validation (interactive — needs both Macs)

**Files:**
- No repo changes; evidence to `/Users/fabricio/Desktop/apple-llms/context-growth-2026-09-18/` (or a new dated dir for EP).

- [ ] **Step 1: Sync the checkout to mac-smb** per `inference/docs/operations.md` (rsync + remote `uv sync --frozen` equivalent for vllm-metal: rsync the repo, ensure the peer's vllm-metal path matches).
- [ ] **Step 2: Two-Mac smoke**: `python tools/jaccl_pp_smoke.py --rank N --peer-ips ... --config tools/jaccl_two_rails.json --model gpt-oss --expert-parallel` on both Macs (rank 0 first). Expected: all checks pass, `tiny_gpt_oss_expert` steps report `argmax_equal: true`.
- [ ] **Step 3: Restart the cluster in experts mode**: `cd ../inference-ui && uv run python cluster.py stop && uv run python cluster.py start --parallelism experts`; confirm `status` shows `"parallelism": "experts"` and `/v1/models` answers.
- [ ] **Step 4: Sanity completions** (short + 1k-token chunked-prefill prompt, temperature 0, matching the distributed.md recipes).
- [ ] **Step 5: Context-growth benchmark**: `python3 <evidence-dir>/ctx_growth.py <evidence-dir>/ctx_growth_experts.csv`; add the EP row to the evidence README comparing pipeline/tensor/experts.
- [ ] **Step 6: If memory allows, try `VLLM_METAL_EXPERT_PARTITION=56,72`** (export before `cluster.py start`) and note the 48GB peer's headroom.
