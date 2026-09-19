# Expert Parallelism (EP2) for GPT-OSS — Design

Date: 2026-09-18
Branch: `feat/expert-parallelism` (off `feat/jaccl-pipeline`)
Status: approved in chat; implementation pending

## Goal

Add a third distributed mode — **experts parallelism** — to vllm-metal and
expose it in the inference-ui parallelism selector alongside Pipeline and
Tensor. Semantics: canonical vLLM TP2+EP. Attention is sharded exactly as
tensor mode today; the 128 routed experts per layer are split by **count**
(default 64/64, optionally uneven e.g. 56/72) instead of by intermediate
width. Both Macs run all 36 layers.

Non-goals (v1): MoE models other than GPT-OSS; EP+PP, EP+DP, TP>2;
EPLB / dynamic rebalancing; pair-filtering dispatch optimization; upstream
mlx-lm changes.

## Why

- Full-width expert kernels (TP halves intermediate width 2880→1440).
- Uneven expert partition matches the asymmetric Macs (48/64 GB) — the
  analog of PP's `VLLM_PP_LAYER_PARTITION=14,22`; TP width-split cannot be
  uneven.
- Path to >2 Macs where TP width-sharding degrades.
- On exactly 2 Macs expect EP ≈ TP (memory and collective count similar);
  the win is the above, not headline decode speed.

## Key facts this builds on (verified 2026-09-18)

- Expert weights are stacked `(128, out, in)`; expert is axis 0
  (`mlx_lm/models/switch_layers.py:99-106`). MXFP4 quant groups live along
  the input dim inside each expert row → axis-0 slicing is
  quantization-valid, no requant.
- Router is replicated and tiny; post-`o_proj`-all_sum activations are
  identical on both ranks (TP correctness already relies on this), so both
  ranks derive identical top-4 indices and softmax weights.
- `MLPBlock.__call__` already ends with `mx.distributed.all_sum(y,
  group=self.sharding_group)` when `sharding_group` is set
  (`mlx_lm/models/gpt_oss.py:163-164`) — the EP combine is free.
- We CANNOT call mlx-lm `Model.shard()` under EP: it width-slices experts
  (`gpt_oss.py:314-319`). Attention sharding must be replicated in
  vllm-metal via the same mlx.nn helpers.
- MLX 0.32.1 has no all_to_all; the design needs none (see Routing).
- `validate_tensor_config` currently REJECTS `enable_expert_parallel`
  (`vllm_metal/distributed/tensor.py:29`).

## Design

### 1. Admission (vllm-metal)

- `distributed/experts.py` (new module, runner-free like pipeline.py).
- `validate_tensor_config` grows an EP branch: `enable_expert_parallel`
  accepted only with TP=2, PP=1, DP=1, ray executor, gpt-oss, sync
  scheduling, explicit JACCL `tensor_transport`, no spec-decode/LoRA/gguf/
  multimodal — the existing TP whitelist verbatim.
- Group bootstrap, `synchronize_tokens`, eager-eval discipline, KV-head
  halving, selective-logits/decode-pipeline disables: ALL unchanged
  (attention stays TP2; `model_runner.py` TP hooks fire as today).

### 2. Weight sharding — `apply_expert_shard(model, group, partition)`

Runs in the same lifecycle slot as `apply_tensor_shard` (right after lazy
load, `vllm_metal/v1/model_runner.py:598-612`), strictly after sanitize.

- Attention per layer: q/k/v `all-to-sharded`, o_proj `sharded-to-all`,
  `num_attention_heads //= 2`, `num_key_value_heads //= 2`, attention
  sinks sliced per rank — mirroring `gpt_oss.py:288-312` (documented
  duplication, compat.py precedent).
- Experts per layer: axis-0 slice `[start:end)` of
  `experts.{gate,up,down}_proj.{weight,scales,bias}`. Router, norms,
  embed_tokens, lm_head untouched.
- Set `mlp.sharding_group = group` per layer.
- Partition: even split by default; `VLLM_METAL_EXPERT_PARTITION="56,72"`
  (comma ints, len == world size, sum == 128) selects uneven. New env in
  `vllm_metal/envs.py` following the existing registry pattern.

### 3. Masked routing (the exact-math core)

Wrap each `MLPBlock.__call__` (installed at load; patching.py
walk-and-wrap precedent):

1. Router logits → `mlx_topk(g, k=4)` → softmax over top-4 — unmodified,
   identical on both ranks.
2. Remap: local index = global − rank_start. Slots whose expert is not
   local → point at any local index (dummy) AND zero their
   `expert_weights` entry.
3. `SwitchGLU` / `mx.gather_qmm` run unchanged on remapped indices; the
   dummy contributions multiply by weight 0.
4. Existing `all_sum` combines partials. `y = Σ wᵢ·Eᵢ(x)` holds exactly.

Static shapes throughout; the sorted-token path (`do_sort` ≥64) survives
because `_gather_sort` is offset-agnostic.

Known cost (accepted for v1): dummy GEMMs for non-local slots ≈ 2×
per-rank expert FLOPs vs perfect dispatch. Optimization (pair filtering)
deferred until benchmarks demand it.

### 4. inference-ui

- Third mode `experts` at the four validation points:
  `cluster_config.py:36-38` (`__post_init__`), `:168-170` (argparse),
  `cluster.py:241-243` (switch whitelist), `runtime_control.py:112-113`
  (UI switch), plus the `parallelism_ui.py:18` dropdown ("Experts").
- `server_command()` branch: PP=1, TP=2, `--enable-expert-parallel`,
  `tensor_transport` additional-config block (same JACCL two-rail JSON),
  own `--gpu-memory-utilization` (start 0.93; measure and adjust).
- `VLLM_METAL_EXPERT_PARTITION` (if set) exported in `environment()` AND
  added to both scrub lists (`cluster.py:86-88` local pop, `:96-99` remote
  `env -u`) so mode switches can't leak it.
- Metrics already render any mode string (`presentation.py:19` →
  "Experts").
- README + tests updated; `test_cluster_config.py` exact-argv assertions
  rewritten for the third branch.

### 5. Validation ladder

1. Unit: admission matrix (EP accepted/rejected combos); shard shapes
   (experts axis-0 per partition, attention halved, router full) on the
   tiny smoke model; routing-wrap exactness — wrapped EP2 forward ==
   unsplit reference logits, prefill + cached decode
   (`test_pp_gpt_oss.py` pattern, `tools/gpt_oss_smoke_model.py` fixture).
2. `tools/check_parity.py` on the tiny model (EXACT bar).
3. Two-Mac JACCL smoke: extend `tools/jaccl_pp_smoke.py` (or sibling) with
   an EP mode; run on both Macs per `docs/distributed.md` recipe.
4. 120B serving on the cluster (UI switch to Experts) + `ctx_growth.py`
   benchmark row for EP vs the 2026-09-18 pipeline/tensor evidence.
5. Hardware testing requires rsync of the checkout to mac-smb first
   (workspace ops doc).

## Risks

- **Silent numerical corruption** if masking is wrong (no crash) — every
  ladder step is exact-match gated before the next.
- **mlx-lm duplication drift**: attention-shard mirroring breaks if
  mlx-lm's shard() changes — pinned exact version; covered by parity tests.
- **Memory**: EP per-Mac weights ≈ TP + replicated router (~1.5MB/layer) —
  negligible; start gpu-mem-util at 0.93 and verify on the 48GB peer.
- **Baseline stacking**: all distributed files are uncommitted WIP on
  `feat/jaccl-pipeline` — commit that baseline first (this repo, DCO `-s`);
  EP work lands on `feat/expert-parallelism` branched from it.
- inference-ui is NOT a git repo — no branch protection; edits stay
  minimal and additive.
