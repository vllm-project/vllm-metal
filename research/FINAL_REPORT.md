# Unified Varlen Paged Attention: Research Report for Metal

> Compiled from 4 parallel research agents + 1 contradiction-resolution agent.
> Date: 2026-03-10 | Target: vllm-metal v2 kernel (Step 1: decode-only)

---

## Executive Summary

We investigated FlashAttention v1-v4, the vLLM Triton unified kernel, Metal vs NVIDIA
platform differences, and 6 other attention implementations (AMD, TPU, CPU, FlashInfer,
xFormers, existing Metal projects). The core finding:

**The algorithmic foundation (online softmax, paged KV, varlen via cu_seqlens) is
universal across ALL backends.** What differs is memory access patterns, tiling, and
synchronization — all of which must be adapted for Metal's specific constraints.

**No unified varlen paged attention kernel exists for Metal anywhere.** We are building
the first one.

---

## Part 1: What Innovations Exist and Are They Portable?

### 1.1 Online Softmax (FA1) — YES, portable

The single-pass tiling algorithm with running max `m` and running sum `l`:
```
For each KV block:
    S = Q @ K^T * scale
    m_new = max(m_old, rowmax(S))
    P = exp(S - m_new)
    l_new = exp(m_old - m_new) * l_old + rowsum(P)
    O_new = exp(m_old - m_new) * O_old + P @ V
Final: O = O / l
```

**Why it matters for us**: Our v1 kernel uses a TWO-PASS approach — it stores ALL QK
logits in threadgroup memory, finds the max, then computes softmax. This requires
`padded_ctx * sizeof(float)` bytes of threadgroup memory. For a 8192-token sequence,
that's exactly 32KB — our entire threadgroup memory budget.

Online softmax replaces this with O(TILE_SIZE) threadgroup memory, enabling arbitrarily
long sequences without partitioning. **This is the #1 algorithmic upgrade for v2.**

Every backend (NVIDIA, AMD, TPU, CPU, FlashInfer, xFormers, MLX's own sdpa_vector)
uses this same algorithm. The merge operation for split-KV is also universal:
```
m_new = max(m_a, m_b)
d_new = d_a * exp(m_a - m_new) + d_b * exp(m_b - m_new)
o_new = (o_a * d_a * exp(m_a - m_new) + o_b * d_b * exp(m_b - m_new)) / d_new
```

### 1.2 Outer-Q / Inner-KV Loop Order (FA2) — YES, portable

FA1 looped outer-KV/inner-Q; FA2 reverses to outer-Q/inner-KV. Each threadgroup
"owns" a Q tile and streams through all KV blocks, keeping the output accumulator
in registers. Written to device memory only once at the end.

The vLLM Triton unified kernel uses this same order. Pure algorithm choice.

### 1.3 Causal Block Skipping (FA2) — YES, portable

For causal attention, ~50% of KV blocks are fully masked. Simply skip them:
```
if tile_start_kv > query_position: skip
```
~2x speedup for causal workloads. Pure control flow.

### 1.4 Varlen via cu_seqlens (FA2 / universal) — YES, portable

Pack all sequences into one tensor, use cumulative sequence lengths for boundaries:
```
cu_seqlens_q = [0, q_len_0, q_len_0 + q_len_1, ..., total_tokens]
```
The vLLM Triton kernel maps global Q-block indices to sequences via binary search.
Used identically by: FlashAttention, FlashInfer, xFormers, TPU, CPU backends.

### 1.5 Paged KV Cache with Block Tables — YES, portable (already implemented)

Read KV from non-contiguous pages via `block_table[seq_idx][logical_block]`:
```
physical_block = block_table[seq][kv_pos / block_size]
offset = kv_pos % block_size
kv_ptr = cache + physical_block * block_stride + offset * token_stride
```
Already working in our v1 kernel. KV cache layout `[num_blocks, block_size, num_kv_heads,
head_size]` matches upstream vLLM.

### 1.6 GQA (Grouped-Query Attention) — YES, portable

The vLLM Triton kernel handles GQA by interleaving query heads within BLOCK_M rows:
- Grid dim 1 = num_kv_heads
- Within each block, `num_queries_per_kv` query heads share K/V loads
- BLOCK_M = max(16, next_power_of_2(num_queries_per_kv))

Pure indexing pattern. Our v1 already supports GQA via `num_kv_heads` parameter.

### 1.7 Split-KV Decode (Flash-Decoding) — YES, portable

For long-context decode (q_len=1), split the KV sequence across multiple threadgroups
for parallelism. Each computes partial (output, max, exp_sum). A reduce kernel merges.

Used by: vLLM Triton (3D kernel), AMD (T_PAR_SIZE=256), FlashInfer, MLX (sdpa_vector_2pass).
Our v1 already supports this via `PARTITION_SIZE`.

### 1.8 Sliding Window — YES, portable

Two parts: (1) skip tiles outside the window range, (2) mask individual positions within
boundary tiles. Pure comparison logic on position indices.
```
if (query_pos - kv_pos) >= sliding_window: mask out
```

### 1.9 Soft Capping — YES, portable

`scores = softcap * tanh(scores / softcap)`, applied after QK, before masking.
Pure arithmetic. The vLLM Triton kernel implements it as:
```
p1 = exp(S/x); p2 = exp(-S/x); result = x * (p1-p2)/(p1+p2)
```

### 1.10 Delayed / Conditional Rescaling (FA2 / FA4) — YES, portable

FA2: defer the final `O /= l` to the end instead of rescaling at every step.
FA4: skip rescaling when `m_new ≈ m_old` (threshold-based). Reduces non-matmul FLOPs.
Both are pure algorithmic optimizations.

### 1.11 LPT Grid Scheduling (FA4) — YES, portable

For varlen+causal, sort sequences longest-first for better load balancing. Reduces
tail latency. CPU-side scheduling, runs before kernel dispatch.

### 1.12 Plan-then-Execute (FlashInfer) — YES, portable

CPU-side planning builds indirection arrays (request_indices, kv_tile_indices), then
a fixed GPU kernel reads them. Decouples scheduling from computation. Enables dynamic
batching without kernel recompilation. **Highly recommended for our design.**

### 1.13 Warp Specialization (FA3) — NO, not portable

Producer warps load data via TMA while consumer warps compute via WGMMA. Requires:
- TMA hardware unit (Hopper-only)
- Named barriers for producer/consumer sync
- Warpgroup-level (128-thread) scheduling

Metal has NONE of these. No equivalent possible.

### 1.14 Pingpong Scheduling (FA3) — NO, not portable

Two warpgroups alternate GEMM and softmax. Requires named barriers and warpgroup
constructs. Metal only has `threadgroup_barrier`.

### 1.15 Async Memory Pipeline (FA3/FA4) — NO, not portable

`cp.async`, `cp.async.bulk`, TMA — hardware DMA units that copy data to shared memory
while compute proceeds. **This is the single largest architectural gap.**

Metal's async copy situation (resolved contradiction):
- `simdgroup_async_copy` exists but is **undocumented** (leaked in Xcode 14.2, removed)
- Accessible only via AIR inline assembly intrinsics
- No fine-grained pipeline depth control (`cp.async.wait_group<N>`)
- MLX does NOT use it anywhere
- On M3+ with shared cache hierarchy, benefit for memory-bound kernels is negligible

**Recommendation**: Do NOT use undocumented async copy. Rely on Metal's SIMD-group
interleaving for latency hiding (high occupancy = automatic overlap).

### 1.16 FP8 Hardware Acceleration — NO, not portable

NVIDIA Hopper/Ada: native FP8 tensor core ops with 2x throughput over FP16.
Metal: no FP8 hardware. Our v1 does software FP8↔float conversion for storage only.

### 1.17 TMEM, 2-CTA MMA, Async Tensor Cores (FA4/Blackwell) — NO, not portable

All Blackwell-specific hardware features with no Metal equivalent.

### 1.18 simdgroup_matrix for QK/PV Matmul — MAYBE, needs investigation

Metal's `simdgroup_matrix<T, 8, 8>` provides 8x8 matrix multiply-accumulate:
- Not dedicated tensor core hardware — uses existing ALU with improved scheduling
- ~2x over scalar FP32 (estimated)
- Our v1 does NOT use it (uses scalar FMA for dot products)
- Could improve prefill throughput where QK and PV are effectively small GEMMs
- For decode (q_len=1), less beneficial since QK is a single dot product per KV token

**Recommendation**: Skip for Step 1 (decode-only). Investigate for Step 2 (prefill).

### 1.19 Double Buffering in Threadgroup Memory — MAYBE, needs benchmarking

Load next KV block while computing on current one. Possible on Metal with manual
buffer management, but:
- No `cp.async` means loads still go through registers
- On M3+, threadgroup and device memory share the same cache — staging may not help
- Metal hides latency via SIMD-group interleaving instead of explicit pipelining

**Recommendation**: Skip initially. Profile first, add if bottleneck is identified.

---

## Part 2: Metal vs NVIDIA — Key Differences

### What's the Same
| Feature | NVIDIA | Metal |
|---------|--------|-------|
| SIMD width | 32 (warp) | 32 (SIMD group) |
| Shuffle ops | `__shfl_xor_sync` etc. | `simd_shuffle_xor` etc. |
| Reduction intrinsics | `__reduce_add/max_sync` | `simd_sum`, `simd_max` |
| Max threadgroup size | 1024 | 1024 |
| FP16/BF16 native | Yes | Yes (BF16 native on M3+, software on M1/M2) |
| Dynamic shared memory | `extern __shared__` | `[[threadgroup(0)]]` |

### What's Different (Critical for kernel design)

| Constraint | NVIDIA | Metal | Impact |
|------------|--------|-------|--------|
| **Threadgroup memory** | 48-100+ KB | **32 KB max** | Limits tile sizes. Online softmax eliminates logits storage. |
| **Async copy** | cp.async, TMA | **None usable** | No compute-memory overlap within threadgroup. Must rely on occupancy for latency hiding. |
| **Matrix acceleration** | Tensor Cores (16x16) | simdgroup_matrix (8x8) | Smaller tiles, not dedicated hardware. |
| **Warp specialization** | Yes (Hopper) | **No** | Can't do producer/consumer patterns. |
| **Cross-threadgroup sync** | Cooperative groups | **No** | Each threadgroup fully independent. |
| **Memory bandwidth** | HBM: 1.5-3.35 TB/s | LPDDR5: 100-800 GB/s | Memory-bound by default. Must minimize reads. |
| **Register control** | `__launch_bounds__` | **Compiler-managed** | Can't manually tune register pressure. |

### What Metal Has That NVIDIA Doesn't

| Feature | Benefit |
|---------|---------|
| **Function constants** | Cleaner dead-code elimination than templates. Compile-time specialization at pipeline creation. |
| **Unified memory** | Zero-copy for metadata (block tables, cu_seqlens). No CPU↔GPU staging. |
| **Large SLC** | 8-96 MB System Level Cache partially compensates for lower DRAM bandwidth. |
| **M3+ dynamic cache** | Threadgroup memory and registers share physical storage. Less of one = more of the other. |
| **Always-converged SIMD** | No warp divergence complexity. No `__syncwarp()` needed. Simpler reduction code. |

---

## Part 3: What Our v2 Kernel Should Look Like (Step 1: Decode-Only)

### Architecture for Step 1

For decode-only (q_len=1 for all sequences), the v2 kernel needs to:
1. Accept the unified interface: `(q, k_cache, v_cache, out, cu_seqlens_q, seqused_k, ...)`
2. Produce identical results to v1 for decode inputs
3. Use online softmax instead of the v1's two-pass approach

**Why online softmax even for decode?** Our v1 allocates `padded_ctx * 4 bytes` of
threadgroup memory for logits. Online softmax eliminates this, freeing threadgroup
memory budget and enabling longer sequences without partitioning.

### Recommended Kernel Structure (Step 1)

```
Grid: (num_heads, num_seqs, 1)  — same as v1
Threadgroup: (256, 1, 1)        — same as v1 (8 SIMD groups)

Each threadgroup:
1. Read cu_seqlens_q to find seq_idx, verify q_len=1
2. Read seqused_k[seq_idx] for kv_len
3. Load Q vector into registers (1 token × head_size)
4. Initialize online softmax state: m=-inf, l=0, O=zeros

5. For each KV block (iterate via block_table):
    a. Load K block from cache (block_size tokens × head_size)
    b. Compute QK dot products → scores[block_size] (in registers)
    c. Apply causal mask (trivial for decode: mask kv_pos >= kv_len)
    d. Update online softmax: m_new, l_new, rescale O
    e. Load V block, accumulate: O += softmax_weights @ V

6. Final normalization: O /= l
7. Write O to output[seq_idx, head_idx, :]
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Loop order | Q-outer (trivial for decode: 1 Q token), KV-inner | Standard FA2 order. One threadgroup owns one (seq, head). |
| Softmax | Online (single-pass) | Eliminates logits storage. Future-proofs for varlen. |
| Threadgroup memory | Minimal: only for cross-SIMD reduction | Q in registers (1 token). No logits array. |
| Partitioning | Not needed for Step 1 | Can add split-KV later if sequences are very long. |
| Function constants | `use_sliding_window`, `use_softcap` (both false for Step 1) | Dead-code elimination. Easy to enable later. |

### Interface Contract (from test file)

```python
metal_unified_attention(
    q,              # [total_q_tokens, num_q_heads, head_size]
    k,              # [num_blocks, block_size, num_kv_heads, head_size]
    v,              # [num_blocks, block_size, num_kv_heads, head_size]
    out,            # [total_q_tokens, num_q_heads, head_size]
    cu_seqlens_q,   # [num_seqs + 1], int32
    seqused_k,      # [num_seqs], int32
    max_seqlen_q,   # int (=1 for decode)
    max_seqlen_k,   # int
    softmax_scale,  # float
    causal,         # bool
    window_size,    # tuple[int,int], (-1,-1) = no window
    block_table,    # [num_seqs, max_blocks_per_seq], int32
    softcap,        # float, 0 = disabled
)
```

For Step 1: `max_seqlen_q=1`, `window_size=(-1,-1)`, `softcap=0`.
The q tensor shape `[total_q_tokens, num_q_heads, head_size]` equals
`[num_seqs, num_q_heads, head_size]` when all q_len=1.

---

## Part 4: Portability Summary Table

| Innovation | Source | Portable? | Priority for v2 |
|------------|--------|-----------|-----------------|
| Online softmax | FA1 | **YES** | Step 1 (critical) |
| Outer-Q / Inner-KV loop | FA2 | **YES** | Step 1 |
| Paged KV via block tables | vLLM | **YES** | Step 1 (already have) |
| cu_seqlens varlen packing | FA2 / universal | **YES** | Step 1 (interface), Step 2 (varlen) |
| Causal block skipping | FA2 | **YES** | Step 1 |
| GQA (shared K/V loads) | FA2 / vLLM | **YES** | Step 1 (already have) |
| Split-KV decode | Flash-Decoding | **YES** | Step 1 if needed, else later |
| Sliding window | vLLM | **YES** | Step 2 |
| Soft capping | Gemma | **YES** | Step 2 |
| Delayed rescaling | FA2 | **YES** | Step 1 (natural with online softmax) |
| Conditional rescaling | FA4 | **YES** | Optimization (later) |
| LPT scheduling | FA4 | **YES** | Optimization (later) |
| Plan-then-execute | FlashInfer | **YES** | Step 2-3 |
| Binary search seq lookup | vLLM Triton | **YES** | Step 2 (varlen) |
| Batch reordering | CPU backend | **YES** | Step 2-3 |
| simdgroup_matrix for QK/PV | Metal-native | **MAYBE** | Investigation (later) |
| Double buffering | FA3 concept | **MAYBE** | Benchmark first |
| Warp specialization | FA3 | **NO** | — |
| TMA / cp.async | FA3 | **NO** | — |
| Pingpong scheduling | FA3 | **NO** | — |
| FP8 tensor cores | FA3 | **NO** | — |
| TMEM / 2-CTA MMA | FA4 | **NO** | — |
| Cooperative groups | CUDA | **NO** | — |

---

## Part 5: Corrections & Resolved Contradictions

### Threadgroup Memory: 32 KB (not 64 KB)
The FA researcher mentioned "up to 64KB on some devices." This is **incorrect for Apple
Silicon**. Verified via `MTLDevice.maxThreadgroupMemoryLength()` and WWDC 2022: the
limit is **32,768 bytes (32 KB)** on all Apple Silicon (M1 through M4).

### Async Copy on Metal: Exists but Unusable
- `simdgroup_async_copy` exists but is **undocumented** (leaked in Xcode 14.2, removed)
- `threadgroup_async_copy` (OpenCL-style) does NOT exist in MSL
- MLX does not use any async copy
- **Recommendation**: Do not use. Rely on SIMD-group interleaving for latency hiding.

### Double Buffering: Not Recommended for Step 1
Without hardware async copy, double buffering requires threads to explicitly load data
through registers, which doesn't truly overlap with compute. On M3+ with shared cache,
the benefit is further reduced. May revisit after profiling.

### Two Kinds of "Two-Pass" (Terminology Clarification)
1. **Softmax two-pass**: compute all QK logits → find max → softmax. This is what our
   v1 does. Online softmax eliminates this.
2. **Split-KV two-pass**: partition KV across threadgroups, each computes partial result,
   then merge. This is MLX's `sdpa_vector_2pass`. Orthogonal to online softmax.

Both can coexist: online softmax WITHIN each partition, multiple partitions merged in
a second kernel.

---

## Individual Research Reports

- [FlashAttention v1-v4](flash_attention.md) — 583 lines
- [vLLM Triton Unified Kernel](vllm_triton_unified.md) — 588 lines
- [Metal vs NVIDIA Platform](metal_vs_nvidia.md) — 715 lines
- [Other Backends (AMD, TPU, CPU, FlashInfer, xFormers, Metal)](other_backends.md) — 607 lines
- [Async Copy Resolution](async_copy_resolution.md) — contradiction resolution
