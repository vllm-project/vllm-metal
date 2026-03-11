# Cross-Validation: Metal Attention Repos vs Our vllm-metal v2

> 3 repos analyzed, cross-referenced against our v1 kernel and FINAL_REPORT.md checklist.

---

## Verdict: Nothing Directly Vendorable. Patterns to Reimplement.

All 3 repos generate Metal shaders at runtime (Swift or C++ string emission).
None produce standalone `.metal` files compatible with our static compilation pipeline.
However, several validated patterns should be reimplemented in our v2 kernel.

---

## Repo Quality Ranking

| Repo | Quality | Useful? | Key Contribution |
|------|---------|---------|------------------|
| **philipturner** | High (well-engineered, MIT) | Patterns only | Online softmax, simdgroup_matrix, Morton order |
| **marcogva** | Medium (AI-assisted but real engineering) | Patterns + validation | Paged KV gather, varlen cu_seqlens, NaN-safe softmax, tile configs |
| **bghira** | Low (wrapper + AI boilerplate) | No | Just wraps philipturner; GLUON additions are broken |

---

## Cross-Validated Patterns (Consistent Across All 3 Repos)

### 1. Online Softmax with Base-2 Arithmetic — CONFIRMED x3

All repos use the same approach (originated by Philip Turner):
```metal
// Premultiply scale by log2(e) once
float scale_log2e = softmax_scale * 1.442695041f;

// In the KV loop:
float m_new = row_max(S);                      // reduce max
float correction = fast::exp2(m_old - m_new);  // rescale factor
m = m_new;
l = l * correction + row_sum_exp;              // rescale sum
// P = fast::exp2(S - m)                       // attention weights
// O = O * correction + P @ V                  // rescale + accumulate
```

Using `fast::exp2` instead of `exp` is faster on Apple GPUs.
**This pattern should be adopted directly in our v2 kernel.**

### 2. simd_shuffle_xor(1) + simd_shuffle_xor(8) Reduction — CONFIRMED x3

The canonical way to reduce across rows within 8x8 simdgroup_matrix fragments:
```metal
// Each 8x8 matrix has 4 threads per row, each holding 2 elements
float val = op(elem[0], elem[1]);          // reduce 2 elements per thread
val = op(val, simd_shuffle_xor(val, 1));   // reduce across column pairs
val = op(val, simd_shuffle_xor(val, 8));   // reduce across quadrant pairs
// val now contains the row-wise reduction
```

Works for both max (online softmax) and sum (exp_sum).

### 3. simdgroup_matrix 8x8 MMA — CONFIRMED x2 (philipturner, marcogva)

All repos that implement the attention GEMM use `simdgroup_multiply_accumulate`
with 8x8 tiles. The wrapper pattern:
```metal
simdgroup_matrix<T, 8, 8> A, B, C;
// Load A, B from registers/threadgroup memory
simdgroup_multiply_accumulate(C, A, B, C);  // C += A @ B
```

**Key finding**: This approach gives 1.5-2x speedup over MLX SDPA for **prefill**
(N_q >> 1), but is NOT beneficial for **decode** (N_q = 1) because a 1-row Q
wastes 7/8 of the 8x8 tile.

### 4. Morton Order Thread Layout — CONFIRMED x2

The mapping from SIMD lane to (row, col) in an 8x8 fragment:
```metal
ushort2 morton_order(ushort lane_id) {
    ushort quad_id = lane_id / 4;
    ushort M_floor = (quad_id / 4) * 4;
    ushort M_in = (lane_id / 2) % 4;
    ushort N_floor = (quad_id & 2) * 2;
    ushort N_in = (lane_id % 2) * 2;
    return ushort2(N_floor + N_in, M_floor + M_in);
}
```

Needed if we use simdgroup_matrix. Not needed for scalar decode kernel.

---

## Feature Gap Analysis

| Feature | philipturner | bghira | marcogva | Our v1 | Needed for v2 |
|---------|:---:|:---:|:---:|:---:|:---:|
| Online softmax | YES | YES* | YES | NO (two-pass) | **Step 1** |
| Paged KV | NO | NO | YES | YES | **Step 1** |
| Varlen / cu_seqlens | NO | NO | YES | NO | Step 1 (interface), Step 2 (impl) |
| GQA | NO | PARTIAL | YES | YES | **Step 1** |
| Causal masking | NO | NO | PARTIAL | NO (external) | **Step 1** |
| Causal block skip | NO | NO | PARTIAL | NO | Step 1 |
| Split-KV decode | NO | NO | YES† | YES | Later |
| Sliding window | NO | YES | YES | NO | Step 2 |
| Soft capping | NO | NO | YES | NO | Step 2 |
| simdgroup_matrix | YES | YES* | YES | NO | Step 2 (prefill) |

*via philipturner's code  †in ccv-derived kernel, not STEEL rewrite

**Critical gap**: No repo has a unified varlen + paged kernel. marcogva has them
as separate kernels. We will be the first to combine them.

---

## Specific Patterns to Reimplement

### For Step 1 (Decode-Only, Online Softmax)

**1. Online softmax core** (from philipturner, validated by all 3):
```metal
// Initialize
float m = -INFINITY;
float l = 0.0f;
float O[HEAD_SIZE] = {0};  // accumulator in registers

// For each KV block:
float scores[BLOCK_SIZE];
// ... compute QK dot products into scores[] ...

// Running max
float block_max = -INFINITY;
for (int i = 0; i < valid_tokens; i++)
    block_max = max(block_max, scores[i]);
// SIMD reduce
block_max = simd_max(block_max);  // or manual shuffle tree

// Rescale
float correction = 1.0f;
if (block_max > m) {
    correction = fast::exp2((m - block_max) * LOG2E);
    m = block_max;
}

// Scale accumulator
for (int d = 0; d < HEAD_SIZE; d++)
    O[d] *= correction;
l *= correction;

// Softmax weights
float weights[BLOCK_SIZE];
for (int i = 0; i < valid_tokens; i++) {
    weights[i] = fast::exp2((scores[i] - m) * LOG2E);
    l += weights[i];
}

// Accumulate V
for (int d = 0; d < HEAD_SIZE; d++)
    for (int i = 0; i < valid_tokens; i++)
        O[d] += weights[i] * V[i][d];

// Final: O /= l
```

**2. NaN-safe sentinel** (from marcogva):
```metal
// When all scores are -inf (fully masked row), m stays -inf
// exp2(-inf - (-inf)) = exp2(NaN) = NaN → corrupts output
// Fix: clamp m to 0 when it's -inf
if (isinf(m) && m < 0) m = 0.0f;
// Now exp2(-inf - 0) = 0 → correct zero contribution
```

**3. Causal mask for decode** (trivial):
```metal
// For decode (q_len=1), the query is at position kv_len-1
// Causal: only attend to positions 0..kv_len-1
// Since all positions are ≤ query position, no masking needed!
// Only mask padding beyond kv_len within the last KV block.
if (block_offset + token_idx >= kv_len) {
    scores[token_idx] = -INFINITY;
}
```

### For Step 2 (Prefill, simdgroup_matrix)

**4. K-transposed / V-row-major shared buffer** (from marcogva):
```
K_smem[d][seq] = K[seq][d]    // transposed for Q @ K^T
V_smem[seq][d] = V[seq][d]    // row-major for P @ V
// Can share same threadgroup memory (time-multiplexed)
```

**5. Tile configurations** (from marcogva, validated on M1 Max):
```
D=128, f16:  BQ=32, BK=16 (M1/M2) or 32 (M3+), 128 threads (4 SIMD groups)
D=64,  f16:  BQ=32, BK=32, 128 threads
D=256, f16:  BQ=16, BK=8,  128 threads  (register pressure)
```

**6. Cooperative paged KV gather** (from marcogva, needs improvement):
```metal
// All threads cooperate to fill BK * D elements from scattered pages
for (int slot = thread_idx; slot < BK * D; slot += TGP_SIZE) {
    int t = slot % BK;         // token within tile
    int d = slot / BK;         // head-dim index
    int global_tok = kb * BK + t;
    int blk = global_tok / block_size;
    int off = global_tok % block_size;
    int phys = block_table[seq * max_blocks + blk];
    K_smem[d * LDK + t] = k_cache[phys * blk_stride + off * tok_stride + kv_head * D + d];
}
```
Note: integer div/mod in inner loop is expensive. Better to precompute on CPU.

---

## Contradictions Resolved

**No significant contradictions** between the 3 repos. They agree on:
- Base-2 exp is preferred on Metal (`fast::exp2` > `exp`)
- 8x8 simdgroup_matrix is the right granularity
- simd_shuffle_xor(1) + simd_shuffle_xor(8) is the row reduction pattern
- 32KB threadgroup memory is the hard limit
- Online softmax eliminates the logits storage bottleneck

**One nuance**: philipturner uses `fast::divide(1, l)` for final normalization
while marcogva uses `1.0f / l`. The `fast::` variant is slightly less accurate
but faster. For fp16 attention, the difference is negligible.

---

## Final Recommendation for Step 1

**Do NOT vendor any code. Reimplement these patterns in a static `.metal` file:**

1. Online softmax with `fast::exp2` and `LOG2E`-premultiplied scale
2. NaN-safe sentinel (clamp -inf max to 0)
3. Keep our v1's scalar decode approach (not simdgroup_matrix — that's for Step 2 prefill)
4. Keep our v1's paged KV block_table lookup
5. Accept the unified interface (cu_seqlens_q, seqused_k) but only handle q_len=1
6. Add causal masking (trivial for decode)
7. Use function constants for sliding_window/softcap (disabled in Step 1)

The kernel should live at `vllm_metal/metal/kernels_v2/unified_attention.metal`.
