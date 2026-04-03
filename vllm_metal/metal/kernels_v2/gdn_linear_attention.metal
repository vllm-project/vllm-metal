// SPDX-License-Identifier: Apache-2.0
// GDN (Gated DeltaNet) linear attention kernel for paged state management.
//
// Reads/writes recurrent state in-place from a managed state pool via
// slot_mapping, following the same pattern as reshape_and_cache.
//
// Threading model (same as mlx_lm gated_delta_kernel):
//   Grid:        (Dv, 1, num_requests * Hv)
//   Threadgroup: (32, 1, 1)
//   Each SIMD group of 32 threads handles Dk elements (n_per_t = Dk/32).
//   simd_sum() reduces across threads for dot products.

#include <metal_stdlib>
using namespace metal;

#define GDN_KERNEL(DATA_T)                                                      \
[[kernel]] void gdn_linear_attention_##DATA_T(                                  \
    const device DATA_T *__restrict__ q          [[buffer(0)]],                 \
    const device DATA_T *__restrict__ k          [[buffer(1)]],                 \
    const device DATA_T *__restrict__ v          [[buffer(2)]],                 \
    const device DATA_T *__restrict__ g          [[buffer(3)]],                 \
    const device DATA_T *__restrict__ beta       [[buffer(4)]],                 \
    device DATA_T *__restrict__ state_pool       [[buffer(5)]],                 \
    const device int *__restrict__ cu_seqlens    [[buffer(6)]],                 \
    const device int *__restrict__ slot_mapping  [[buffer(7)]],                 \
    device DATA_T *__restrict__ y                [[buffer(8)]],                 \
    constant int &num_requests                   [[buffer(9)]],                 \
    constant int &Hk                             [[buffer(10)]],                \
    constant int &Hv                             [[buffer(11)]],                \
    constant int &Dk                             [[buffer(12)]],                \
    constant int &Dv                             [[buffer(13)]],                \
    uint3 gid [[threadgroup_position_in_grid]],                                 \
    uint tid [[thread_index_in_simdgroup]])                                     \
{                                                                               \
    const int req_idx = gid.z / Hv;                                             \
    const int hv_idx = gid.z % Hv;                                              \
    const int dv_idx = gid.x;                                                   \
    const int dk_idx = tid;                                                     \
                                                                                \
    if (req_idx >= num_requests || dv_idx >= Dv) return;                        \
                                                                                \
    const int hk_idx = hv_idx / (Hv / Hk);                                     \
    const int seq_start = cu_seqlens[req_idx];                                  \
    const int seq_end = cu_seqlens[req_idx + 1];                                \
    const int T = seq_end - seq_start;                                          \
    const int slot = slot_mapping[req_idx];                                     \
                                                                                \
    device DATA_T *state_ptr = state_pool                                       \
        + ((slot * Hv + hv_idx) * Dv + dv_idx) * Dk;                           \
                                                                                \
    const int n_per_t = Dk / 32;                                                \
    float state[8];                                                             \
    for (int i = 0; i < n_per_t; ++i) {                                        \
        int s_idx = n_per_t * dk_idx + i;                                      \
        state[i] = (s_idx < Dk)                                                \
            ? static_cast<float>(state_ptr[s_idx]) : 0.0f;                     \
    }                                                                           \
                                                                                \
    const device DATA_T *q_ = q + seq_start * Hk * Dk + hk_idx * Dk;           \
    const device DATA_T *k_ = k + seq_start * Hk * Dk + hk_idx * Dk;           \
    const device DATA_T *v_ = v + seq_start * Hv * Dv + hv_idx * Dv;           \
    const device DATA_T *g_ = g + seq_start * Hv;                              \
    const device DATA_T *beta_ = beta + seq_start * Hv;                        \
    device DATA_T *y_ = y + seq_start * Hv * Dv + hv_idx * Dv;                 \
                                                                                \
    for (int t = 0; t < T; ++t) {                                              \
        float g_val = static_cast<float>(g_[hv_idx]);                          \
        float kv_mem = 0.0f;                                                   \
        for (int i = 0; i < n_per_t; ++i) {                                   \
            int s_idx = n_per_t * dk_idx + i;                                  \
            state[i] *= g_val;                                                 \
            kv_mem += state[i] * static_cast<float>(k_[s_idx]);                \
        }                                                                       \
        kv_mem = simd_sum(kv_mem);                                             \
                                                                                \
        float delta = (static_cast<float>(v_[dv_idx]) - kv_mem)               \
                      * static_cast<float>(beta_[hv_idx]);                     \
                                                                                \
        float out = 0.0f;                                                      \
        for (int i = 0; i < n_per_t; ++i) {                                   \
            int s_idx = n_per_t * dk_idx + i;                                  \
            state[i] += static_cast<float>(k_[s_idx]) * delta;                 \
            out += state[i] * static_cast<float>(q_[s_idx]);                   \
        }                                                                       \
        out = simd_sum(out);                                                   \
        if (dk_idx == 0) {                                                     \
            y_[dv_idx] = static_cast<DATA_T>(out);                             \
        }                                                                       \
                                                                                \
        q_ += Hk * Dk;                                                         \
        k_ += Hk * Dk;                                                         \
        v_ += Hv * Dv;                                                         \
        y_ += Hv * Dv;                                                         \
        g_ += Hv;                                                              \
        beta_ += Hv;                                                           \
    }                                                                           \
                                                                                \
    for (int i = 0; i < n_per_t; ++i) {                                        \
        int s_idx = n_per_t * dk_idx + i;                                      \
        if (s_idx < Dk) {                                                      \
            state_ptr[s_idx] = static_cast<DATA_T>(state[i]);                  \
        }                                                                       \
    }                                                                           \
}

GDN_KERNEL(float)
GDN_KERNEL(half)
#if __METAL_VERSION__ >= 310
GDN_KERNEL(bfloat16_t)
#endif
