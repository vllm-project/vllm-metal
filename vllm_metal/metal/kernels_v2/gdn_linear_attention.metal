#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void gdn_linear_attention(
    const device T *__restrict__ q          [[buffer(0)]],
    const device T *__restrict__ k          [[buffer(1)]],
    const device T *__restrict__ v          [[buffer(2)]],
    const device T *__restrict__ g          [[buffer(3)]],
    const device T *__restrict__ beta       [[buffer(4)]],
    device T *__restrict__ state_pool       [[buffer(5)]],
    const device int *__restrict__ cu_seqlens    [[buffer(6)]],
    const device int *__restrict__ slot_mapping  [[buffer(7)]],
    device T *__restrict__ y                [[buffer(8)]],
    constant int &num_requests              [[buffer(9)]],
    constant int &Hk                        [[buffer(10)]],
    constant int &Hv                        [[buffer(11)]],
    constant int &Dk                        [[buffer(12)]],
    constant int &Dv                        [[buffer(13)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_simdgroup]])
{
    const int req_idx = gid.z / Hv;
    const int hv_idx = gid.z % Hv;
    const int dv_idx = gid.x;
    const int dk_idx = tid;

    if (req_idx >= num_requests || dv_idx >= Dv) return;

    // Map value head -> key head (GQA-style grouping)
    const int hk_idx = hv_idx / (Hv / Hk);

    // Sequence boundaries for this request
    const int seq_start = cu_seqlens[req_idx];
    const int seq_end = cu_seqlens[req_idx + 1];
    const int seq_len = seq_end - seq_start;

    // State pool slot for this request: [max_seqs, Hv, Dv, Dk]
    const int slot = slot_mapping[req_idx];
    device T *state_ptr = state_pool
        + ((slot * Hv + hv_idx) * Dv + dv_idx) * Dk;

    // n_per_t = Dk / 32 elements per thread (supports Dk up to 256)
    const int n_per_t = Dk / 32;
    float state[8];
    for (int i = 0; i < n_per_t; ++i) {
        int s_idx = n_per_t * dk_idx + i;
        state[i] = (s_idx < Dk)
            ? static_cast<float>(state_ptr[s_idx]) : 0.0f;
    }

    // Pointers into packed input tensors (offset by seq_start)
    const device T *q_ = q + seq_start * Hk * Dk + hk_idx * Dk;
    const device T *k_ = k + seq_start * Hk * Dk + hk_idx * Dk;
    const device T *v_ = v + seq_start * Hv * Dv + hv_idx * Dv;
    const device T *g_ = g + seq_start * Hv;
    const device T *beta_ = beta + seq_start * Hv;
    device T *y_ = y + seq_start * Hv * Dv + hv_idx * Dv;

    // === Main recurrence loop ===
    for (int t = 0; t < seq_len; ++t) {
        float g_val = static_cast<float>(g_[hv_idx]);

        // Decay + compute k . state dot product
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            int s_idx = n_per_t * dk_idx + i;
            state[i] *= g_val;
            kv_mem += state[i] * static_cast<float>(k_[s_idx]);
        }
        kv_mem = simd_sum(kv_mem);

        // Delta update: delta = (v - k . state) * beta
        float delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
                      * static_cast<float>(beta_[hv_idx]);

        // State update + compute q . state output
        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            int s_idx = n_per_t * dk_idx + i;
            state[i] += static_cast<float>(k_[s_idx]) * delta;
            out += state[i] * static_cast<float>(q_[s_idx]);
        }
        out = simd_sum(out);

        // First thread in SIMD group writes output
        if (dk_idx == 0) {
            y_[dv_idx] = static_cast<T>(out);
        }

        // Advance to next timestep
        q_ += Hk * Dk;
        k_ += Hk * Dk;
        v_ += Hv * Dv;
        y_ += Hv * Dv;
        g_ += Hv;
        beta_ += Hv;
    }

    // Write state back in-place to pool
    for (int i = 0; i < n_per_t; ++i) {
        int s_idx = n_per_t * dk_idx + i;
        if (s_idx < Dk) {
            state_ptr[s_idx] = static_cast<T>(state[i]);
        }
    }
}

// Template instantiations — same pattern as reshape_and_cache.metal
#define instantiate_gdn(type)                                      \
  template [[host_name("gdn_linear_attention_" #type)]]            \
  [[kernel]] void gdn_linear_attention<type>(                      \
      const device type*, const device type*,                      \
      const device type*, const device type*,                      \
      const device type*, device type*,                            \
      const device int*, const device int*,                        \
      device type*,                                                \
      constant int&, constant int&, constant int&,                 \
      constant int&, constant int&,                                \
      uint3, uint);

instantiate_gdn(float);
instantiate_gdn(half);
instantiate_gdn(bfloat16_t);
