// Template params: T (input/output dtype), StT (state dtype), Dk, Dv, Hk, Hv
// Inputs: q, k, v, g, beta, state_in, cu_seqlens, slot_mapping, num_requests
// Outputs: y, state_out
// Grid: (32, Dv, num_requests * Hv), Threadgroup: (32, 4, 1)
//
// This is the lazy-graph counterpart of gdn_linear_attention.metal for
// prefill-containing batches.  It keeps the per-request sequence loop but emits
// compact state updates for active requests instead of mutating the full state
// pool in-place, allowing MLX to schedule the recurrent work lazily.

auto n = thread_position_in_grid.z;
auto req_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
auto dv_idx = thread_position_in_grid.y;
auto dk_idx = thread_position_in_threadgroup.x;
constexpr int n_per_t = Dk / 32;

if (req_idx >= (uint)num_requests || dv_idx >= (uint)Dv) return;

uint slot_idx = (uint)slot_mapping[req_idx];
int state_base = ((slot_idx * Hv + hv_idx) * Dv + dv_idx) * Dk;
int state_out_base = ((req_idx * Hv + hv_idx) * Dv + dv_idx) * Dk;

int seq_start = cu_seqlens[req_idx];
int seq_end = cu_seqlens[req_idx + 1];
int seq_len = seq_end - seq_start;

float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
    int s_idx = n_per_t * dk_idx + i;
    state[i] = static_cast<float>(state_in[state_base + s_idx]);
}

auto q_ = q + seq_start * Hk * Dk + hk_idx * Dk;
auto k_ = k + seq_start * Hk * Dk + hk_idx * Dk;
auto v_ = v + seq_start * Hv * Dv + hv_idx * Dv;
auto g_ = g + seq_start * Hv;
auto beta_ = beta + seq_start * Hv;
auto y_ = y + seq_start * Hv * Dv + hv_idx * Dv;

for (int t = 0; t < seq_len; ++t) {
    float g_val = static_cast<float>(g_[hv_idx]);

    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        int s_idx = n_per_t * dk_idx + i;
        state[i] *= g_val;
        kv_mem += state[i] * static_cast<float>(k_[s_idx]);
    }
    kv_mem = simd_sum(kv_mem);

    float delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
                  * static_cast<float>(beta_[hv_idx]);

    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        int s_idx = n_per_t * dk_idx + i;
        state[i] += static_cast<float>(k_[s_idx]) * delta;
        out += state[i] * static_cast<float>(q_[s_idx]);
    }
    out = simd_sum(out);

    if (thread_index_in_simdgroup == 0) {
        y_[dv_idx] = static_cast<T>(out);
    }

    q_ += Hk * Dk;
    k_ += Hk * Dk;
    v_ += Hv * Dv;
    y_ += Hv * Dv;
    g_ += Hv;
    beta_ += Hv;
}

for (int i = 0; i < n_per_t; ++i) {
    int s_idx = n_per_t * dk_idx + i;
    state_out[state_out_base + s_idx] = static_cast<StT>(state[i]);
}
