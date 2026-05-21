// Grid: (32, Dv, num_requests * Hv), Threadgroup: (32, TG_DV, 1)
// TG_DV is selected by the Python wrapper for the active GDN state layout.
// Each SIMD group of 32 threads handles Dk elements for one (request, hv, dv).
auto n = thread_position_in_grid.z;
auto req_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
auto dv_idx = thread_position_in_grid.y;
auto dk_idx = thread_position_in_threadgroup.x;
constexpr int n_per_t = Dk / 32;

if (req_idx >= (uint)num_requests || dv_idx >= (uint)Dv) return;

// State: [MAX_SEQS, Hv, Dv, Dk]
auto slot_idx = (uint)slot_mapping[req_idx];
int state_base = ((slot_idx * Hv + hv_idx) * Dv + dv_idx) * Dk;
int state_out_base = ((req_idx * Hv + hv_idx) * Dv + dv_idx) * Dk;

// Load state into registers
float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
    int s_idx = n_per_t * dk_idx + i;
    state[i] = static_cast<float>(state_in[state_base + s_idx]);
}

auto q_ = q + req_idx * Hk * Dk + hk_idx * Dk;
auto k_ = k + req_idx * Hk * Dk + hk_idx * Dk;
auto v_ = v + req_idx * Hv * Dv + hv_idx * Dv;
auto g_ = g + req_idx * Hv;
auto beta_ = beta + req_idx * Hv;

float g_val = static_cast<float>(g_[hv_idx]);

// Decay + k . state dot product
float kv_mem = 0.0f;
for (int i = 0; i < n_per_t; ++i) {
    int s_idx = n_per_t * dk_idx + i;
    state[i] *= g_val;
    kv_mem += state[i] * static_cast<float>(k_[s_idx]);
}
kv_mem = simd_sum(kv_mem);

// Delta update
float delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
              * static_cast<float>(beta_[hv_idx]);

// State update + q . state output
float out = 0.0f;
for (int i = 0; i < n_per_t; ++i) {
    int s_idx = n_per_t * dk_idx + i;
    state[i] += static_cast<float>(k_[s_idx]) * delta;
    out += state[i] * static_cast<float>(q_[s_idx]);
}
out = simd_sum(out);

if (thread_index_in_simdgroup == 0) {
    y[req_idx * Hv * Dv + hv_idx * Dv + dv_idx] = static_cast<T>(out);
}

// Write compact state updates for active requests only.
for (int i = 0; i < n_per_t; ++i) {
    int s_idx = n_per_t * dk_idx + i;
    state_out[state_out_base + s_idx] = static_cast<StT>(state[i]);
}
