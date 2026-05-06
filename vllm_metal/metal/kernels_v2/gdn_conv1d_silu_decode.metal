// Template params: T (dtype), CONV_DIM, KERNEL_SIZE, MAX_SEQS
// Inputs: input, conv_state_in, weights, slot_mapping, num_requests
// Outputs: output, conv_state_out
// Grid: (MAX_SEQS * CONV_DIM, 1, 1) — one thread per (slot, channel)

uint tid = thread_position_in_grid.x;
uint slot = tid / CONV_DIM;
uint c = tid % CONV_DIM;

if (slot >= MAX_SEQS) return;

constexpr int state_len = KERNEL_SIZE - 1;
int state_base = slot * state_len * CONV_DIM + c;
int weight_base = c * KERNEL_SIZE;

// Find if this slot has an active request
int req_idx = -1;
for (uint r = 0; r < (uint)num_requests; ++r) {
    if (slot_mapping[r] == (int)slot) {
        req_idx = r;
        break;
    }
}

if (req_idx >= 0) {
    // Active slot: compute conv + SiLU and shift state
    float inp = static_cast<float>(input[req_idx * CONV_DIM + c]);

    float acc = 0.0f;
    for (int t = 0; t < state_len; ++t) {
        acc += static_cast<float>(conv_state_in[state_base + t * CONV_DIM])
             * static_cast<float>(weights[weight_base + t]);
    }
    acc += inp * static_cast<float>(weights[weight_base + state_len]);

    output[req_idx * CONV_DIM + c] = static_cast<T>(acc / (1.0f + metal::exp(-acc)));

    for (int t = 0; t < state_len - 1; ++t) {
        conv_state_out[state_base + t * CONV_DIM] =
            conv_state_in[state_base + (t + 1) * CONV_DIM];
    }
    conv_state_out[state_base + (state_len - 1) * CONV_DIM] = static_cast<T>(inp);
} else {
    // Inactive slot: identity copy
    for (int t = 0; t < state_len; ++t) {
        conv_state_out[state_base + t * CONV_DIM] =
            conv_state_in[state_base + t * CONV_DIM];
    }
}
