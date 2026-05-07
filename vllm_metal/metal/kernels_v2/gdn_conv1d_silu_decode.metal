// Template params: T (dtype), CONV_DIM, KERNEL_SIZE
// Inputs: input, conv_state_in, weights, slot_mapping, num_requests
// Outputs: output, conv_state_out
// Grid: (num_requests * CONV_DIM, 1, 1) — one thread per (request, channel)

uint tid = thread_position_in_grid.x;
uint req_idx = tid / CONV_DIM;
uint c = tid % CONV_DIM;

if (req_idx >= (uint)num_requests) return;

constexpr int state_len = KERNEL_SIZE - 1;
uint slot = (uint)slot_mapping[req_idx];
int state_base = slot * state_len * CONV_DIM + c;
int state_out_base = req_idx * state_len * CONV_DIM + c;
int weight_base = c * KERNEL_SIZE;

// Active request: compute conv + SiLU and emit compact state update.
float inp = static_cast<float>(input[req_idx * CONV_DIM + c]);

float acc = 0.0f;
for (int t = 0; t < state_len; ++t) {
    acc += static_cast<float>(conv_state_in[state_base + t * CONV_DIM])
         * static_cast<float>(weights[weight_base + t]);
}
acc += inp * static_cast<float>(weights[weight_base + state_len]);

output[req_idx * CONV_DIM + c] = static_cast<T>(acc / (1.0f + metal::exp(-acc)));

for (int t = 0; t < state_len - 1; ++t) {
    conv_state_out[state_out_base + t * CONV_DIM] =
        conv_state_in[state_base + (t + 1) * CONV_DIM];
}
conv_state_out[state_out_base + (state_len - 1) * CONV_DIM] = static_cast<T>(inp);
