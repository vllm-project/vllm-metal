// Template params: T (output/input dtype), StT (state dtype), CONV_DIM, KERNEL_SIZE
// Inputs: input, conv_state_in, weights, cu_seqlens, slot_mapping, num_requests, total_tokens
// Outputs: output, conv_state_out
// Grid: ((total_tokens + num_requests * (KERNEL_SIZE - 1)) * CONV_DIM, 1, 1)
//
// Prefill-containing depthwise conv counterpart to gdn_conv1d_silu_decode.metal.
// The first total_tokens * CONV_DIM threads compute SiLU(depthwise conv) for
// packed prefill tokens.  The tail threads emit compact updated conv state for
// active request slots.

uint tid = thread_position_in_grid.x;
constexpr int state_len = KERNEL_SIZE - 1;
uint output_threads = (uint)total_tokens * CONV_DIM;

if (tid < output_threads) {
    uint token_idx = tid / CONV_DIM;
    uint c = tid % CONV_DIM;

    int req_idx = 0;
    for (int i = 0; i < num_requests; ++i) {
        if (token_idx >= (uint)cu_seqlens[i] && token_idx < (uint)cu_seqlens[i + 1]) {
            req_idx = i;
            break;
        }
    }

    int seq_start = cu_seqlens[req_idx];
    int local_t = (int)token_idx - seq_start;
    uint slot = (uint)slot_mapping[req_idx];
    int state_base = slot * state_len * CONV_DIM + c;
    int weight_base = c * KERNEL_SIZE;

    float acc = 0.0f;
    for (int j = 0; j < KERNEL_SIZE; ++j) {
        int conv_pos = local_t + j;
        float val;
        if (conv_pos < state_len) {
            val = static_cast<float>(conv_state_in[state_base + conv_pos * CONV_DIM]);
        } else {
            int input_t = seq_start + conv_pos - state_len;
            val = static_cast<float>(input[input_t * CONV_DIM + c]);
        }
        acc += val * static_cast<float>(weights[weight_base + j]);
    }

    output[token_idx * CONV_DIM + c] = static_cast<T>(acc / (1.0f + metal::exp(-acc)));
    return;
}

uint state_tid = tid - output_threads;
uint state_linear = state_tid / CONV_DIM;
uint c = state_tid % CONV_DIM;
uint req_idx = state_linear / state_len;
uint state_pos = state_linear % state_len;
if (req_idx >= (uint)num_requests) return;

int seq_start = cu_seqlens[req_idx];
int seq_end = cu_seqlens[req_idx + 1];
int seq_len = seq_end - seq_start;
uint slot = (uint)slot_mapping[req_idx];
int state_base = slot * state_len * CONV_DIM + c;
int state_out_base = (req_idx * state_len + state_pos) * CONV_DIM + c;

int conv_pos = seq_len + (int)state_pos;
if (conv_pos < state_len) {
    conv_state_out[state_out_base] = conv_state_in[state_base + conv_pos * CONV_DIM];
} else {
    int input_t = seq_start + conv_pos - state_len;
    conv_state_out[state_out_base] = static_cast<StT>(input[input_t * CONV_DIM + c]);
}
