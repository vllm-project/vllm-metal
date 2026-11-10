// SPDX-License-Identifier: Apache-2.0 AND MIT
// Portions adapted from mlx-lm/models/gated_delta.py (mlx-lm#1559).
// Copyright © 2025 Apple Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Template params: T (compute/output dtype), StT (state dtype), Dk, Dv, Hk,
// Hv, Packed (eight value rows per SIMD-group instead of one).
// Inputs: q, k, v, g, beta, state_in, cu_seqlens, slot_mapping, num_requests
// Outputs: y, state_out
// Generic grid: (32, Dv, num_requests * Hv), threadgroup: (32, 4, 1)
// Packed grid: (32, Dv / 8, num_requests * Hv), threadgroup: (32, 2, 1)
//
// Both layouts keep state in registers across a ragged sequence and emit compact
// updates. Packed retains the generic four-element partial sums, then performs
// the first three reduction levels locally and the last two across four lanes.
// Generic retains simd_sum; its compiler lowering need not match the explicit
// packed tree bit-for-bit.

constexpr int lanes_per_row = Packed ? 4 : 32;
constexpr int rows_per_simdgroup = 32 / lanes_per_row;
constexpr int values_per_lane = Dk / lanes_per_row;
constexpr int partials_per_lane = Packed ? 8 : 1;
constexpr int elements_per_partial = values_per_lane / partials_per_lane;
static_assert(!Packed || (Dk == 128 && Dv % 8 == 0),
              "packed prefill requires Dk == 128 and Dv divisible by eight");

auto reduce_partials = [](thread const float* part) {
    if constexpr (Packed) {
        float sum = ((part[0] + part[1]) + (part[2] + part[3])) +
                    ((part[4] + part[5]) + (part[6] + part[7]));
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        return sum;
    } else {
        return simd_sum(part[0]);
    }
};

auto n = thread_position_in_grid.z;
auto req_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
auto lane = thread_index_in_simdgroup;
auto lane_in_row = lane % lanes_per_row;
auto dv_idx = thread_position_in_grid.y * rows_per_simdgroup
    + lane / lanes_per_row;
auto dk_offset = lane_in_row * values_per_lane;

if (req_idx >= (uint)num_requests || dv_idx >= (uint)Dv) return;

uint slot_idx = (uint)slot_mapping[req_idx];
int64_t state_base = int64_t(slot_idx) * state_in_strides[0]
    + (hv_idx * Dv + dv_idx) * Dk + dk_offset;
int state_out_base =
    ((req_idx * Hv + hv_idx) * Dv + dv_idx) * Dk + dk_offset;

int seq_start = cu_seqlens[req_idx];
int seq_end = cu_seqlens[req_idx + 1];
int seq_len = seq_end - seq_start;

float state[values_per_lane];
for (int i = 0; i < values_per_lane; ++i) {
    state[i] = static_cast<float>(state_in[state_base + i]);
}

auto q_ = q + (seq_start * Hk + hk_idx) * Dk + dk_offset;
auto k_ = k + (seq_start * Hk + hk_idx) * Dk + dk_offset;
auto v_ = v + seq_start * Hv * Dv + hv_idx * Dv;
auto g_ = g + seq_start * Hv;
auto beta_ = beta + seq_start * Hv;
auto y_ = y + seq_start * Hv * Dv + hv_idx * Dv;

for (int t = 0; t < seq_len; ++t) {
    float g_val = static_cast<float>(g_[hv_idx]);
    float part[partials_per_lane];
    for (int pb = 0; pb < partials_per_lane; ++pb) {
        float acc = 0.0f;
        for (int i = 0; i < elements_per_partial; ++i) {
            int e = pb * elements_per_partial + i;
            state[e] = state[e] * g_val;
            acc += state[e] * static_cast<float>(k_[e]);
        }
        part[pb] = acc;
    }
    float kv_mem = reduce_partials(part);
    float delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
                  * static_cast<float>(beta_[hv_idx]);

    for (int pb = 0; pb < partials_per_lane; ++pb) {
        float acc = 0.0f;
        for (int i = 0; i < elements_per_partial; ++i) {
            int e = pb * elements_per_partial + i;
            state[e] = state[e] + static_cast<float>(k_[e]) * delta;
            acc += state[e] * static_cast<float>(q_[e]);
        }
        part[pb] = acc;
    }
    float out = reduce_partials(part);
    if (lane_in_row == 0) {
        y_[dv_idx] = static_cast<T>(out);
    }

    q_ += Hk * Dk;
    k_ += Hk * Dk;
    v_ += Hv * Dv;
    y_ += Hv * Dv;
    g_ += Hv;
    beta_ += Hv;
}

for (int i = 0; i < values_per_lane; ++i) {
    state_out[state_out_base + i] = static_cast<StT>(state[i]);
}
