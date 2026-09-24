// SPDX-License-Identifier: Apache-2.0
//! Device-resident caches: paged K/V per attention layer and per-sequence
//! recurrent state pools (conv + SSM) per linear-attention layer.

use crate::backend::Backend;
use crate::ffi::*;
use crate::graph::{F16, F32, T};
use crate::models::ModelInfo;
use anyhow::Result;
use std::ptr;

pub struct Cache {
    ctx: *mut ggml_context,
    buf: ggml_backend_buffer_t,
    /// `[head_dim * n_kv_heads, num_blocks * block_size]` F16 per KV layer.
    pub k: Vec<T>,
    pub v: Vec<T>,
    /// `[conv_row, n_state_slots]` F32 per state layer.
    pub conv: Vec<T>,
    /// `[ssm_row, n_state_slots]` F32 per state layer.
    pub ssm: Vec<T>,
    pub num_blocks: usize,
    pub block_size: usize,
    pub n_state_slots: usize,
}

unsafe impl Send for Cache {}

impl Cache {
    pub fn new(
        backend: &Backend,
        info: &ModelInfo,
        num_blocks: usize,
        block_size: usize,
        n_state_slots: usize,
    ) -> Result<Self> {
        let n_tensors = 2 * (info.kv_layers.len() + info.state_layers.len()) + 1;
        unsafe {
            let params = ggml_init_params {
                mem_size: ggml_tensor_overhead() * n_tensors,
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ggml_init(params);
            let n_slots = (num_blocks * block_size) as i64;
            let mut k = Vec::new();
            let mut v = Vec::new();
            for l in &info.kv_layers {
                let row = (l.head_dim * l.n_kv_heads) as i64;
                k.push(ggml_new_tensor_2d(ctx, F16, row, n_slots));
                v.push(ggml_new_tensor_2d(ctx, F16, row, n_slots));
            }
            let mut conv = Vec::new();
            let mut ssm = Vec::new();
            for s in &info.state_layers {
                conv.push(ggml_new_tensor_2d(
                    ctx,
                    F32,
                    s.conv_row as i64,
                    n_state_slots as i64,
                ));
                ssm.push(ggml_new_tensor_2d(
                    ctx,
                    F32,
                    s.ssm_row as i64,
                    n_state_slots as i64,
                ));
            }
            let buf = if k.is_empty() && conv.is_empty() {
                ptr::null_mut()
            } else {
                let b = backend.alloc_ctx(ctx, false)?;
                ggml_backend_buffer_clear(b, 0);
                b
            };
            Ok(Self {
                ctx,
                buf,
                k,
                v,
                conv,
                ssm,
                num_blocks,
                block_size,
                n_state_slots,
            })
        }
    }

    pub fn bytes(&self) -> usize {
        if self.buf.is_null() {
            0
        } else {
            unsafe { ggml_backend_buffer_get_size(self.buf) }
        }
    }

    /// Zero the recurrent state of `slot` in every state layer.
    pub fn reset_state_slot(&self, slot: usize) {
        unsafe {
            for &t in self.conv.iter().chain(self.ssm.iter()) {
                let row = (*t).nb[1];
                ggml_backend_tensor_memset(t, 0, slot * row, row);
            }
        }
    }

    /// Bytes of KV cache per token (all layers).
    pub fn kv_bytes_per_token(info: &ModelInfo) -> usize {
        info.kv_layers
            .iter()
            .map(|l| 2 * 2 * l.head_dim * l.n_kv_heads)
            .sum()
    }

    /// Bytes of recurrent state per sequence slot (all layers).
    pub fn state_bytes_per_slot(info: &ModelInfo) -> usize {
        info.state_layers
            .iter()
            .map(|s| 4 * (s.conv_row + s.ssm_row))
            .sum()
    }
}

impl Drop for Cache {
    fn drop(&mut self) {
        unsafe {
            if !self.buf.is_null() {
                ggml_backend_buffer_free(self.buf);
            }
            ggml_free(self.ctx);
        }
    }
}
