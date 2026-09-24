// SPDX-License-Identifier: Apache-2.0
//! Thin wrapper over a no-alloc ggml context used to build one forward graph.
//!
//! All tensors are raw `*mut ggml_tensor`; the context owns their metadata
//! and is freed when the builder is dropped. Input tensors are recorded with
//! their host data and uploaded after the scheduler allocates the graph.

use crate::ffi::*;
use std::ptr;

pub type T = *mut ggml_tensor;

pub const F32: ggml_type = ggml_type_GGML_TYPE_F32;
pub const F16: ggml_type = ggml_type_GGML_TYPE_F16;
pub const BF16: ggml_type = ggml_type_GGML_TYPE_BF16;
pub const I32: ggml_type = ggml_type_GGML_TYPE_I32;
pub const I64: ggml_type = ggml_type_GGML_TYPE_I64;

pub const ROPE_NEOX: i32 = GGML_ROPE_TYPE_NEOX as i32;

pub struct Graph {
    pub ctx: *mut ggml_context,
    pub gf: *mut ggml_cgraph,
    uploads: Vec<(T, Vec<u8>)>,
}

impl Graph {
    pub fn new(max_nodes: usize) -> Self {
        unsafe {
            let mem = ggml_tensor_overhead() * max_nodes * 2
                + ggml_graph_overhead_custom(max_nodes, false);
            let params = ggml_init_params {
                mem_size: mem,
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ggml_init(params);
            assert!(!ctx.is_null(), "ggml_init failed");
            let gf = ggml_new_graph_custom(ctx, max_nodes, false);
            Self {
                ctx,
                gf,
                uploads: Vec::new(),
            }
        }
    }

    pub fn expand(&mut self, t: T) {
        unsafe { ggml_build_forward_expand(self.gf, t) }
    }

    pub fn n_nodes(&self) -> i32 {
        unsafe { ggml_graph_n_nodes(self.gf) }
    }

    // ---- inputs -------------------------------------------------------

    fn input(&mut self, ty: ggml_type, ne: &[i64], bytes: Vec<u8>) -> T {
        unsafe {
            let t = match ne.len() {
                1 => ggml_new_tensor_1d(self.ctx, ty, ne[0]),
                2 => ggml_new_tensor_2d(self.ctx, ty, ne[0], ne[1]),
                3 => ggml_new_tensor_3d(self.ctx, ty, ne[0], ne[1], ne[2]),
                _ => ggml_new_tensor_4d(self.ctx, ty, ne[0], ne[1], ne[2], ne[3]),
            };
            ggml_set_input(t);
            debug_assert_eq!(ggml_nbytes(t), bytes.len());
            self.uploads.push((t, bytes));
            t
        }
    }

    pub fn input_i32(&mut self, data: &[i32]) -> T {
        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.input(I32, &[data.len() as i64], bytes)
    }

    pub fn input_i64(&mut self, data: &[i64]) -> T {
        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.input(I64, &[data.len() as i64], bytes)
    }

    pub fn input_f32(&mut self, ne: &[i64], data: &[f32]) -> T {
        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.input(F32, ne, bytes)
    }

    pub fn input_f16_bits(&mut self, ne: &[i64], data: &[u16]) -> T {
        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.input(F16, ne, bytes)
    }

    /// Upload recorded input data. Call after the graph has been allocated.
    pub fn upload_inputs(&mut self) {
        for (t, data) in self.uploads.drain(..) {
            unsafe {
                ggml_backend_tensor_set(t, data.as_ptr() as *const _, 0, data.len());
            }
        }
    }

    // ---- ops ----------------------------------------------------------

    pub fn mul_mat(&self, w: T, x: T) -> T {
        unsafe { ggml_mul_mat(self.ctx, w, x) }
    }
    pub fn add(&self, a: T, b: T) -> T {
        unsafe { ggml_add(self.ctx, a, b) }
    }
    pub fn mul(&self, a: T, b: T) -> T {
        unsafe { ggml_mul(self.ctx, a, b) }
    }
    pub fn scale(&self, a: T, s: f32) -> T {
        unsafe { ggml_scale(self.ctx, a, s) }
    }
    pub fn rms_norm(&self, a: T, eps: f32) -> T {
        unsafe { ggml_rms_norm(self.ctx, a, eps) }
    }
    /// rms_norm(a) * w (w broadcast along rows).
    pub fn rms_norm_w(&self, a: T, w: T, eps: f32) -> T {
        self.mul(self.rms_norm(a, eps), w)
    }
    /// x / sqrt(sum(x^2) + eps) along dim 0 (HF-style l2norm).
    pub fn l2_norm(&self, a: T, eps: f32) -> T {
        let n = unsafe { (*a).ne[0] } as f32;
        self.scale(self.rms_norm(a, eps / n), 1.0 / n.sqrt())
    }
    pub fn silu(&self, a: T) -> T {
        unsafe { ggml_silu(self.ctx, a) }
    }
    pub fn sigmoid(&self, a: T) -> T {
        unsafe { ggml_sigmoid(self.ctx, a) }
    }
    pub fn softplus(&self, a: T) -> T {
        unsafe { ggml_softplus(self.ctx, a) }
    }
    pub fn gelu(&self, a: T) -> T {
        unsafe { ggml_gelu(self.ctx, a) }
    }
    pub fn tanh(&self, a: T) -> T {
        unsafe { ggml_tanh(self.ctx, a) }
    }
    pub fn swiglu_split(&self, gate: T, up: T) -> T {
        unsafe { ggml_swiglu_split(self.ctx, gate, up) }
    }
    pub fn geglu_split(&self, gate: T, up: T) -> T {
        unsafe { ggml_geglu_split(self.ctx, gate, up) }
    }
    pub fn cont(&self, a: T) -> T {
        unsafe { ggml_cont(self.ctx, a) }
    }
    pub fn transpose(&self, a: T) -> T {
        unsafe { ggml_transpose(self.ctx, a) }
    }
    pub fn permute(&self, a: T, a0: i32, a1: i32, a2: i32, a3: i32) -> T {
        unsafe { ggml_permute(self.ctx, a, a0, a1, a2, a3) }
    }
    pub fn concat(&self, a: T, b: T, dim: i32) -> T {
        unsafe { ggml_concat(self.ctx, a, b, dim) }
    }
    pub fn reshape_2d(&self, a: T, n0: i64, n1: i64) -> T {
        unsafe { ggml_reshape_2d(self.ctx, a, n0, n1) }
    }
    pub fn reshape_3d(&self, a: T, n0: i64, n1: i64, n2: i64) -> T {
        unsafe { ggml_reshape_3d(self.ctx, a, n0, n1, n2) }
    }
    pub fn reshape_4d(&self, a: T, n0: i64, n1: i64, n2: i64, n3: i64) -> T {
        unsafe { ggml_reshape_4d(self.ctx, a, n0, n1, n2, n3) }
    }
    pub fn view_2d(&self, a: T, n0: i64, n1: i64, nb1: usize, off: usize) -> T {
        unsafe { ggml_view_2d(self.ctx, a, n0, n1, nb1, off) }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn view_3d(
        &self,
        a: T,
        n0: i64,
        n1: i64,
        n2: i64,
        nb1: usize,
        nb2: usize,
        off: usize,
    ) -> T {
        unsafe { ggml_view_3d(self.ctx, a, n0, n1, n2, nb1, nb2, off) }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn view_4d(
        &self,
        a: T,
        n0: i64,
        n1: i64,
        n2: i64,
        n3: i64,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        off: usize,
    ) -> T {
        unsafe { ggml_view_4d(self.ctx, a, n0, n1, n2, n3, nb1, nb2, nb3, off) }
    }
    pub fn get_rows(&self, a: T, idx: T) -> T {
        unsafe { ggml_get_rows(self.ctx, a, idx) }
    }
    pub fn set_rows(&self, dst: T, src: T, idx: T) -> T {
        unsafe { ggml_set_rows(self.ctx, dst, src, idx) }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rope(&self, a: T, pos: T, freq_factors: T, n_dims: i32, mode: i32, base: f32) -> T {
        unsafe {
            ggml_rope_ext(
                self.ctx,
                a,
                pos,
                freq_factors,
                n_dims,
                mode,
                0,
                base,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            )
        }
    }
    pub fn flash_attn(&self, q: T, k: T, v: T, mask: T, scale: f32) -> T {
        unsafe {
            let r = ggml_flash_attn_ext(self.ctx, q, k, v, mask, scale, 0.0, 0.0);
            ggml_flash_attn_ext_set_prec(r, ggml_prec_GGML_PREC_F32);
            r
        }
    }
    pub fn ssm_conv(&self, sx: T, c: T) -> T {
        unsafe { ggml_ssm_conv(self.ctx, sx, c) }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_net(&self, q: T, k: T, v: T, g: T, beta: T, state: T) -> T {
        unsafe { ggml_gated_delta_net(self.ctx, q, k, v, g, beta, state, 1) }
    }
    pub fn repeat_4d(&self, a: T, n0: i64, n1: i64, n2: i64, n3: i64) -> T {
        unsafe { ggml_repeat_4d(self.ctx, a, n0, n1, n2, n3) }
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { ggml_free(self.ctx) }
    }
}

pub fn ne(t: T, i: usize) -> i64 {
    unsafe { (*t).ne[i] }
}
pub fn nb(t: T, i: usize) -> usize {
    unsafe { (*t).nb[i] }
}
pub fn elsize(t: T) -> usize {
    unsafe { ggml_element_size(t) }
}
