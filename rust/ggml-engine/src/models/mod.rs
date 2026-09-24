// SPDX-License-Identifier: Apache-2.0
//! Model definitions (ggml graphs) and shared building blocks.

pub mod gemma4;
pub mod qwen35;

use crate::backend::Backend;
use crate::batch::MicroBatch;
use crate::cache::Cache;
use crate::ffi::*;
use crate::graph::{ne, Graph, BF16, F16, F32, T};
use crate::st::{DType, SafeTensors};
use anyhow::{bail, Result};
use serde_json::Value;
use std::path::Path;
use std::ptr;

#[derive(Clone, Debug)]
pub struct KvLayerSpec {
    /// Decoder layer index that owns (writes) this cache.
    pub layer: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub sliding_window: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct StateLayerSpec {
    pub layer: usize,
    /// F32 elements of convolution state per sequence.
    pub conv_row: usize,
    /// F32 elements of recurrent (SSM) state per sequence.
    pub ssm_row: usize,
}

#[derive(Clone, Debug, Default)]
pub struct ModelInfo {
    pub arch: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub max_position_embeddings: usize,
    pub kv_layers: Vec<KvLayerSpec>,
    pub state_layers: Vec<StateLayerSpec>,
    pub weight_bytes: usize,
}

pub trait Model: Send {
    fn info(&self) -> &ModelInfo;
    /// Build the forward graph for `mb`; returns logits `[vocab, n_rows]`
    /// for `rows` (token indices within the micro-batch).
    fn build(&self, g: &mut Graph, mb: &MicroBatch, cache: &Cache, rows: &[i32]) -> Result<T>;
    /// Graph node budget per layer and per (layer, group).
    fn nodes_per_layer(&self) -> (usize, usize) {
        (64, 32)
    }
}

pub fn load(dir: &Path, backend: &Backend) -> Result<Box<dyn Model>> {
    let cfg: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let model_type = cfg["model_type"].as_str().unwrap_or("");
    let st = SafeTensors::open_dir(dir)?;
    match model_type {
        "qwen3_5" | "qwen3_5_text" => Ok(Box::new(qwen35::Qwen35::load(&cfg, st, backend)?)),
        "gemma4" | "gemma4_text" => Ok(Box::new(gemma4::Gemma4::load(&cfg, st, backend)?)),
        other => {
            bail!("ggml backend: unsupported model_type '{other}' (supported: qwen3_5, gemma4)")
        }
    }
}

/// The language-model sub-config (`text_config` for multimodal checkpoints).
pub fn text_config(cfg: &Value) -> &Value {
    if cfg.get("text_config").is_some_and(|v| v.is_object()) {
        &cfg["text_config"]
    } else {
        cfg
    }
}

pub fn cfg_usize(c: &Value, key: &str) -> Result<usize> {
    c[key]
        .as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| anyhow::anyhow!("config missing '{key}'"))
}

pub fn cfg_f32(c: &Value, key: &str, default: f32) -> f32 {
    c[key].as_f64().map(|v| v as f32).unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

enum Source {
    Raw(String),
    F32(Vec<f32>),
}

/// Declares weight tensors, then allocates them in one device buffer.
pub struct WeightBuilder {
    pub st: SafeTensors,
    pub prefix: String,
    ctx: *mut ggml_context,
    pending: Vec<(T, Source)>,
}

pub struct Weights {
    ctx: *mut ggml_context,
    buf: ggml_backend_buffer_t,
    pub bytes: usize,
}

unsafe impl Send for Weights {}

impl Drop for Weights {
    fn drop(&mut self) {
        unsafe {
            ggml_backend_buffer_free(self.buf);
            ggml_free(self.ctx);
        }
    }
}

impl WeightBuilder {
    pub fn new(st: SafeTensors, prefix: &str) -> Self {
        unsafe {
            let params = ggml_init_params {
                mem_size: ggml_tensor_overhead() * 8192,
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            };
            Self {
                st,
                prefix: prefix.to_string(),
                ctx: ggml_init(params),
                pending: Vec::new(),
            }
        }
    }

    pub fn name(&self, key: &str) -> String {
        format!("{}{}", self.prefix, key)
    }

    pub fn has(&self, key: &str) -> bool {
        self.st.contains(&self.name(key))
    }

    /// A linear weight `[out, in]` kept in its checkpoint dtype, as ggml `[in, out]`.
    pub fn matrix(&mut self, key: &str) -> Result<T> {
        self.matrix_full_name(&self.name(key))
    }

    pub fn matrix_full_name(&mut self, name: &str) -> Result<T> {
        let info = self.st.info(name)?.clone();
        if info.shape.len() != 2 {
            bail!("{name}: expected 2-D weight, got {:?}", info.shape);
        }
        let ty = match info.dtype {
            DType::BF16 => BF16,
            DType::F16 => F16,
            DType::F32 => F32,
        };
        let t =
            unsafe { ggml_new_tensor_2d(self.ctx, ty, info.shape[1] as i64, info.shape[0] as i64) };
        self.pending.push((t, Source::Raw(name.to_string())));
        Ok(t)
    }

    /// A tensor converted to F32 with a host-side transform, stored with shape `ne`.
    pub fn f32_with(&mut self, key: &str, ne: &[i64], f: impl Fn(f32) -> f32) -> Result<T> {
        let data: Vec<f32> = self.st.f32(&self.name(key))?.into_iter().map(f).collect();
        self.f32_data(ne, data)
    }

    pub fn f32(&mut self, key: &str, ne: &[i64]) -> Result<T> {
        self.f32_with(key, ne, |x| x)
    }

    pub fn f32_data(&mut self, ne: &[i64], data: Vec<f32>) -> Result<T> {
        let n: i64 = ne.iter().product();
        if n as usize != data.len() {
            bail!("f32 tensor size mismatch: {:?} vs {}", ne, data.len());
        }
        let t = unsafe {
            match ne.len() {
                1 => ggml_new_tensor_1d(self.ctx, F32, ne[0]),
                2 => ggml_new_tensor_2d(self.ctx, F32, ne[0], ne[1]),
                3 => ggml_new_tensor_3d(self.ctx, F32, ne[0], ne[1], ne[2]),
                _ => ggml_new_tensor_4d(self.ctx, F32, ne[0], ne[1], ne[2], ne[3]),
            }
        };
        self.pending.push((t, Source::F32(data)));
        Ok(t)
    }

    /// Read a scalar weight (e.g. Gemma's per-layer `layer_scalar`).
    pub fn scalar(&self, key: &str) -> Result<f32> {
        Ok(self.st.f32(&self.name(key))?[0])
    }

    /// Allocate on the device and upload. Returns the store and the safetensors
    /// handle (still needed by models that read host-side tables).
    pub fn finish(self, backend: &Backend) -> Result<(Weights, SafeTensors)> {
        let buf = backend.alloc_ctx(self.ctx, true)?;
        for (t, src) in &self.pending {
            unsafe {
                match src {
                    Source::Raw(name) => {
                        let (_, raw) = self.st.bytes(name)?;
                        if raw.len() != ggml_nbytes(*t) {
                            bail!("{name}: size mismatch");
                        }
                        ggml_backend_tensor_set(*t, raw.as_ptr() as *const _, 0, raw.len());
                    }
                    Source::F32(data) => {
                        ggml_backend_tensor_set(*t, data.as_ptr() as *const _, 0, data.len() * 4);
                    }
                }
            }
        }
        let bytes = unsafe { ggml_backend_buffer_get_size(buf) };
        Ok((
            Weights {
                ctx: self.ctx,
                buf,
                bytes,
            },
            self.st,
        ))
    }
}

// ---------------------------------------------------------------------------
// Paged attention over ggml
// ---------------------------------------------------------------------------

/// Per-group gather indices and masks for one attention "kind" (causal or a
/// particular sliding window). Shared by every layer of that kind.
pub struct AttnInputs {
    pub groups: Vec<(T, T, usize)>, // (gather idx I32, mask F16, n_kv)
}

impl AttnInputs {
    pub fn new(g: &mut Graph, mb: &MicroBatch, window: Option<usize>) -> Result<Self> {
        let mut groups = Vec::with_capacity(mb.groups.len());
        for grp in &mb.groups {
            let meta = mb.attn_meta(grp, window)?;
            let idx = g.input_i32(&meta.gather);
            let mask = g.input_f16_bits(
                &[meta.n_kv as i64, grp.q_len as i64, 1, grp.n_seqs as i64],
                &meta.mask,
            );
            groups.push((idx, mask, meta.n_kv));
        }
        Ok(Self { groups })
    }
}

/// Write this step's K/V (`[head_dim, n_kv_heads, n_tokens]`) into the paged
/// cache; returns the cache views that carry the write dependency.
pub fn write_kv(g: &mut Graph, kc: T, vc: T, k: T, v: T, slots: T) -> (T, T) {
    let n_tok = ne(k, 2);
    let row = ne(k, 0) * ne(k, 1);
    let k2 = g.reshape_2d(g.cont(k), row, n_tok);
    let v2 = g.reshape_2d(g.cont(v), row, n_tok);
    let kw = g.set_rows(kc, k2, slots);
    let vw = g.set_rows(vc, v2, slots);
    (kw, vw)
}

/// Gather each group's context K/V from the cache:
/// `[head_dim, n_kv, n_kv_heads, n_seqs]` (F32) per group.
pub fn gather_kv(
    g: &mut Graph,
    mb: &MicroBatch,
    inputs: &AttnInputs,
    kc: T,
    vc: T,
    head_dim: i64,
    n_kv_heads: i64,
) -> Vec<(T, T)> {
    mb.groups
        .iter()
        .zip(&inputs.groups)
        .map(|(grp, &(idx, _, n_kv))| {
            let f = |c: T| {
                let r = g.get_rows(c, idx);
                let r = g.reshape_4d(r, head_dim, n_kv_heads, n_kv as i64, grp.n_seqs as i64);
                g.permute(r, 0, 2, 1, 3)
            };
            (f(kc), f(vc))
        })
        .collect()
}

/// Attention for all groups. `q` is `[head_dim, n_head, n_tokens]`;
/// returns `[head_dim * n_head, n_tokens]`.
pub fn attend(
    g: &mut Graph,
    mb: &MicroBatch,
    inputs: &AttnInputs,
    kv: &[(T, T)],
    q: T,
    scale: f32,
) -> T {
    let d = ne(q, 0);
    let h = ne(q, 1);
    let q = g.cont(q);
    let nb1 = crate::graph::nb(q, 1);
    let nb2 = crate::graph::nb(q, 2);
    let mut out: Option<T> = None;
    for (gi, grp) in mb.groups.iter().enumerate() {
        let ql = grp.q_len as i64;
        let s = grp.n_seqs as i64;
        let qg = g.view_4d(
            q,
            d,
            h,
            ql,
            s,
            nb1,
            nb2,
            nb2 * ql as usize,
            nb2 * grp.tok_start,
        );
        let qg = g.permute(qg, 0, 2, 1, 3); // [d, q_len, h, n_seqs]
        let (k, v) = kv[gi];
        let mask = inputs.groups[gi].1;
        let o = g.flash_attn(qg, k, v, mask, scale); // [d, h, q_len, n_seqs]
        let o = g.reshape_2d(o, d * h, ql * s);
        out = Some(match out {
            None => o,
            Some(prev) => g.concat(prev, o, 1),
        });
    }
    out.expect("empty micro-batch")
}
