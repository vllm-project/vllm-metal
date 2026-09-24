// SPDX-License-Identifier: Apache-2.0
//! Gemma 4 (dense, E2B/E4B style) text model loaded from HF safetensors.
//!
//! Features: sliding/full attention with different head dims and RoPE
//! (default vs. proportional), Q/K/V RMSNorm with scale 1.0 attention,
//! cross-layer KV sharing, per-layer embeddings (PLE, looked up on the host
//! from the memory-mapped table), GeGLU MLP (double-wide on shared layers),
//! per-layer output scalar and final logit soft-capping.
//!
//! Reference: transformers `modeling_gemma4.py`, mlx-lm `gemma4_text.py`.

use super::*;
use crate::batch::MicroBatch;
use crate::graph::ROPE_NEOX;
use half::bf16;

struct KvProj {
    wk: T,
    wv: T,
    k_norm: T,
    kv_idx: usize,
}

struct Layer {
    sliding: bool,
    head_dim: i64,
    attn_norm: T,
    wq: T,
    q_norm: T,
    kv: Option<KvProj>,
    /// KV cache index this layer attends over (own, or shared source).
    kv_src: usize,
    wo: T,
    post_attn_norm: T,
    pre_ff_norm: T,
    post_ff_norm: T,
    gate: T,
    up: T,
    down: T,
    pl_gate: Option<T>,
    pl_proj: Option<T>,
    pl_norm: Option<T>,
    scalar: f32,
}

pub struct Gemma4 {
    info: ModelInfo,
    _weights: Weights,
    st: SafeTensors,
    prefix: String,
    embed: T,
    lm_head: T,
    norm: T,
    embed_scale: f32,
    layers: Vec<Layer>,
    eps: f32,
    n_head: i64,
    n_kv_head: i64,
    sliding_window: usize,
    rope_sliding: f32,
    rope_full: f32,
    full_freq_factors: Option<T>,
    softcap: Option<f32>,
    // Per-layer embeddings.
    ple_dim: i64,
    pl_model_proj: Option<T>,
    pl_proj_norm: Option<T>,
}

fn bf16_round(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

impl Gemma4 {
    pub fn load(cfg: &Value, st: SafeTensors, backend: &Backend) -> Result<Self> {
        let c = text_config(cfg);
        let hidden = cfg_usize(c, "hidden_size")?;
        let n_layers = cfg_usize(c, "num_hidden_layers")?;
        let n_head = cfg_usize(c, "num_attention_heads")? as i64;
        let n_kv_head = cfg_usize(c, "num_key_value_heads")? as i64;
        let head_dim = cfg_usize(c, "head_dim")? as i64;
        let global_head_dim = c["global_head_dim"]
            .as_u64()
            .map(|v| v as i64)
            .unwrap_or(head_dim);
        let eps = cfg_f32(c, "rms_norm_eps", 1e-6);
        let sliding_window = c["sliding_window"].as_u64().unwrap_or(512) as usize;
        let n_shared = c["num_kv_shared_layers"].as_u64().unwrap_or(0) as usize;
        let ple_dim = c["hidden_size_per_layer_input"].as_u64().unwrap_or(0) as i64;
        if c["enable_moe_block"].as_bool().unwrap_or(false) {
            bail!("gemma4 MoE variants are not supported by the ggml backend yet");
        }
        if c["attention_k_eq_v"].as_bool().unwrap_or(false) {
            bail!("gemma4 attention_k_eq_v variants are not supported by the ggml backend yet");
        }
        if let Some(n) = c["num_global_key_value_heads"].as_u64() {
            if n as i64 != n_kv_head {
                bail!("gemma4: distinct global KV head counts are not supported yet");
            }
        }
        let layer_types: Vec<bool> = c["layer_types"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("gemma4 config missing layer_types"))?
            .iter()
            .map(|v| v.as_str() == Some("sliding_attention"))
            .collect();

        let rp = &c["rope_parameters"];
        let rope_sliding = rp["sliding_attention"]["rope_theta"]
            .as_f64()
            .unwrap_or(10000.0) as f32;
        let rope_full = rp["full_attention"]["rope_theta"].as_f64().unwrap_or(1e6) as f32;
        let full_rope_type = rp["full_attention"]["rope_type"]
            .as_str()
            .unwrap_or("default");
        let full_partial = rp["full_attention"]["partial_rotary_factor"]
            .as_f64()
            .unwrap_or(1.0);
        if rp["full_attention"]["factor"].as_f64().unwrap_or(1.0) != 1.0 {
            bail!("gemma4: rope scaling factor != 1 is not supported yet");
        }

        let prefix = if st.contains("model.language_model.embed_tokens.weight") {
            "model.language_model."
        } else {
            "model."
        };
        let mut wb = WeightBuilder::new(st, prefix);
        let h = hidden as i64;

        let embed = wb.matrix("embed_tokens.weight")?;
        let tied = c["tie_word_embeddings"]
            .as_bool()
            .or(cfg["tie_word_embeddings"].as_bool())
            .unwrap_or(true);
        let lm_head = if !tied && wb.st.contains("lm_head.weight") {
            wb.matrix_full_name("lm_head.weight")?
        } else {
            embed
        };
        let norm = wb.f32("norm.weight", &[h])?;

        // Proportional RoPE: frequencies computed over the full head dim, only
        // the first `partial * head_dim` dims rotate; the rest are identity.
        let full_freq_factors = if full_rope_type == "proportional" && full_partial < 1.0 {
            let half = (global_head_dim / 2) as usize;
            let rot = ((global_head_dim as f64 * full_partial) as usize) / 2;
            let ff: Vec<f32> = (0..half)
                .map(|i| if i < rot { 1.0 } else { 1e30 })
                .collect();
            Some(wb.f32_data(&[half as i64], ff)?)
        } else if full_rope_type != "default" && full_rope_type != "proportional" {
            bail!("gemma4: unsupported full-attention rope_type '{full_rope_type}'");
        } else {
            None
        };

        let (pl_model_proj, pl_proj_norm) = if ple_dim > 0 {
            (
                Some(wb.matrix("per_layer_model_projection.weight")?),
                Some(wb.f32("per_layer_projection_norm.weight", &[ple_dim])?),
            )
        } else {
            (None, None)
        };

        let mut info = ModelInfo {
            arch: "gemma4".into(),
            vocab_size: cfg_usize(c, "vocab_size")?,
            hidden_size: hidden,
            num_layers: n_layers,
            max_position_embeddings: c["max_position_embeddings"].as_u64().unwrap_or(131072)
                as usize,
            ..Default::default()
        };

        let first_shared = n_layers - n_shared;
        let mut last_kv_by_type: [Option<usize>; 2] = [None, None];
        let mut layers = Vec::with_capacity(n_layers);
        if layer_types.len() != n_layers {
            bail!(
                "layer_types has {} entries, expected {n_layers}",
                layer_types.len()
            );
        }
        for (i, &sliding) in layer_types.iter().enumerate() {
            let p = format!("layers.{i}.");
            let a = format!("{p}self_attn.");
            let hd = if sliding { head_dim } else { global_head_dim };
            let has_kv = i < first_shared;
            let (kv, kv_src) = if has_kv {
                info.kv_layers.push(KvLayerSpec {
                    layer: i,
                    n_kv_heads: n_kv_head as usize,
                    head_dim: hd as usize,
                    sliding_window: if sliding { Some(sliding_window) } else { None },
                });
                let idx = info.kv_layers.len() - 1;
                last_kv_by_type[sliding as usize] = Some(idx);
                (
                    Some(KvProj {
                        wk: wb.matrix(&format!("{a}k_proj.weight"))?,
                        wv: wb.matrix(&format!("{a}v_proj.weight"))?,
                        k_norm: wb.f32(&format!("{a}k_norm.weight"), &[hd])?,
                        kv_idx: idx,
                    }),
                    idx,
                )
            } else {
                let src = last_kv_by_type[sliding as usize].ok_or_else(|| {
                    anyhow::anyhow!("gemma4: no KV source layer for shared layer {i}")
                })?;
                (None, src)
            };
            let m = format!("{p}mlp.");
            let gate = wb.matrix(&format!("{m}gate_proj.weight"))?;
            // Double-wide MLPs on KV-shared layers: shapes come from the checkpoint.
            let (pl_gate, pl_proj, pl_norm) = if ple_dim > 0 {
                (
                    Some(wb.matrix(&format!("{p}per_layer_input_gate.weight"))?),
                    Some(wb.matrix(&format!("{p}per_layer_projection.weight"))?),
                    Some(wb.f32(&format!("{p}post_per_layer_input_norm.weight"), &[h])?),
                )
            } else {
                (None, None, None)
            };
            let scalar = if wb.has(&format!("{p}layer_scalar")) {
                wb.scalar(&format!("{p}layer_scalar"))?
            } else {
                1.0
            };
            layers.push(Layer {
                sliding,
                head_dim: hd,
                attn_norm: wb.f32(&format!("{p}input_layernorm.weight"), &[h])?,
                wq: wb.matrix(&format!("{a}q_proj.weight"))?,
                q_norm: wb.f32(&format!("{a}q_norm.weight"), &[hd])?,
                kv,
                kv_src,
                wo: wb.matrix(&format!("{a}o_proj.weight"))?,
                post_attn_norm: wb.f32(&format!("{p}post_attention_layernorm.weight"), &[h])?,
                pre_ff_norm: wb.f32(&format!("{p}pre_feedforward_layernorm.weight"), &[h])?,
                post_ff_norm: wb.f32(&format!("{p}post_feedforward_layernorm.weight"), &[h])?,
                gate,
                up: wb.matrix(&format!("{m}up_proj.weight"))?,
                down: wb.matrix(&format!("{m}down_proj.weight"))?,
                pl_gate,
                pl_proj,
                pl_norm,
                scalar,
            });
        }

        let (weights, st) = wb.finish(backend)?;
        info.weight_bytes = weights.bytes;
        let softcap = c["final_logit_softcapping"].as_f64().map(|v| v as f32);
        Ok(Self {
            info,
            _weights: weights,
            st,
            prefix: prefix.to_string(),
            embed,
            lm_head,
            norm,
            // HF casts sqrt(hidden) to the weight dtype (bf16) before scaling.
            embed_scale: bf16_round((hidden as f32).sqrt()),
            layers,
            eps,
            n_head,
            n_kv_head,
            sliding_window,
            rope_sliding,
            rope_full,
            full_freq_factors,
            softcap,
            ple_dim,
            pl_model_proj,
            pl_proj_norm,
        })
    }

    /// Host-side per-layer-embedding lookup: `[ple_dim * n_layers, n_tokens]` f32.
    fn ple_lookup(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        let name = format!("{}embed_tokens_per_layer.weight", self.prefix);
        let row = (self.ple_dim as usize) * self.layers.len();
        let scale = bf16_round((self.ple_dim as f32).sqrt());
        let mut out = Vec::with_capacity(row * tokens.len());
        for &t in tokens {
            let (info, raw) = self.st.row_bytes(&name, t as usize)?;
            if info.shape[1] != row {
                bail!("embed_tokens_per_layer width {} != {}", info.shape[1], row);
            }
            out.extend(
                crate::st::to_f32(info.dtype, raw)
                    .into_iter()
                    .map(|v| bf16_round(v * scale)),
            );
        }
        Ok(out)
    }
}

// SAFETY: see Qwen35.
unsafe impl Send for Gemma4 {}

impl Model for Gemma4 {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn nodes_per_layer(&self) -> (usize, usize) {
        (72, 24)
    }

    fn build(&self, g: &mut Graph, mb: &MicroBatch, cache: &Cache, rows: &[i32]) -> Result<T> {
        let n = mb.n_tokens() as i64;
        let n_layers = self.layers.len() as i64;
        let tokens = g.input_i32(&mb.tokens);
        let pos = g.input_i32(&mb.positions());
        let slots = g.input_i64(&mb.slot_mapping()?);
        let in_full = AttnInputs::new(g, mb, None)?;
        let in_sliding = AttnInputs::new(g, mb, Some(self.sliding_window))?;

        let mut x = g.scale(g.get_rows(self.embed, tokens), self.embed_scale); // [hidden, n]

        let ple = if self.ple_dim > 0 {
            let pd = self.ple_dim;
            let tok = self.ple_lookup(&mb.tokens)?;
            let tok = g.input_f32(&[pd * n_layers, n], &tok);
            let proj = g.mul_mat(self.pl_model_proj.unwrap(), x);
            let proj = g.scale(proj, 1.0 / (self.info.hidden_size as f32).sqrt());
            let proj = g.reshape_3d(proj, pd, n_layers, n);
            let proj = g.rms_norm_w(proj, self.pl_proj_norm.unwrap(), self.eps);
            let sum = g.add(proj, g.reshape_3d(tok, pd, n_layers, n));
            Some(g.scale(sum, std::f32::consts::FRAC_1_SQRT_2))
        } else {
            None
        };

        let null = ptr::null_mut();
        let mut gathered: Vec<Option<Vec<(T, T)>>> = vec![None; cache.k.len()];
        for (li, l) in self.layers.iter().enumerate() {
            let hd = l.head_dim;
            let inputs = if l.sliding { &in_sliding } else { &in_full };
            let (base, ff, n_dims) = if l.sliding {
                (self.rope_sliding, null, hd as i32)
            } else {
                (
                    self.rope_full,
                    self.full_freq_factors.unwrap_or(null),
                    hd as i32,
                )
            };

            let h = g.rms_norm_w(x, l.attn_norm, self.eps);
            let q = g.reshape_3d(g.mul_mat(l.wq, h), hd, self.n_head, n);
            let q = g.rms_norm_w(q, l.q_norm, self.eps);
            let q = g.rope(q, pos, ff, n_dims, ROPE_NEOX, base);

            if let Some(kv) = &l.kv {
                let k = g.reshape_3d(g.mul_mat(kv.wk, h), hd, self.n_kv_head, n);
                let k = g.rms_norm_w(k, kv.k_norm, self.eps);
                let k = g.rope(k, pos, ff, n_dims, ROPE_NEOX, base);
                let v = g.reshape_3d(g.mul_mat(kv.wv, h), hd, self.n_kv_head, n);
                let v = g.rms_norm(v, self.eps);
                let (kc, vc) = write_kv(g, cache.k[kv.kv_idx], cache.v[kv.kv_idx], k, v, slots);
                gathered[kv.kv_idx] = Some(gather_kv(g, mb, inputs, kc, vc, hd, self.n_kv_head));
            }
            let kvs = gathered[l.kv_src]
                .as_ref()
                .ok_or_else(|| {
                    anyhow::anyhow!("gemma4: KV source {} not built before layer {li}", l.kv_src)
                })?
                .clone();
            let o = attend(g, mb, inputs, &kvs, q, 1.0);
            let o = g.mul_mat(l.wo, o);
            let o = g.rms_norm_w(o, l.post_attn_norm, self.eps);
            x = g.add(x, o);

            let hff = g.rms_norm_w(x, l.pre_ff_norm, self.eps);
            let m = g.geglu_split(g.mul_mat(l.gate, hff), g.mul_mat(l.up, hff));
            let m = g.rms_norm_w(g.mul_mat(l.down, m), l.post_ff_norm, self.eps);
            x = g.add(x, m);

            if let (Some(ple), Some(pg), Some(pp), Some(pn)) =
                (ple, l.pl_gate, l.pl_proj, l.pl_norm)
            {
                let pd = self.ple_dim;
                let pli = g.cont(g.view_2d(
                    ple,
                    pd,
                    n,
                    crate::graph::nb(ple, 2),
                    crate::graph::nb(ple, 1) * li,
                ));
                let gt = g.mul(g.gelu(g.mul_mat(pg, x)), pli);
                let gt = g.rms_norm_w(g.mul_mat(pp, gt), pn, self.eps);
                x = g.add(x, gt);
            }
            if l.scalar != 1.0 {
                x = g.scale(x, l.scalar);
            }
        }

        let rows_t = g.input_i32(rows);
        let x = g.get_rows(x, rows_t);
        let x = g.rms_norm_w(x, self.norm, self.eps);
        let mut logits = g.mul_mat(self.lm_head, x);
        if let Some(cap) = self.softcap {
            logits = g.scale(g.tanh(g.scale(logits, 1.0 / cap)), cap);
        }
        Ok(logits)
    }
}
