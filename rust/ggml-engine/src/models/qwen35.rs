// SPDX-License-Identifier: Apache-2.0
//! Qwen3.5 (dense) text model: hybrid Gated DeltaNet linear attention +
//! gated full attention, loaded from HF safetensors.
//!
//! Reference: transformers `modeling_qwen3_5.py`, mlx-lm `qwen3_5.py`,
//! llama.cpp `models/qwen35.cpp`.

use super::*;
use crate::batch::MicroBatch;
use crate::graph::{nb, ROPE_NEOX};

struct FullAttn {
    wq: T,
    wk: T,
    wv: T,
    wo: T,
    q_norm: T,
    k_norm: T,
    kv_idx: usize,
}

struct LinearAttn {
    wqkv: T,
    wz: T,
    wb: T,
    wa: T,
    conv: T,    // F32 [K, conv_dim]
    dt_bias: T, // F32 [Hv]
    ssm_a: T,   // F32 [Hv] = -exp(A_log)
    norm: T,    // F32 [Dv]
    wout: T,
    st_idx: usize,
}

enum Mixer {
    Full(FullAttn),
    Linear(LinearAttn),
}

struct Layer {
    attn_norm: T,
    post_norm: T,
    mixer: Mixer,
    gate: T,
    up: T,
    down: T,
}

pub struct Qwen35 {
    info: ModelInfo,
    _weights: Weights,
    embed: T,
    lm_head: T,
    norm: T,
    layers: Vec<Layer>,
    eps: f32,
    n_head: i64,
    n_kv_head: i64,
    head_dim: i64,
    n_rot: i32,
    rope_base: f32,
    hk: i64,
    hv: i64,
    dk: i64,
    dv: i64,
    conv_k: i64,
}

impl Qwen35 {
    pub fn load(cfg: &Value, st: SafeTensors, backend: &Backend) -> Result<Self> {
        let c = text_config(cfg);
        let hidden = cfg_usize(c, "hidden_size")?;
        let n_layers = cfg_usize(c, "num_hidden_layers")?;
        let n_head = cfg_usize(c, "num_attention_heads")? as i64;
        let n_kv_head = cfg_usize(c, "num_key_value_heads")? as i64;
        let head_dim = c["head_dim"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(hidden / n_head as usize) as i64;
        let eps = cfg_f32(c, "rms_norm_eps", 1e-6);
        let rope = &c["rope_parameters"];
        let rope_base = rope["rope_theta"]
            .as_f64()
            .or(c["rope_theta"].as_f64())
            .unwrap_or(1e7) as f32;
        let partial = rope["partial_rotary_factor"]
            .as_f64()
            .or(c["partial_rotary_factor"].as_f64())
            .unwrap_or(0.25);
        let n_rot = (head_dim as f64 * partial) as i32;
        let hk = cfg_usize(c, "linear_num_key_heads")? as i64;
        let hv = cfg_usize(c, "linear_num_value_heads")? as i64;
        let dk = cfg_usize(c, "linear_key_head_dim")? as i64;
        let dv = cfg_usize(c, "linear_value_head_dim")? as i64;
        let conv_k = cfg_usize(c, "linear_conv_kernel_dim")? as i64;
        if dk != dv {
            bail!("qwen3_5: linear key/value head dims must match (got {dk} vs {dv})");
        }
        if hv % hk != 0 {
            bail!("qwen3_5: linear value heads must be a multiple of key heads");
        }
        if c["num_experts"].as_u64().unwrap_or(0) > 0 {
            bail!("qwen3_5 MoE variants are not supported by the ggml backend yet");
        }
        let full_interval = c["full_attention_interval"].as_u64().unwrap_or(4) as usize;
        let layer_types: Vec<bool> = match c["layer_types"].as_array() {
            Some(a) => a
                .iter()
                .map(|v| v.as_str() == Some("full_attention"))
                .collect(),
            None => (0..n_layers)
                .map(|i| (i + 1) % full_interval == 0)
                .collect(),
        };
        let conv_dim = 2 * dk * hk + dv * hv;

        let prefix = if st.contains("model.language_model.embed_tokens.weight") {
            "model.language_model."
        } else {
            "model."
        };
        let mut wb = WeightBuilder::new(st, prefix);
        // Qwen3.5 RMSNorm scales by (1 + w).
        let plus1 = |x: f32| x + 1.0;
        let h = hidden as i64;

        let embed = wb.matrix("embed_tokens.weight")?;
        let tied = c["tie_word_embeddings"]
            .as_bool()
            .or(cfg["tie_word_embeddings"].as_bool())
            .unwrap_or(false);
        let lm_head = if !tied && wb.st.contains("lm_head.weight") {
            wb.matrix_full_name("lm_head.weight")?
        } else {
            embed
        };
        let norm = wb.f32_with("norm.weight", &[h], plus1)?;

        let mut info = ModelInfo {
            arch: "qwen3_5".into(),
            vocab_size: cfg_usize(c, "vocab_size")?,
            hidden_size: hidden,
            num_layers: n_layers,
            max_position_embeddings: c["max_position_embeddings"].as_u64().unwrap_or(262144)
                as usize,
            ..Default::default()
        };

        let mut layers = Vec::with_capacity(n_layers);
        if layer_types.len() != n_layers {
            bail!(
                "layer_types has {} entries, expected {n_layers}",
                layer_types.len()
            );
        }
        for (i, &is_full) in layer_types.iter().enumerate() {
            let p = format!("layers.{i}.");
            let attn_norm = wb.f32_with(&format!("{p}input_layernorm.weight"), &[h], plus1)?;
            let post_norm =
                wb.f32_with(&format!("{p}post_attention_layernorm.weight"), &[h], plus1)?;
            let mixer = if is_full {
                let a = format!("{p}self_attn.");
                info.kv_layers.push(KvLayerSpec {
                    layer: i,
                    n_kv_heads: n_kv_head as usize,
                    head_dim: head_dim as usize,
                    sliding_window: None,
                });
                Mixer::Full(FullAttn {
                    wq: wb.matrix(&format!("{a}q_proj.weight"))?,
                    wk: wb.matrix(&format!("{a}k_proj.weight"))?,
                    wv: wb.matrix(&format!("{a}v_proj.weight"))?,
                    wo: wb.matrix(&format!("{a}o_proj.weight"))?,
                    q_norm: wb.f32_with(&format!("{a}q_norm.weight"), &[head_dim], plus1)?,
                    k_norm: wb.f32_with(&format!("{a}k_norm.weight"), &[head_dim], plus1)?,
                    kv_idx: info.kv_layers.len() - 1,
                })
            } else {
                let a = format!("{p}linear_attn.");
                info.state_layers.push(StateLayerSpec {
                    layer: i,
                    conv_row: (conv_dim * (conv_k - 1)) as usize,
                    ssm_row: (dv * dv * hv) as usize,
                });
                // HF conv1d weight [C, 1, K] is ggml [K, C] as-is.
                let conv = wb.f32(&format!("{a}conv1d.weight"), &[conv_k, conv_dim])?;
                Mixer::Linear(LinearAttn {
                    wqkv: wb.matrix(&format!("{a}in_proj_qkv.weight"))?,
                    wz: wb.matrix(&format!("{a}in_proj_z.weight"))?,
                    wb: wb.matrix(&format!("{a}in_proj_b.weight"))?,
                    wa: wb.matrix(&format!("{a}in_proj_a.weight"))?,
                    conv,
                    dt_bias: wb.f32(&format!("{a}dt_bias"), &[hv])?,
                    ssm_a: wb.f32_with(&format!("{a}A_log"), &[hv], |x| -x.exp())?,
                    norm: wb.f32(&format!("{a}norm.weight"), &[dv])?,
                    wout: wb.matrix(&format!("{a}out_proj.weight"))?,
                    st_idx: info.state_layers.len() - 1,
                })
            };
            let m = format!("{p}mlp.");
            layers.push(Layer {
                attn_norm,
                post_norm,
                mixer,
                gate: wb.matrix(&format!("{m}gate_proj.weight"))?,
                up: wb.matrix(&format!("{m}up_proj.weight"))?,
                down: wb.matrix(&format!("{m}down_proj.weight"))?,
            });
        }

        let (weights, _st) = wb.finish(backend)?;
        info.weight_bytes = weights.bytes;
        Ok(Self {
            info,
            _weights: weights,
            embed,
            lm_head,
            norm,
            layers,
            eps,
            n_head,
            n_kv_head,
            head_dim,
            n_rot,
            rope_base,
            hk,
            hv,
            dk,
            dv,
            conv_k,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn full_attn(
        &self,
        g: &mut Graph,
        mb: &MicroBatch,
        cache: &Cache,
        a: &FullAttn,
        x: T,
        pos: T,
        slots: T,
        inputs: &AttnInputs,
    ) -> T {
        let (d, hq, hkv) = (self.head_dim, self.n_head, self.n_kv_head);
        let n = ne(x, 1);
        let qg = g.mul_mat(a.wq, x); // [2*d*hq, n]: per head [q | gate]
        let es = crate::graph::elsize(qg);
        let q = g.view_3d(qg, d, hq, n, 2 * d as usize * es, nb(qg, 1), 0);
        let q = g.rms_norm_w(q, a.q_norm, self.eps);
        let gate = g.view_3d(
            qg,
            d,
            hq,
            n,
            2 * d as usize * es,
            nb(qg, 1),
            d as usize * es,
        );
        let gate = g.reshape_2d(g.cont(gate), d * hq, n);

        let k = g.reshape_3d(g.mul_mat(a.wk, x), d, hkv, n);
        let k = g.rms_norm_w(k, a.k_norm, self.eps);
        let v = g.reshape_3d(g.mul_mat(a.wv, x), d, hkv, n);

        let null = ptr::null_mut();
        let q = g.rope(q, pos, null, self.n_rot, ROPE_NEOX, self.rope_base);
        let k = g.rope(k, pos, null, self.n_rot, ROPE_NEOX, self.rope_base);

        let (kc, vc) = write_kv(g, cache.k[a.kv_idx], cache.v[a.kv_idx], k, v, slots);
        let kv = gather_kv(g, mb, inputs, kc, vc, d, hkv);
        let o = attend(g, mb, inputs, &kv, q, 1.0 / (d as f32).sqrt());
        let o = g.mul(o, g.sigmoid(gate));
        g.mul_mat(a.wo, o)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attn(
        &self,
        g: &mut Graph,
        mb: &MicroBatch,
        cache: &Cache,
        a: &LinearAttn,
        x: T,
        slot_i32: &[T],
        slot_i64: &[T],
    ) -> T {
        let (hk, hv, dk, dv, kc) = (self.hk, self.hv, self.dk, self.dv, self.conv_k);
        let n = ne(x, 1);
        let conv_dim = 2 * dk * hk + dv * hv;
        let qkv = g.mul_mat(a.wqkv, x); // [conv_dim, n]
        let z = g.mul_mat(a.wz, x); // [dv*hv, n]
        let beta = g.sigmoid(g.mul_mat(a.wb, x)); // [hv, n]
        let alpha = g.mul_mat(a.wa, x);
        let gate = g.mul(g.softplus(g.add(alpha, a.dt_bias)), a.ssm_a); // log-decay [hv, n]

        let conv_pool = cache.conv[a.st_idx];
        let ssm_pool = cache.ssm[a.st_idx];
        let mut out: Option<T> = None;
        for (gi, grp) in mb.groups.iter().enumerate() {
            let q = grp.q_len as i64;
            let s = grp.n_seqs as i64;
            let t0 = grp.tok_start;

            // --- causal depthwise conv with carried state ---------------
            let qkv_g = g.view_3d(
                qkv,
                conv_dim,
                q,
                s,
                nb(qkv, 1),
                nb(qkv, 1) * q as usize,
                nb(qkv, 1) * t0,
            );
            let cs = g.get_rows(conv_pool, slot_i32[gi]); // [conv_dim*(K-1), s]
            let cs = g.reshape_3d(cs, conv_dim, kc - 1, s);
            let xin = g.concat(cs, qkv_g, 1); // [conv_dim, K-1+q, s]
            let new_cs = g.view_3d(
                xin,
                conv_dim,
                kc - 1,
                s,
                nb(xin, 1),
                nb(xin, 2),
                nb(xin, 1) * q as usize,
            );
            let new_cs = g.reshape_2d(g.cont(new_cs), conv_dim * (kc - 1), s);
            let w = g.set_rows(conv_pool, new_cs, slot_i64[gi]);
            g.expand(w);
            let sx = g.cont(g.transpose(xin)); // [K-1+q, conv_dim, s]
            let conv = g.silu(g.ssm_conv(sx, a.conv)); // [conv_dim, q, s]

            let es = crate::graph::elsize(conv);
            let (cb1, cb2) = (nb(conv, 1), nb(conv, 2));
            let qc = g.view_4d(conv, dk, hk, q, s, dk as usize * es, cb1, cb2, 0);
            let kcv = g.view_4d(
                conv,
                dk,
                hk,
                q,
                s,
                dk as usize * es,
                cb1,
                cb2,
                (dk * hk) as usize * es,
            );
            let vc = g.view_4d(
                conv,
                dv,
                hv,
                q,
                s,
                dv as usize * es,
                cb1,
                cb2,
                (2 * dk * hk) as usize * es,
            );
            let mut qc = g.l2_norm(qc, self.eps);
            let mut kcv = g.l2_norm(kcv, self.eps);
            if hv != hk {
                // HF repeats key heads interleaved (value head h uses key head h / r).
                let r = hv / hk;
                let rep = |t: T| {
                    let t = g.reshape_4d(t, dk, 1, hk, q * s);
                    let t = g.repeat_4d(t, dk, r, hk, q * s);
                    g.reshape_4d(t, dk, hv, q, s)
                };
                qc = rep(qc);
                kcv = rep(kcv);
            }

            // --- gated delta rule ---------------------------------------
            let st = g.get_rows(ssm_pool, slot_i32[gi]);
            let st = g.reshape_4d(st, dv, dv, hv, s);
            let gf = 4usize;
            let gg = g.view_4d(
                gate,
                1,
                hv,
                q,
                s,
                gf,
                hv as usize * gf,
                (hv * q) as usize * gf,
                hv as usize * gf * t0,
            );
            let bg = g.view_4d(
                beta,
                1,
                hv,
                q,
                s,
                gf,
                hv as usize * gf,
                (hv * q) as usize * gf,
                hv as usize * gf * t0,
            );
            let r = g.gated_delta_net(qc, kcv, vc, gg, bg, st);
            let o_elems = (dv * hv * q * s) as usize;
            let o = g.view_2d(r, dv * hv, q * s, (dv * hv) as usize * 4, 0);
            let new_st = g.view_2d(r, dv * dv * hv, s, (dv * dv * hv) as usize * 4, o_elems * 4);
            let w = g.set_rows(ssm_pool, new_st, slot_i64[gi]);
            g.expand(w);
            out = Some(match out {
                None => o,
                Some(prev) => g.concat(prev, o, 1),
            });
        }
        let o = out.expect("empty micro-batch");
        let o = g.reshape_2d(g.cont(o), dv, hv * n);
        let z = g.reshape_2d(z, dv, hv * n);
        let o = g.mul(g.rms_norm_w(o, a.norm, self.eps), g.silu(z));
        let o = g.reshape_2d(o, dv * hv, n);
        g.mul_mat(a.wout, o)
    }
}

// SAFETY: tensor handles point into the model-owned ggml context; all use is
// serialized by the engine mutex.
unsafe impl Send for Qwen35 {}

impl Model for Qwen35 {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn build(&self, g: &mut Graph, mb: &MicroBatch, cache: &Cache, rows: &[i32]) -> Result<T> {
        let tokens = g.input_i32(&mb.tokens);
        let pos = g.input_i32(&mb.positions());
        let slots = g.input_i64(&mb.slot_mapping()?);
        let inputs = AttnInputs::new(g, mb, None)?;
        let mut slot_i32 = Vec::new();
        let mut slot_i64 = Vec::new();
        for grp in &mb.groups {
            let ss: Vec<i32> = mb.seqs[grp.seq_start..grp.seq_start + grp.n_seqs]
                .iter()
                .map(|s| s.state_slot)
                .collect();
            slot_i32.push(g.input_i32(&ss));
            slot_i64.push(g.input_i64(&ss.iter().map(|&v| v as i64).collect::<Vec<_>>()));
        }

        let mut x = g.get_rows(self.embed, tokens); // [hidden, n]
        for layer in &self.layers {
            let h = g.rms_norm_w(x, layer.attn_norm, self.eps);
            let a = match &layer.mixer {
                Mixer::Full(fa) => self.full_attn(g, mb, cache, fa, h, pos, slots, &inputs),
                Mixer::Linear(la) => self.linear_attn(g, mb, cache, la, h, &slot_i32, &slot_i64),
            };
            x = g.add(x, a);
            let h = g.rms_norm_w(x, layer.post_norm, self.eps);
            let m = g.swiglu_split(g.mul_mat(layer.gate, h), g.mul_mat(layer.up, h));
            x = g.add(x, g.mul_mat(layer.down, m));
        }
        let rows_t = g.input_i32(rows);
        let x = g.get_rows(x, rows_t);
        let x = g.rms_norm_w(x, self.norm, self.eps);
        Ok(g.mul_mat(self.lm_head, x))
    }
}
