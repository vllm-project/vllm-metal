// SPDX-License-Identifier: Apache-2.0
//! Engine: owns the backend, model weights and caches; runs forward passes.

use crate::backend::Backend;
use crate::batch::{plan, Limits, Seq};
use crate::cache::Cache;
use crate::ffi::*;
use crate::graph::Graph;
use crate::models::{self, Model, ModelInfo};
use anyhow::{anyhow, bail, Result};
use std::path::Path;
use std::time::Instant;

/// Scheduler graph capacity (nodes). Micro-batches are sized to fit.
const GRAPH_SIZE: usize = 65536;
const MAX_GROUPS_CAP: usize = 32;
/// Default byte budget for per-micro-batch attention temporaries.
const DEFAULT_ATTN_BUDGET: usize = 1 << 30;

pub struct Engine {
    // Field order matters for Drop: caches and weights before the backend.
    cache: Option<Cache>,
    model: Box<dyn Model>,
    backend: Backend,
    max_groups: usize,
    profile: bool,
    attn_budget: usize,
}

impl Engine {
    pub fn new(model_dir: &Path, force_cpu: bool) -> Result<Self> {
        let backend = Backend::new(GRAPH_SIZE, force_cpu)?;
        let model = models::load(model_dir, &backend)?;
        let info = model.info();
        let (npl, npg) = model.nodes_per_layer();
        let l = info.num_layers.max(1);
        let budget = GRAPH_SIZE.saturating_sub(512 + l * npl);
        let max_groups = (budget / (l * npg)).clamp(1, MAX_GROUPS_CAP);
        Ok(Self {
            cache: None,
            model,
            backend,
            max_groups,
            profile: std::env::var("VLLM_METAL_GGML_PROFILE").as_deref() == Ok("1"),
            attn_budget: DEFAULT_ATTN_BUDGET,
        })
    }

    pub fn info(&self) -> &ModelInfo {
        self.model.info()
    }

    pub fn backend_name(&self) -> String {
        self.backend.name()
    }

    pub fn memory(&self) -> (usize, usize) {
        self.backend.memory()
    }

    pub fn max_groups(&self) -> usize {
        self.max_groups
    }

    fn limits(&self) -> Limits {
        let info = self.model.info();
        let kv_row = info
            .kv_layers
            .iter()
            .map(|l| l.head_dim * l.n_kv_heads)
            .max()
            .unwrap_or(0);
        let kinds = 1 + info.kv_layers.iter().any(|l| l.sliding_window.is_some()) as usize;
        Limits {
            max_groups: self.max_groups,
            attn_bytes: self.attn_budget,
            kv_row,
            kinds,
        }
    }

    pub fn set_attn_budget(&mut self, bytes: usize) {
        self.attn_budget = bytes.max(1);
    }

    pub fn set_max_groups(&mut self, n: usize) {
        self.max_groups = n.clamp(1, MAX_GROUPS_CAP);
    }

    pub fn init_cache(
        &mut self,
        num_blocks: usize,
        block_size: usize,
        n_state_slots: usize,
    ) -> Result<usize> {
        if block_size == 0 || num_blocks == 0 {
            bail!("num_blocks and block_size must be positive");
        }
        self.cache = None; // free the previous allocation first
        let cache = Cache::new(
            &self.backend,
            self.model.info(),
            num_blocks,
            block_size,
            n_state_slots,
        )?;
        let bytes = cache.bytes();
        self.cache = Some(cache);
        Ok(bytes)
    }

    pub fn kv_bytes_per_token(&self) -> usize {
        Cache::kv_bytes_per_token(self.model.info())
    }

    pub fn state_bytes_per_slot(&self) -> usize {
        Cache::state_bytes_per_slot(self.model.info())
    }

    fn validate(&self, cache: &Cache, seqs: &[Seq]) -> Result<()> {
        let has_state = !self.model.info().state_layers.is_empty();
        for (i, s) in seqs.iter().enumerate() {
            let need = s.ctx_len.div_ceil(cache.block_size);
            if s.blocks.len() < need {
                bail!(
                    "seq {i}: block table has {} blocks, need {need}",
                    s.blocks.len()
                );
            }
            if let Some(&b) = s.blocks[..need]
                .iter()
                .find(|&&b| b < 0 || b as usize >= cache.num_blocks)
            {
                bail!(
                    "seq {i}: block id {b} out of range (num_blocks={})",
                    cache.num_blocks
                );
            }
            if has_state && (s.state_slot < 0 || s.state_slot as usize >= cache.n_state_slots) {
                bail!(
                    "seq {i}: state slot {} out of range ({} slots)",
                    s.state_slot,
                    cache.n_state_slots
                );
            }
        }
        Ok(())
    }

    /// Run one scheduler step. Writes `sum(min(n_logits, q_len))` rows of
    /// `vocab_size` logits into `out` (sequence order) and returns the row count.
    pub fn forward(&mut self, tokens: &[i32], seqs: Vec<Seq>, out: &mut [f32]) -> Result<usize> {
        let cache = self
            .cache
            .as_ref()
            .ok_or_else(|| anyhow!("init_cache() has not been called"))?;
        self.validate(cache, &seqs)?;
        let vocab = self.model.info().vocab_size;
        let total_rows: usize = seqs.iter().map(|s| s.n_logits.min(s.q_len)).sum();
        if out.len() < total_rows * vocab {
            bail!(
                "output buffer too small: {} < {}",
                out.len(),
                total_rows * vocab
            );
        }
        let n_layers = self.model.info().num_layers;
        let (npl, npg) = self.model.nodes_per_layer();

        let mut row_off = 0usize;
        for mb in plan(tokens, seqs, cache.block_size, self.limits())? {
            for s in &mb.seqs {
                if s.reset_state && !cache.conv.is_empty() {
                    cache.reset_state_slot(s.state_slot as usize);
                }
            }
            let rows = mb.logit_rows();
            let discard = rows.is_empty();
            let rows_eff = if discard {
                vec![mb.n_tokens() as i32 - 1]
            } else {
                rows
            };

            let t0 = Instant::now();
            let max_nodes = (512 + n_layers * (npl + npg * mb.groups.len())).min(GRAPH_SIZE);
            let mut g = Graph::new(max_nodes);
            let logits = self.model.build(&mut g, &mb, cache, &rows_eff)?;
            unsafe {
                ggml_set_output(logits);
            }
            g.expand(logits);
            let t_build = t0.elapsed();
            let (t_alloc, t_upload, t_compute);
            unsafe {
                let sched = self.backend.sched;
                ggml_backend_sched_reset(sched);
                if !ggml_backend_sched_alloc_graph(sched, g.gf) {
                    bail!(
                        "ggml: failed to allocate compute graph ({} nodes)",
                        g.n_nodes()
                    );
                }
                if self.backend.is_gpu {
                    check_no_cpu_nodes(&self.backend, &g)?;
                }
                t_alloc = t0.elapsed();
                g.upload_inputs();
                t_upload = t0.elapsed();
                let st = ggml_backend_sched_graph_compute(sched, g.gf);
                if st != ggml_status_GGML_STATUS_SUCCESS {
                    bail!("ggml graph compute failed with status {st}");
                }
                t_compute = t0.elapsed();
                if !discard {
                    let n = rows_eff.len() * vocab;
                    let dst = &mut out[row_off * vocab..row_off * vocab + n];
                    ggml_backend_tensor_get(logits, dst.as_mut_ptr() as *mut _, 0, n * 4);
                    row_off += rows_eff.len();
                }
            }
            if self.profile {
                let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
                eprintln!(
                    "[ggml] tokens={} seqs={} groups={} nodes={} splits={} | build {:.2} alloc {:.2} upload {:.2} compute {:.2} read {:.2} ms",
                    mb.n_tokens(),
                    mb.seqs.len(),
                    mb.groups.len(),
                    g.n_nodes(),
                    unsafe { ggml_backend_sched_get_n_splits(self.backend.sched) },
                    ms(t_build),
                    ms(t_alloc - t_build),
                    ms(t_upload - t_alloc),
                    ms(t_compute - t_upload),
                    ms(t0.elapsed() - t_compute),
                );
            }
        }
        debug_assert_eq!(row_off, total_rows);
        Ok(total_rows)
    }
}

/// Verify every node runs on the GPU backend. The Homebrew ggml CPU backend
/// uses its own libomp; running CPU kernels in a process that has also
/// initialized PyTorch's bundled libomp aborts the process, so a silent CPU
/// fallback must surface as an error instead.
#[allow(non_upper_case_globals)]
fn check_no_cpu_nodes(backend: &Backend, g: &Graph) -> Result<()> {
    let mut cpu_ops: Vec<String> = Vec::new();
    unsafe {
        for i in 0..g.n_nodes() {
            let node = ggml_graph_node(g.gf, i);
            // View-type ops execute no kernel (e.g. reshapes of host inputs).
            let is_view = matches!(
                (*node).op,
                ggml_op_GGML_OP_NONE
                    | ggml_op_GGML_OP_RESHAPE
                    | ggml_op_GGML_OP_VIEW
                    | ggml_op_GGML_OP_PERMUTE
                    | ggml_op_GGML_OP_TRANSPOSE
            );
            if !is_view && ggml_backend_sched_get_tensor_backend(backend.sched, node) == backend.cpu
            {
                let op = std::ffi::CStr::from_ptr(ggml_op_desc(node))
                    .to_string_lossy()
                    .into_owned();
                if !cpu_ops.contains(&op) {
                    cpu_ops.push(op);
                }
            }
        }
    }
    if !cpu_ops.is_empty()
        && std::env::var("VLLM_METAL_GGML_ALLOW_CPU_FALLBACK").as_deref() != Ok("1")
    {
        bail!(
            "ggml scheduled ops on the CPU backend ({}); set VLLM_METAL_GGML_ALLOW_CPU_FALLBACK=1 to allow",
            cpu_ops.join(", ")
        );
    }
    Ok(())
}
