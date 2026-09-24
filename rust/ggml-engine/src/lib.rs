// SPDX-License-Identifier: Apache-2.0
//! `vllm_metal.ggml._ggml_engine`: a ggml (Metal) model executor for
//! vllm-metal. Python owns scheduling/sampling; this module owns weights,
//! the paged KV cache, recurrent state and the forward graph.

mod backend;
mod batch;
mod cache;
mod engine;
mod ffi;
mod graph;
mod models;
mod st;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;
use std::sync::Mutex;

fn err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

fn read_i32(py: Python<'_>, obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<i32>> {
    let buf = PyBuffer::<i32>::get(obj).map_err(|e| {
        PyValueError::new_err(format!("{name}: expected a contiguous int32 buffer ({e})"))
    })?;
    buf.to_vec(py)
}

#[pyclass(name = "Engine", module = "vllm_metal.ggml._ggml_engine")]
struct PyEngine {
    inner: Mutex<engine::Engine>,
}

#[pymethods]
impl PyEngine {
    /// Load a HF safetensors checkpoint directory onto the ggml backend.
    /// `device` is "metal" (default) or "cpu".
    #[new]
    #[pyo3(signature = (model_dir, device = "metal"))]
    fn new(py: Python<'_>, model_dir: PathBuf, device: &str) -> PyResult<Self> {
        let force_cpu = match device {
            "metal" | "gpu" => false,
            "cpu" => true,
            other => return Err(PyValueError::new_err(format!("unknown device '{other}'"))),
        };
        let eng = py
            .detach(|| engine::Engine::new(&model_dir, force_cpu))
            .map_err(err)?;
        Ok(Self {
            inner: Mutex::new(eng),
        })
    }

    /// Static model / backend description.
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let e = self.inner.lock().unwrap();
        let i = e.info();
        let d = PyDict::new(py);
        d.set_item("arch", &i.arch)?;
        d.set_item("vocab_size", i.vocab_size)?;
        d.set_item("hidden_size", i.hidden_size)?;
        d.set_item("num_layers", i.num_layers)?;
        d.set_item("max_position_embeddings", i.max_position_embeddings)?;
        d.set_item("weight_bytes", i.weight_bytes)?;
        d.set_item("backend", e.backend_name())?;
        d.set_item("kv_bytes_per_token", e.kv_bytes_per_token())?;
        d.set_item("state_bytes_per_slot", e.state_bytes_per_slot())?;
        d.set_item("max_groups", e.max_groups())?;
        let kv = PyList::empty(py);
        for l in &i.kv_layers {
            let x = PyDict::new(py);
            x.set_item("layer", l.layer)?;
            x.set_item("num_kv_heads", l.n_kv_heads)?;
            x.set_item("head_dim", l.head_dim)?;
            x.set_item("sliding_window", l.sliding_window)?;
            kv.append(x)?;
        }
        d.set_item("kv_layers", kv)?;
        let stl = PyList::empty(py);
        for s in &i.state_layers {
            let x = PyDict::new(py);
            x.set_item("layer", s.layer)?;
            x.set_item("conv_elems", s.conv_row)?;
            x.set_item("ssm_elems", s.ssm_row)?;
            stl.append(x)?;
        }
        d.set_item("state_layers", stl)?;
        Ok(d)
    }

    /// (free, total) device memory in bytes.
    fn memory(&self) -> (usize, usize) {
        self.inner.lock().unwrap().memory()
    }

    fn set_max_groups(&self, n: usize) {
        self.inner.lock().unwrap().set_max_groups(n)
    }

    /// Byte budget for attention temporaries (gathered K/V + masks) per
    /// micro-batch; larger steps are split into several graphs.
    fn set_attn_budget(&self, bytes: usize) {
        self.inner.lock().unwrap().set_attn_budget(bytes)
    }

    /// Allocate the paged KV cache and recurrent-state pools. Returns bytes.
    fn init_cache(
        &self,
        py: Python<'_>,
        num_blocks: usize,
        block_size: usize,
        num_state_slots: usize,
    ) -> PyResult<usize> {
        py.detach(|| {
            self.inner
                .lock()
                .unwrap()
                .init_cache(num_blocks, block_size, num_state_slots)
        })
        .map_err(err)
    }

    /// Run one step.
    ///
    /// token_ids:   int32 [T], packed sequence-major
    /// q_lens:      int32 [S], tokens of each sequence in this step
    /// ctx_lens:    int32 [S], context length after this step
    /// block_table: int32 [S, max_blocks] (row-major, padded)
    /// state_slots: int32 [S], recurrent-state slot per sequence
    /// reset_state: int32 [S], nonzero = zero the slot before running
    /// n_logits:    int32 [S], logits for the last n tokens of each sequence
    /// out:         writable float32 buffer >= sum(n_logits) * vocab
    /// Returns the number of logits rows written.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        py: Python<'_>,
        token_ids: &Bound<'_, PyAny>,
        q_lens: &Bound<'_, PyAny>,
        ctx_lens: &Bound<'_, PyAny>,
        block_table: &Bound<'_, PyAny>,
        state_slots: &Bound<'_, PyAny>,
        reset_state: &Bound<'_, PyAny>,
        n_logits: &Bound<'_, PyAny>,
        out: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let tokens = read_i32(py, token_ids, "token_ids")?;
        let q = read_i32(py, q_lens, "q_lens")?;
        let c = read_i32(py, ctx_lens, "ctx_lens")?;
        let bt = read_i32(py, block_table, "block_table")?;
        let ss = read_i32(py, state_slots, "state_slots")?;
        let rs = read_i32(py, reset_state, "reset_state")?;
        let nl = read_i32(py, n_logits, "n_logits")?;
        let n = q.len();
        if [c.len(), ss.len(), rs.len(), nl.len()]
            .iter()
            .any(|&l| l != n)
        {
            return Err(PyValueError::new_err(
                "per-sequence arrays must have equal length",
            ));
        }
        if n == 0 {
            return Ok(0);
        }
        if bt.len() % n != 0 {
            return Err(PyValueError::new_err(
                "block_table size must be a multiple of the number of sequences",
            ));
        }
        let mb = bt.len() / n;
        let seqs: Vec<batch::Seq> = (0..n)
            .map(|i| batch::Seq {
                q_len: q[i].max(0) as usize,
                ctx_len: c[i].max(0) as usize,
                blocks: bt[i * mb..(i + 1) * mb].to_vec(),
                state_slot: ss[i],
                reset_state: rs[i] != 0,
                n_logits: nl[i].max(0) as usize,
            })
            .collect();

        let out_buf = PyBuffer::<f32>::get(out).map_err(|e| {
            PyValueError::new_err(format!("out: expected a contiguous float32 buffer ({e})"))
        })?;
        if out_buf.readonly() || !out_buf.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "out must be a writable C-contiguous float32 buffer",
            ));
        }
        let ptr = out_buf.buf_ptr() as usize;
        let len = out_buf.item_count();
        py.detach(move || {
            // SAFETY: `out` is kept alive by the caller for the duration of the
            // call and its buffer is exported as writable and contiguous.
            let dst = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, len) };
            self.inner.lock().unwrap().forward(&tokens, seqs, dst)
        })
        .map_err(err)
    }
}

#[pymodule]
fn _ggml_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    Ok(())
}
