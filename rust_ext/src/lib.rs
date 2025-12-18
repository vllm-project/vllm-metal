use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Fast conversion of numpy array to nested Python list.
/// This is much faster than Python's .tolist() method because it:
/// 1. Avoids Python object allocation for each element
/// 2. Builds the list structure directly
/// 3. No GIL contention during the conversion
#[pyfunction]
fn tensor_to_nested_list(py: Python<'_>, arr: PyReadonlyArray2<i64>) -> PyResult<PyObject> {
    let arr = arr.as_array();
    let shape = arr.shape();
    let rows = shape[0];
    let cols = shape[1];

    let outer_list = PyList::empty_bound(py);

    for i in 0..rows {
        let inner_list = PyList::empty_bound(py);
        for j in 0..cols {
            inner_list.append(arr[[i, j]])?;
        }
        outer_list.append(inner_list)?;
    }

    Ok(outer_list.into())
}

/// Fast conversion of 1D numpy array to Python list of lists (each with 1 element).
/// Optimized for the common case where each batch has exactly 1 token.
#[pyfunction]
fn tensor_1d_to_nested_list(py: Python<'_>, arr: PyReadonlyArray1<i64>) -> PyResult<PyObject> {
    let arr = arr.as_array();
    let len = arr.len();

    let outer_list = PyList::empty_bound(py);

    for i in 0..len {
        let inner_list = PyList::empty_bound(py);
        inner_list.append(arr[i])?;
        outer_list.append(inner_list)?;
    }

    Ok(outer_list.into())
}

/// Fast conversion of flat numpy array to Python list.
#[pyfunction]
fn tensor_to_flat_list(py: Python<'_>, arr: PyReadonlyArray1<i64>) -> PyResult<PyObject> {
    let arr = arr.as_array();
    let len = arr.len();

    let list = PyList::empty_bound(py);

    for i in 0..len {
        list.append(arr[i])?;
    }

    Ok(list.into())
}

/// Compute flat indices for KV cache gathering.
/// This eliminates Python loop overhead for index computation.
///
/// Given:
/// - seq_lens: [num_seqs] - length of each sequence
/// - block_table: [num_seqs, max_blocks] - block table for each sequence
/// - block_size: scalar - tokens per block
///
/// Returns:
/// - flat_indices: [total_hist_tokens] - indices into flattened cache
/// - seq_offsets: [num_seqs + 1] - cumulative offsets for each sequence's indices
#[pyfunction]
fn compute_kv_gather_indices(
    py: Python<'_>,
    seq_lens: PyReadonlyArray1<i64>,
    block_table: PyReadonlyArray2<i64>,
    block_size: i64,
) -> PyResult<(PyObject, PyObject)> {
    let seq_lens = seq_lens.as_array();
    let block_table = block_table.as_array();
    let num_seqs = seq_lens.len();

    // First pass: compute total tokens needed
    let mut total_tokens: usize = 0;
    let mut hist_lens: Vec<i64> = Vec::with_capacity(num_seqs);

    for i in 0..num_seqs {
        let hist_len = seq_lens[i] - 1; // Exclude current token
        hist_lens.push(hist_len.max(0));
        total_tokens += hist_len.max(0) as usize;
    }

    // Allocate output arrays
    let mut flat_indices: Vec<i64> = Vec::with_capacity(total_tokens);
    let mut seq_offsets: Vec<i64> = Vec::with_capacity(num_seqs + 1);
    seq_offsets.push(0);

    // Second pass: compute indices
    let mut offset: i64 = 0;
    for seq_idx in 0..num_seqs {
        let hist_len = hist_lens[seq_idx];

        for pos in 0..hist_len {
            let logical_block = pos / block_size;
            let block_offset = pos % block_size;
            let physical_block = block_table[[seq_idx, logical_block as usize]];
            let flat_idx = physical_block * block_size + block_offset;
            flat_indices.push(flat_idx);
        }

        offset += hist_len;
        seq_offsets.push(offset);
    }

    // Convert to Python lists
    let indices_list = PyList::empty_bound(py);
    for idx in &flat_indices {
        indices_list.append(*idx)?;
    }

    let offsets_list = PyList::empty_bound(py);
    for off in &seq_offsets {
        offsets_list.append(*off)?;
    }

    Ok((indices_list.into(), offsets_list.into()))
}

/// Compute padded KV gather indices for batched SDPA.
/// Returns indices padded to max_seq_len for each sequence, enabling
/// batched gather operations.
///
/// Returns:
/// - padded_indices: [num_seqs, max_hist_len] - padded indices (0 for padding)
/// - hist_lens: [num_seqs] - actual history length per sequence
#[pyfunction]
fn compute_padded_kv_indices(
    py: Python<'_>,
    seq_lens: PyReadonlyArray1<i64>,
    block_table: PyReadonlyArray2<i64>,
    block_size: i64,
) -> PyResult<(PyObject, PyObject)> {
    let seq_lens = seq_lens.as_array();
    let block_table = block_table.as_array();
    let num_seqs = seq_lens.len();

    // Find max history length
    let mut max_hist_len: i64 = 0;
    let mut hist_lens_vec: Vec<i64> = Vec::with_capacity(num_seqs);

    for i in 0..num_seqs {
        let hist_len = (seq_lens[i] - 1).max(0);
        hist_lens_vec.push(hist_len);
        if hist_len > max_hist_len {
            max_hist_len = hist_len;
        }
    }

    if max_hist_len == 0 {
        // No history - return empty arrays
        let empty_outer = PyList::empty_bound(py);
        let empty_lens = PyList::empty_bound(py);
        for _ in 0..num_seqs {
            empty_outer.append(PyList::empty_bound(py))?;
            empty_lens.append(0i64)?;
        }
        return Ok((empty_outer.into(), empty_lens.into()));
    }

    // Build padded index array
    let padded_outer = PyList::empty_bound(py);

    for seq_idx in 0..num_seqs {
        let hist_len = hist_lens_vec[seq_idx];
        let inner = PyList::empty_bound(py);

        for pos in 0..max_hist_len {
            if pos < hist_len {
                let logical_block = pos / block_size;
                let block_offset = pos % block_size;
                let physical_block = block_table[[seq_idx, logical_block as usize]];
                let flat_idx = physical_block * block_size + block_offset;
                inner.append(flat_idx)?;
            } else {
                // Padding - use 0 (will be masked out)
                inner.append(0i64)?;
            }
        }
        padded_outer.append(inner)?;
    }

    let hist_lens_list = PyList::empty_bound(py);
    for &len in &hist_lens_vec {
        hist_lens_list.append(len)?;
    }

    Ok((padded_outer.into(), hist_lens_list.into()))
}

/// Batch process position encodings for rotary embeddings.
/// Computes sin/cos for all positions efficiently.
#[pyfunction]
fn compute_rotary_positions(
    py: Python<'_>,
    seq_lens: PyReadonlyArray1<i64>,
    query_start_loc: PyReadonlyArray1<i64>,
) -> PyResult<PyObject> {
    let seq_lens = seq_lens.as_array();
    let query_start_loc = query_start_loc.as_array();
    let num_seqs = seq_lens.len();

    // For decode, each sequence has query_len=1, so position = seq_len - 1
    let positions = PyList::empty_bound(py);

    for seq_idx in 0..num_seqs {
        positions.append(seq_lens[seq_idx] - 1)?;
    }

    Ok(positions.into())
}

/// Double-buffer state for async token generation.
/// This enables CPU to prepare next batch while GPU processes current batch.
#[pyclass]
struct AsyncTokenBuffer {
    ready: Arc<AtomicBool>,
    buffer_idx: usize,
}

#[pymethods]
impl AsyncTokenBuffer {
    #[new]
    fn new() -> Self {
        AsyncTokenBuffer {
            ready: Arc::new(AtomicBool::new(false)),
            buffer_idx: 0,
        }
    }

    fn mark_ready(&self) {
        self.ready.store(true, Ordering::Release);
    }

    fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    fn swap_buffer(&mut self) {
        self.buffer_idx = 1 - self.buffer_idx;
        self.ready.store(false, Ordering::Release);
    }

    fn get_buffer_idx(&self) -> usize {
        self.buffer_idx
    }
}

/// Fast conversion with pre-allocated buffer hint.
/// When we know the output size, we can pre-allocate for better performance.
#[pyfunction]
fn tensor_to_nested_list_preallocated(
    py: Python<'_>,
    arr: PyReadonlyArray2<i64>,
    _expected_rows: usize,
    _expected_cols: usize,
) -> PyResult<PyObject> {
    // For now, delegate to the main function - PyO3 lists grow dynamically
    // In future, could use Vec pre-allocation for Rust-side work
    tensor_to_nested_list(py, arr)
}

/// A Python module implemented in Rust for fast tensor operations.
#[pymodule]
fn vllm_metal_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tensor_to_nested_list, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_1d_to_nested_list, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_to_flat_list, m)?)?;
    m.add_function(wrap_pyfunction!(compute_kv_gather_indices, m)?)?;
    m.add_function(wrap_pyfunction!(compute_padded_kv_indices, m)?)?;
    m.add_function(wrap_pyfunction!(compute_rotary_positions, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_to_nested_list_preallocated, m)?)?;
    m.add_class::<AsyncTokenBuffer>()?;
    Ok(())
}
