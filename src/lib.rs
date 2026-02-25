// SPDX-License-Identifier: Apache-2.0
//! High-performance Rust extension for vLLM Metal.
//!
//! This module provides optimized implementations of performance-critical
//! operations that are bottlenecks in the Python code:
//!
//! 1. Block allocation with O(1) operations
//! 2. Input preparation with numpy arrays (zero-copy)
//! 3. KV cache block retrieval with vectorized indexing

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use std::collections::{HashMap, VecDeque};

/// High-performance block allocator with O(1) allocation and deallocation.
///
/// Uses a VecDeque for O(1) pop_front operations instead of Python's list.pop(0)
/// which is O(n).
#[pyclass]
pub struct BlockAllocator {
    /// Free blocks stored in a deque for O(1) operations
    free_blocks: VecDeque<usize>,
    /// Mapping from sequence ID to allocated blocks
    sequence_blocks: HashMap<String, Vec<usize>>,
    /// Total number of blocks
    num_blocks: usize,
}

#[pymethods]
impl BlockAllocator {
    /// Create a new block allocator.
    ///
    /// # Arguments
    /// * `num_blocks` - Total number of blocks to manage
    #[new]
    pub fn new(num_blocks: usize) -> Self {
        let free_blocks: VecDeque<usize> = (0..num_blocks).collect();
        BlockAllocator {
            free_blocks,
            sequence_blocks: HashMap::new(),
            num_blocks,
        }
    }

    /// Allocate blocks for a sequence.
    ///
    /// # Arguments
    /// * `seq_id` - Sequence identifier
    /// * `num_blocks` - Number of blocks to allocate
    ///
    /// # Returns
    /// List of allocated block indices
    ///
    /// # Raises
    /// RuntimeError if not enough free blocks
    pub fn allocate_blocks(&mut self, seq_id: String, num_blocks: usize) -> PyResult<Vec<usize>> {
        if self.free_blocks.len() < num_blocks {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Not enough free blocks: need {}, have {}",
                num_blocks,
                self.free_blocks.len()
            )));
        }

        let mut allocated = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            // O(1) operation!
            if let Some(block_idx) = self.free_blocks.pop_front() {
                allocated.push(block_idx);
            }
        }

        // Update sequence blocks
        self.sequence_blocks
            .entry(seq_id)
            .or_insert_with(Vec::new)
            .extend(allocated.iter().copied());

        Ok(allocated)
    }

    /// Free all blocks for a sequence.
    ///
    /// # Arguments
    /// * `seq_id` - Sequence identifier
    pub fn free_sequence(&mut self, seq_id: String) {
        if let Some(blocks) = self.sequence_blocks.remove(&seq_id) {
            // Return blocks to the free pool
            for block_idx in blocks {
                self.free_blocks.push_back(block_idx);
            }
        }
    }

    /// Get blocks for a sequence.
    ///
    /// # Arguments
    /// * `seq_id` - Sequence identifier
    ///
    /// # Returns
    /// List of block indices for the sequence
    pub fn get_sequence_blocks(&self, seq_id: String) -> Vec<usize> {
        self.sequence_blocks
            .get(&seq_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get the number of free blocks.
    #[getter]
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get the total number of blocks.
    #[getter]
    pub fn total_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Check if sequence has blocks allocated.
    pub fn has_sequence(&self, seq_id: String) -> bool {
        self.sequence_blocks.contains_key(&seq_id)
    }

    /// Reset the allocator to initial state.
    pub fn reset(&mut self) {
        self.free_blocks = (0..self.num_blocks).collect();
        self.sequence_blocks.clear();
    }

    /// Get all sequence blocks as a dictionary.
    pub fn get_all_sequence_blocks(&self) -> HashMap<String, Vec<usize>> {
        self.sequence_blocks.clone()
    }
}

/// Fast input preparation for model inference with numpy output.
///
/// Prepares input_ids and positions arrays from sequence data
/// with pre-allocated buffers and returns numpy arrays.
#[pyclass]
pub struct InputPreparer {
    /// Pre-allocated buffer for input IDs
    input_ids_buffer: Vec<i32>,
    /// Pre-allocated buffer for positions
    positions_buffer: Vec<i32>,
}

#[pymethods]
impl InputPreparer {
    /// Create a new input preparer.
    ///
    /// # Arguments
    /// * `initial_capacity` - Initial buffer capacity (will grow as needed)
    #[new]
    #[pyo3(signature = (initial_capacity=4096))]
    pub fn new(initial_capacity: usize) -> Self {
        InputPreparer {
            input_ids_buffer: Vec::with_capacity(initial_capacity),
            positions_buffer: Vec::with_capacity(initial_capacity),
        }
    }

    /// Prepare inputs and return numpy arrays.
    ///
    /// # Arguments
    /// * `sequences` - List of (token_ids, is_prompt) tuples
    ///
    /// # Returns
    /// Tuple of (input_ids, positions) as numpy arrays
    pub fn prepare_numpy<'py>(
        &mut self,
        py: Python<'py>,
        sequences: Vec<(Vec<i32>, bool)>,
    ) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>)> {
        // Clear and reuse buffers
        self.input_ids_buffer.clear();
        self.positions_buffer.clear();

        for (token_ids, is_prompt) in sequences {
            if is_prompt {
                // Prefill: use all tokens
                let num_tokens = token_ids.len();
                self.input_ids_buffer.extend(token_ids);
                self.positions_buffer.extend(0..num_tokens as i32);
            } else {
                // Decode: use only last token
                if let Some(&last_token) = token_ids.last() {
                    self.input_ids_buffer.push(last_token);
                    self.positions_buffer.push(token_ids.len() as i32 - 1);
                }
            }
        }

        // Create numpy arrays from the buffers
        let input_ids = PyArray1::from_slice(py, &self.input_ids_buffer);
        let positions = PyArray1::from_slice(py, &self.positions_buffer);

        Ok((input_ids, positions))
    }

    /// Prepare inputs from numpy arrays directly (zero-copy input).
    ///
    /// # Arguments
    /// * `token_ids_list` - List of numpy arrays containing token IDs
    /// * `is_prompt_list` - List of booleans indicating if each sequence is prefill
    ///
    /// # Returns
    /// Tuple of (input_ids, positions) as numpy arrays
    pub fn prepare_from_numpy<'py>(
        &mut self,
        py: Python<'py>,
        token_ids_list: Vec<Bound<'py, PyArray1<i32>>>,
        is_prompt_list: Vec<bool>,
    ) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>)> {
        // Clear and reuse buffers
        self.input_ids_buffer.clear();
        self.positions_buffer.clear();

        for (token_ids, is_prompt) in token_ids_list.iter().zip(is_prompt_list.iter()) {
            let tokens = unsafe { token_ids.as_slice()? };
            let num_tokens = tokens.len();

            if *is_prompt {
                // Prefill: use all tokens
                self.input_ids_buffer.extend_from_slice(tokens);
                self.positions_buffer.extend(0..num_tokens as i32);
            } else {
                // Decode: use only last token
                if let Some(&last_token) = tokens.last() {
                    self.input_ids_buffer.push(last_token);
                    self.positions_buffer.push(num_tokens as i32 - 1);
                }
            }
        }

        // Create numpy arrays from the buffers
        let input_ids = PyArray1::from_slice(py, &self.input_ids_buffer);
        let positions = PyArray1::from_slice(py, &self.positions_buffer);

        Ok((input_ids, positions))
    }

    /// Get current buffer sizes (for debugging).
    pub fn buffer_sizes(&self) -> (usize, usize) {
        (self.input_ids_buffer.capacity(), self.positions_buffer.capacity())
    }
}

/// Compute block indices for KV cache retrieval.
///
/// Given a list of block indices and sequence length, returns the
/// (block_idx, slot_start, slot_end) tuples needed to gather KV pairs.
#[pyfunction]
pub fn compute_kv_block_indices(
    blocks: Vec<usize>,
    seq_len: usize,
    block_size: usize,
) -> Vec<(usize, usize, usize)> {
    if blocks.is_empty() || seq_len == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(blocks.len());
    let mut remaining = seq_len;

    for block_idx in blocks {
        let tokens_in_block = remaining.min(block_size);
        result.push((block_idx, 0, tokens_in_block));
        remaining = remaining.saturating_sub(block_size);
        if remaining == 0 {
            break;
        }
    }

    result
}

/// Batch compute block indices for multiple sequences.
#[pyfunction]
pub fn batch_compute_kv_indices(
    sequence_blocks: HashMap<String, Vec<usize>>,
    seq_lens: HashMap<String, usize>,
    block_size: usize,
) -> HashMap<String, Vec<(usize, usize, usize)>> {
    let mut result = HashMap::with_capacity(sequence_blocks.len());

    for (seq_id, blocks) in sequence_blocks {
        if let Some(&seq_len) = seq_lens.get(&seq_id) {
            let indices = compute_kv_block_indices(blocks, seq_len, block_size);
            result.insert(seq_id, indices);
        }
    }

    result
}

/// Flatten multiple token sequences into a numpy array.
#[pyfunction]
pub fn flatten_token_ids_numpy<'py>(
    py: Python<'py>,
    sequences: Vec<Vec<i32>>,
) -> Bound<'py, PyArray1<i32>> {
    let total_len: usize = sequences.iter().map(|s| s.len()).sum();
    let mut result = Vec::with_capacity(total_len);
    for seq in sequences {
        result.extend(seq);
    }
    PyArray1::from_vec(py, result)
}

/// Compute position indices and return as numpy array.
#[pyfunction]
pub fn compute_positions_numpy<'py>(
    py: Python<'py>,
    seq_lens: Vec<usize>,
    is_prefill: Vec<bool>,
) -> Bound<'py, PyArray1<i32>> {
    let total_len: usize = seq_lens
        .iter()
        .zip(is_prefill.iter())
        .map(|(&len, &prefill)| if prefill { len } else { 1 })
        .sum();

    let mut result = Vec::with_capacity(total_len);

    for (&len, &prefill) in seq_lens.iter().zip(is_prefill.iter()) {
        if prefill {
            result.extend(0..len as i32);
        } else {
            result.push(len as i32 - 1);
        }
    }

    PyArray1::from_vec(py, result)
}

/// High-performance request state manager for vLLM-Metal.
///
/// Manages token sequences for requests with O(1) operations for common
/// operations like getting last tokens, updating tokens, and batch operations.
#[pyclass]
pub struct RequestStateManager {
    /// Token sequences per request ID
    token_sequences: HashMap<String, Vec<i32>>,
    /// Generated token counts per request
    generated_counts: HashMap<String, usize>,
}

#[pymethods]
impl RequestStateManager {
    /// Create a new request state manager.
    #[new]
    pub fn new() -> Self {
        RequestStateManager {
            token_sequences: HashMap::new(),
            generated_counts: HashMap::new(),
        }
    }

    /// Add a new request with its initial tokens.
    ///
    /// # Arguments
    /// * `req_id` - Request identifier
    /// * `token_ids` - Initial token sequence
    pub fn add_request(&mut self, req_id: String, token_ids: Vec<i32>) {
        self.token_sequences.insert(req_id.clone(), token_ids);
        self.generated_counts.insert(req_id, 0);
    }

    /// Get the last token for a request.
    ///
    /// # Returns
    /// The last token ID, or 0 if request not found or empty
    pub fn get_last_token(&self, req_id: &str) -> i32 {
        self.token_sequences
            .get(req_id)
            .and_then(|tokens| tokens.last().copied())
            .unwrap_or(0)
    }

    /// Get last tokens for multiple requests in batch.
    ///
    /// # Arguments
    /// * `req_ids` - List of request IDs
    ///
    /// # Returns
    /// List of last tokens (0 for missing requests)
    pub fn get_last_tokens_batch(&self, req_ids: Vec<String>) -> Vec<i32> {
        req_ids
            .iter()
            .map(|req_id| self.get_last_token(req_id))
            .collect()
    }

    /// Append a token to a request's sequence.
    ///
    /// # Arguments
    /// * `req_id` - Request identifier
    /// * `token` - Token to append
    pub fn append_token(&mut self, req_id: &str, token: i32) {
        if let Some(tokens) = self.token_sequences.get_mut(req_id) {
            tokens.push(token);
        }
        if let Some(count) = self.generated_counts.get_mut(req_id) {
            *count += 1;
        }
    }

    /// Batch append tokens to multiple requests.
    ///
    /// # Arguments
    /// * `req_ids` - List of request IDs
    /// * `tokens` - List of tokens to append (parallel with req_ids)
    pub fn append_tokens_batch(&mut self, req_ids: Vec<String>, tokens: Vec<i32>) {
        for (req_id, token) in req_ids.iter().zip(tokens.iter()) {
            self.append_token(req_id, *token);
        }
    }

    /// Remove a request from the manager.
    ///
    /// # Arguments
    /// * `req_id` - Request identifier
    pub fn remove_request(&mut self, req_id: &str) {
        self.token_sequences.remove(req_id);
        self.generated_counts.remove(req_id);
    }

    /// Remove multiple requests in batch.
    ///
    /// # Arguments
    /// * `req_ids` - List of request IDs to remove
    pub fn remove_requests_batch(&mut self, req_ids: Vec<String>) {
        for req_id in req_ids {
            self.remove_request(&req_id);
        }
    }

    /// Check if a request exists.
    pub fn has_request(&self, req_id: &str) -> bool {
        self.token_sequences.contains_key(req_id)
    }

    /// Get the token sequence for a request.
    ///
    /// # Returns
    /// Token sequence, or empty list if not found
    pub fn get_tokens(&self, req_id: &str) -> Vec<i32> {
        self.token_sequences.get(req_id).cloned().unwrap_or_default()
    }

    /// Get the generated token count for a request.
    pub fn get_generated_count(&self, req_id: &str) -> usize {
        self.generated_counts.get(req_id).copied().unwrap_or(0)
    }

    /// Get the number of active requests.
    #[getter]
    pub fn num_requests(&self) -> usize {
        self.token_sequences.len()
    }

    /// Get all request IDs.
    pub fn get_all_request_ids(&self) -> Vec<String> {
        self.token_sequences.keys().cloned().collect()
    }

    /// Clear all requests.
    pub fn clear(&mut self) {
        self.token_sequences.clear();
        self.generated_counts.clear();
    }
}

impl Default for RequestStateManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Python module definition.
/// This module is imported as `vllm_metal._rs` in Python.
#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BlockAllocator>()?;
    m.add_class::<InputPreparer>()?;
    m.add_class::<RequestStateManager>()?;
    m.add_function(wrap_pyfunction!(compute_kv_block_indices, m)?)?;
    m.add_function(wrap_pyfunction!(batch_compute_kv_indices, m)?)?;
    m.add_function(wrap_pyfunction!(flatten_token_ids_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(compute_positions_numpy, m)?)?;
    Ok(())
}
