// SPDX-License-Identifier: Apache-2.0
//! High-performance Rust extension for vLLM Metal.
//!
//! This module provides the `RequestStateManager`, an optimized implementation
//! for managing token sequences and request state with O(1) operations.

use pyo3::prelude::*;
use std::collections::{HashMap};


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
    m.add_class::<RequestStateManager>()?;
    Ok(())
}
