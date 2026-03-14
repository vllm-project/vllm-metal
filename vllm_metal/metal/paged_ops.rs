// SPDX-License-Identifier: Apache-2.0
//! Rust + PyO3 bridge for paged attention Metal kernels.
//!
//! Dispatches reshape_and_cache and paged_attention_{v1,v2} through MLX's
//! Metal command encoder via a thin C FFI shim (mlx_ffi.cpp), eliminating
//! the nanobind dependency.
//!
//! This is a 1:1 Rust rewrite of paged_ops.cpp.

use pyo3::prelude::*;
use pyo3::ffi::PyObject;
use std::ffi::CString;

// ---------------------------------------------------------------------------
// FFI declarations — links against mlx_ffi.cpp
// ---------------------------------------------------------------------------

type MlxArrayHandle = *mut std::ffi::c_void;

extern "C" {
    fn mlx_ffi_init_libraries(reshape_src: *const i8, paged_attn_src: *const i8);
    fn mlx_ffi_init_v2_library(v2_src: *const i8);

    fn mlx_ffi_extract_array(py_object: *mut std::ffi::c_void) -> MlxArrayHandle;

    fn mlx_ffi_reshape_and_cache(
        key: MlxArrayHandle,
        value: MlxArrayHandle,
        key_cache: MlxArrayHandle,
        value_cache: MlxArrayHandle,
        slot_mapping: MlxArrayHandle,
    );

    fn mlx_ffi_paged_attention_v1(
        out: MlxArrayHandle,
        query: MlxArrayHandle,
        key_cache: MlxArrayHandle,
        value_cache: MlxArrayHandle,
        num_kv_heads: i32,
        scale: f32,
        block_tables: MlxArrayHandle,
        seq_lens: MlxArrayHandle,
        block_size: i32,
        max_seq_len: i32,
    );

    fn mlx_ffi_paged_attention_v2_online(
        out: MlxArrayHandle,
        query: MlxArrayHandle,
        key_cache: MlxArrayHandle,
        value_cache: MlxArrayHandle,
        num_kv_heads: i32,
        scale: f32,
        block_tables: MlxArrayHandle,
        seq_lens: MlxArrayHandle,
        block_size: i32,
        max_seq_len: i32,
    );
}

// ---------------------------------------------------------------------------
// Helper: extract the underlying mlx::core::array* from a Python mlx.core.array
// ---------------------------------------------------------------------------

/// Extract the C++ `mlx::core::array*` from a Python `mlx.core.array` object,
/// returning an opaque handle for use with the FFI dispatch functions.
///
/// # Safety
/// The returned handle is only valid as long as the Python object is alive.
/// MLX's `add_temporary` in the dispatch functions ensures the GPU retains
/// references until the command buffer completes.
#[inline]
unsafe fn extract_array(obj: &Bound<'_, PyAny>) -> MlxArrayHandle {
    let py_obj: *mut PyObject = obj.as_ptr();
    mlx_ffi_extract_array(py_obj as *mut std::ffi::c_void)
}

// ---------------------------------------------------------------------------
// Python-exposed functions
// ---------------------------------------------------------------------------

/// JIT-compile the vendored Metal shaders for reshape_cache and paged_attention.
#[pyfunction]
fn init_libraries(reshape_src: &str, paged_attn_src: &str) -> PyResult<()> {
    let reshape_c = CString::new(reshape_src)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let paged_c = CString::new(paged_attn_src)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    unsafe {
        mlx_ffi_init_libraries(reshape_c.as_ptr(), paged_c.as_ptr());
    }
    Ok(())
}

/// JIT-compile the v2 online-softmax Metal shader.
#[pyfunction]
fn init_v2_library(v2_src: &str) -> PyResult<()> {
    let v2_c = CString::new(v2_src)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    unsafe {
        mlx_ffi_init_v2_library(v2_c.as_ptr());
    }
    Ok(())
}

/// Write projected K/V into the paged cache.
#[pyfunction]
fn reshape_and_cache(
    key: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
    key_cache: &Bound<'_, PyAny>,
    value_cache: &Bound<'_, PyAny>,
    slot_mapping: &Bound<'_, PyAny>,
) -> PyResult<()> {
    unsafe {
        let key_h = extract_array(key);
        let value_h = extract_array(value);
        let key_cache_h = extract_array(key_cache);
        let value_cache_h = extract_array(value_cache);
        let slot_mapping_h = extract_array(slot_mapping);

        mlx_ffi_reshape_and_cache(
            key_h,
            value_h,
            key_cache_h,
            value_cache_h,
            slot_mapping_h,
        );
    }
    Ok(())
}

/// Zero-copy paged attention (v1, no partitioning).
#[pyfunction]
fn paged_attention_v1(
    out: &Bound<'_, PyAny>,
    query: &Bound<'_, PyAny>,
    key_cache: &Bound<'_, PyAny>,
    value_cache: &Bound<'_, PyAny>,
    num_kv_heads: i32,
    scale: f32,
    block_tables: &Bound<'_, PyAny>,
    seq_lens: &Bound<'_, PyAny>,
    block_size: i32,
    max_seq_len: i32,
) -> PyResult<()> {
    unsafe {
        let out_h = extract_array(out);
        let query_h = extract_array(query);
        let key_cache_h = extract_array(key_cache);
        let value_cache_h = extract_array(value_cache);
        let block_tables_h = extract_array(block_tables);
        let seq_lens_h = extract_array(seq_lens);

        mlx_ffi_paged_attention_v1(
            out_h,
            query_h,
            key_cache_h,
            value_cache_h,
            num_kv_heads,
            scale,
            block_tables_h,
            seq_lens_h,
            block_size,
            max_seq_len,
        );
    }
    Ok(())
}

/// Online-softmax paged attention (v2, decode-only).
#[pyfunction]
fn paged_attention_v2_online(
    out: &Bound<'_, PyAny>,
    query: &Bound<'_, PyAny>,
    key_cache: &Bound<'_, PyAny>,
    value_cache: &Bound<'_, PyAny>,
    num_kv_heads: i32,
    scale: f32,
    block_tables: &Bound<'_, PyAny>,
    seq_lens: &Bound<'_, PyAny>,
    block_size: i32,
    max_seq_len: i32,
) -> PyResult<()> {
    unsafe {
        let out_h = extract_array(out);
        let query_h = extract_array(query);
        let key_cache_h = extract_array(key_cache);
        let value_cache_h = extract_array(value_cache);
        let block_tables_h = extract_array(block_tables);
        let seq_lens_h = extract_array(seq_lens);

        mlx_ffi_paged_attention_v2_online(
            out_h,
            query_h,
            key_cache_h,
            value_cache_h,
            num_kv_heads,
            scale,
            block_tables_h,
            seq_lens_h,
            block_size,
            max_seq_len,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

/// Python module: _paged_ops (Rust rewrite)
#[pymodule]
fn _paged_ops(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_libraries, m)?)?;
    m.add_function(wrap_pyfunction!(init_v2_library, m)?)?;
    m.add_function(wrap_pyfunction!(reshape_and_cache, m)?)?;
    m.add_function(wrap_pyfunction!(paged_attention_v1, m)?)?;
    m.add_function(wrap_pyfunction!(paged_attention_v2_online, m)?)?;
    Ok(())
}
