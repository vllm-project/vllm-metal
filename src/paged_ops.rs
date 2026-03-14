// SPDX-License-Identifier: Apache-2.0
//! Rust PyO3 bindings for the paged-attention Metal kernels.
//!
//! All heavy lifting is delegated to the C ABI bridge in `paged_ops_bridge.cpp`
//! which calls MLX's Metal command encoder directly.  This module owns only the
//! Python-facing interface: argument unpacking and the `#[pymodule]` export.

use pyo3::prelude::*;
use pyo3::ffi::PyObject;

// ---------------------------------------------------------------------------
// Raw C bridge declarations
// ---------------------------------------------------------------------------

extern "C" {
    fn paged_ops_init_libraries(reshape_src: *const std::ffi::c_char,
                                paged_attn_src: *const std::ffi::c_char);
    fn paged_ops_init_v2_library(v2_src: *const std::ffi::c_char);

    fn paged_ops_reshape_and_cache(key: *mut PyObject,
                                   value: *mut PyObject,
                                   key_cache: *mut PyObject,
                                   value_cache: *mut PyObject,
                                   slot_mapping: *mut PyObject);

    fn paged_ops_attention_v1(out: *mut PyObject,
                               query: *mut PyObject,
                               key_cache: *mut PyObject,
                               value_cache: *mut PyObject,
                               num_kv_heads: std::ffi::c_int,
                               scale: f32,
                               block_tables: *mut PyObject,
                               seq_lens: *mut PyObject,
                               block_size: std::ffi::c_int,
                               max_seq_len: std::ffi::c_int);

    fn paged_ops_attention_v2_online(out: *mut PyObject,
                                     query: *mut PyObject,
                                     key_cache: *mut PyObject,
                                     value_cache: *mut PyObject,
                                     num_kv_heads: std::ffi::c_int,
                                     scale: f32,
                                     block_tables: *mut PyObject,
                                     seq_lens: *mut PyObject,
                                     block_size: std::ffi::c_int,
                                     max_seq_len: std::ffi::c_int);
}

// ---------------------------------------------------------------------------
// Python-exposed functions
// ---------------------------------------------------------------------------

/// JIT-compile the reshape-and-cache + v1 paged-attention Metal libraries.
#[pyfunction]
pub fn init_libraries(reshape_src: &str, paged_attn_src: &str) -> PyResult<()> {
    let reshape_cstr = std::ffi::CString::new(reshape_src)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let paged_cstr = std::ffi::CString::new(paged_attn_src)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    unsafe {
        paged_ops_init_libraries(reshape_cstr.as_ptr(), paged_cstr.as_ptr());
    }
    Ok(())
}

/// JIT-compile the v2 online-softmax Metal library.
#[pyfunction]
pub fn init_v2_library(v2_src: &str) -> PyResult<()> {
    let cstr = std::ffi::CString::new(v2_src)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    unsafe {
        paged_ops_init_v2_library(cstr.as_ptr());
    }
    Ok(())
}

/// Write projected K/V tokens into the paged KV cache.
#[pyfunction]
pub fn reshape_and_cache<'py>(
    key: Bound<'py, PyAny>,
    value: Bound<'py, PyAny>,
    key_cache: Bound<'py, PyAny>,
    value_cache: Bound<'py, PyAny>,
    slot_mapping: Bound<'py, PyAny>,
) -> PyResult<()> {
    unsafe {
        paged_ops_reshape_and_cache(
            key.as_ptr(),
            value.as_ptr(),
            key_cache.as_ptr(),
            value_cache.as_ptr(),
            slot_mapping.as_ptr(),
        );
    }
    Ok(())
}

/// Zero-copy paged attention v1 (no partitioning).
#[pyfunction]
pub fn paged_attention_v1<'py>(
    out: Bound<'py, PyAny>,
    query: Bound<'py, PyAny>,
    key_cache: Bound<'py, PyAny>,
    value_cache: Bound<'py, PyAny>,
    num_kv_heads: i32,
    scale: f32,
    block_tables: Bound<'py, PyAny>,
    seq_lens: Bound<'py, PyAny>,
    block_size: i32,
    max_seq_len: i32,
) -> PyResult<()> {
    unsafe {
        paged_ops_attention_v1(
            out.as_ptr(),
            query.as_ptr(),
            key_cache.as_ptr(),
            value_cache.as_ptr(),
            num_kv_heads,
            scale,
            block_tables.as_ptr(),
            seq_lens.as_ptr(),
            block_size,
            max_seq_len,
        );
    }
    Ok(())
}

/// Online-softmax paged attention v2 (decode-only).
#[pyfunction]
pub fn paged_attention_v2_online<'py>(
    out: Bound<'py, PyAny>,
    query: Bound<'py, PyAny>,
    key_cache: Bound<'py, PyAny>,
    value_cache: Bound<'py, PyAny>,
    num_kv_heads: i32,
    scale: f32,
    block_tables: Bound<'py, PyAny>,
    seq_lens: Bound<'py, PyAny>,
    block_size: i32,
    max_seq_len: i32,
) -> PyResult<()> {
    unsafe {
        paged_ops_attention_v2_online(
            out.as_ptr(),
            query.as_ptr(),
            key_cache.as_ptr(),
            value_cache.as_ptr(),
            num_kv_heads,
            scale,
            block_tables.as_ptr(),
            seq_lens.as_ptr(),
            block_size,
            max_seq_len,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sub-module registration (called from the top-level _rs module)
// ---------------------------------------------------------------------------

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "paged_ops")?;
    m.add_function(wrap_pyfunction!(init_libraries, &m)?)?;
    m.add_function(wrap_pyfunction!(init_v2_library, &m)?)?;
    m.add_function(wrap_pyfunction!(reshape_and_cache, &m)?)?;
    m.add_function(wrap_pyfunction!(paged_attention_v1, &m)?)?;
    m.add_function(wrap_pyfunction!(paged_attention_v2_online, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
