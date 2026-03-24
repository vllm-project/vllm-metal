// SPDX-License-Identifier: Apache-2.0
//! Rust extension for vLLM Metal.
//!
//! This module provides the PyO3 build skeleton for future
//! performance-critical Rust extensions.

use pyo3::prelude::*;

#[pymodule]
fn _rs(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
