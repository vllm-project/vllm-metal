// SPDX-License-Identifier: Apache-2.0
//! Raw bindgen output for ggml / ggml-backend / ggml-alloc.
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/ggml_bindings.rs"));
