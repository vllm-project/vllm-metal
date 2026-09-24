// SPDX-License-Identifier: Apache-2.0
//! ggml backend bring-up: dynamic backend loading, Metal + CPU scheduler.

use crate::ffi::*;
use anyhow::{bail, Result};
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;
use std::sync::Once;

static LOAD_BACKENDS: Once = Once::new();

unsafe extern "C" fn log_callback(level: ggml_log_level, text: *const c_char, _: *mut c_void) {
    let verbose = std::env::var("VLLM_METAL_GGML_VERBOSE")
        .map(|v| v == "1")
        .unwrap_or(false);
    if (verbose || level >= ggml_log_level_GGML_LOG_LEVEL_WARN) && !text.is_null() {
        eprint!("{}", CStr::from_ptr(text).to_string_lossy());
    }
}

fn backend_dir() -> String {
    std::env::var("GGML_BACKEND_DIR")
        .unwrap_or_else(|_| env!("GGML_BACKEND_DIR_DEFAULT").to_string())
}

pub fn ensure_backends_loaded() {
    LOAD_BACKENDS.call_once(|| unsafe {
        ggml_log_set(Some(log_callback), ptr::null_mut());
        let dir = CString::new(backend_dir()).unwrap();
        ggml_backend_load_all_from_path(dir.as_ptr());
    });
}

pub struct Backend {
    /// Primary compute backend (Metal, or CPU when forced).
    pub main: ggml_backend_t,
    /// CPU backend used by the scheduler as a fallback for unsupported ops.
    pub cpu: ggml_backend_t,
    pub sched: ggml_backend_sched_t,
    pub is_gpu: bool,
}

// The engine serializes all access behind a Mutex; ggml handles themselves
// are not tied to a thread.
unsafe impl Send for Backend {}

impl Backend {
    pub fn new(graph_size: usize, force_cpu: bool) -> Result<Self> {
        ensure_backends_loaded();
        unsafe {
            let cpu = ggml_backend_init_by_type(
                ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU,
                ptr::null(),
            );
            if cpu.is_null() {
                bail!(
                    "failed to initialize the ggml CPU backend (backend dir: {})",
                    backend_dir()
                );
            }
            let mut main = ptr::null_mut();
            if !force_cpu {
                main = ggml_backend_init_by_type(
                    ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU,
                    ptr::null(),
                );
                if main.is_null() {
                    main = ggml_backend_init_by_type(
                        ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU,
                        ptr::null(),
                    );
                }
                if main.is_null() {
                    bail!(
                        "no ggml GPU (Metal) backend found in {} (set GGML_BACKEND_DIR)",
                        backend_dir()
                    );
                }
            }
            let is_gpu = !main.is_null();
            let mut backends = if is_gpu { vec![main, cpu] } else { vec![cpu] };
            if !is_gpu {
                main = cpu;
            }
            let sched = ggml_backend_sched_new(
                backends.as_mut_ptr(),
                ptr::null_mut(),
                backends.len() as i32,
                graph_size,
                false,
                is_gpu,
            );
            if sched.is_null() {
                bail!("ggml_backend_sched_new failed");
            }
            Ok(Self {
                main,
                cpu,
                sched,
                is_gpu,
            })
        }
    }

    pub fn name(&self) -> String {
        unsafe {
            CStr::from_ptr(ggml_backend_name(self.main))
                .to_string_lossy()
                .into_owned()
        }
    }

    /// (free, total) bytes as reported by the primary device.
    pub fn memory(&self) -> (usize, usize) {
        let mut free = 0usize;
        let mut total = 0usize;
        unsafe {
            let dev = ggml_backend_get_device(self.main);
            if !dev.is_null() {
                ggml_backend_dev_memory(dev, &mut free, &mut total);
            }
        }
        (free, total)
    }

    /// Allocate every tensor of `ctx` in one buffer on the primary backend.
    pub fn alloc_ctx(
        &self,
        ctx: *mut ggml_context,
        usage_weights: bool,
    ) -> Result<ggml_backend_buffer_t> {
        unsafe {
            let buf = ggml_backend_alloc_ctx_tensors(ctx, self.main);
            if buf.is_null() {
                bail!("failed to allocate ggml backend buffer");
            }
            if usage_weights {
                ggml_backend_buffer_set_usage(
                    buf,
                    ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
                );
            }
            Ok(buf)
        }
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        unsafe {
            ggml_backend_sched_free(self.sched);
            if self.is_gpu {
                ggml_backend_free(self.main);
            }
            ggml_backend_free(self.cpu);
        }
    }
}
