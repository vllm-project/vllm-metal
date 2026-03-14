// SPDX-License-Identifier: Apache-2.0
// Minimal C FFI shim for MLX Metal operations used by paged_ops.rs.
//
// This header exposes a C-linkage API around the subset of MLX's C++ API
// needed for paged attention dispatch: library caching, command encoding,
// buffer binding, and kernel dispatch.

#ifndef MLX_FFI_H
#define MLX_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

/// Opaque handle to an mlx::core::array.
typedef void* MlxArrayHandle;

// ---------------------------------------------------------------------------
// Library / kernel caching
// ---------------------------------------------------------------------------

/// Register and JIT-compile the reshape_cache and paged_attention Metal
/// shader libraries.  Must be called once before dispatching kernels.
void mlx_ffi_init_libraries(const char* reshape_src, const char* paged_attn_src);

/// Register and JIT-compile the v2 online-softmax Metal shader library.
void mlx_ffi_init_v2_library(const char* v2_src);

// ---------------------------------------------------------------------------
// reshape_and_cache dispatch
// ---------------------------------------------------------------------------

/// Dispatch the reshape_and_cache Metal kernel.
///
/// All MlxArrayHandle arguments must be pointers obtained via
/// `nb::inst_ptr<array>()` or the PyO3 equivalent extraction.
void mlx_ffi_reshape_and_cache(
    MlxArrayHandle key,
    MlxArrayHandle value,
    MlxArrayHandle key_cache,
    MlxArrayHandle value_cache,
    MlxArrayHandle slot_mapping
);

// ---------------------------------------------------------------------------
// paged_attention_v1 dispatch
// ---------------------------------------------------------------------------

void mlx_ffi_paged_attention_v1(
    MlxArrayHandle out,
    MlxArrayHandle query,
    MlxArrayHandle key_cache,
    MlxArrayHandle value_cache,
    int32_t num_kv_heads,
    float scale,
    MlxArrayHandle block_tables,
    MlxArrayHandle seq_lens,
    int32_t block_size,
    int32_t max_seq_len
);

// ---------------------------------------------------------------------------
// paged_attention_v2_online dispatch
// ---------------------------------------------------------------------------

void mlx_ffi_paged_attention_v2_online(
    MlxArrayHandle out,
    MlxArrayHandle query,
    MlxArrayHandle key_cache,
    MlxArrayHandle value_cache,
    int32_t num_kv_heads,
    float scale,
    MlxArrayHandle block_tables,
    MlxArrayHandle seq_lens,
    int32_t block_size,
    int32_t max_seq_len
);

// ---------------------------------------------------------------------------
// Utility: extract mlx::core::array* from a Python object pointer.
//
// The caller passes the raw `PyObject*`; this function uses nanobind's
// `nb::inst_ptr<array>()` to resolve the underlying C++ object, returning
// an opaque handle that the dispatch functions above accept.
// ---------------------------------------------------------------------------

MlxArrayHandle mlx_ffi_extract_array(void* py_object);

#ifdef __cplusplus
}
#endif

#endif // MLX_FFI_H
