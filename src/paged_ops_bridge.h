// SPDX-License-Identifier: Apache-2.0
// C ABI bridge for MLX paged-attention Metal kernels.
//
// All functions use extern "C" linkage so they can be called from Rust FFI.
// Array arguments are raw PyObject* pointers; the .cpp implementation uses
// nanobind only to recover the underlying mlx::core::array* from those
// existing Python objects (no new Python types are registered here).

#pragma once

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

/// JIT-compile and cache the reshape-and-cache + v1 attention Metal libraries.
void paged_ops_init_libraries(const char* reshape_src,
                              const char* paged_attn_src);

/// JIT-compile and cache the v2 online-softmax attention Metal library.
void paged_ops_init_v2_library(const char* v2_src);

/// Write projected K/V tokens into the paged KV cache.
void paged_ops_reshape_and_cache(PyObject* key,
                                 PyObject* value,
                                 PyObject* key_cache,
                                 PyObject* value_cache,
                                 PyObject* slot_mapping);

/// Zero-copy paged attention v1 (no partitioning, full softmax in one pass).
void paged_ops_attention_v1(PyObject* out,
                             PyObject* query,
                             PyObject* key_cache,
                             PyObject* value_cache,
                             int num_kv_heads,
                             float scale,
                             PyObject* block_tables,
                             PyObject* seq_lens,
                             int block_size,
                             int max_seq_len);

/// Online-softmax paged attention v2 (decode-only).
void paged_ops_attention_v2_online(PyObject* out,
                                   PyObject* query,
                                   PyObject* key_cache,
                                   PyObject* value_cache,
                                   int num_kv_heads,
                                   float scale,
                                   PyObject* block_tables,
                                   PyObject* seq_lens,
                                   int block_size,
                                   int max_seq_len);

#ifdef __cplusplus
}
#endif
