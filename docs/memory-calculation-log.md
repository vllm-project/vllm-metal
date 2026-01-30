# Memory Calculation Log Analysis

## Log Output (Qwen/Qwen3-0.6B, max_model_len=2048)

### 1. System Memory Info
```
[Memory] System: total=32.00GB, available=13.26GB
```
- **Source**: `worker.py:110` (`init_device`)
- **Logic**: `psutil.virtual_memory().total` and `.available`

### 2. Metal Device Constraints
```
[Memory] Metal max_buffer_size=18.72GB
```
- **Source**: `worker.py:128` (`init_device`)
- **Logic**: `mx.metal.device_info().get("max_buffer_length", 0)`
- **Note**: This is the maximum single buffer allocation allowed by Metal

### 3. Initial MLX Memory Limit
```
[Memory] Initial MLX memory limit set to 11.23GB (60% of max_buffer_size)
```
- **Source**: `worker.py:138` (`init_device`)
- **Logic**: `memory_limit = int(max_buffer_size * 0.60)`
- **Calculation**: `18.72GB * 0.60 = 11.23GB`

### 4. Model Memory After Loading
```
[Memory] Auto mode calculation: total_system=32.00GB, model_loaded=1.11GB
```
- **Source**: `worker.py:270` (`_set_auto_memory_limit`)
- **Logic**: `mx.get_active_memory()` after model load

### 5. KV Cache Parameters
```
[Memory] KV cache params: block_size=16 tokens (1835008 bytes), max_model_len=2048, blocks_per_seq=128, max_num_seqs=256
```
- **Source**: `worker.py:287` (`_set_auto_memory_limit`)
- **Calculation**:
  - `block_size_tokens = 16` (from config)
  - `block_size_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size`
  - `block_size_bytes = 2 * 28 * 16 * 8 * 128 * 2 = 1,835,008 bytes` (~1.75MB per block)
  - `blocks_per_seq = ceil(max_model_len / block_size) = ceil(2048 / 16) = 128`
  - `max_num_seqs = 256` (from scheduler config)

### 6. Auto Mode MLX Memory Limit
```
Auto mode: set MLX memory limit to 12.06GB (model=1.19GB, kv_cache=16.54GB, max_num_seqs=256)
```
- **Source**: `worker.py:346` (`_set_auto_memory_limit`)
- **Logic**:
  - Calculated limit: `(model_memory + kv_cache_memory) * 1.05`
  - But capped at `max_buffer_size * 0.60` if exceeds 90% of max_buffer_size
  - Final: `12.06GB`

### 7. Available Memory for KV Cache
```
Auto memory mode: model=1.19GB, max_model_len=2048, max_num_seqs=256, blocks_per_seq=128, target_blocks=9011, cache_memory=14.93GB
Metal available memory for KV cache: 14.93 GB
```
- **Source**: `worker.py:427, 446` (`determine_available_memory`)
- **Calculation**:
  - `avg_blocks_per_seq = blocks_per_seq // 4 = 128 // 4 = 32`
  - `target_blocks = max_num_seqs * avg_blocks_per_seq * 1.1 = 256 * 32 * 1.1 = 9,011`
  - `available_for_cache = (total_memory - model_memory) * 0.45`
  - `cache_memory = min(target_cache_memory, available_for_cache) = 14.93GB`

### 8. KV Cache Allocation
```
[Memory] KV cache allocation: num_blocks=8133, shape=(8133, 28, 2, 16, 8, 128), total=13.90GB
```
- **Source**: `cache.py:175` (`PagedKVCache.__init__`)
- **Calculation**:
  - `num_blocks = 8133` (calculated by vLLM core from available memory)
  - `shape = (num_blocks, num_layers, 2, block_size, num_kv_heads, head_dim)`
  - `shape = (8133, 28, 2, 16, 8, 128)`
  - `total_bytes = 8133 * 28 * 2 * 16 * 8 * 128 * 2 = 14,927,056,896 bytes = 13.90GB`

### 9. Final Cache Ready
```
[Memory] PagedKVCache ready: 8133 blocks (13.90GB), block_size=16, layers=28, kv_heads=8, head_dim=128
```
- **Source**: `model_runner.py:573` (`initialize_kv_cache`)

---

## Summary Table

| Metric | Value |
|--------|-------|
| System Total Memory | 32.00 GB |
| System Available Memory | 13.26 GB |
| Metal max_buffer_length | 18.72 GB |
| Initial MLX Memory Limit | 11.23 GB (60% of max_buffer) |
| Model Memory | 1.11-1.19 GB |
| Max Model Length | 2048 tokens |
| Block Size | 16 tokens |
| Block Size (bytes) | 1.75 MB |
| Blocks per Sequence | 128 |
| Max Concurrent Sequences | 256 |
| Target Blocks | 9,011 |
| Actual Blocks Allocated | 8,133 |
| KV Cache Memory | 13.90 GB |
| Final MLX Memory Limit | 12.06 GB |

---

## Memory Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     SYSTEM MEMORY (32GB)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Metal max_buffer_length (18.72GB)           │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │         MLX Memory Limit (12.06GB)                 │  │  │
│  │  │  ┌──────────┐  ┌─────────────────────────────────┐ │  │  │
│  │  │  │  Model   │  │      KV Cache (13.90GB)         │ │  │  │
│  │  │  │ (1.19GB) │  │      8133 blocks                │ │  │  │
│  │  │  └──────────┘  └─────────────────────────────────┘ │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Available for other apps: ~13GB                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Code Locations

| Log Message | File | Line | Function |
|-------------|------|------|----------|
| System memory | `worker.py` | 110 | `init_device` |
| max_buffer_length | `worker.py` | 128 | `init_device` |
| Initial MLX limit | `worker.py` | 138 | `init_device` |
| Auto mode calculation | `worker.py` | 270 | `_set_auto_memory_limit` |
| KV cache params | `worker.py` | 287 | `_set_auto_memory_limit` |
| Auto MLX memory limit | `worker.py` | 346 | `_set_auto_memory_limit` |
| Available memory | `worker.py` | 427, 446 | `determine_available_memory` |
| Cache allocation | `cache.py` | 175 | `PagedKVCache.__init__` |
| Cache ready | `model_runner.py` | 573 | `initialize_kv_cache` |

---

## Bug Fixes Applied

1. **max_buffer_size key name** (Fixed): Changed from `max_buffer_size` to `max_buffer_length` to match MLX API.
   - Files fixed: `utils.py`, `v1/worker.py`, `v1/model_runner.py`, `pytorch_backend/tensor_bridge.py`
   - Before: `0 GB` (key not found)
   - After: `18.72 GB` (correct value)

---

## Model-Specific Parameters (Qwen3-0.6B)

| Parameter | Value |
|-----------|-------|
| num_layers | 28 |
| num_kv_heads | 8 |
| head_dim | 128 |
| hidden_size | 1024 |
| dtype | float16 (2 bytes) |
