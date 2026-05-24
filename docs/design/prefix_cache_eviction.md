# RFC: `PrefixCacheManager` Eviction Policy

> Status: **Draft / discussion** &middot; Scope: `vllm_metal.v1.contiguous_cache.PrefixCacheManager` &middot; Author: @fxdv &middot; Companion PR: #393 (characterisation tests)

This document is an internal design note, not user-facing documentation. It is intentionally **not** registered in `mkdocs.yaml`.

## 1. Summary

`PrefixCacheManager` uses a `CachedPrefix.ref_count` integer as its eviction
priority key. The name and surrounding code (`restore_cache`, dataclass
default of `0`) strongly suggest *pin-during-use* semantics. The runtime
behavior is in fact a **monotonic hit counter**: it is initialised to `1`
on insert, incremented on every `lookup` hit, and **never decremented
anywhere in the codebase**. The resulting eviction policy is therefore
*LFU-forever*: prefixes that accumulate hits early in a run pin themselves
into the cache and cannot be displaced by later-popular prefixes.

This RFC documents the verified behavior, the resulting failure modes,
the trade-offs of three remediation options, and a recommended migration
path.

## 2. Blast radius

`PrefixCacheManager` is instantiated only by `MetalModelRunner.__init__`
under one condition (`vllm_metal/v1/model_runner.py:315-316`):

```python
self._prefix_cache: PrefixCacheManager | None = None
if _PREFIX_CACHE_ENABLED:
    self._prefix_cache = PrefixCacheManager(model_adapter=self._model_adapter)
```

`_PREFIX_CACHE_ENABLED` is evaluated at module import time from
`envs.VLLM_METAL_PREFIX_CACHE`, which is a **presence-only** check
(`vllm_metal/envs.py:68`). Once the manager exists, it is consumed in
exactly one call site &mdash; `MetalModelRunner._prefill_single` &mdash; which is
itself only reachable on the **legacy non-paged code path** (when
`self._paged_attention_backend is None`, i.e. when the user has set
`VLLM_METAL_USE_PAGED_ATTENTION=0`).

Therefore the affected configuration is:

- `VLLM_METAL_USE_PAGED_ATTENTION=0` (opt-out of the v0.2.0 default), **and**
- `VLLM_METAL_PREFIX_CACHE` set in the environment **before any
  `vllm_metal` import** (any value, including `"0"`, enables it).

The default v0.2.0 deployment is unaffected.

## 3. Verified current behavior

### 3.1 Hot-counter mechanics

| Event | Effect on `entry.ref_count` |
|---|---|
| `insert(new)` | created with value `1` |
| `lookup(existing)` hit | `+= 1` |
| `lookup(missing)` | no change |
| `restore_cache(entry)` | no change |
| request completion / engine shutdown | no change |

There is no `release`, `unpin`, `release_request`, or decrement of any
kind. Confirmed by repository-wide `grep` for `ref_count` (`vllm_metal/v1/contiguous_cache.py:131, 158, 178, 222` and one test fixture in `tests/test_vlm_forward_model.py:67`).

### 3.2 Eviction loop

```python
while self._current_bytes + needed_bytes > self._max_bytes and self._cache:
    min_hash, min_entry = min(self._cache.items(), key=lambda x: x[1].ref_count)
    self._current_bytes -= min_entry.size_bytes
    del self._cache[min_hash]
```

- One eviction per loop iteration.
- The eviction selector walks the entire dict on every iteration (`O(N)` per
  step; `O(k·N)` total to free `k` slots).
- Under ties, `min()` returns the first matching element in dict iteration
  order &mdash; insertion order in CPython 3.7+. So ties are broken FIFO (oldest
  loses).

### 3.3 Pathological steady-state

Take a cache that fits two entries. Workload: one hot system prompt `A`,
then a stream of cold prefixes `B, C, D, ...`.

1. Request 1 (`A`): miss &rarr; `insert(A)` &rarr; `{A: rc=1}`.
2. Requests 2&ndash;101 (`A` &times; 100): hits &rarr; `{A: rc=101}`.
3. Request 102 (`B`): miss &rarr; `insert(B)` (no eviction yet, capacity = 2)
   &rarr; `{A: 101, B: 1}`.
4. Request 103 (`C`): miss &rarr; eviction selects `B` (`rc=1`), `B` evicted,
   `insert(C)` &rarr; `{A: 101, C: 1}`.
5. Request 104 (`B` again): miss (re-inserted from scratch, full forward
   pass) &rarr; eviction selects `C` (`rc=1`) &rarr; `{A: 101, B: 1}`.
6. &hellip; ad infinitum. `A` is pinned forever; the remaining slot rotates
   through every new prefix.

The cache asymptotes to "one frozen early winner plus N&minus;1 rotating slots
that never accumulate." A new prefix that becomes hot later in the run
needs to land at least N&minus;1 hits before its first eviction-round
participation, otherwise it is replaced by the next miss.

## 4. Why it matters

1. **Wasted GPU work.** Every miss on a recently-evicted prefix pays a full
   `_forward_model` prefill cost. Under shifting workloads this defeats the
   feature's purpose.
2. **Misleading name invites a UAF-introducing "fix".** The next reader is
   very likely to add `cached.ref_count -= 1` in `_cleanup_finished_requests`
   to implement what the name suggests. Today that would not crash because
   `restore_cache` calls `mx.array(k)` / `mx.array(v)` (an MLX-side copy at
   `vllm_metal/v1/contiguous_cache.py:240`), so live requests do not alias
   evicted buffers. That property is undocumented; a future "zero-copy
   optimization" combined with the natural-looking rename would introduce a
   use-after-free between request handlers and the eviction loop.
3. **Policy is fully untested.** `tests/test_prefix_cache.py::TestPrefixCacheEviction`
   asserts only `len(_cache)` and `_current_bytes` post-eviction; it never
   checks *which* entry was evicted. The current test suite would pass
   under LRU, LFU, FIFO, random, or LFU-forever indistinguishably.
4. **`O(N)` eviction selector.** Acceptable at &le; tens of entries, painful
   at hundreds &mdash; particularly because the loop runs inside `insert`, on
   the prefill latency path.

The first two are the real concerns. The third is addressed by the tests
added alongside this RFC (see &sect;7).

## 5. Options

### Option A &mdash; Honest rename, keep semantics

Rename `ref_count` &rarr; `hit_count`. Update the docstring on `CachedPrefix`
to say:

> `hit_count` is a monotonically non-decreasing eviction priority key
> updated on every `lookup` hit. Entries are **not** pinned by in-flight
> use; the `mx.array` copies inside `restore_cache` make eviction safe
> while requests are still consuming the restored cache.

**Pros**

- Minimal diff. No behavior change. No risk of regression.
- The footgun (someone adding a misleading decrement) is closed by the
  name and docstring.

**Cons**

- Preserves the LFU-forever pathology. Cache still under-performs on
  shifting workloads.
- Still `O(N)` per eviction.

### Option B &mdash; Switch to LRU via `OrderedDict`

Replace the dict with `collections.OrderedDict`. On `lookup` hit, call
`self._cache.move_to_end(prefix_hash)`. In `_evict_until_fits`, call
`self._cache.popitem(last=False)` to evict the oldest entry. Drop
`ref_count` entirely.

**Pros**

- `O(1)` per `lookup` and per eviction.
- Matches the *de facto* contract users expect from a prefix cache.
- Eviction policy is locally well-known and testable.

**Cons**

- Changes user-visible behavior: hot-but-old prefixes that have not been
  hit recently can now be evicted. For workloads where the original
  policy happened to do the right thing (a single dominant system
  prompt), there is no regression because that prompt is constantly hit.
- Requires re-running benchmarks to confirm we are not regressing the
  one configuration this matters for. The existing `_prefill_single`
  serving path has no benchmark coverage; we would add one.

### Option C &mdash; Hybrid LRU + pin counter

Add a real `pin_count` to `CachedPrefix`, incremented inside `restore_cache`
and decremented from a new `release(req_id)` API called by
`_cleanup_finished_requests`. Use LRU for recency, but skip pinned entries
in eviction.

**Pros**

- Strongest correctness story. Eliminates "evict-while-in-use" races
  forever, even if `restore_cache` is ever switched to a zero-copy
  alias.

**Cons**

- Largest surface area. Requires wiring through request lifecycle
  (`_cleanup_finished_requests` does not today track which requests
  consumed which prefixes).
- Today's `_prefill_single` path is single-threaded per worker and the
  `mx.array` copy in `restore_cache` already isolates eviction from
  in-flight requests; the extra machinery is not load-bearing for the
  current code.
- Pin leaks (forgotten `release`) silently disable eviction on the
  affected entry. New class of bug.

## 6. Recommendation

**Adopt Option A immediately.** It is a one-commit fix that closes the
UAF-on-future-fix footgun and aligns the code's vocabulary with its
behavior. The remediation cost is approximately one rename plus a one-line
comment in `restore_cache`.

**Defer Option B** until we have a numbers-driven motivation. The
non-paged path is opt-out; the prefix cache is opt-in within it. Until
someone reports a real workload where eviction order matters, the
intersection is too small to justify changing the steady-state semantics.

**Do not implement Option C** unless we (a) move `_prefill_single` to a
zero-copy `restore_cache`, or (b) introduce concurrency on the legacy
path. Today's contract already guarantees that evicted buffers are not
aliased by live requests, so the additional machinery would be
load-bearing for nothing.

## 7. Test plan

Landing in the companion PR (#393) as
`tests/test_prefix_cache.py::TestPrefixCacheEvictionPolicy`:

- `test_insert_initializes_ref_count_to_one`
- `test_lookup_hit_increments_ref_count`
- `test_lookup_miss_does_not_touch_ref_count`
- `test_restore_cache_does_not_decrement_ref_count`
- `test_eviction_drops_lowest_ref_count_entry`
- `test_hot_prefix_pins_out_cold_prefixes`
- `test_ref_count_is_unbounded_monotonic`

These pin the current behavior so that any future eviction-policy change
shows up as an explicit test update. If we adopt **Option A** (rename
only), these tests should be updated to reference `hit_count` and the
docstrings adjusted; no semantic change is required.

If we ever adopt **Option B**, the LFU-forever tests must be replaced by
their LRU equivalents:

| LFU-forever test (today) | LRU replacement (Option B) |
|---|---|
| `test_eviction_drops_lowest_ref_count_entry` | `test_eviction_drops_least_recently_used_entry` |
| `test_hot_prefix_pins_out_cold_prefixes` | `test_recently_used_prefix_survives_eviction` |
| `test_ref_count_is_unbounded_monotonic` | *delete* (field removed) |
| `test_lookup_hit_increments_ref_count` | *replace with* `test_lookup_hit_moves_entry_to_mru` |

## 8. Migration notes (Option A)

1. Rename `ref_count` &rarr; `hit_count` in `CachedPrefix`, in `lookup`, in
   `insert`, and in `_evict_until_fits`.
2. Update the two test files (`tests/test_prefix_cache.py`,
   `tests/test_vlm_forward_model.py`) to match.
3. Add a comment to `restore_cache` explaining that the `mx.array(k)` /
   `mx.array(v)` copies are load-bearing for eviction safety.
4. Add a short note to `docs/configuration.md` (or wherever
   `VLLM_METAL_PREFIX_CACHE` is documented) clarifying that eviction is
   LFU-by-cumulative-hits and that the cache is intentionally biased
   toward long-lived shared system prompts. *(The current docs do not
   document the eviction policy at all.)*

## 9. Out of scope

- The presence-only enable check on `VLLM_METAL_PREFIX_CACHE` (`H5` in
  the review &mdash; setting `=0` still enables it). Tracked separately.
- The `array("I", token_ids).tobytes()` hash on platforms where C `unsigned int`
  is 2 bytes. Not observed on macOS arm64; tracked separately if we ever
  add Linux dev support.
- The `_prefill_single` non-paged code path itself. The longer-term plan
  is to retire it; that decision is independent of the eviction policy
  choice.
