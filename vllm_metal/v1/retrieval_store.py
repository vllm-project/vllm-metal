# SPDX-License-Identifier: Apache-2.0
"""Bounded memory of past tool invocations, for retrieval-augmented drafting.

This is the store half of ToolSpec (arXiv 2604.13519). The grammar proposer
drafts the text a request's schema has *already* determined and stops at every
value; measured, that covers 37% of emitted tokens over API-Bank and 50% over
BFCL, and the rest is argument *content* it never touches. ToolSpec's
observation is that
this content is not arbitrary either: tool-calling traffic repeats itself across
requests, so an argument value a past invocation already produced is a good
guess for the one being generated now.

Two stages, both from the paper:

* **Which past invocations are relevant.** Each finished request contributes its
  output tokens keyed by a *question vector* -- the target's hidden state at the
  last prompt token, which is the cheapest available summary of what was asked.
  :meth:`RetrievalStore.retrieve` ranks stored vectors by cosine similarity
  against the current request's. Vectors are unit-normalised on insert so the
  ranking is a single mat-vec rather than a per-entry norm.

* **What to draft from them.** :meth:`RetrievalStore.match` n-gram suffix-matches
  the request's committed output against the retrieved traces, longest needle
  first, and returns the continuation. Long needles are checked before short
  ones so a confident 7-token match beats a coincidental 5-token one.

Why a group key
---------------
The paper searches the whole memory. This adds an optional ``group_key`` (the
request's grammar identity) and prefers entries that share it, because two
requests under *different* schemas can still have similar prompts, and a
continuation drafted from the wrong schema is rejected at verification -- legal
to propose, but it wastes the wider step that proposing it costs. When no stored
entry shares the key the search falls back to the whole memory, so this only
ever narrows a search that would otherwise be noisier.

What this cannot affect
-----------------------
Drafts are verified against the target exactly like every other Metal
speculative method, so a bad retrieval costs a wider decode step and nothing
else -- never output. That matters here more than for the other proposers
because the memory is shared across requests: one request's output can seed
another's draft. It changes *what is guessed*, never *what is emitted*.

The store is per-worker and bounded. It is a drafting cache, not a datastore:
:meth:`clear` empties it, and eviction is FIFO once ``capacity`` is reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

# Defaults follow ToolSpec's `find_candidate_pred_tokens`: needles from 7 tokens
# down to 5. Shorter needles than 5 match almost anything in JSON-shaped text
# (`": "` and friends recur constantly) and mostly produce rejected drafts.
DEFAULT_NGRAM_MAX = 7
DEFAULT_NGRAM_MIN = 5
DEFAULT_TOP_K = 4
DEFAULT_CAPACITY = 512
# Traces longer than this are truncated on insert. A tool-calling trace is tens
# of tokens; anything far larger is a prose response that will never be matched
# by a 5..7-token needle anyway, and it would only cost search time.
DEFAULT_MAX_TRACE_TOKENS = 512


@dataclass(slots=True)
class InvocationRecord:
    """One finished request's output, keyed by its question vector."""

    output_ids: tuple[int, ...]
    group_key: object | None
    req_id: str


@dataclass
class RetrievalStoreStats:
    """Counters the benchmark harness reads. Cheap to maintain, never reset."""

    inserts: int = 0
    evictions: int = 0
    retrieve_calls: int = 0
    # Calls that found at least one candidate trace to search.
    retrieve_hits: int = 0
    match_calls: int = 0
    match_hits: int = 0
    # Needle length that produced each hit, so the ngram window can be tuned
    # against evidence rather than guessed.
    hits_by_ngram: dict[int, int] = field(default_factory=dict)
    # Retrievals that fell back to searching the whole memory because no stored
    # entry shared the query's group key.
    group_fallbacks: int = 0


class RetrievalStore:
    """Bounded, per-worker memory of past invocations with cosine retrieval."""

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_CAPACITY,
        top_k: int = DEFAULT_TOP_K,
        ngram_max: int = DEFAULT_NGRAM_MAX,
        ngram_min: int = DEFAULT_NGRAM_MIN,
        max_trace_tokens: int = DEFAULT_MAX_TRACE_TOKENS,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if ngram_min <= 0 or ngram_max < ngram_min:
            raise ValueError(
                f"require 0 < ngram_min <= ngram_max, got {ngram_min}/{ngram_max}"
            )
        self._capacity = capacity
        self._top_k = top_k
        self._ngram_max = ngram_max
        self._ngram_min = ngram_min
        self._max_trace_tokens = max_trace_tokens

        self._records: list[InvocationRecord] = []
        # Unit-normalised question vectors, row i belonging to _records[i].
        # Allocated on first insert, when the hidden width becomes known.
        self._vectors: np.ndarray | None = None
        self._size = 0
        self.stats = RetrievalStoreStats()

    # -- introspection -------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def ngram_window(self) -> tuple[int, int]:
        return (self._ngram_min, self._ngram_max)

    def clear(self) -> None:
        self._records.clear()
        self._vectors = None
        self._size = 0

    # -- writing -------------------------------------------------------------

    def add(
        self,
        *,
        vector: np.ndarray,
        output_ids: Sequence[int],
        group_key: object | None = None,
        req_id: str = "",
    ) -> bool:
        """Store one finished invocation. Returns whether it was kept.

        A trace shorter than the smallest needle can never be matched, so it is
        dropped rather than allowed to consume a slot.
        """
        if len(output_ids) <= self._ngram_min:
            return False
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm == 0.0:
            # A zero or non-finite question vector carries no similarity signal
            # and would poison the ranking with NaNs.
            return False
        vec = vec / norm

        if self._vectors is None:
            self._vectors = np.zeros((self._capacity, vec.shape[0]), dtype=np.float32)
        elif vec.shape[0] != self._vectors.shape[1]:
            # Hidden width changed mid-process. Nothing legitimate does this;
            # refuse rather than corrupt the matrix.
            return False

        record = InvocationRecord(
            output_ids=tuple(output_ids[: self._max_trace_tokens]),
            group_key=group_key,
            req_id=req_id,
        )
        if self._size < self._capacity:
            self._vectors[self._size] = vec
            self._records.append(record)
            self._size += 1
        else:
            # FIFO: drop the oldest so a long-running server keeps drafting from
            # recent traffic rather than from whatever it saw at startup.
            self._vectors[:-1] = self._vectors[1:]
            self._vectors[-1] = vec
            self._records.pop(0)
            self._records.append(record)
            self.stats.evictions += 1
        self.stats.inserts += 1
        return True

    # -- reading -------------------------------------------------------------

    def retrieve(
        self,
        *,
        vector: np.ndarray,
        group_key: object | None = None,
        top_k: int | None = None,
    ) -> list[InvocationRecord]:
        """Return the most similar stored invocations, most similar first."""
        self.stats.retrieve_calls += 1
        if self._size == 0 or self._vectors is None:
            return []
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self._vectors.shape[1]:
            return []
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm == 0.0:
            return []
        vec = vec / norm

        indices = np.arange(self._size)
        if group_key is not None:
            same = np.array(
                [self._records[i].group_key == group_key for i in range(self._size)],
                dtype=bool,
            )
            if bool(same.any()):
                indices = indices[same]
            else:
                self.stats.group_fallbacks += 1

        # Both sides are unit-normalised, so the dot product is the cosine.
        scores = self._vectors[indices] @ vec
        k = min(self._top_k if top_k is None else top_k, indices.shape[0])
        if k <= 0:
            return []
        # argpartition picks the top k without ordering the rest; the small k
        # slice is then sorted descending.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        self.stats.retrieve_hits += 1
        return [self._records[int(indices[i])] for i in top]

    def match(
        self,
        *,
        context: Sequence[int],
        records: Sequence[InvocationRecord],
        max_tokens: int,
    ) -> list[int]:
        """N-gram suffix match ``context`` inside ``records``, longest needle first.

        Returns the continuation that followed the match, capped at
        ``max_tokens``, or an empty list when nothing matched.
        """
        self.stats.match_calls += 1
        if max_tokens <= 0 or not records or not context:
            return []
        longest = min(self._ngram_max, len(context))
        for size in range(longest, self._ngram_min - 1, -1):
            needle = tuple(context[-size:])
            for record in records:
                start = _find_last(record.output_ids, needle)
                if start < 0:
                    continue
                tail = record.output_ids[start + size : start + size + max_tokens]
                if not tail:
                    continue
                self.stats.match_hits += 1
                self.stats.hits_by_ngram[size] = (
                    self.stats.hits_by_ngram.get(size, 0) + 1
                )
                return list(tail)
        return []


def _find_last(haystack: tuple[int, ...], needle: tuple[int, ...]) -> int:
    """Index of the last occurrence of ``needle`` in ``haystack``, or -1.

    Last rather than first: when a trace repeats a pattern, the most recent
    occurrence is the one whose continuation is most likely to still apply.
    """
    n = len(needle)
    if n == 0 or n > len(haystack):
        return -1
    for start in range(len(haystack) - n, -1, -1):
        if haystack[start : start + n] == needle:
            return start
    return -1
