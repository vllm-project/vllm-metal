# SPDX-License-Identifier: Apache-2.0
"""Tests for the KV-cache memory-safety clamp (cache_policy.compute_safe_kv_budget).

The clamp exists because fraction-derived KV budgets are wired (non-pageable)
and know nothing about host state: fraction 0.8 on a 128GB machine wired ~90GB
for a 1.5B model and drove free memory to 11% (observed live, 2026-07-17).
"""

from vllm_metal.v1.cache_policy import WorkerCachePlanner

GB = 1 << 30
SLACK = WorkerCachePlanner.MEMORY_GUARD_SLACK_BYTES


def clamp(kv, other, avail, total, frac=0.15, disabled=False):
    return WorkerCachePlanner.compute_safe_kv_budget(
        kv,
        planned_other_bytes=other,
        available_bytes=avail,
        total_bytes=total,
        min_free_fraction=frac,
        guard_disabled=disabled,
    )


def test_small_plan_untouched() -> None:
    """A modest plan on a big machine passes through unchanged."""
    budget, reason = clamp(10 * GB, 2 * GB, 110 * GB, 137 * GB)
    assert budget == 10 * GB and reason is None


def test_studio_incident_is_clamped() -> None:
    """The observed live incident: 90.4GB budget, 114GB available, 137GB
    total. The soft floor (15% = 20.6GB) must clamp the budget so free
    memory never approaches the observed 11%."""
    budget, reason = clamp(int(90.4 * GB), 2 * GB, 114 * GB, 137 * GB)
    assert budget < int(90.4 * GB)
    assert reason and "clamped" in reason
    # Post-clamp: wired total leaves at least the floor free.
    floor = max(4 * GB, int(137 * GB * 0.15))
    assert budget + 2 * GB + SLACK <= 114 * GB - floor


def test_hard_limit_when_plan_exceeds_available() -> None:
    """A plan bigger than available memory is cut regardless of the guard
    (serving would thrash immediately)."""
    budget, reason = clamp(120 * GB, 2 * GB, 100 * GB, 137 * GB, disabled=True)
    assert budget + 2 * GB + SLACK <= 100 * GB - 1 * GB
    assert reason and "cannot-fit" in reason


def test_disable_env_skips_soft_floor_only() -> None:
    """Opt-out disables the floor clamp but never the cannot-fit cut."""
    kv = int(90.4 * GB)
    budget, reason = clamp(kv, 2 * GB, 114 * GB, 137 * GB, disabled=True)
    assert budget == kv and reason is None  # fits in available: untouched


def test_small_machine_recipe_mostly_preserved() -> None:
    """fraction 0.8 on a 16GB CI Mac: the clamp trims rather than guts the
    budget (floor = max(4GB, 2.4GB) = 4GB)."""
    # 0.8 * 14.5GB metal limit ~ 11.6GB usable; model+overhead ~1GB
    budget, reason = clamp(int(10.6 * GB), 1 * GB, 13 * GB, 16 * GB)
    assert reason is not None  # small machines do get clamped
    assert budget > 3 * GB  # but a serving-viable budget remains


def test_floor_fraction_scales() -> None:
    """A stricter floor clamps more."""
    loose, _ = clamp(int(90.4 * GB), 2 * GB, 114 * GB, 137 * GB, frac=0.10)
    strict, _ = clamp(int(90.4 * GB), 2 * GB, 114 * GB, 137 * GB, frac=0.30)
    assert strict < loose


def test_never_negative() -> None:
    budget, _ = clamp(50 * GB, 10 * GB, 8 * GB, 16 * GB)
    assert budget == 0
