"""Freshness-aware completeness checks for partitioned artifacts.

Orchestrators like Luigi treat a task as "complete" when its output file
exists. That is one of the biggest silent-failure classes in partitioned
pipelines: a parquet with only a fraction of its expected rows (e.g. 700
when the month should have ~20,000) is marked complete, and every
downstream stage silently consumes the partial data.

This module provides pure functions that look **beyond file existence**:

- :func:`manifest_coverage_ok` — did the writer record coverage_ok=True?
- :func:`manifest_expected_present` — are all expected periods actually in
  the manifest's ``present_months``/``present_periods`` list?
- :func:`manifest_min_row_count` — does the manifest report at least N rows?
- :func:`manifest_freshness_ok` — is the manifest file itself younger than
  ``max_age_seconds``?
- :func:`upstream_newer_than` — were any upstream manifests updated after
  the self manifest (i.e. we are stale relative to inputs)?

The functions are intentionally small and orchestrator-agnostic. The Luigi
integration is in :mod:`pitight.luigi_adapter` (``FreshCompleteMixin``).

Design notes:

- **Missing manifest is not silently-complete**. Every function returns
  ``False`` (or raises) when the manifest is absent. Callers who want to
  tolerate that should check existence first.
- **Clock-skew tolerance**: ``upstream_newer_than`` accepts a
  ``tolerance_seconds`` slack since cron jobs on the same host can fire in
  any order within a second or two.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ============================================================
# Manifest readers (pure)
# ============================================================


def _load_manifest(manifest_path: Path) -> dict | None:
    """Load a manifest.json; return None if missing or unreadable."""
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cannot read manifest %s: %s", manifest_path, exc)
        return None


def manifest_coverage_ok(manifest_path: Path) -> bool:
    """Return True iff the manifest records ``coverage.coverage_ok == True``."""
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        return False
    coverage = manifest.get("coverage") or {}
    return bool(coverage.get("coverage_ok", False))


def manifest_expected_present(
    manifest_path: Path,
    expected: Iterable[str],
) -> tuple[bool, list[str]]:
    """Check every ``expected`` period appears in the manifest's present list.

    Returns ``(ok, missing)``. ``missing`` is non-empty when ``ok`` is False.
    Accepts both ``present_periods`` (the current key written by
    :func:`pitight.partition.build_manifest`) and the legacy
    ``present_months`` key for backward compatibility.
    """
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        return False, list(expected)
    coverage = manifest.get("coverage") or {}
    present = coverage.get("present_periods") or coverage.get("present_months") or []
    present_set = set(present)
    missing = [p for p in expected if p not in present_set]
    return (len(missing) == 0), missing


def manifest_min_row_count(manifest_path: Path, minimum: int) -> bool:
    """Return True iff ``stats_rollup.row_count_total >= minimum``.

    A stale manifest from a partial rebuild fails this when the subclass
    declares a sane minimum (e.g. expected_rows_per_day * expected_days).
    """
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        return False
    stats = manifest.get("stats_rollup") or {}
    total = int(stats.get("row_count_total", 0))
    return total >= minimum


# ============================================================
# Freshness (mtime-based)
# ============================================================


def manifest_freshness_ok(manifest_path: Path, max_age_seconds: float) -> bool:
    """Return True iff the manifest was written within ``max_age_seconds``.

    ``False`` when the file is missing. When ``max_age_seconds <= 0`` the
    check is skipped and the function returns True (freshness not enforced).
    """
    if max_age_seconds <= 0:
        return True
    if not manifest_path.exists():
        return False
    age = time.time() - manifest_path.stat().st_mtime
    return age <= max_age_seconds


def upstream_newer_than(
    self_manifest: Path,
    upstream_manifests: Iterable[Path],
    tolerance_seconds: float = 2.0,
) -> list[Path]:
    """Return the list of upstream manifests newer than ``self_manifest``.

    An empty list means we are up-to-date relative to every listed upstream.
    A non-empty list means at least one input has been rebuilt after us, so
    we should be rebuilt too.

    ``tolerance_seconds`` absorbs same-second cron races.
    """
    if not self_manifest.exists():
        # We have no timestamp, treat every upstream as newer.
        return [p for p in upstream_manifests if p.exists()]
    self_mtime = self_manifest.stat().st_mtime
    newer: list[Path] = []
    for up in upstream_manifests:
        if not up.exists():
            continue
        if up.stat().st_mtime > self_mtime + tolerance_seconds:
            newer.append(up)
    return newer
