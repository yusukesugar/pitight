"""Partition management for pipeline artifacts.

Provides orchestrator-agnostic logic for time-partitioned data:
- Period enumeration (monthly, daily, weekly, yearly)
- Hive-style path generation (year=YYYY/month=MM/...)
- Coverage tracking (expected vs present partitions)
- Manifest building for artifact completeness

Usage:
    from pitight.partition import (
        resolve_expected_periods,
        compute_coverage,
        hive_path,
        scan_present_months,
        build_manifest,
    )

    periods = resolve_expected_periods("2025-01", "2025-06", freq="M")
    missing, ok = compute_coverage(periods, present_months)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# Period enumeration
# ============================================================


def resolve_expected_periods(
    start: str,
    end: str,
    freq: str = "M",
) -> list[str]:
    """Return list of expected period strings for a date range.

    Args:
        start: Start period (format depends on freq).
            Monthly: "YYYY-MM", Daily: "YYYY-MM-DD"
        end: End period (same format as start).
        freq: Partition frequency. "M" (monthly), "D" (daily).

    Returns:
        Sorted list of period strings.

    Raises:
        ValueError: If freq is unsupported or dates are invalid.
    """
    if freq == "M":
        return _enumerate_months(start, end)
    if freq == "D":
        return _enumerate_days(start, end)
    raise ValueError(f"Unsupported freq: {freq!r}. Use 'M' or 'D'.")


def _enumerate_months(start_ym: str, end_ym: str) -> list[str]:
    start_dt = datetime.strptime(start_ym, "%Y-%m")
    end_dt = datetime.strptime(end_ym, "%Y-%m")
    result: list[str] = []
    curr = start_dt
    while curr <= end_dt:
        result.append(curr.strftime("%Y-%m"))
        if curr.month == 12:
            curr = curr.replace(year=curr.year + 1, month=1)
        else:
            curr = curr.replace(month=curr.month + 1)
    return result


def _enumerate_days(start_date: str, end_date: str) -> list[str]:
    from datetime import timedelta

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    result: list[str] = []
    curr = start_dt
    while curr <= end_dt:
        result.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)
    return result


# ============================================================
# Hive-style path utilities
# ============================================================


def hive_path(
    root: Path,
    period: str,
    freq: str = "M",
    base_name: str = "part",
) -> Path:
    """Generate a hive-partitioned file path for a given period.

    Args:
        root: Root directory of the artifact data.
        period: Period string ("2025-01" for monthly, "2025-01-15" for daily).
        freq: Partition frequency ("M" or "D").
        base_name: Base name for the parquet file.

    Returns:
        Path like root/data/year=2025/month=01/part-2025-01.parquet
    """
    if freq == "M":
        y, m = period.split("-")
        return (
            root
            / "data"
            / f"year={y}"
            / f"month={m.zfill(2)}"
            / f"{base_name}-{period}.parquet"
        )
    if freq == "D":
        parts = period.split("-")
        y, m, d = parts[0], parts[1], parts[2]
        return (
            root
            / "data"
            / f"year={y}"
            / f"month={m.zfill(2)}"
            / f"day={d.zfill(2)}"
            / f"{base_name}-{period}.parquet"
        )
    raise ValueError(f"Unsupported freq: {freq!r}")


def scan_present_months(data_dir: Path) -> list[str]:
    """Scan a hive-partitioned data directory for present months.

    Expects structure: data_dir/year=YYYY/month=MM/*.parquet

    Args:
        data_dir: The "data" directory to scan.

    Returns:
        Sorted list of "YYYY-MM" strings for months with parquet files.
    """
    if not data_dir.exists():
        return []

    present: set[str] = set()
    for y_dir in data_dir.glob("year=*"):
        year = y_dir.name.split("=")[1]
        for m_dir in y_dir.glob("month=*"):
            month = m_dir.name.split("=")[1]
            if list(m_dir.glob("*.parquet")):
                present.add(f"{year}-{month.zfill(2)}")
    return sorted(present)


# ============================================================
# Coverage
# ============================================================


def compute_coverage(
    expected: list[str],
    present: list[str],
) -> tuple[list[str], bool]:
    """Compute missing partitions and overall coverage status.

    Args:
        expected: List of expected period strings.
        present: List of actually present period strings.

    Returns:
        Tuple of (missing_periods, coverage_ok).
    """
    present_set = set(present)
    missing = [p for p in expected if p not in present_set]
    return missing, len(missing) == 0


def rollup_stats(monthly_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-partition stats into a summary.

    Args:
        monthly_stats: List of per-partition stat dicts (each should have "row_count").

    Returns:
        Rollup dict with row_count_total and updated_at.
    """
    row_count_total = sum(s.get("row_count", 0) for s in monthly_stats)
    return {
        "row_count_total": row_count_total,
        "updated_at": datetime.now().isoformat(),
    }


# ============================================================
# Manifest
# ============================================================


def build_manifest(
    artifact_id: str,
    identity_params: dict[str, Any],
    request: dict[str, str],
    expected_periods: list[str],
    present_periods: list[str],
    missing_periods: list[str],
    coverage_ok: bool,
    stats_rollup: dict[str, Any],
    checks: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest dict describing artifact completeness.

    Args:
        artifact_id: Unique identifier for the artifact.
        identity_params: Parameters that determine the artifact identity.
        request: Request parameters (e.g., {"start_ym": ..., "end_ym": ...}).
        expected_periods: Periods that should exist.
        present_periods: Periods that actually exist.
        missing_periods: Periods that are missing.
        coverage_ok: Whether all expected periods are present.
        stats_rollup: Aggregated statistics.
        checks: Optional validation check results.
        meta: Optional additional metadata.

    Returns:
        Manifest dict ready for JSON serialization.
    """
    result: dict[str, Any] = {
        "artifact_id": artifact_id,
        "policy": "incremental_partitioned",
        "identity_params": identity_params,
        "request": request,
        "coverage": {
            "expected_periods": expected_periods,
            "present_periods": present_periods,
            "missing_periods": missing_periods,
            "coverage_ok": coverage_ok,
            "expected_resolver": "range",
        },
        "stats_rollup": stats_rollup,
        "checks": checks or {"schema_ok": True, "notes": []},
    }
    if meta:
        result["meta"] = meta
    return result


def write_manifest(
    manifest_dir: Path,
    manifest: dict[str, Any],
    filename: str = "manifest.json",
) -> Path:
    """Write manifest dict to a JSON file atomically.

    Args:
        manifest_dir: Directory to write the manifest into.
        manifest: Manifest dict to serialize.
        filename: Output filename (default: "manifest.json").

    Returns:
        Path to the written manifest file.
    """
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_path = manifest_dir / filename
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    tmp_path.rename(out_path)
    return out_path


# ============================================================
# Identity & Schema Hash
# ============================================================


def compute_schema_hash(output_schema: dict[str, Any]) -> str:
    """Compute a deterministic SHA256 hash of an output schema.

    Args:
        output_schema: Mapping of column_name -> dtype_string.

    Returns:
        64-char hex digest.

    Raises:
        ValueError: If output_schema is empty.
    """
    if not output_schema:
        raise ValueError("output_schema is empty")
    sorted_cols = sorted(output_schema.keys())
    payload = "|".join(f"{col}:{output_schema[col]}" for col in sorted_cols)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def encode_identity(identity: dict[str, Any]) -> Path:
    """Encode identity parameters as a stable directory path.

    Produces path segments like ``schema_hash=abc123/version=v1``.
    The ``schema_hash`` key is placed first if present; other keys are sorted.

    Args:
        identity: Dict of identity key-value pairs.

    Returns:
        Relative Path with one segment per key.
    """
    if not identity:
        return Path()

    keys = sorted(identity.keys())
    if "schema_hash" in identity:
        keys = ["schema_hash"] + [k for k in keys if k != "schema_hash"]

    parts = [f"{k}={identity[k]}" for k in keys]
    return Path(*parts)


# ============================================================
# Success marker
# ============================================================


def update_success_marker(success_path: Path, ok: bool) -> None:
    """Create or remove a _SUCCESS marker file.

    Args:
        success_path: Path to the _SUCCESS file.
        ok: If True, create the marker. If False, remove it.
    """
    if ok:
        success_path.parent.mkdir(parents=True, exist_ok=True)
        success_path.write_text("ok")
    else:
        if success_path.exists():
            success_path.unlink()
