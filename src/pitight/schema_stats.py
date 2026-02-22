"""Schema and statistics inference for DataFrame artifacts.

Automatically generates metadata alongside your pipeline outputs:
  - schema.json: column types, nullability, ranges, unique counts
  - stats.json: row/column counts, date ranges, git hash, timestamps

Usage:
    from pitight.schema_stats import infer_schema, infer_stats, write_schema_and_stats

    write_schema_and_stats(df, Path("output/_meta"), "part-2025-01")
"""

from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def git_hash() -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def infer_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Infer column-level schema from a DataFrame.

    Returns:
        Dict with 'columns' key mapping column names to type info:
        - dtype: string representation of pandas dtype
        - nullable: whether column contains any nulls
        - min/max: for numeric columns
        - approx_nunique: for string/categorical columns
    """
    cols: dict[str, Any] = {}
    for c in df.columns:
        s = df[c]
        info: dict[str, Any] = {
            "dtype": str(s.dtype),
            "nullable": bool(s.isna().any()),
        }
        if pd.api.types.is_numeric_dtype(s):
            info["min"] = None if s.dropna().empty else float(s.min())
            info["max"] = None if s.dropna().empty else float(s.max())
        if pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            info["approx_nunique"] = int(s.nunique(dropna=True))
        cols[c] = info
    return {"columns": cols}


def infer_stats(
    df: pd.DataFrame,
    date_col: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Infer aggregate statistics from a DataFrame.

    Returns:
        Dict with n_rows, n_cols, created_at, git_hash, and optionally
        date_min/date_max if date_col is provided.
    """
    stats: dict[str, Any] = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash": git_hash(),
    }
    if date_col and date_col in df.columns:
        ds = pd.to_datetime(df[date_col], errors="coerce")
        if not ds.dropna().empty:
            stats["date_min"] = ds.min().isoformat()
            stats["date_max"] = ds.max().isoformat()
    if extra:
        stats.update(extra)
    return stats


def write_schema_and_stats(
    df: pd.DataFrame,
    out_dir: Path,
    base_name: str,
    *,
    date_col: str | None = None,
    extra_stats: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write schema.json and stats.json for a DataFrame artifact.

    Args:
        df: The DataFrame to inspect.
        out_dir: Directory to write metadata files.
        base_name: File prefix (e.g., 'part-2025-01').
        date_col: Optional date column for date range stats.
        extra_stats: Additional key-value pairs to include in stats.

    Returns:
        Tuple of (schema_path, stats_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = infer_schema(df)
    stats = infer_stats(df, date_col=date_col, extra=extra_stats)

    schema_path = out_dir / f"{base_name}.schema.json"
    stats_path = out_dir / f"{base_name}.stats.json"

    schema_path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return schema_path, stats_path
