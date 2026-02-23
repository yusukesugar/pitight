"""Schema and metadata validation for pipeline artifacts.

Validates that artifacts have the expected metadata files (schema.json, stats.json,
manifest.json) and that DataFrame contents match declared schemas.

Usage:
    from pitight.validation import validate_source_meta, validate_df_schema

    # Check that metadata files exist and dates are consistent
    validate_source_meta(Path("data/features/"), start_ym="2025-01", end_ym="2025-06")

    # Validate a DataFrame against its schema.json
    validate_df_schema(df, Path("data/features/_meta/part.schema.json"))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from pitight.assertions import SchemaViolationError, assert_has_columns

logger = logging.getLogger(__name__)


class SourceValidationError(Exception):
    """Raised when source metadata validation fails."""


class IdentityCalculationError(Exception):
    """Raised when schema identity hash is missing for output paths."""


class ExecutionError(Exception):
    """Raised when execution must fail fast due to governance rules."""


def validate_source_meta(
    root_dir: Path,
    base_name: str = "part",
    start_ym: str | None = None,
    end_ym: str | None = None,
    is_partitioned: bool = False,
    tag: str = "SourceMeta",
) -> None:
    """Check presence of manifest/schema/stats files and date range consistency.

    Args:
        root_dir: Root directory of the artifact.
        base_name: Base name for schema/stats files (default: "part").
        start_ym: Expected start month (YYYY-MM format).
        end_ym: Expected end month (YYYY-MM format).
        is_partitioned: Whether this is a partitioned artifact (changes meta location).
        tag: Descriptive tag for error messages.

    Raises:
        SourceValidationError: If required metadata files are missing.
    """
    meta_dir = root_dir / "_meta" if is_partitioned else root_dir.parent

    # 1. Presence check
    if is_partitioned:
        manifest_path = meta_dir / "manifest.json"
        if not manifest_path.exists():
            raise SourceValidationError(f"[{tag}] Manifest missing: {manifest_path}")
        schema_path = meta_dir / "all.schema.json"
        stats_path = meta_dir / "all.stats.json"
    else:
        schema_path = meta_dir / f"{base_name}.schema.json"
        stats_path = meta_dir / f"{base_name}.stats.json"

    if not schema_path.exists():
        raise SourceValidationError(f"[{tag}] Schema missing: {schema_path}")
    if not stats_path.exists():
        raise SourceValidationError(f"[{tag}] Stats missing: {stats_path}")

    logger.info(
        "[%s] Source meta check passed: %s (partitioned=%s)",
        tag,
        root_dir,
        is_partitioned,
    )

    # 2. Date range consistency
    if stats_path.exists() and (start_ym or end_ym):
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        _check_date_range(stats, start_ym, end_ym, tag)


def _normalize_ym(ym: str) -> str:
    """Normalize YYYYMM to YYYY-MM format."""
    if len(ym) == 6 and ym.isdigit():
        return f"{ym[:4]}-{ym[4:]}"
    return ym


def _check_date_range(
    stats: dict,
    start_ym: str | None,
    end_ym: str | None,
    tag: str,
) -> None:
    """Check that stats date range is consistent with expected range."""
    date_min = stats.get("date_min")
    date_max = stats.get("date_max")

    if date_min and start_ym:
        s_ym = pd.to_datetime(_normalize_ym(start_ym)).to_period("M")
        d_min_ym = pd.to_datetime(date_min).to_period("M")
        if d_min_ym < s_ym:
            logger.warning(
                "[%s] date_min (%s) is earlier than start_ym (%s)",
                tag,
                date_min,
                start_ym,
            )
        elif d_min_ym > s_ym:
            logger.warning(
                "[%s] date_min (%s) is later than start_ym (%s)",
                tag,
                date_min,
                start_ym,
            )

    if date_max and end_ym:
        e_ym = pd.to_datetime(_normalize_ym(end_ym)).to_period("M")
        d_max_ym = pd.to_datetime(date_max).to_period("M")
        if d_max_ym > e_ym:
            logger.warning(
                "[%s] date_max (%s) is later than end_ym (%s)",
                tag,
                date_max,
                end_ym,
            )


def validate_df_schema(
    df: pd.DataFrame,
    schema_path: Path,
    tag: str = "SchemaValidation",
    columns: list[str] | None = None,
) -> None:
    """Validate a DataFrame against a schema.json file.

    Checks:
    1. Required columns are present (if specified).
    2. Dtypes match (warning on mismatch, since pandas dtype naming varies).
    3. Non-nullable columns contain no NULLs (raises on violation).

    Args:
        df: DataFrame to validate.
        schema_path: Path to schema.json file.
        tag: Descriptive tag for error messages.
        columns: Optional subset of columns to check presence for.

    Raises:
        SourceValidationError: If columns are missing or non-nullable columns have NULLs.
    """
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    cols_spec = schema.get("columns", {})

    # 1. Column presence
    if columns:
        try:
            assert_has_columns(df, columns, tag=tag)
        except SchemaViolationError as e:
            raise SourceValidationError(
                f"[{tag}] Missing requested columns for {schema_path}: {e}"
            ) from e

    # 2. Dtype and nullable checks
    for col in df.columns:
        if col not in cols_spec:
            continue

        spec = cols_spec[col]

        # Dtype check (warn only — pandas dtype naming is inconsistent)
        expected_dtype = spec.get("dtype")
        actual_dtype = str(df[col].dtype)
        if expected_dtype and actual_dtype != expected_dtype:
            logger.warning(
                "[%s] Dtype mismatch for '%s' in %s: expected %s, got %s",
                tag,
                col,
                schema_path.name,
                expected_dtype,
                actual_dtype,
            )

        # Nullable check (fail)
        if not spec.get("nullable", True):
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                raise SourceValidationError(
                    f"[{tag}] Column '{col}' is not nullable but contains "
                    f"{null_count} NULLs (file: {schema_path})"
                )

    logger.info(
        "[%s] Schema validation passed: %s (cols=%s)",
        tag,
        schema_path.name,
        list(df.columns),
    )
