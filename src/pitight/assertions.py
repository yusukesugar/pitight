"""DataFrame assertions for pipeline boundary validation.

Fail-fast checks at I/O boundaries. Every assertion includes a `tag` parameter
for traceable error messages — when something breaks, you know exactly where.

Usage:
    from pitight.assertions import assert_has_columns, assert_unique_key

    assert_has_columns(df, ["user_id", "timestamp"], tag="LoadUsers")
    assert_unique_key(df, on=["user_id", "timestamp"], tag="LoadUsers")
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SchemaViolationError(Exception):
    """Raised when a DataFrame violates its declared schema contract."""


class EmptyDataError(Exception):
    """Raised when an empty DataFrame is not allowed."""


def assert_has_columns(
    df: pd.DataFrame, required: Iterable[str], tag: str
) -> None:
    """Fail if any required columns are missing."""
    miss = set(required) - set(df.columns)
    if miss:
        raise SchemaViolationError(f"[{tag}] missing columns: {sorted(miss)}")


def assert_columns_match(
    df: pd.DataFrame,
    expected_cols: Iterable[str],
    tag: str,
    ignore_order: bool = True,
) -> None:
    """Fail if DataFrame columns don't exactly match expected set."""
    expected = list(expected_cols)
    if ignore_order:
        dc = set(df.columns)
        ec = set(expected)
        if dc != ec:
            raise SchemaViolationError(
                f"[{tag}] columns mismatch. "
                f"df_only={sorted(dc - ec)}, "
                f"expected_only={sorted(ec - dc)}"
            )
    else:
        if list(df.columns) != expected:
            raise SchemaViolationError(
                f"[{tag}] column order mismatch. "
                f"df_cols={list(df.columns)}, "
                f"expected_cols={expected}"
            )


def assert_unique_key(
    df: pd.DataFrame, on: Iterable[str], tag: str, sample: int = 20
) -> None:
    """Fail if duplicate rows exist for the given key columns."""
    on = list(on)
    dup = df.loc[df.duplicated(on, keep=False), on].head(sample)
    if not dup.empty:
        raise SchemaViolationError(
            f"[{tag}] duplicate keys on {on}. sample={dup.to_dict('records')}"
        )


def assert_keyset_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Iterable[str],
    tag: str,
    sample: int = 20,
) -> None:
    """Fail if the key sets of two DataFrames are not identical."""
    on = list(on)
    left_keys = left[on].drop_duplicates()
    right_keys = right[on].drop_duplicates()

    lost_l = left_keys.merge(right_keys, on=on, how="left", indicator=True)
    lost_l = lost_l[lost_l["_merge"] == "left_only"][on].head(sample)

    lost_r = right_keys.merge(left_keys, on=on, how="left", indicator=True)
    lost_r = lost_r[lost_r["_merge"] == "left_only"][on].head(sample)

    if not lost_l.empty or not lost_r.empty:
        raise SchemaViolationError(
            f"[{tag}] keyset mismatch on {on}. "
            f"left_only(sample)={lost_l.to_dict('records')}, "
            f"right_only(sample)={lost_r.to_dict('records')}"
        )


def assert_keyset_subset(
    df_small: pd.DataFrame,
    df_large: pd.DataFrame,
    on: Iterable[str],
    tag: str,
    sample: int = 20,
) -> None:
    """Fail if small's key set is not a subset of large's."""
    on = list(on)
    s = df_small[on].drop_duplicates()
    l_keys = df_large[on].drop_duplicates()

    lost = s.merge(l_keys, on=on, how="left", indicator=True)
    lost = lost[lost["_merge"] == "left_only"][on].head(sample)

    if not lost.empty:
        raise SchemaViolationError(
            f"[{tag}] keyset not subset on {on}. "
            f"small_only(sample)={lost.to_dict('records')}"
        )


def assert_no_nulls(
    df: pd.DataFrame, cols: Iterable[str], tag: str
) -> None:
    """Fail if nulls found in specified columns."""
    bad = [c for c in cols if df[c].isna().any()]
    if bad:
        raise SchemaViolationError(f"[{tag}] nulls not allowed in columns: {bad}")


def assert_prob_range(df: pd.DataFrame, col: str, tag: str) -> None:
    """Fail if values in column are outside [0, 1]."""
    vals = df[col].to_numpy(dtype=float)
    ok = (vals >= 0.0) & (vals <= 1.0)
    if not ok.all():
        offenders = np.where(~ok)[0][:10]
        raise SchemaViolationError(
            f"[{tag}] probability column '{col}' must be within [0,1]. "
            f"offending rows: {offenders.tolist()}"
        )


def assert_not_empty(df: pd.DataFrame, tag: str) -> None:
    """Fail if DataFrame is empty."""
    if df.empty:
        raise EmptyDataError(f"[{tag}] DataFrame is empty (0 rows)")


def safe_merge(
    left: pd.DataFrame, right: pd.DataFrame, **kwargs: Any
) -> pd.DataFrame:
    """pd.merge with suffixes=(None, None) — column collision raises ValueError.

    This prevents silent column shadowing (e.g., 'col_x', 'col_y') which is
    a common source of subtle bugs in ML pipelines.
    """
    kwargs["suffixes"] = (None, None)
    return pd.merge(left, right, **kwargs)


# ============================================================
# Temporal / period assertions
# ============================================================


def assert_single_period(
    df: pd.DataFrame,
    period: str,
    tag: str,
    date_col: str = "date",
    freq: str = "M",
) -> None:
    """Fail if DataFrame contains rows outside the expected period.

    Args:
        df: DataFrame to validate.
        period: Expected period string (e.g. ``'2025-01'`` for monthly).
        tag: Context label for error messages.
        date_col: Name of the date/datetime column.
        freq: Pandas period frequency (``'M'`` for monthly, ``'D'`` for daily).

    Raises:
        SchemaViolationError: If rows fall outside the expected period.

    Example:
        >>> df = pd.DataFrame({"date": ["2025-01-15", "2025-01-20"], "revenue": [100, 200]})
        >>> assert_single_period(df, "2025-01", tag="MonthlySales")
    """
    if df.empty:
        return
    if date_col not in df.columns:
        raise SchemaViolationError(
            f"[{tag}] missing date_col={date_col!r}, columns={list(df.columns)}"
        )
    periods = pd.to_datetime(df[date_col]).dt.to_period(freq).astype(str)
    bad = df.loc[periods != period, [date_col]].head(20)
    if not bad.empty:
        raise SchemaViolationError(
            f"[{tag}] out-of-period rows: expected={period!r} (freq={freq}) "
            f"sample={bad.to_dict('records')}"
        )


def assert_period_coverage(
    df: pd.DataFrame,
    date_col: str,
    start: str,
    end: str,
    tag: str,
    freq: str = "M",
) -> None:
    """Fail if DataFrame is missing too many expected periods.

    For single-period ranges, all periods must be present. For multi-period
    ranges, fails if more than half the periods are missing.

    Args:
        df: DataFrame to validate.
        date_col: Name of the date/datetime column.
        start: Start period (e.g. ``'2025-01'``).
        end: End period (e.g. ``'2025-06'``).
        tag: Context label for error messages.
        freq: Pandas period frequency (``'M'``, ``'D'``, etc.).

    Raises:
        SchemaViolationError: If coverage is insufficient.

    Example:
        >>> import pandas as pd
        >>> dates = ["2025-01-15", "2025-02-10", "2025-03-05"]
        >>> df = pd.DataFrame({"date": dates, "sessions": [10, 20, 30]})
        >>> assert_period_coverage(df, "date", "2025-01", "2025-03", tag="UserSessions")
    """
    expected = pd.period_range(start, end, freq=freq)
    got = set(pd.to_datetime(df[date_col]).dt.to_period(freq))
    miss = [p for p in expected if p not in got]

    if len(got) == 0:
        raise SchemaViolationError(
            f"[{tag}] period {start}..{end} is empty (0 rows)"
        )
    if len(expected) == 1:
        if miss:
            raise SchemaViolationError(
                f"[{tag}] period {start}..{end} is missing"
            )
    else:
        if len(miss) >= len(expected) // 2:
            raise SchemaViolationError(
                f"[{tag}] period coverage too thin: "
                f"missing {len(miss)}/{len(expected)} e.g. {[str(m) for m in miss[:6]]}"
            )


# ============================================================
# Numeric value assertions
# ============================================================


def assert_finite(
    df: pd.DataFrame,
    cols: Iterable[str],
    tag: str,
    allow_nan: bool = False,
) -> None:
    """Fail if non-finite values (inf, -inf, optionally NaN) are found.

    Args:
        df: DataFrame to validate.
        cols: Columns to check.
        tag: Context label for error messages.
        allow_nan: If True, NaN is permitted; only inf/-inf are rejected.

    Raises:
        SchemaViolationError: If non-finite values are found.

    Example:
        >>> df = pd.DataFrame({"score": [0.5, 0.8, 0.3]})
        >>> assert_finite(df, ["score"], tag="Predictions")
    """
    bad: list[str] = []
    for c in cols:
        vals = df[c].to_numpy(dtype=float)
        if allow_nan:
            ok = np.isfinite(vals) | np.isnan(vals)
        else:
            ok = np.isfinite(vals)
        if not ok.all():
            bad.append(c)
    if bad:
        raise SchemaViolationError(
            f"[{tag}] non-finite values found in columns: {bad}"
        )


def assert_positive(
    df: pd.DataFrame, col: str, tag: str
) -> None:
    """Fail if any value in column is not strictly positive.

    Raises:
        SchemaViolationError: If non-positive values exist.

    Example:
        >>> df = pd.DataFrame({"price": [10.0, 25.0, 5.0]})
        >>> assert_positive(df, "price", tag="Products")
    """
    vals = df[col].to_numpy(dtype=float)
    if not (vals > 0).all():
        n_bad = int((vals <= 0).sum())
        raise SchemaViolationError(
            f"[{tag}] column {col!r} must be strictly positive "
            f"({n_bad} non-positive values)"
        )


# ============================================================
# Diagnostic utilities
# ============================================================


def anti_join_report(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Iterable[str],
    tag: str,
) -> int:
    """Log rows in *left* that have no match in *right*.

    Returns the count of unmatched rows (0 means perfect join).
    This is a diagnostic helper, not a hard assertion — it logs at INFO level.

    Example:
        >>> orders = pd.DataFrame({"user_id": [1, 2, 3], "amount": [10, 20, 30]})
        >>> users = pd.DataFrame({"user_id": [1, 2]})
        >>> n = anti_join_report(orders, users, on=["user_id"], tag="OrderUsers")
    """
    on_list = list(on)
    probe = left.merge(right[on_list], on=on_list, how="left", indicator=True)
    lost = probe[probe["_merge"] == "left_only"]
    n = len(lost)
    if n > 0:
        samp = lost[on_list].head(5).to_dict("records")
        logger.info("[%s] anti-join: left-only=%s sample=%s", tag, n, samp)
    return n
