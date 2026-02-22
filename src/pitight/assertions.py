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
    import numpy as np

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
