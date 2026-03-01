import numpy as np
import pandas as pd
import pytest

from pitight.assertions import (
    EmptyDataError,
    SchemaViolationError,
    anti_join_report,
    assert_finite,
    assert_has_columns,
    assert_no_nulls,
    assert_not_empty,
    assert_period_coverage,
    assert_positive,
    assert_single_period,
    assert_unique_key,
    safe_merge,
)


@pytest.fixture()
def sample_df():
    return pd.DataFrame(
        {"user_id": [1, 2, 3], "score": [0.5, 0.8, 0.3], "label": ["a", "b", "c"]}
    )


class TestAssertHasColumns:
    def test_pass(self, sample_df):
        assert_has_columns(sample_df, ["user_id", "score"], tag="test")

    def test_fail(self, sample_df):
        with pytest.raises(SchemaViolationError, match="missing columns"):
            assert_has_columns(sample_df, ["user_id", "nonexistent"], tag="test")


class TestAssertUniqueKey:
    def test_pass(self, sample_df):
        assert_unique_key(sample_df, on=["user_id"], tag="test")

    def test_fail(self):
        df = pd.DataFrame({"id": [1, 1, 2], "val": [10, 20, 30]})
        with pytest.raises(SchemaViolationError, match="duplicate keys"):
            assert_unique_key(df, on=["id"], tag="test")


class TestAssertNoNulls:
    def test_pass(self, sample_df):
        assert_no_nulls(sample_df, ["user_id", "score"], tag="test")

    def test_fail(self):
        df = pd.DataFrame({"x": [1, None, 3]})
        with pytest.raises(SchemaViolationError, match="nulls not allowed"):
            assert_no_nulls(df, ["x"], tag="test")


class TestAssertNotEmpty:
    def test_pass(self, sample_df):
        assert_not_empty(sample_df, tag="test")

    def test_fail(self):
        with pytest.raises(EmptyDataError):
            assert_not_empty(pd.DataFrame(), tag="test")


class TestSafeMerge:
    def test_no_collision(self):
        left = pd.DataFrame({"key": [1], "val_a": [10]})
        right = pd.DataFrame({"key": [1], "val_b": [20]})
        result = safe_merge(left, right, on="key")
        assert list(result.columns) == ["key", "val_a", "val_b"]

    def test_collision_raises(self):
        left = pd.DataFrame({"key": [1], "val": [10]})
        right = pd.DataFrame({"key": [1], "val": [20]})
        with pytest.raises(ValueError):
            safe_merge(left, right, on="key")


# ============================================================
# Temporal / period assertions
# ============================================================


class TestAssertSinglePeriod:
    def test_pass_monthly(self):
        df = pd.DataFrame({"date": ["2025-01-05", "2025-01-20"], "revenue": [100, 200]})
        assert_single_period(df, "2025-01", tag="test")

    def test_fail_mixed_months(self):
        df = pd.DataFrame({"date": ["2025-01-05", "2025-02-10"], "revenue": [100, 200]})
        with pytest.raises(SchemaViolationError, match="out-of-period"):
            assert_single_period(df, "2025-01", tag="test")

    def test_pass_daily(self):
        df = pd.DataFrame({"date": ["2025-01-15 09:00", "2025-01-15 17:00"], "val": [1, 2]})
        assert_single_period(df, "2025-01-15", tag="test", freq="D")

    def test_fail_daily(self):
        df = pd.DataFrame({"date": ["2025-01-15", "2025-01-16"], "val": [1, 2]})
        with pytest.raises(SchemaViolationError, match="out-of-period"):
            assert_single_period(df, "2025-01-15", tag="test", freq="D")

    def test_empty_df_passes(self):
        df = pd.DataFrame({"date": [], "val": []})
        assert_single_period(df, "2025-01", tag="test")

    def test_missing_date_col(self):
        df = pd.DataFrame({"timestamp": ["2025-01-01"], "val": [1]})
        with pytest.raises(SchemaViolationError, match="missing date_col"):
            assert_single_period(df, "2025-01", tag="test")


class TestAssertPeriodCoverage:
    def test_pass_full_coverage(self):
        dates = ["2025-01-15", "2025-02-10", "2025-03-05"]
        df = pd.DataFrame({"date": dates, "sessions": [10, 20, 30]})
        assert_period_coverage(df, "date", "2025-01", "2025-03", tag="test")

    def test_fail_empty(self):
        df = pd.DataFrame({"date": pd.Series([], dtype="str"), "val": []})
        with pytest.raises(SchemaViolationError, match="empty"):
            assert_period_coverage(df, "date", "2025-01", "2025-03", tag="test")

    def test_fail_too_many_missing(self):
        # 6 months expected, only 2 present → 4 missing >= 3 (6//2)
        df = pd.DataFrame({"date": ["2025-01-01", "2025-02-01"], "val": [1, 2]})
        with pytest.raises(SchemaViolationError, match="coverage too thin"):
            assert_period_coverage(df, "date", "2025-01", "2025-06", tag="test")

    def test_pass_single_period(self):
        df = pd.DataFrame({"date": ["2025-03-10"], "val": [1]})
        assert_period_coverage(df, "date", "2025-03", "2025-03", tag="test")

    def test_fail_single_period_missing(self):
        df = pd.DataFrame({"date": ["2025-04-10"], "val": [1]})
        with pytest.raises(SchemaViolationError, match="missing"):
            assert_period_coverage(df, "date", "2025-03", "2025-03", tag="test")


# ============================================================
# Numeric value assertions
# ============================================================


class TestAssertFinite:
    def test_pass(self):
        df = pd.DataFrame({"score": [0.5, 0.8, 0.3]})
        assert_finite(df, ["score"], tag="test")

    def test_fail_inf(self):
        df = pd.DataFrame({"score": [0.5, np.inf, 0.3]})
        with pytest.raises(SchemaViolationError, match="non-finite"):
            assert_finite(df, ["score"], tag="test")

    def test_fail_nan(self):
        df = pd.DataFrame({"score": [0.5, np.nan, 0.3]})
        with pytest.raises(SchemaViolationError, match="non-finite"):
            assert_finite(df, ["score"], tag="test")

    def test_allow_nan(self):
        df = pd.DataFrame({"score": [0.5, np.nan, 0.3]})
        assert_finite(df, ["score"], tag="test", allow_nan=True)

    def test_allow_nan_still_rejects_inf(self):
        df = pd.DataFrame({"score": [0.5, np.inf, 0.3]})
        with pytest.raises(SchemaViolationError, match="non-finite"):
            assert_finite(df, ["score"], tag="test", allow_nan=True)


class TestAssertPositive:
    def test_pass(self):
        df = pd.DataFrame({"price": [10.0, 25.0, 5.0]})
        assert_positive(df, "price", tag="test")

    def test_fail_zero(self):
        df = pd.DataFrame({"price": [10.0, 0.0, 5.0]})
        with pytest.raises(SchemaViolationError, match="strictly positive"):
            assert_positive(df, "price", tag="test")

    def test_fail_negative(self):
        df = pd.DataFrame({"price": [10.0, -1.0, 5.0]})
        with pytest.raises(SchemaViolationError, match="strictly positive"):
            assert_positive(df, "price", tag="test")


# ============================================================
# Diagnostic utilities
# ============================================================


class TestAntiJoinReport:
    def test_no_orphans(self):
        orders = pd.DataFrame({"user_id": [1, 2], "amount": [10, 20]})
        users = pd.DataFrame({"user_id": [1, 2]})
        assert anti_join_report(orders, users, on=["user_id"], tag="test") == 0

    def test_with_orphans(self):
        orders = pd.DataFrame({"user_id": [1, 2, 3], "amount": [10, 20, 30]})
        users = pd.DataFrame({"user_id": [1, 2]})
        assert anti_join_report(orders, users, on=["user_id"], tag="test") == 1
