import pandas as pd
import pytest

from pitight.assertions import (
    EmptyDataError,
    SchemaViolationError,
    assert_has_columns,
    assert_no_nulls,
    assert_not_empty,
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
