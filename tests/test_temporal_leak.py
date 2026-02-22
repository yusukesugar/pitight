import pandas as pd
import pytest

from pitight.temporal_leak import (
    TemporalBoundary,
    TemporalLeakError,
    check_leak,
    no_leak,
)


@pytest.fixture()
def boundary():
    return TemporalBoundary(
        forbidden_columns=frozenset({"odds", "payout", "result"}),
        forbidden_prefixes=("outcome_",),
        forbidden_substrings=("_rank",),
        allowed_stages=frozenset({"s70", "s80"}),
    )


class TestTemporalBoundary:
    def test_clean_columns(self, boundary):
        violations = boundary.check(["race_key", "prob", "feature_1"], stage="s50")
        assert violations == []

    def test_exact_match(self, boundary):
        violations = boundary.check(["race_key", "odds", "prob"], stage="s50")
        assert violations == ["odds"]

    def test_prefix_match(self, boundary):
        violations = boundary.check(["outcome_win", "feature_1"], stage="s50")
        assert violations == ["outcome_win"]

    def test_substring_match(self, boundary):
        violations = boundary.check(["player_rank", "feature_1"], stage="s50")
        assert violations == ["player_rank"]

    def test_allowed_stage_passes(self, boundary):
        violations = boundary.check(["odds", "payout", "result"], stage="s70")
        assert violations == []

    def test_multiple_violations(self, boundary):
        violations = boundary.check(["odds", "payout", "outcome_x"], stage="s50")
        assert len(violations) == 3


class TestCheckLeak:
    def test_raises(self, boundary):
        df = pd.DataFrame({"race_key": [1], "odds": [5.0]})
        with pytest.raises(TemporalLeakError, match="Forbidden columns"):
            check_leak(df, boundary, stage="s50")

    def test_no_raise(self, boundary):
        df = pd.DataFrame({"race_key": [1], "odds": [5.0]})
        violations = check_leak(df, boundary, stage="s50", raise_on_violation=False)
        assert violations == ["odds"]

    def test_clean_passes(self, boundary):
        df = pd.DataFrame({"race_key": [1], "prob": [0.5]})
        violations = check_leak(df, boundary, stage="s50")
        assert violations == []


class TestNoLeakDecorator:
    def test_input_check(self, boundary):
        @no_leak(boundary, stage="s50")
        def compute(df):
            return df

        clean_df = pd.DataFrame({"race_key": [1], "prob": [0.5]})
        result = compute(clean_df)
        assert len(result) == 1

    def test_input_leak_raises(self, boundary):
        @no_leak(boundary, stage="s50")
        def compute(df):
            return df

        dirty_df = pd.DataFrame({"race_key": [1], "payout": [1000]})
        with pytest.raises(TemporalLeakError):
            compute(dirty_df)

    def test_output_check(self, boundary):
        @no_leak(boundary, stage="s50", check_output=True)
        def compute(df):
            df = df.copy()
            df["result"] = 1  # leak in output
            return df

        clean_df = pd.DataFrame({"race_key": [1], "prob": [0.5]})
        with pytest.raises(TemporalLeakError):
            compute(clean_df)

    def test_kwargs_checked(self, boundary):
        @no_leak(boundary, stage="s50")
        def compute(train_df=None):
            return train_df

        dirty_df = pd.DataFrame({"odds": [5.0]})
        with pytest.raises(TemporalLeakError):
            compute(train_df=dirty_df)
