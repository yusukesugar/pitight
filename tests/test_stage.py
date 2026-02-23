"""Tests for pitight.stage — Stage declarative lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pitight.assertions import EmptyDataError, SchemaViolationError
from pitight.stage import EmptyPolicy, InputSpec, Stage
from pitight.temporal_leak import TemporalBoundary, TemporalLeakError


# ============================================================
# Concrete test stage
# ============================================================


class AddOneStage(Stage):
    """Minimal concrete stage for testing."""

    artifact_name = "test_add_one"
    OUTPUT_SCHEMA = {"id": "int64", "value": "float64"}
    REQUIRED_INPUTS = {
        "source": InputSpec(required_cols=["id", "raw_value"]),
    }
    EMPTY_POLICY = EmptyPolicy.FAIL

    def compute(
        self, inputs: dict[str, pd.DataFrame], period: str
    ) -> pd.DataFrame:
        df = inputs["source"].copy()
        df["value"] = df["raw_value"] + 1
        return df[["id", "value"]]


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture()
def source_df() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2, 3], "raw_value": [10.0, 20.0, 30.0]})


@pytest.fixture()
def stage(tmp_path: Path) -> AddOneStage:
    return AddOneStage(data_root=tmp_path)


# ============================================================
# Tests
# ============================================================


class TestBasicLifecycle:
    def test_basic_lifecycle(
        self, stage: AddOneStage, source_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """compute → writes parquet + _meta."""
        out = stage.run("2025-01", {"source": source_df})

        assert out.exists()
        assert out.suffix == ".parquet"

        result = pd.read_parquet(out)
        assert list(result.columns) == ["id", "value"]
        assert len(result) == 3
        assert result["value"].tolist() == [11.0, 21.0, 31.0]

        meta_dir = out.parent / "_meta"
        base = out.stem  # e.g. "part-2025-01"
        assert (meta_dir / f"{base}.schema.json").exists()
        assert (meta_dir / f"{base}.stats.json").exists()


class TestIsComplete:
    def test_not_complete_before_run(self, stage: AddOneStage) -> None:
        """Before run(), is_complete returns False."""
        assert stage.is_complete("2025-01") is False

    def test_complete_after_run(
        self, stage: AddOneStage, source_df: pd.DataFrame
    ) -> None:
        """After run(), is_complete returns True."""
        stage.run("2025-01", {"source": source_df})
        assert stage.is_complete("2025-01") is True

    def test_other_period_still_incomplete(
        self, stage: AddOneStage, source_df: pd.DataFrame
    ) -> None:
        """Running one period doesn't complete another."""
        stage.run("2025-01", {"source": source_df})
        assert stage.is_complete("2025-01") is True
        assert stage.is_complete("2025-02") is False

    def test_output_path_matches_run(
        self, stage: AddOneStage, source_df: pd.DataFrame
    ) -> None:
        """output_path() returns the same path that run() writes to."""
        out = stage.run("2025-01", {"source": source_df})
        assert stage.output_path("2025-01") == out


class TestInputValidation:
    def test_missing_required_column(
        self, stage: AddOneStage, tmp_path: Path
    ) -> None:
        """Missing required col → SchemaViolationError."""
        bad_df = pd.DataFrame({"id": [1], "wrong_col": [10.0]})
        with pytest.raises(SchemaViolationError, match="missing columns"):
            stage.run("2025-01", {"source": bad_df})

    def test_missing_input_key(
        self, stage: AddOneStage, source_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Missing input key → SchemaViolationError."""
        with pytest.raises(SchemaViolationError, match="missing input keys"):
            stage.run("2025-01", {"wrong_key": source_df})

    def test_empty_input_raises(self, stage: AddOneStage, tmp_path: Path) -> None:
        """Empty df with allow_empty=False → EmptyDataError."""
        empty_df = pd.DataFrame({"id": pd.Series([], dtype="int64"), "raw_value": pd.Series([], dtype="float64")})
        with pytest.raises(EmptyDataError, match="input 'source' is empty"):
            stage.run("2025-01", {"source": empty_df})

    def test_empty_input_allowed(self, tmp_path: Path) -> None:
        """Empty df with allow_empty=True passes input validation."""

        class AllowEmptyStage(Stage):
            artifact_name = "test_allow_empty"
            OUTPUT_SCHEMA = {"id": "int64"}
            REQUIRED_INPUTS = {
                "source": InputSpec(required_cols=["id"], allow_empty=True),
            }
            EMPTY_POLICY = EmptyPolicy.WRITE_EMPTY

            def compute(
                self, inputs: dict[str, pd.DataFrame], period: str
            ) -> pd.DataFrame:
                return inputs["source"][["id"]]

        s = AllowEmptyStage(data_root=tmp_path)
        empty_df = pd.DataFrame({"id": pd.Series([], dtype="int64")})
        out = s.run("2025-01", {"source": empty_df})
        assert out.exists()


class TestOutputSchemaEnforcement:
    def test_missing_output_column(self, tmp_path: Path) -> None:
        """Missing output col → SchemaViolationError."""

        class BadOutputStage(Stage):
            artifact_name = "test_bad_output"
            OUTPUT_SCHEMA = {"id": "int64", "value": "float64", "extra": "string"}
            REQUIRED_INPUTS = {
                "source": InputSpec(required_cols=["id", "raw_value"]),
            }
            EMPTY_POLICY = EmptyPolicy.FAIL

            def compute(
                self, inputs: dict[str, pd.DataFrame], period: str
            ) -> pd.DataFrame:
                df = inputs["source"].copy()
                df["value"] = df["raw_value"] + 1
                return df[["id", "value"]]  # missing "extra"

        s = BadOutputStage(data_root=tmp_path)
        source_df = pd.DataFrame({"id": [1], "raw_value": [10.0]})
        with pytest.raises(SchemaViolationError, match="output missing columns"):
            s.run("2025-01", {"source": source_df})

    def test_column_order_enforced(
        self, tmp_path: Path, source_df: pd.DataFrame
    ) -> None:
        """Output columns match OUTPUT_SCHEMA order."""

        class ReverseStage(Stage):
            artifact_name = "test_reverse"
            OUTPUT_SCHEMA = {"id": "int64", "value": "float64"}
            REQUIRED_INPUTS = {
                "source": InputSpec(required_cols=["id", "raw_value"]),
            }
            EMPTY_POLICY = EmptyPolicy.FAIL

            def compute(
                self, inputs: dict[str, pd.DataFrame], period: str
            ) -> pd.DataFrame:
                df = inputs["source"].copy()
                df["value"] = df["raw_value"] + 1
                # Return columns in reversed order
                return df[["value", "id"]]

        s = ReverseStage(data_root=tmp_path)
        out = s.run("2025-01", {"source": source_df})
        result = pd.read_parquet(out)
        assert list(result.columns) == ["id", "value"]


class TestTemporalLeak:
    def test_temporal_leak_raises(self, tmp_path: Path) -> None:
        """Forbidden column → TemporalLeakError."""
        boundary = TemporalBoundary(
            forbidden_columns=frozenset({"value"}),
        )

        class LeakyStage(Stage):
            artifact_name = "test_leaky"
            OUTPUT_SCHEMA = {"id": "int64", "value": "float64"}
            REQUIRED_INPUTS = {
                "source": InputSpec(required_cols=["id", "raw_value"]),
            }
            EMPTY_POLICY = EmptyPolicy.FAIL

            def compute(
                self, inputs: dict[str, pd.DataFrame], period: str
            ) -> pd.DataFrame:
                df = inputs["source"].copy()
                df["value"] = df["raw_value"] + 1
                return df[["id", "value"]]

        s = LeakyStage(data_root=tmp_path, boundary=boundary)
        source_df = pd.DataFrame({"id": [1], "raw_value": [10.0]})
        with pytest.raises(TemporalLeakError):
            s.run("2025-01", {"source": source_df})

    def test_no_boundary_skips_check(
        self, stage: AddOneStage, source_df: pd.DataFrame
    ) -> None:
        """boundary=None → no error even with columns that would be forbidden."""
        # stage has boundary=None by default
        out = stage.run("2025-01", {"source": source_df})
        assert out.exists()


class TestEmptyPolicy:
    def test_empty_policy_fail(self, tmp_path: Path) -> None:
        """Empty output with FAIL policy → EmptyDataError."""

        class EmptyOutputStage(Stage):
            artifact_name = "test_empty_fail"
            OUTPUT_SCHEMA = {"id": "int64", "value": "float64"}
            REQUIRED_INPUTS = {
                "source": InputSpec(required_cols=["id", "raw_value"], allow_empty=True),
            }
            EMPTY_POLICY = EmptyPolicy.FAIL

            def compute(
                self, inputs: dict[str, pd.DataFrame], period: str
            ) -> pd.DataFrame:
                return inputs["source"][["id"]].rename(columns={}).head(0).assign(
                    id=pd.Series([], dtype="int64"),
                    value=pd.Series([], dtype="float64"),
                )

        s = EmptyOutputStage(data_root=tmp_path)
        empty_df = pd.DataFrame({"id": pd.Series([], dtype="int64"), "raw_value": pd.Series([], dtype="float64")})
        with pytest.raises(EmptyDataError, match="compute returned empty"):
            s.run("2025-01", {"source": empty_df})

    def test_empty_policy_write_empty(self, tmp_path: Path) -> None:
        """Empty output with WRITE_EMPTY → writes 0-row parquet."""

        class WriteEmptyStage(Stage):
            artifact_name = "test_write_empty"
            OUTPUT_SCHEMA = {"id": "int64", "value": "float64"}
            REQUIRED_INPUTS = {
                "source": InputSpec(required_cols=["id"], allow_empty=True),
            }
            EMPTY_POLICY = EmptyPolicy.WRITE_EMPTY

            def compute(
                self, inputs: dict[str, pd.DataFrame], period: str
            ) -> pd.DataFrame:
                return pd.DataFrame(
                    {"id": pd.Series([], dtype="int64"), "value": pd.Series([], dtype="float64")}
                )

        s = WriteEmptyStage(data_root=tmp_path)
        empty_df = pd.DataFrame({"id": pd.Series([], dtype="int64")})
        out = s.run("2025-01", {"source": empty_df})
        assert out.exists()
        result = pd.read_parquet(out)
        assert len(result) == 0

    def test_empty_policy_upstream_empty_true(self, tmp_path: Path) -> None:
        """upstream_empty=True with WRITE_EMPTY_IF_UPSTREAM_EMPTY → writes."""

        class ConditionalStage(Stage):
            artifact_name = "test_cond"
            OUTPUT_SCHEMA = {"id": "int64"}
            REQUIRED_INPUTS = {
                "source": InputSpec(required_cols=["id"], allow_empty=True),
            }
            EMPTY_POLICY = EmptyPolicy.WRITE_EMPTY_IF_UPSTREAM_EMPTY

            def compute(
                self, inputs: dict[str, pd.DataFrame], period: str
            ) -> pd.DataFrame:
                return pd.DataFrame({"id": pd.Series([], dtype="int64")})

        s = ConditionalStage(data_root=tmp_path)
        empty_df = pd.DataFrame({"id": pd.Series([], dtype="int64")})
        out = s.run("2025-01", {"source": empty_df}, upstream_empty=True)
        assert out.exists()

    def test_empty_policy_upstream_empty_false_raises(
        self, tmp_path: Path
    ) -> None:
        """upstream_empty=False with WRITE_EMPTY_IF_UPSTREAM_EMPTY → raises."""

        class ConditionalStage(Stage):
            artifact_name = "test_cond2"
            OUTPUT_SCHEMA = {"id": "int64"}
            REQUIRED_INPUTS = {
                "source": InputSpec(required_cols=["id"], allow_empty=True),
            }
            EMPTY_POLICY = EmptyPolicy.WRITE_EMPTY_IF_UPSTREAM_EMPTY

            def compute(
                self, inputs: dict[str, pd.DataFrame], period: str
            ) -> pd.DataFrame:
                return pd.DataFrame({"id": pd.Series([], dtype="int64")})

        s = ConditionalStage(data_root=tmp_path)
        empty_df = pd.DataFrame({"id": pd.Series([], dtype="int64")})
        with pytest.raises(EmptyDataError, match="upstream_empty=False"):
            s.run("2025-01", {"source": empty_df}, upstream_empty=False)


class TestHooks:
    def test_hooks_called(
        self, tmp_path: Path, source_df: pd.DataFrame
    ) -> None:
        """preprocess, validate, postprocess, build_meta all called."""
        call_log: list[str] = []

        class HookedStage(Stage):
            artifact_name = "test_hooked"
            OUTPUT_SCHEMA = {"id": "int64", "value": "float64"}
            REQUIRED_INPUTS = {
                "source": InputSpec(required_cols=["id", "raw_value"]),
            }
            EMPTY_POLICY = EmptyPolicy.FAIL

            def preprocess_inputs(
                self, inputs: dict[str, pd.DataFrame], period: str
            ) -> dict[str, pd.DataFrame]:
                call_log.append("preprocess")
                return inputs

            def compute(
                self, inputs: dict[str, pd.DataFrame], period: str
            ) -> pd.DataFrame:
                call_log.append("compute")
                df = inputs["source"].copy()
                df["value"] = df["raw_value"] + 1
                return df[["id", "value"]]

            def validate(self, df: pd.DataFrame, period: str) -> None:
                call_log.append("validate")

            def postprocess(
                self, df: pd.DataFrame, period: str
            ) -> pd.DataFrame:
                call_log.append("postprocess")
                return df

            def build_meta(
                self, df: pd.DataFrame, period: str
            ) -> dict[str, Any] | None:
                call_log.append("build_meta")
                return {"custom": True}

        s = HookedStage(data_root=tmp_path)
        s.run("2025-01", {"source": source_df})

        assert call_log == [
            "preprocess",
            "compute",
            "validate",
            "postprocess",
            "build_meta",
        ]
