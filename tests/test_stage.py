"""Tests for pitight.stage — PartitionedArtifact declarative lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pitight.assertions import EmptyDataError, SchemaViolationError
from pitight.stage import EmptyPolicy, InputSpec, PartitionedArtifact
from pitight.temporal_leak import TemporalBoundary, TemporalLeakError


# ============================================================
# Concrete test artifact
# ============================================================


class AddOneArtifact(PartitionedArtifact):
    """Minimal concrete artifact for testing."""

    artifact_name = "test_stage/add_one"
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
def artifact(tmp_path: Path) -> AddOneArtifact:
    return AddOneArtifact(data_root=tmp_path)


# ============================================================
# Tests
# ============================================================


class TestBasicLifecycle:
    def test_basic_lifecycle(
        self, artifact: AddOneArtifact, source_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """compute → writes parquet + _meta."""
        out = artifact.run("2025-01", {"source": source_df})

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

    def test_hierarchical_artifact_name(
        self, artifact: AddOneArtifact, source_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """artifact_name with stage prefix creates nested directory."""
        out = artifact.run("2025-01", {"source": source_df})
        # test_stage/add_one/data/year=2025/month=01/...
        assert "test_stage" in str(out)
        assert "add_one" in str(out)


class TestIsComplete:
    def test_not_complete_before_run(self, artifact: AddOneArtifact) -> None:
        """Before run(), is_complete returns False."""
        assert artifact.is_complete("2025-01") is False

    def test_complete_after_run(
        self, artifact: AddOneArtifact, source_df: pd.DataFrame
    ) -> None:
        """After run(), is_complete returns True."""
        artifact.run("2025-01", {"source": source_df})
        assert artifact.is_complete("2025-01") is True

    def test_other_period_still_incomplete(
        self, artifact: AddOneArtifact, source_df: pd.DataFrame
    ) -> None:
        """Running one period doesn't complete another."""
        artifact.run("2025-01", {"source": source_df})
        assert artifact.is_complete("2025-01") is True
        assert artifact.is_complete("2025-02") is False

    def test_output_path_matches_run(
        self, artifact: AddOneArtifact, source_df: pd.DataFrame
    ) -> None:
        """output_path() returns the same path that run() writes to."""
        out = artifact.run("2025-01", {"source": source_df})
        assert artifact.output_path("2025-01") == out


class TestInputValidation:
    def test_missing_required_column(
        self, artifact: AddOneArtifact
    ) -> None:
        """Missing required col → SchemaViolationError."""
        bad_df = pd.DataFrame({"id": [1], "wrong_col": [10.0]})
        with pytest.raises(SchemaViolationError, match="missing columns"):
            artifact.run("2025-01", {"source": bad_df})

    def test_missing_input_key(
        self, artifact: AddOneArtifact, source_df: pd.DataFrame
    ) -> None:
        """Missing input key → SchemaViolationError."""
        with pytest.raises(SchemaViolationError, match="missing input keys"):
            artifact.run("2025-01", {"wrong_key": source_df})

    def test_empty_input_raises(self, artifact: AddOneArtifact) -> None:
        """Empty df with allow_empty=False → EmptyDataError."""
        empty_df = pd.DataFrame({"id": pd.Series([], dtype="int64"), "raw_value": pd.Series([], dtype="float64")})
        with pytest.raises(EmptyDataError, match="input 'source' is empty"):
            artifact.run("2025-01", {"source": empty_df})

    def test_empty_input_allowed(self, tmp_path: Path) -> None:
        """Empty df with allow_empty=True passes input validation."""

        class AllowEmptyArtifact(PartitionedArtifact):
            artifact_name = "test_stage/allow_empty"
            OUTPUT_SCHEMA = {"id": "int64"}
            REQUIRED_INPUTS = {
                "source": InputSpec(required_cols=["id"], allow_empty=True),
            }
            EMPTY_POLICY = EmptyPolicy.WRITE_EMPTY

            def compute(
                self, inputs: dict[str, pd.DataFrame], period: str
            ) -> pd.DataFrame:
                return inputs["source"][["id"]]

        a = AllowEmptyArtifact(data_root=tmp_path)
        empty_df = pd.DataFrame({"id": pd.Series([], dtype="int64")})
        out = a.run("2025-01", {"source": empty_df})
        assert out.exists()


class TestOutputSchemaEnforcement:
    def test_missing_output_column(self, tmp_path: Path) -> None:
        """Missing output col → SchemaViolationError."""

        class BadOutputArtifact(PartitionedArtifact):
            artifact_name = "test_stage/bad_output"
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

        a = BadOutputArtifact(data_root=tmp_path)
        source_df = pd.DataFrame({"id": [1], "raw_value": [10.0]})
        with pytest.raises(SchemaViolationError, match="output missing columns"):
            a.run("2025-01", {"source": source_df})

    def test_column_order_enforced(
        self, tmp_path: Path, source_df: pd.DataFrame
    ) -> None:
        """Output columns match OUTPUT_SCHEMA order."""

        class ReverseArtifact(PartitionedArtifact):
            artifact_name = "test_stage/reverse"
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

        a = ReverseArtifact(data_root=tmp_path)
        out = a.run("2025-01", {"source": source_df})
        result = pd.read_parquet(out)
        assert list(result.columns) == ["id", "value"]


class TestTemporalLeak:
    def test_temporal_leak_raises(self, tmp_path: Path) -> None:
        """Forbidden column → TemporalLeakError."""
        boundary = TemporalBoundary(
            forbidden_columns=frozenset({"value"}),
        )

        class LeakyArtifact(PartitionedArtifact):
            artifact_name = "test_stage/leaky"
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

        a = LeakyArtifact(data_root=tmp_path, boundary=boundary)
        source_df = pd.DataFrame({"id": [1], "raw_value": [10.0]})
        with pytest.raises(TemporalLeakError):
            a.run("2025-01", {"source": source_df})

    def test_no_boundary_skips_check(
        self, artifact: AddOneArtifact, source_df: pd.DataFrame
    ) -> None:
        """boundary=None → no error even with columns that would be forbidden."""
        out = artifact.run("2025-01", {"source": source_df})
        assert out.exists()


class TestEmptyPolicy:
    def test_empty_policy_fail(self, tmp_path: Path) -> None:
        """Empty output with FAIL policy → EmptyDataError."""

        class EmptyOutputArtifact(PartitionedArtifact):
            artifact_name = "test_stage/empty_fail"
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

        a = EmptyOutputArtifact(data_root=tmp_path)
        empty_df = pd.DataFrame({"id": pd.Series([], dtype="int64"), "raw_value": pd.Series([], dtype="float64")})
        with pytest.raises(EmptyDataError, match="compute returned empty"):
            a.run("2025-01", {"source": empty_df})

    def test_empty_policy_write_empty(self, tmp_path: Path) -> None:
        """Empty output with WRITE_EMPTY → writes 0-row parquet."""

        class WriteEmptyArtifact(PartitionedArtifact):
            artifact_name = "test_stage/write_empty"
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

        a = WriteEmptyArtifact(data_root=tmp_path)
        empty_df = pd.DataFrame({"id": pd.Series([], dtype="int64")})
        out = a.run("2025-01", {"source": empty_df})
        assert out.exists()
        result = pd.read_parquet(out)
        assert len(result) == 0

    def test_empty_policy_upstream_empty_true(self, tmp_path: Path) -> None:
        """upstream_empty=True with WRITE_EMPTY_IF_UPSTREAM_EMPTY → writes."""

        class ConditionalArtifact(PartitionedArtifact):
            artifact_name = "test_stage/cond"
            OUTPUT_SCHEMA = {"id": "int64"}
            REQUIRED_INPUTS = {
                "source": InputSpec(required_cols=["id"], allow_empty=True),
            }
            EMPTY_POLICY = EmptyPolicy.WRITE_EMPTY_IF_UPSTREAM_EMPTY

            def compute(
                self, inputs: dict[str, pd.DataFrame], period: str
            ) -> pd.DataFrame:
                return pd.DataFrame({"id": pd.Series([], dtype="int64")})

        a = ConditionalArtifact(data_root=tmp_path)
        empty_df = pd.DataFrame({"id": pd.Series([], dtype="int64")})
        out = a.run("2025-01", {"source": empty_df}, upstream_empty=True)
        assert out.exists()

    def test_empty_policy_upstream_empty_false_raises(
        self, tmp_path: Path
    ) -> None:
        """upstream_empty=False with WRITE_EMPTY_IF_UPSTREAM_EMPTY → raises."""

        class ConditionalArtifact(PartitionedArtifact):
            artifact_name = "test_stage/cond2"
            OUTPUT_SCHEMA = {"id": "int64"}
            REQUIRED_INPUTS = {
                "source": InputSpec(required_cols=["id"], allow_empty=True),
            }
            EMPTY_POLICY = EmptyPolicy.WRITE_EMPTY_IF_UPSTREAM_EMPTY

            def compute(
                self, inputs: dict[str, pd.DataFrame], period: str
            ) -> pd.DataFrame:
                return pd.DataFrame({"id": pd.Series([], dtype="int64")})

        a = ConditionalArtifact(data_root=tmp_path)
        empty_df = pd.DataFrame({"id": pd.Series([], dtype="int64")})
        with pytest.raises(EmptyDataError, match="upstream_empty=False"):
            a.run("2025-01", {"source": empty_df}, upstream_empty=False)


class TestHooks:
    def test_hooks_called(
        self, tmp_path: Path, source_df: pd.DataFrame
    ) -> None:
        """preprocess, validate, postprocess, build_meta all called."""
        call_log: list[str] = []

        class HookedArtifact(PartitionedArtifact):
            artifact_name = "test_stage/hooked"
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

        a = HookedArtifact(data_root=tmp_path)
        a.run("2025-01", {"source": source_df})

        assert call_log == [
            "preprocess",
            "compute",
            "validate",
            "postprocess",
            "build_meta",
        ]


class TestBackwardCompatibility:
    def test_stage_alias(self) -> None:
        """Stage is an alias for PartitionedArtifact."""
        from pitight.stage import Stage

        assert Stage is PartitionedArtifact
