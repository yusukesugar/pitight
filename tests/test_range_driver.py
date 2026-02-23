"""Tests for pitight.range_driver — RangeDriver multi-period execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pitight.partitioned import EmptyPolicy, InputSpec, PartitionedArtifact
from pitight.range_driver import RangeDriver, RunResult


# ============================================================
# Concrete test artifact
# ============================================================


class DoubleArtifact(PartitionedArtifact):
    """Minimal artifact: doubles the 'value' column."""

    artifact_name = "test_range/double"
    OUTPUT_SCHEMA = {"id": "int64", "value": "float64"}
    REQUIRED_INPUTS = {
        "source": InputSpec(required_cols=["id", "value"]),
    }
    EMPTY_POLICY = EmptyPolicy.FAIL

    def compute(
        self, inputs: dict[str, pd.DataFrame], period: str
    ) -> pd.DataFrame:
        df = inputs["source"].copy()
        df["value"] = df["value"] * 2
        return df[["id", "value"]]


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture()
def artifact(tmp_path: Path) -> DoubleArtifact:
    return DoubleArtifact(data_root=tmp_path)


@pytest.fixture()
def source_df() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})


def make_input_fn(
    source_df: pd.DataFrame,
) -> Any:
    """Create an input_fn that returns the same source for every period."""

    def input_fn(period: str) -> dict[str, pd.DataFrame]:
        return {"source": source_df}

    return input_fn


# ============================================================
# Tests: Query methods
# ============================================================


class TestExpectedPeriods:
    def test_expected_periods(self, artifact: DoubleArtifact) -> None:
        driver = RangeDriver(artifact, "2025-01", "2025-03")
        assert driver.expected_periods() == ["2025-01", "2025-02", "2025-03"]

    def test_single_period(self, artifact: DoubleArtifact) -> None:
        driver = RangeDriver(artifact, "2025-06", "2025-06")
        assert driver.expected_periods() == ["2025-06"]


class TestPendingPeriods:
    def test_pending_all_missing(self, artifact: DoubleArtifact) -> None:
        driver = RangeDriver(artifact, "2025-01", "2025-03")
        assert driver.pending_periods() == ["2025-01", "2025-02", "2025-03"]

    def test_pending_skips_complete(
        self, artifact: DoubleArtifact, source_df: pd.DataFrame
    ) -> None:
        """After running one period, it's excluded from pending."""
        driver = RangeDriver(artifact, "2025-01", "2025-03")
        driver.run_period("2025-01", {"source": source_df})
        assert driver.pending_periods() == ["2025-02", "2025-03"]

    def test_pending_force(
        self, artifact: DoubleArtifact, source_df: pd.DataFrame
    ) -> None:
        """force=True returns all periods regardless of completion."""
        driver = RangeDriver(artifact, "2025-01", "2025-03")
        driver.run_period("2025-01", {"source": source_df})
        assert driver.pending_periods(force=True) == [
            "2025-01",
            "2025-02",
            "2025-03",
        ]


# ============================================================
# Tests: Execution
# ============================================================


class TestRunPeriod:
    def test_run_period_basic(
        self, artifact: DoubleArtifact, source_df: pd.DataFrame
    ) -> None:
        driver = RangeDriver(artifact, "2025-01", "2025-03")
        result = driver.run_period("2025-01", {"source": source_df})

        assert isinstance(result, RunResult)
        assert result.period == "2025-01"
        assert result.output_path.exists()
        assert result.row_count == 3

        df = pd.read_parquet(result.output_path)
        assert df["value"].tolist() == [20.0, 40.0, 60.0]

    def test_run_period_out_of_range(
        self, artifact: DoubleArtifact, source_df: pd.DataFrame
    ) -> None:
        driver = RangeDriver(artifact, "2025-01", "2025-03")
        with pytest.raises(ValueError, match="outside range"):
            driver.run_period("2025-06", {"source": source_df})


# ============================================================
# Tests: Finalization
# ============================================================


class TestFinalize:
    def test_finalize_all_complete(
        self, artifact: DoubleArtifact, source_df: pd.DataFrame
    ) -> None:
        driver = RangeDriver(artifact, "2025-01", "2025-02")

        driver.run_period("2025-01", {"source": source_df})
        driver.run_period("2025-02", {"source": source_df})
        manifest_path = driver.finalize()

        assert manifest_path.exists()
        assert manifest_path.name == "manifest.json"

        # _SUCCESS should exist
        success = manifest_path.parent / "_SUCCESS"
        assert success.exists()

    def test_finalize_partial(
        self, artifact: DoubleArtifact, source_df: pd.DataFrame
    ) -> None:
        """Partial completion → coverage_ok=False, no _SUCCESS."""
        driver = RangeDriver(artifact, "2025-01", "2025-03")
        driver.run_period("2025-01", {"source": source_df})
        manifest_path = driver.finalize()

        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["coverage"]["coverage_ok"] is False
        assert "2025-02" in manifest["coverage"]["missing_periods"]
        assert "2025-03" in manifest["coverage"]["missing_periods"]

        success = manifest_path.parent / "_SUCCESS"
        assert not success.exists()


# ============================================================
# Tests: run_all
# ============================================================


class TestRunAll:
    def test_run_all(
        self, artifact: DoubleArtifact, source_df: pd.DataFrame
    ) -> None:
        driver = RangeDriver(artifact, "2025-01", "2025-02")
        results = driver.run_all(make_input_fn(source_df))

        assert len(results) == 2
        assert results[0].period == "2025-01"
        assert results[1].period == "2025-02"

        # finalize was called — manifest should exist
        run_dir = (
            artifact.data_root
            / artifact.artifact_name
            / "_runs"
            / "start=2025-01__end=2025-02"
        )
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "_SUCCESS").exists()

    def test_run_all_skips_complete(
        self, artifact: DoubleArtifact, source_df: pd.DataFrame
    ) -> None:
        """run_all skips already-completed periods."""
        driver = RangeDriver(artifact, "2025-01", "2025-03")

        # Pre-run one period
        driver.run_period("2025-01", {"source": source_df})

        # run_all should only run 2025-02 and 2025-03
        results = driver.run_all(make_input_fn(source_df))
        assert len(results) == 2
        assert [r.period for r in results] == ["2025-02", "2025-03"]


# ============================================================
# Tests: Manifest content
# ============================================================


class TestManifestContent:
    def test_manifest_content(
        self, artifact: DoubleArtifact, source_df: pd.DataFrame
    ) -> None:
        driver = RangeDriver(artifact, "2025-01", "2025-02")
        driver.run_period("2025-01", {"source": source_df})
        driver.run_period("2025-02", {"source": source_df})
        manifest_path = driver.finalize()

        manifest = json.loads(manifest_path.read_text())

        assert manifest["artifact_id"] == "test_range/double"
        assert manifest["request"] == {"start": "2025-01", "end": "2025-02"}
        assert manifest["coverage"]["expected_periods"] == [
            "2025-01",
            "2025-02",
        ]
        assert manifest["coverage"]["coverage_ok"] is True
        assert manifest["stats_rollup"]["row_count_total"] == 6

    def test_identity_params_in_manifest(
        self, artifact: DoubleArtifact, source_df: pd.DataFrame
    ) -> None:
        driver = RangeDriver(
            artifact,
            "2025-01",
            "2025-01",
            identity_params={"version": "v2", "model": "lgbm"},
        )
        driver.run_period("2025-01", {"source": source_df})
        manifest_path = driver.finalize()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["identity_params"] == {
            "version": "v2",
            "model": "lgbm",
        }
