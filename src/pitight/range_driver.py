"""RangeDriver — orchestrator-agnostic multi-period execution for PartitionedArtifact.

Composes PartitionedArtifact with partition utilities to provide:
- Period enumeration and pending detection
- Sequential single-period execution
- Finalization (manifest + _SUCCESS)

Does NOT import any orchestrator (Luigi, Airflow, Dagster).
Scheduling and parallelism are the caller's responsibility.

Usage:
    from pitight import PartitionedArtifact, RangeDriver

    class MyArtifact(PartitionedArtifact):
        ...

    artifact = MyArtifact(data_root="/data/output")
    driver = RangeDriver(artifact, start="2025-01", end="2025-06")

    for period in driver.pending_periods():
        inputs = load_inputs(period)
        driver.run_period(period, inputs)

    driver.finalize()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pitight.partition import (
    build_manifest,
    compute_coverage,
    resolve_expected_periods,
    rollup_stats,
    scan_present_months,
    update_success_marker,
    write_manifest,
)
from pitight.partitioned import PartitionedArtifact

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of a single period execution."""

    period: str
    output_path: Path
    row_count: int


class RangeDriver:
    """Orchestrator-agnostic driver for running a PartitionedArtifact over a date range.

    Args:
        artifact: The PartitionedArtifact instance to drive.
        start: Start period string (e.g. "2025-01" for monthly).
        end: End period string (e.g. "2025-06" for monthly).
        identity_params: Optional identity parameters included in manifest.
    """

    def __init__(
        self,
        artifact: PartitionedArtifact,
        start: str,
        end: str,
        *,
        identity_params: dict[str, Any] | None = None,
    ) -> None:
        self._artifact = artifact
        self._start = start
        self._end = end
        self._identity_params = identity_params or {}
        self._results: list[RunResult] = []

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def expected_periods(self) -> list[str]:
        """Return all periods in the [start, end] range."""
        return resolve_expected_periods(
            self._start, self._end, freq=self._artifact.freq
        )

    def pending_periods(self, *, force: bool = False) -> list[str]:
        """Return periods that have not been completed yet.

        Args:
            force: If True, return all expected periods regardless of completion.
        """
        expected = self.expected_periods()
        if force:
            return expected
        return [p for p in expected if not self._artifact.is_complete(p)]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_period(
        self,
        period: str,
        inputs: dict[str, Any],
        *,
        upstream_empty: bool = False,
    ) -> RunResult:
        """Run the artifact for a single period and record the result.

        Args:
            period: Period string (must be within [start, end]).
            inputs: Named input DataFrames for PartitionedArtifact.run().
            upstream_empty: Passed through to PartitionedArtifact.run().

        Returns:
            RunResult with output path and row count.

        Raises:
            ValueError: If period is outside the [start, end] range.
        """
        expected = self.expected_periods()
        if period not in expected:
            raise ValueError(
                f"Period {period!r} is outside range [{self._start}, {self._end}]"
            )

        output_path = self._artifact.run(
            period, inputs, upstream_empty=upstream_empty
        )
        row_count = self._read_row_count(output_path)
        result = RunResult(
            period=period, output_path=output_path, row_count=row_count
        )
        self._results.append(result)
        logger.info(
            "RangeDriver: %s [%s] → %d rows",
            self._artifact.artifact_name,
            period,
            row_count,
        )
        return result

    def run_all(
        self,
        input_fn: Callable[[str], dict[str, Any]],
        *,
        force: bool = False,
    ) -> list[RunResult]:
        """Run all pending periods sequentially and finalize.

        Args:
            input_fn: Callable that takes a period string and returns inputs dict.
            force: If True, re-run all periods regardless of completion.

        Returns:
            List of RunResult for each executed period.
        """
        results: list[RunResult] = []
        for period in self.pending_periods(force=force):
            inputs = input_fn(period)
            result = self.run_period(period, inputs)
            results.append(result)
        self.finalize()
        return results

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self) -> Path:
        """Write manifest.json and _SUCCESS marker based on disk state.

        Scans the artifact data directory to determine present periods,
        computes coverage, collects per-period stats, and writes the manifest.

        Returns:
            Path to the written manifest.json.
        """
        expected = self.expected_periods()
        data_dir = self._artifact.data_root / self._artifact.artifact_name / "data"
        present = scan_present_months(data_dir)
        missing, coverage_ok = compute_coverage(expected, present)

        stats_list = self._collect_stats(present)
        stats = rollup_stats(stats_list)

        manifest = build_manifest(
            artifact_id=self._artifact.artifact_name,
            identity_params=self._identity_params,
            request={"start": self._start, "end": self._end},
            expected_periods=expected,
            present_periods=present,
            missing_periods=missing,
            coverage_ok=coverage_ok,
            stats_rollup=stats,
        )

        run_dir = self._run_dir()
        manifest_path = write_manifest(run_dir, manifest)

        success_path = run_dir / "_SUCCESS"
        update_success_marker(success_path, coverage_ok)

        status = "COMPLETE" if coverage_ok else "PARTIAL"
        logger.info(
            "RangeDriver: finalize %s [%s..%s] → %s (%d/%d periods)",
            self._artifact.artifact_name,
            self._start,
            self._end,
            status,
            len(present),
            len(expected),
        )
        return manifest_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_dir(self) -> Path:
        """Return the run directory for this range."""
        return (
            self._artifact.data_root
            / self._artifact.artifact_name
            / "_runs"
            / f"start={self._start}__end={self._end}"
        )

    def _read_row_count(self, parquet_path: Path) -> int:
        """Read row count from the stats.json next to the parquet file."""
        meta_dir = parquet_path.parent / "_meta"
        stats_path = meta_dir / f"{parquet_path.stem}.stats.json"
        if stats_path.exists():
            with open(stats_path, encoding="utf-8") as f:
                stats = json.load(f)
            return stats.get("n_rows", 0)
        return 0

    def _collect_stats(self, periods: list[str]) -> list[dict[str, Any]]:
        """Collect per-period stats.json and normalize to rollup_stats format.

        rollup_stats expects ``row_count`` key; infer_stats writes ``n_rows``.
        This method bridges the two conventions.
        """
        result: list[dict[str, Any]] = []
        for period in periods:
            parquet_path = self._artifact.output_path(period)
            n_rows = self._read_row_count(parquet_path)
            result.append({"row_count": n_rows})
        return result
