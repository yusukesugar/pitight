"""Tests for pitight.partition module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pitight.partition import (
    build_manifest,
    compute_coverage,
    compute_schema_hash,
    encode_identity,
    hive_path,
    resolve_expected_periods,
    rollup_stats,
    scan_present_months,
    update_success_marker,
    write_manifest,
)


# ============================================================
# resolve_expected_periods
# ============================================================


class TestResolveExpectedPeriods:
    def test_monthly_basic(self) -> None:
        result = resolve_expected_periods("2025-01", "2025-03", freq="M")
        assert result == ["2025-01", "2025-02", "2025-03"]

    def test_monthly_cross_year(self) -> None:
        result = resolve_expected_periods("2024-11", "2025-02", freq="M")
        assert result == ["2024-11", "2024-12", "2025-01", "2025-02"]

    def test_monthly_single(self) -> None:
        result = resolve_expected_periods("2025-06", "2025-06", freq="M")
        assert result == ["2025-06"]

    def test_daily_basic(self) -> None:
        result = resolve_expected_periods("2025-01-28", "2025-02-02", freq="D")
        assert result == [
            "2025-01-28",
            "2025-01-29",
            "2025-01-30",
            "2025-01-31",
            "2025-02-01",
            "2025-02-02",
        ]

    def test_daily_single(self) -> None:
        result = resolve_expected_periods("2025-03-15", "2025-03-15", freq="D")
        assert result == ["2025-03-15"]

    def test_unsupported_freq(self) -> None:
        with pytest.raises(ValueError, match="Unsupported freq"):
            resolve_expected_periods("2025-01", "2025-03", freq="W")


# ============================================================
# hive_path
# ============================================================


class TestHivePath:
    def test_monthly(self) -> None:
        root = Path("/data/features")
        result = hive_path(root, "2025-01", freq="M")
        assert result == Path("/data/features/data/year=2025/month=01/part-2025-01.parquet")

    def test_monthly_custom_base(self) -> None:
        root = Path("/data/features")
        result = hive_path(root, "2025-01", freq="M", base_name="output")
        assert result == Path(
            "/data/features/data/year=2025/month=01/output-2025-01.parquet"
        )

    def test_daily(self) -> None:
        root = Path("/data/logs")
        result = hive_path(root, "2025-01-05", freq="D")
        assert result == Path(
            "/data/logs/data/year=2025/month=01/day=05/part-2025-01-05.parquet"
        )

    def test_unsupported_freq(self) -> None:
        with pytest.raises(ValueError, match="Unsupported freq"):
            hive_path(Path("/tmp"), "2025-W01", freq="W")


# ============================================================
# scan_present_months
# ============================================================


class TestScanPresentMonths:
    def test_finds_months(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        for ym in ["2025-01", "2025-03"]:
            y, m = ym.split("-")
            d = data_dir / f"year={y}" / f"month={m}"
            d.mkdir(parents=True)
            (d / f"part-{ym}.parquet").write_text("fake")

        result = scan_present_months(data_dir)
        assert result == ["2025-01", "2025-03"]

    def test_empty_dir(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        assert scan_present_months(data_dir) == []

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        assert scan_present_months(tmp_path / "nope") == []

    def test_ignores_empty_month_dirs(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        d = data_dir / "year=2025" / "month=01"
        d.mkdir(parents=True)
        # No parquet file
        assert scan_present_months(data_dir) == []


# ============================================================
# compute_coverage
# ============================================================


class TestComputeCoverage:
    def test_all_present(self) -> None:
        missing, ok = compute_coverage(["2025-01", "2025-02"], ["2025-01", "2025-02"])
        assert ok is True
        assert missing == []

    def test_some_missing(self) -> None:
        missing, ok = compute_coverage(
            ["2025-01", "2025-02", "2025-03"], ["2025-01", "2025-03"]
        )
        assert ok is False
        assert missing == ["2025-02"]

    def test_extra_present_ignored(self) -> None:
        missing, ok = compute_coverage(
            ["2025-01"], ["2025-01", "2025-02", "2025-03"]
        )
        assert ok is True
        assert missing == []


# ============================================================
# rollup_stats
# ============================================================


class TestRollupStats:
    def test_basic(self) -> None:
        stats = [{"row_count": 100}, {"row_count": 200}]
        result = rollup_stats(stats)
        assert result["row_count_total"] == 300
        assert "updated_at" in result

    def test_empty(self) -> None:
        result = rollup_stats([])
        assert result["row_count_total"] == 0


# ============================================================
# build_manifest & write_manifest
# ============================================================


class TestManifest:
    def test_build_manifest(self) -> None:
        m = build_manifest(
            artifact_id="test",
            identity_params={"version": "v1"},
            request={"start_ym": "2025-01", "end_ym": "2025-03"},
            expected_periods=["2025-01", "2025-02", "2025-03"],
            present_periods=["2025-01", "2025-02", "2025-03"],
            missing_periods=[],
            coverage_ok=True,
            stats_rollup={"row_count_total": 1000},
        )
        assert m["coverage"]["coverage_ok"] is True
        assert m["policy"] == "incremental_partitioned"

    def test_write_manifest(self, tmp_path: Path) -> None:
        m = {"artifact_id": "test", "coverage": {"coverage_ok": True}}
        path = write_manifest(tmp_path / "meta", m)
        assert path.exists()
        import json

        loaded = json.loads(path.read_text())
        assert loaded["artifact_id"] == "test"


# ============================================================
# compute_schema_hash & encode_identity
# ============================================================


class TestIdentity:
    def test_schema_hash_deterministic(self) -> None:
        schema = {"col_a": "int64", "col_b": "float64"}
        h1 = compute_schema_hash(schema)
        h2 = compute_schema_hash(schema)
        assert h1 == h2
        assert len(h1) == 64

    def test_schema_hash_order_independent(self) -> None:
        h1 = compute_schema_hash({"b": "float64", "a": "int64"})
        h2 = compute_schema_hash({"a": "int64", "b": "float64"})
        assert h1 == h2

    def test_schema_hash_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_schema_hash({})

    def test_encode_identity_basic(self) -> None:
        p = encode_identity({"version": "v1", "algo": "lgbm"})
        assert str(p) == "algo=lgbm/version=v1"

    def test_encode_identity_schema_hash_first(self) -> None:
        p = encode_identity({"version": "v1", "schema_hash": "abc123"})
        assert str(p) == "schema_hash=abc123/version=v1"

    def test_encode_identity_empty(self) -> None:
        assert encode_identity({}) == Path()


# ============================================================
# update_success_marker
# ============================================================


class TestSuccessMarker:
    def test_create(self, tmp_path: Path) -> None:
        marker = tmp_path / "run" / "_SUCCESS"
        update_success_marker(marker, ok=True)
        assert marker.exists()
        assert marker.read_text() == "ok"

    def test_remove(self, tmp_path: Path) -> None:
        marker = tmp_path / "_SUCCESS"
        marker.write_text("ok")
        update_success_marker(marker, ok=False)
        assert not marker.exists()

    def test_remove_nonexistent(self, tmp_path: Path) -> None:
        marker = tmp_path / "_SUCCESS"
        update_success_marker(marker, ok=False)  # should not raise
