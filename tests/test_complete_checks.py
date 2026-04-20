"""Tests for pitight.complete_checks."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from pitight.complete_checks import (
    manifest_coverage_ok,
    manifest_expected_present,
    manifest_freshness_ok,
    manifest_min_row_count,
    upstream_newer_than,
)


def _write_manifest(path: Path, body: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


class TestManifestCoverageOk:
    def test_missing_file_returns_false(self, tmp_path):
        assert manifest_coverage_ok(tmp_path / "nope.json") is False

    def test_coverage_ok_true(self, tmp_path):
        m = _write_manifest(
            tmp_path / "manifest.json",
            {"coverage": {"coverage_ok": True}},
        )
        assert manifest_coverage_ok(m) is True

    def test_coverage_ok_false(self, tmp_path):
        m = _write_manifest(
            tmp_path / "manifest.json",
            {"coverage": {"coverage_ok": False}},
        )
        assert manifest_coverage_ok(m) is False

    def test_missing_coverage_key_returns_false(self, tmp_path):
        m = _write_manifest(tmp_path / "manifest.json", {})
        assert manifest_coverage_ok(m) is False

    def test_unreadable_json_returns_false(self, tmp_path):
        m = tmp_path / "manifest.json"
        m.write_text("{not json", encoding="utf-8")
        assert manifest_coverage_ok(m) is False


class TestManifestExpectedPresent:
    def test_all_present_months(self, tmp_path):
        m = _write_manifest(
            tmp_path / "manifest.json",
            {"coverage": {"present_months": ["2026-01", "2026-02", "2026-03"]}},
        )
        ok, missing = manifest_expected_present(m, ["2026-01", "2026-02"])
        assert ok is True
        assert missing == []

    def test_missing_period_reported(self, tmp_path):
        m = _write_manifest(
            tmp_path / "manifest.json",
            {"coverage": {"present_months": ["2026-01"]}},
        )
        ok, missing = manifest_expected_present(m, ["2026-01", "2026-02"])
        assert ok is False
        assert missing == ["2026-02"]

    def test_accepts_present_periods_key(self, tmp_path):
        m = _write_manifest(
            tmp_path / "manifest.json",
            {"coverage": {"present_periods": ["2026-03-01", "2026-03-02"]}},
        )
        ok, missing = manifest_expected_present(m, ["2026-03-01"])
        assert ok is True
        assert missing == []

    def test_missing_manifest_everything_missing(self, tmp_path):
        ok, missing = manifest_expected_present(
            tmp_path / "nope.json", ["2026-01", "2026-02"]
        )
        assert ok is False
        assert missing == ["2026-01", "2026-02"]


class TestManifestMinRowCount:
    def test_row_count_meets_minimum(self, tmp_path):
        m = _write_manifest(
            tmp_path / "manifest.json",
            {"stats_rollup": {"row_count_total": 10000}},
        )
        assert manifest_min_row_count(m, 10000) is True

    def test_row_count_below_minimum(self, tmp_path):
        m = _write_manifest(
            tmp_path / "manifest.json",
            {"stats_rollup": {"row_count_total": 720}},
        )
        # Partial-rebuild shape: writer claims done but only a fraction
        # of expected rows landed.
        assert manifest_min_row_count(m, 20000) is False

    def test_missing_stats_rollup(self, tmp_path):
        m = _write_manifest(tmp_path / "manifest.json", {})
        assert manifest_min_row_count(m, 1) is False


class TestManifestFreshnessOk:
    def test_disabled_when_threshold_zero(self, tmp_path):
        assert manifest_freshness_ok(tmp_path / "nope.json", 0) is True

    def test_fresh_file(self, tmp_path):
        m = _write_manifest(tmp_path / "manifest.json", {})
        assert manifest_freshness_ok(m, 3600) is True

    def test_stale_file(self, tmp_path):
        m = _write_manifest(tmp_path / "manifest.json", {})
        # backdate the manifest by 2 hours
        past = time.time() - 7200
        os.utime(m, (past, past))
        assert manifest_freshness_ok(m, 3600) is False

    def test_missing_file_is_not_fresh(self, tmp_path):
        assert manifest_freshness_ok(tmp_path / "nope.json", 3600) is False


class TestUpstreamNewerThan:
    def test_no_upstream_is_newer(self, tmp_path):
        self_m = _write_manifest(tmp_path / "self.json", {})
        up_m = _write_manifest(tmp_path / "up.json", {})
        # backdate upstream
        past = time.time() - 3600
        os.utime(up_m, (past, past))
        assert upstream_newer_than(self_m, [up_m]) == []

    def test_upstream_newer(self, tmp_path):
        self_m = _write_manifest(tmp_path / "self.json", {})
        # backdate self
        past = time.time() - 3600
        os.utime(self_m, (past, past))
        up_m = _write_manifest(tmp_path / "up.json", {})
        newer = upstream_newer_than(self_m, [up_m])
        assert up_m in newer

    def test_tolerance_absorbs_small_skew(self, tmp_path):
        self_m = _write_manifest(tmp_path / "self.json", {})
        up_m = _write_manifest(tmp_path / "up.json", {})
        # upstream only 1 second newer — should be within default 2s tolerance
        future = time.time() + 1.0
        os.utime(up_m, (future, future))
        assert upstream_newer_than(self_m, [up_m], tolerance_seconds=2.0) == []

    def test_missing_self_everything_is_newer(self, tmp_path):
        up_m = _write_manifest(tmp_path / "up.json", {})
        newer = upstream_newer_than(tmp_path / "self.json", [up_m])
        assert newer == [up_m]

    def test_missing_upstream_skipped(self, tmp_path):
        self_m = _write_manifest(tmp_path / "self.json", {})
        assert upstream_newer_than(self_m, [tmp_path / "missing.json"]) == []
