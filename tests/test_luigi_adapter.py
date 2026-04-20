import json
import os
import time
from pathlib import Path

import pytest

luigi = pytest.importorskip("luigi")

from pitight.artifact import ArtifactRegistry
from pitight.luigi_adapter import (
    FreshCompleteMixin,
    artifact_from_luigi_task,
    register_luigi_output,
)


class DummyTask(luigi.Task):
    ym = luigi.Parameter(default="2025-01")
    version = luigi.Parameter(default="v1")

    def output(self):
        return luigi.LocalTarget(f"/tmp/data/{self.ym}.parquet")


class TestArtifactFromLuigiTask:
    def test_basic(self):
        task = DummyTask(ym="2025-03")
        art = artifact_from_luigi_task(task, config_hash="abc123")
        assert art.name.endswith("DummyTask")
        assert art.path == "/tmp/data/2025-03.parquet"
        assert art.config_hash == "abc123"

    def test_params_in_metadata(self):
        task = DummyTask(ym="2025-03", version="v2")
        art = artifact_from_luigi_task(task, config_hash="abc123")
        assert art.metadata["luigi_params"]["ym"] == "2025-03"
        assert art.metadata["luigi_params"]["version"] == "v2"


class TestRegisterLuigiOutput:
    def test_register(self, tmp_path):
        registry = ArtifactRegistry(str(tmp_path / "artifacts.json"))
        task = DummyTask(ym="2025-06")
        art = register_luigi_output(task, registry, config_hash="def456")
        assert len(registry.list_all()) == 1
        assert art.config_hash == "def456"


# ============================================================
# FreshCompleteMixin
# ============================================================


def _make_task(
    tmp_path: Path,
    *,
    ym: str = "2026-04",
    min_row_count: int = 0,
    max_age_seconds: float = 0.0,
    expected_periods: tuple[str, ...] = (),
    upstream_manifests: tuple[Path, ...] = (),
    write_parquet: bool = True,
):
    """Build an anonymous FreshCompleteMixin + luigi.Task class with
    configurable hooks. The Luigi output lives at
    ``tmp_path/root/data/year=Y/month=M/part-YM.parquet`` and the
    manifest at ``tmp_path/root/manifest.json``.
    """
    root = tmp_path / "root"
    y, m = ym.split("-")
    parquet = root / "data" / f"year={y}" / f"month={m}" / f"part-{ym}.parquet"
    if write_parquet:
        parquet.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(b"stub")
    manifest = root / "manifest.json"

    class Task(FreshCompleteMixin, luigi.Task):
        MIN_ROW_COUNT = min_row_count
        MAX_MANIFEST_AGE_SECONDS = max_age_seconds

        def output(self):
            return luigi.LocalTarget(parquet.as_posix())

        def expected_periods(self):
            return expected_periods

        def upstream_manifests(self):
            return upstream_manifests

    return Task(), manifest, parquet


def _write_manifest(path: Path, body: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


class TestFreshCompleteMixinBasic:
    def test_missing_output_is_not_complete(self, tmp_path):
        task, _, _ = _make_task(tmp_path, write_parquet=False)
        assert task.complete() is False

    def test_output_exists_but_no_manifest_is_not_complete(self, tmp_path):
        task, manifest, _ = _make_task(tmp_path)
        assert not manifest.exists()
        assert task.complete() is False

    def test_coverage_ok_true_is_complete(self, tmp_path):
        task, manifest, _ = _make_task(tmp_path)
        _write_manifest(manifest, {"coverage": {"coverage_ok": True}})
        assert task.complete() is True

    def test_coverage_ok_false_is_not_complete(self, tmp_path):
        task, manifest, _ = _make_task(tmp_path)
        _write_manifest(manifest, {"coverage": {"coverage_ok": False}})
        assert task.complete() is False


class TestFreshCompleteMixinExpectedPeriods:
    def test_missing_expected_period(self, tmp_path):
        task, manifest, _ = _make_task(
            tmp_path,
            expected_periods=("2026-04-01", "2026-04-02", "2026-04-03"),
        )
        _write_manifest(
            manifest,
            {
                "coverage": {
                    "coverage_ok": True,
                    "present_periods": ["2026-04-01"],
                }
            },
        )
        assert task.complete() is False

    def test_all_expected_present(self, tmp_path):
        task, manifest, _ = _make_task(
            tmp_path,
            expected_periods=("2026-04-01", "2026-04-02"),
        )
        _write_manifest(
            manifest,
            {
                "coverage": {
                    "coverage_ok": True,
                    "present_periods": ["2026-04-01", "2026-04-02", "2026-04-03"],
                }
            },
        )
        assert task.complete() is True


class TestFreshCompleteMixinRowCount:
    def test_row_count_below_minimum(self, tmp_path):
        # Shape: writer claims coverage_ok but only produced a fraction
        # of the expected rows (partial rebuild / stale input).
        task, manifest, _ = _make_task(tmp_path, min_row_count=20000)
        _write_manifest(
            manifest,
            {
                "coverage": {"coverage_ok": True},
                "stats_rollup": {"row_count_total": 720},
            },
        )
        assert task.complete() is False

    def test_row_count_meets_minimum(self, tmp_path):
        task, manifest, _ = _make_task(tmp_path, min_row_count=20000)
        _write_manifest(
            manifest,
            {
                "coverage": {"coverage_ok": True},
                "stats_rollup": {"row_count_total": 24594},
            },
        )
        assert task.complete() is True


class TestFreshCompleteMixinFreshness:
    def test_stale_manifest_is_not_complete(self, tmp_path):
        task, manifest, _ = _make_task(tmp_path, max_age_seconds=3600)
        _write_manifest(manifest, {"coverage": {"coverage_ok": True}})
        past = time.time() - 7200
        os.utime(manifest, (past, past))
        assert task.complete() is False

    def test_fresh_manifest_is_complete(self, tmp_path):
        task, manifest, _ = _make_task(tmp_path, max_age_seconds=3600)
        _write_manifest(manifest, {"coverage": {"coverage_ok": True}})
        assert task.complete() is True


class TestFreshCompleteMixinUpstream:
    def test_upstream_newer_is_not_complete(self, tmp_path):
        upstream = _write_manifest(
            tmp_path / "up.json", {"coverage": {"coverage_ok": True}}
        )
        task, manifest, _ = _make_task(
            tmp_path, upstream_manifests=(upstream,)
        )
        _write_manifest(manifest, {"coverage": {"coverage_ok": True}})
        # Make upstream 1 hour newer than self.
        past = time.time() - 3600
        os.utime(manifest, (past, past))
        assert task.complete() is False

    def test_upstream_older_is_complete(self, tmp_path):
        upstream = _write_manifest(
            tmp_path / "up.json", {"coverage": {"coverage_ok": True}}
        )
        # Backdate upstream.
        past = time.time() - 3600
        os.utime(upstream, (past, past))
        task, manifest, _ = _make_task(
            tmp_path, upstream_manifests=(upstream,)
        )
        _write_manifest(manifest, {"coverage": {"coverage_ok": True}})
        assert task.complete() is True
