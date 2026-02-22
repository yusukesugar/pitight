"""Tests for pitight.validation module."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pitight.validation import (
    SourceValidationError,
    validate_df_schema,
    validate_source_meta,
)


# ============================================================
# validate_source_meta
# ============================================================


class TestValidateSourceMeta:
    """Tests for metadata file presence and date range checks."""

    def test_non_partitioned_passes(self, tmp_path: Path) -> None:
        """Non-partitioned artifact with schema + stats passes."""
        data_dir = tmp_path / "features"
        data_dir.mkdir()
        (tmp_path / "part.schema.json").write_text("{}")
        (tmp_path / "part.stats.json").write_text("{}")

        # root_dir is the file-level dir; meta lives in parent
        validate_source_meta(data_dir, base_name="part")

    def test_non_partitioned_missing_schema(self, tmp_path: Path) -> None:
        """Missing schema.json raises SourceValidationError."""
        data_dir = tmp_path / "features"
        data_dir.mkdir()
        (tmp_path / "part.stats.json").write_text("{}")

        with pytest.raises(SourceValidationError, match="Schema missing"):
            validate_source_meta(data_dir, base_name="part")

    def test_non_partitioned_missing_stats(self, tmp_path: Path) -> None:
        """Missing stats.json raises SourceValidationError."""
        data_dir = tmp_path / "features"
        data_dir.mkdir()
        (tmp_path / "part.schema.json").write_text("{}")

        with pytest.raises(SourceValidationError, match="Stats missing"):
            validate_source_meta(data_dir, base_name="part")

    def test_partitioned_passes(self, tmp_path: Path) -> None:
        """Partitioned artifact with manifest + schema + stats passes."""
        meta_dir = tmp_path / "_meta"
        meta_dir.mkdir()
        (meta_dir / "manifest.json").write_text("{}")
        (meta_dir / "all.schema.json").write_text("{}")
        (meta_dir / "all.stats.json").write_text("{}")

        validate_source_meta(tmp_path, is_partitioned=True)

    def test_partitioned_missing_manifest(self, tmp_path: Path) -> None:
        """Missing manifest.json for partitioned raises."""
        meta_dir = tmp_path / "_meta"
        meta_dir.mkdir()
        (meta_dir / "all.schema.json").write_text("{}")
        (meta_dir / "all.stats.json").write_text("{}")

        with pytest.raises(SourceValidationError, match="Manifest missing"):
            validate_source_meta(tmp_path, is_partitioned=True)

    def test_date_range_warn_early_date_min(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warn when date_min is earlier than start_ym."""
        data_dir = tmp_path / "features"
        data_dir.mkdir()
        (tmp_path / "part.schema.json").write_text("{}")
        stats = {"date_min": "2024-06-01", "date_max": "2025-06-30"}
        (tmp_path / "part.stats.json").write_text(json.dumps(stats))

        with caplog.at_level("WARNING"):
            validate_source_meta(data_dir, start_ym="2025-01", end_ym="2025-06")

        assert "earlier than start_ym" in caplog.text

    def test_date_range_warn_late_date_max(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warn when date_max is later than end_ym."""
        data_dir = tmp_path / "features"
        data_dir.mkdir()
        (tmp_path / "part.schema.json").write_text("{}")
        stats = {"date_min": "2025-01-01", "date_max": "2025-12-31"}
        (tmp_path / "part.stats.json").write_text(json.dumps(stats))

        with caplog.at_level("WARNING"):
            validate_source_meta(data_dir, start_ym="2025-01", end_ym="2025-06")

        assert "later than end_ym" in caplog.text

    def test_yyyymm_format_normalized(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """YYYYMM format (no dash) is correctly normalized."""
        data_dir = tmp_path / "features"
        data_dir.mkdir()
        (tmp_path / "part.schema.json").write_text("{}")
        stats = {"date_min": "2025-01-15", "date_max": "2025-06-15"}
        (tmp_path / "part.stats.json").write_text(json.dumps(stats))

        # Should not raise — 202501 → 2025-01
        validate_source_meta(data_dir, start_ym="202501", end_ym="202506")


# ============================================================
# validate_df_schema
# ============================================================


class TestValidateDfSchema:
    """Tests for DataFrame vs schema.json validation."""

    @pytest.fixture()
    def schema_path(self, tmp_path: Path) -> Path:
        """Create a sample schema.json."""
        schema = {
            "columns": {
                "id": {"dtype": "int64", "nullable": False},
                "name": {"dtype": "object", "nullable": True},
                "score": {"dtype": "float64", "nullable": False},
            }
        }
        p = tmp_path / "part.schema.json"
        p.write_text(json.dumps(schema))
        return p

    def test_valid_df_passes(self, schema_path: Path) -> None:
        """DataFrame matching schema passes silently."""
        df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "score": [0.5, 0.9]})
        validate_df_schema(df, schema_path)

    def test_missing_requested_columns(self, schema_path: Path) -> None:
        """Requesting columns that don't exist raises."""
        df = pd.DataFrame({"id": [1], "name": ["a"]})

        with pytest.raises(SourceValidationError, match="Missing requested columns"):
            validate_df_schema(df, schema_path, columns=["id", "name", "score"])

    def test_nullable_violation(self, schema_path: Path) -> None:
        """Non-nullable column with NULLs raises."""
        df = pd.DataFrame(
            {"id": [1, None], "name": ["a", "b"], "score": [0.5, 0.9]}
        )

        with pytest.raises(SourceValidationError, match="not nullable"):
            validate_df_schema(df, schema_path)

    def test_dtype_mismatch_warns(
        self, schema_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Dtype mismatch produces a warning but does not raise."""
        df = pd.DataFrame(
            {"id": [1, 2], "name": ["a", "b"], "score": ["x", "y"]}  # object, not float64
        )

        with caplog.at_level("WARNING"):
            validate_df_schema(df, schema_path)

        assert "Dtype mismatch" in caplog.text
        assert "score" in caplog.text

    def test_extra_columns_ignored(self, tmp_path: Path) -> None:
        """Columns in DataFrame but not in schema are silently ignored."""
        schema = {"columns": {"id": {"dtype": "int64", "nullable": False}}}
        p = tmp_path / "schema.json"
        p.write_text(json.dumps(schema))

        df = pd.DataFrame({"id": [1], "extra": ["x"]})
        validate_df_schema(df, p)  # should not raise
