import json
from pathlib import Path

import pandas as pd

from pitight.schema_stats import infer_schema, infer_stats, write_schema_and_stats


class TestInferSchema:
    def test_numeric_columns(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10, 20, 30]})
        schema = infer_schema(df)
        assert "x" in schema["columns"]
        assert schema["columns"]["x"]["min"] == 1.0
        assert schema["columns"]["x"]["max"] == 3.0
        assert schema["columns"]["x"]["nullable"] is False

    def test_nullable(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0]})
        schema = infer_schema(df)
        assert schema["columns"]["x"]["nullable"] is True

    def test_string_columns(self):
        df = pd.DataFrame({"name": ["a", "b", "a"]})
        schema = infer_schema(df)
        assert schema["columns"]["name"]["approx_nunique"] == 2


class TestInferStats:
    def test_basic(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        stats = infer_stats(df)
        assert stats["n_rows"] == 3
        assert stats["n_cols"] == 1
        assert "created_at" in stats
        assert "git_hash" in stats

    def test_date_range(self):
        df = pd.DataFrame({"date": ["2025-01-01", "2025-01-31"], "val": [1, 2]})
        stats = infer_stats(df, date_col="date")
        assert "date_min" in stats
        assert "date_max" in stats

    def test_extra(self):
        df = pd.DataFrame({"x": [1]})
        stats = infer_stats(df, extra={"custom_key": "custom_value"})
        assert stats["custom_key"] == "custom_value"


class TestWriteSchemaAndStats:
    def test_writes_files(self, tmp_path):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
        schema_path, stats_path = write_schema_and_stats(
            df, tmp_path / "_meta", "part-2025-01"
        )
        assert schema_path.exists()
        assert stats_path.exists()

        schema = json.loads(schema_path.read_text())
        assert "x" in schema["columns"]

        stats = json.loads(stats_path.read_text())
        assert stats["n_rows"] == 2
