"""Tests for pitight.json_utils — atomic JSON writing."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from pitight.json_utils import json_serial, write_json_atomically


class TestJsonSerial:
    def test_datetime(self) -> None:
        dt = datetime.datetime(2025, 1, 15, 12, 30, 0)
        assert json_serial(dt) == "2025-01-15T12:30:00"

    def test_date(self) -> None:
        d = datetime.date(2025, 6, 1)
        assert json_serial(d) == "2025-06-01"

    def test_unsupported_type(self) -> None:
        with pytest.raises(TypeError, match="not serializable"):
            json_serial(42)


class TestWriteJsonAtomically:
    def test_basic(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        write_json_atomically(path, {"key": "value"})

        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == {"key": "value"}

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deep" / "out.json"
        write_json_atomically(path, {"a": 1})

        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["a"] == 1

    def test_no_tmp_file_left(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        write_json_atomically(path, {"a": 1})

        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "out.json"

    def test_sort_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "sorted.json"
        write_json_atomically(path, {"z": 1, "a": 2}, sort_keys=True)

        text = path.read_text()
        assert text.index('"a"') < text.index('"z"')

    def test_default_serializer_handles_datetime(self, tmp_path: Path) -> None:
        path = tmp_path / "dt.json"
        write_json_atomically(path, {"ts": datetime.datetime(2025, 1, 1)})

        loaded = json.loads(path.read_text())
        assert loaded["ts"] == "2025-01-01T00:00:00"

    def test_custom_default(self, tmp_path: Path) -> None:
        path = tmp_path / "custom.json"

        def custom(obj: object) -> str:
            if isinstance(obj, set):
                return sorted(obj)  # type: ignore[return-value]
            raise TypeError

        write_json_atomically(path, {"s": {3, 1, 2}}, default=custom)

        loaded = json.loads(path.read_text())
        assert loaded["s"] == [1, 2, 3]

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        write_json_atomically(path, {"v": 1})
        write_json_atomically(path, {"v": 2})

        loaded = json.loads(path.read_text())
        assert loaded["v"] == 2

    def test_ensure_ascii_false(self, tmp_path: Path) -> None:
        path = tmp_path / "unicode.json"
        write_json_atomically(path, {"name": "日本語"})

        text = path.read_text(encoding="utf-8")
        assert "日本語" in text
