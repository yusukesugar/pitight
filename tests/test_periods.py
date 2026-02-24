"""Tests for pitight.periods — YYYY-MM string utilities."""

from __future__ import annotations

from datetime import date

import pytest

from pitight.periods import (
    add_months,
    date_to_ym,
    format_ym,
    parse_ym,
    prev_ym,
    prev_ym_str,
    ym_from_date_str,
    ym_to_date,
)


class TestParseYm:
    def test_basic(self) -> None:
        assert parse_ym("2025-01") == 202501

    def test_december(self) -> None:
        assert parse_ym("2024-12") == 202412

    def test_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Invalid ym format"):
            parse_ym("2025/01")

    def test_too_short(self) -> None:
        with pytest.raises(ValueError, match="Invalid ym format"):
            parse_ym("2025-1")

    def test_garbage(self) -> None:
        with pytest.raises(ValueError, match="Invalid ym format"):
            parse_ym("hello")


class TestFormatYm:
    def test_basic(self) -> None:
        assert format_ym(202501) == "2025-01"

    def test_december(self) -> None:
        assert format_ym(202412) == "2024-12"

    def test_roundtrip(self) -> None:
        assert format_ym(parse_ym("2025-06")) == "2025-06"


class TestAddMonths:
    def test_forward(self) -> None:
        assert add_months("2025-01", 3) == "2025-04"

    def test_backward(self) -> None:
        assert add_months("2025-03", -2) == "2025-01"

    def test_cross_year_forward(self) -> None:
        assert add_months("2024-11", 3) == "2025-02"

    def test_cross_year_backward(self) -> None:
        assert add_months("2025-02", -3) == "2024-11"

    def test_zero(self) -> None:
        assert add_months("2025-06", 0) == "2025-06"

    def test_twelve_months(self) -> None:
        assert add_months("2025-01", 12) == "2026-01"


class TestPrevYmStr:
    def test_basic(self) -> None:
        assert prev_ym_str("2025-02") == "2025-01"

    def test_january(self) -> None:
        assert prev_ym_str("2025-01") == "2024-12"


class TestYmToDate:
    def test_basic(self) -> None:
        assert ym_to_date(202501) == date(2025, 1, 1)


class TestDateToYm:
    def test_basic(self) -> None:
        assert date_to_ym(date(2025, 6, 15)) == 202506


class TestPrevYm:
    def test_basic(self) -> None:
        assert prev_ym(202502) == 202501

    def test_january(self) -> None:
        assert prev_ym(202501) == 202412


class TestYmFromDateStr:
    def test_basic(self) -> None:
        assert ym_from_date_str("2025-06-15") == "2025-06"

    def test_first_day(self) -> None:
        assert ym_from_date_str("2025-01-01") == "2025-01"
