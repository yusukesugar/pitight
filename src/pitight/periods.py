"""Period string utilities for YYYY-MM partition keys.

Complements ``pitight.partition.resolve_expected_periods`` with arithmetic
and conversion helpers for the "YYYY-MM" string format used throughout
pitight's partition management.

Usage:
    from pitight.periods import add_months, prev_ym_str, parse_ym, format_ym

    next_month = add_months("2025-01", 1)   # "2025-02"
    prev_month = prev_ym_str("2025-01")     # "2024-12"
    ym_int = parse_ym("2025-06")            # 202506
    ym_str = format_ym(202506)              # "2025-06"
"""

from __future__ import annotations

import re
from datetime import date, timedelta

_YM_RE = re.compile(r"^\d{4}-\d{2}$")


def parse_ym(s: str) -> int:
    """Parse ``'YYYY-MM'`` to integer ``YYYYMM``.

    Raises:
        ValueError: If *s* doesn't match ``YYYY-MM`` format.
    """
    if not _YM_RE.match(s):
        raise ValueError(f"Invalid ym format: {s!r}")
    y, m = s.split("-")
    return int(y) * 100 + int(m)


def format_ym(ym_int: int) -> str:
    """Format integer ``YYYYMM`` to ``'YYYY-MM'``."""
    y = ym_int // 100
    m = ym_int % 100
    return f"{y:04d}-{m:02d}"


def add_months(s: str, n: int) -> str:
    """Add *n* months to a ``'YYYY-MM'`` string (negative *n* subtracts)."""
    ym_int = parse_ym(s)
    y = ym_int // 100
    m = ym_int % 100

    total_months = y * 12 + (m - 1) + n
    new_y = total_months // 12
    new_m = (total_months % 12) + 1
    return f"{new_y:04d}-{new_m:02d}"


def prev_ym_str(s: str) -> str:
    """Return the previous month as ``'YYYY-MM'``."""
    return add_months(s, -1)


def ym_to_date(ym: int) -> date:
    """Convert integer ``YYYYMM`` to ``date(YYYY, MM, 1)``."""
    y = ym // 100
    m = ym % 100
    return date(y, m, 1)


def date_to_ym(d: date) -> int:
    """Convert a ``date`` to integer ``YYYYMM``."""
    return d.year * 100 + d.month


def prev_ym(ym: int) -> int:
    """Return previous month as integer ``YYYYMM``."""
    d = ym_to_date(ym)
    prev_month_last_day = d.replace(day=1) - timedelta(days=1)
    return date_to_ym(prev_month_last_day)


def ym_from_date_str(date_str: str) -> str:
    """Extract ``'YYYY-MM'`` from ``'YYYY-MM-DD'``."""
    return str(date_str)[:7]
