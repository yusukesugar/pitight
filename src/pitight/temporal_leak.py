"""Temporal leak detection for ML pipelines.

Prevents future information from leaking into training/prediction stages.
The core idea: define which columns are "outcome" data (results, payouts, odds)
and which pipeline stages are "decision-time" (must not see outcomes).

Usage:
    from pitight.temporal_leak import TemporalBoundary, check_leak

    # Define boundaries
    boundary = TemporalBoundary(
        forbidden_columns={"odds", "payout", "result", "rank"},
        forbidden_prefixes={"outcome_", "post_"},
    )

    # Check a DataFrame at a decision-time stage
    check_leak(df, boundary, stage="s50_probs")

    # Or use as a decorator
    @no_leak(boundary, stage="s50_probs")
    def compute_probabilities(df):
        ...
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


class TemporalLeakError(Exception):
    """Raised when outcome data is detected in a decision-time stage."""


@dataclass(frozen=True)
class TemporalBoundary:
    """Defines what columns are forbidden at decision-time stages.

    Attributes:
        forbidden_columns: Exact column names that must not appear.
        forbidden_prefixes: Column name prefixes that must not appear.
        forbidden_substrings: Substrings in column names that must not appear.
        allowed_stages: Stages where these columns ARE allowed (e.g., ["s70", "s80"]).
    """

    forbidden_columns: frozenset[str] = field(default_factory=frozenset)
    forbidden_prefixes: tuple[str, ...] = ()
    forbidden_substrings: tuple[str, ...] = ()
    allowed_stages: frozenset[str] = field(default_factory=frozenset)

    def check(self, columns: list[str], stage: str) -> list[str]:
        """Check columns against this boundary.

        Args:
            columns: DataFrame column names to check.
            stage: Pipeline stage name (e.g., "s50_probs").

        Returns:
            List of violating column names (empty if clean).
        """
        if stage in self.allowed_stages:
            return []

        violations: list[str] = []
        for col in columns:
            col_lower = col.lower()
            if col in self.forbidden_columns or col_lower in self.forbidden_columns:
                violations.append(col)
                continue
            if any(col_lower.startswith(p) for p in self.forbidden_prefixes):
                violations.append(col)
                continue
            if any(s in col_lower for s in self.forbidden_substrings):
                violations.append(col)

        return violations


def check_leak(
    df: pd.DataFrame,
    boundary: TemporalBoundary,
    stage: str,
    raise_on_violation: bool = True,
) -> list[str]:
    """Check a DataFrame for temporal leaks.

    Args:
        df: DataFrame to check.
        boundary: The temporal boundary rules.
        stage: Current pipeline stage name.
        raise_on_violation: If True, raise TemporalLeakError on violations.

    Returns:
        List of violating column names.

    Raises:
        TemporalLeakError: If violations found and raise_on_violation is True.
    """
    violations = boundary.check(list(df.columns), stage)

    if violations:
        msg = (
            f"[{stage}] Temporal leak detected! "
            f"Forbidden columns found: {violations}. "
            f"These columns contain outcome/future data that must not be "
            f"available at decision time."
        )
        if raise_on_violation:
            raise TemporalLeakError(msg)
        logger.warning(msg)

    return violations


def no_leak(
    boundary: TemporalBoundary,
    stage: str,
    check_inputs: bool = True,
    check_output: bool = False,
) -> Callable[..., Any]:
    """Decorator that checks for temporal leaks on function inputs/outputs.

    Args:
        boundary: The temporal boundary rules.
        stage: Pipeline stage name.
        check_inputs: Check DataFrame arguments for leaks.
        check_output: Check the returned DataFrame for leaks.

    Example:
        @no_leak(boundary, stage="s50_probs")
        def predict(train_df, test_df):
            ...  # train_df and test_df are checked for forbidden columns
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if check_inputs:
                for i, arg in enumerate(args):
                    if isinstance(arg, pd.DataFrame):
                        check_leak(arg, boundary, stage=f"{stage}/arg[{i}]")
                for key, val in kwargs.items():
                    if isinstance(val, pd.DataFrame):
                        check_leak(val, boundary, stage=f"{stage}/kwarg[{key}]")

            result = func(*args, **kwargs)

            if check_output and isinstance(result, pd.DataFrame):
                check_leak(result, boundary, stage=f"{stage}/output")

            return result

        return wrapper

    return decorator
