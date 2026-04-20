"""Luigi adapter — bridge between Luigi tasks and pitight's Artifact Registry.

Usage:
    from pitight.luigi_adapter import artifact_from_luigi_task, register_luigi_output

    # After a Luigi task completes:
    artifact = artifact_from_luigi_task(my_task, config_hash="abc123...")
    registry.register(artifact)

    # Or as a one-liner:
    register_luigi_output(my_task, registry, config_hash="abc123...")

For freshness-aware completion checking (i.e. "file exists" is not enough),
use :class:`FreshCompleteMixin`:

    from pitight.luigi_adapter import FreshCompleteMixin

    class MyMonthlyTask(FreshCompleteMixin, luigi.Task):
        def manifest_path(self) -> Path: ...
        def expected_periods(self) -> list[str]: ...
        MIN_ROW_COUNT = 10000  # optional

Requires: pip install pitight[luigi]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from pitight.artifact import Artifact, ArtifactRegistry
from pitight.complete_checks import (
    manifest_coverage_ok,
    manifest_expected_present,
    manifest_freshness_ok,
    manifest_min_row_count,
    upstream_newer_than,
)

try:
    import luigi
except ImportError:
    raise ImportError(
        "Luigi is required for the luigi adapter. "
        "Install it with: pip install pitight[luigi]"
    )


def artifact_from_luigi_task(
    task: luigi.Task,
    config_hash: str,
    metadata: dict[str, Any] | None = None,
) -> Artifact:
    """Create an Artifact from a Luigi task.

    Extracts the task name and output path from the Luigi task,
    and associates it with the given config hash.

    Args:
        task: A Luigi task instance (must have output() returning a LocalTarget).
        config_hash: The config hash that produced this artifact.
        metadata: Optional additional metadata.

    Returns:
        An Artifact instance ready for registration.
    """
    output = task.output()

    # Handle LocalTarget
    if hasattr(output, "path"):
        path = output.path
    else:
        raise TypeError(
            f"Cannot extract path from task output: {type(output).__name__}. "
            "Expected a target with a .path attribute (e.g., LocalTarget)."
        )

    name = _task_name(task)
    meta = metadata or {}

    # Capture Luigi params as metadata
    params = task.param_kwargs
    if params:
        meta["luigi_params"] = {k: str(v) for k, v in params.items()}

    return Artifact(
        name=name,
        path=str(path),
        config_hash=config_hash,
        metadata=meta,
    )


def register_luigi_output(
    task: luigi.Task,
    registry: ArtifactRegistry,
    config_hash: str,
    metadata: dict[str, Any] | None = None,
) -> Artifact:
    """Create and register an Artifact from a Luigi task in one step.

    Args:
        task: A Luigi task instance.
        registry: The ArtifactRegistry to register with.
        config_hash: The config hash that produced this artifact.
        metadata: Optional additional metadata.

    Returns:
        The registered Artifact.
    """
    artifact = artifact_from_luigi_task(task, config_hash, metadata)
    registry.register(artifact)
    return artifact


def _task_name(task: luigi.Task) -> str:
    """Derive a human-readable artifact name from a Luigi task."""
    cls = type(task)
    module = cls.__module__ or ""
    # Strip common prefixes for readability
    for prefix in ("__main__.", "tasks.", "task."):
        if module.startswith(prefix):
            module = module[len(prefix):]
    if module:
        return f"{module}.{cls.__name__}"
    return cls.__name__


# ============================================================
# FreshCompleteMixin — multi-axis completeness for Luigi tasks
# ============================================================


class FreshCompleteMixin:
    """Luigi Task mixin that treats a task as complete only when:

    1. The Luigi output target exists (standard Luigi check).
    2. A manifest file exists at ``self.manifest_path()``.
    3. The manifest records ``coverage.coverage_ok == True``.
    4. Every period from ``self.expected_periods()`` is in the manifest's
       ``present_periods`` / ``present_months`` list.
    5. ``stats_rollup.row_count_total >= self.MIN_ROW_COUNT`` (if set).
    6. The manifest is younger than ``self.MAX_MANIFEST_AGE_SECONDS`` (if > 0).
    7. No upstream manifest from ``self.upstream_manifests()`` is newer than
       the self manifest (if the method is overridden).

    Subclasses **must** override ``manifest_path()``. Everything else is
    optional — defaults are permissive (no-op) to keep the mixin backward
    compatible when retrofitted to existing tasks.

    The mixin is deliberately additive: if you inherit from it without
    overriding ``manifest_path``, ``complete()`` falls back to the standard
    "output exists" semantics, so it is safe to mix in early.
    """

    # Class-level knobs. Override in subclasses as needed.
    MIN_ROW_COUNT: int = 0
    MAX_MANIFEST_AGE_SECONDS: float = 0.0  # 0 = disabled
    UPSTREAM_TOLERANCE_SECONDS: float = 2.0

    def min_row_count_required(self) -> int:
        """Return the row-count minimum to consider this task complete.

        Default: returns the class-level :attr:`MIN_ROW_COUNT`. A value of
        ``0`` disables the row-count check.

        Override to compute dynamically — for example, from the elapsed
        days in the task's period::

            from pitight.complete_checks import expected_rows_for_period

            class MonthlyTask(FreshCompleteMixin, luigi.Task):
                ym = luigi.Parameter()

                def min_row_count_required(self) -> int:
                    return expected_rows_for_period(
                        self.ym,
                        rows_per_day=700,    # observed baseline
                        safety_factor=0.5,   # allow 50% headroom
                    )

        A date-aware threshold prevents the rot that static thresholds
        accumulate — a number that was "right" in January may be wrong in
        April because row rates drift, schemas change, or the partition is
        mid-build.
        """
        return self.MIN_ROW_COUNT

    def manifest_path(self) -> Path | None:
        """Return the path to this task's manifest.json, or None to disable.

        Default implementation looks next to the Luigi output target. If the
        output is at ``.../data/year=Y/month=M/part-YM.parquet``, the manifest
        is assumed at ``.../manifest.json`` (two dirs up from the parquet).

        Subclasses that use a different layout should override this.
        """
        try:
            out = self.output()  # type: ignore[attr-defined]
        except Exception:
            return None
        path_attr = getattr(out, "path", None)
        if path_attr is None:
            return None
        parquet = Path(str(path_attr))
        # .../<root>/data/year=Y/month=M/part-YM.parquet  →  .../<root>/manifest.json
        # parent chain: month=M → year=Y → data → <root>
        candidate = parquet.parent.parent.parent.parent / "manifest.json"
        if candidate.exists():
            return candidate
        # Fallback: manifest alongside the parquet.
        sibling = parquet.parent / "manifest.json"
        if sibling.exists():
            return sibling
        return candidate  # return the expected location even if missing

    def expected_periods(self) -> Iterable[str]:
        """Return periods that must appear in the manifest's present list.

        Default: empty (= skip the coverage-expected check). Subclasses with
        range-driven builds should override to return the monthly/daily list.
        """
        return ()

    def upstream_manifests(self) -> Iterable[Path]:
        """Return paths to upstream manifests that must not be newer than self.

        Default: empty (= skip the upstream-newer check). Subclasses with
        explicit input tracking should override.
        """
        return ()

    # ------- complete() orchestration -------

    def complete(self) -> bool:  # type: ignore[override]
        # Step 1: standard Luigi existence check.
        try:
            target = self.output()  # type: ignore[attr-defined]
        except Exception:
            return False
        if not getattr(target, "exists", lambda: False)():
            return False

        manifest = self.manifest_path()
        if manifest is None:
            # Subclass opted out of manifest-based checks. Fall back to
            # standard Luigi semantics.
            return True

        # Step 2: manifest must exist and record coverage_ok.
        if not manifest.exists():
            return False
        if not manifest_coverage_ok(manifest):
            return False

        # Step 3: expected periods must all be present.
        expected = list(self.expected_periods())
        if expected:
            ok, _ = manifest_expected_present(manifest, expected)
            if not ok:
                return False

        # Step 4: row-count sanity (if configured).
        min_rc = self.min_row_count_required()
        if min_rc > 0:
            if not manifest_min_row_count(manifest, min_rc):
                return False

        # Step 5: freshness (if configured).
        if self.MAX_MANIFEST_AGE_SECONDS > 0:
            if not manifest_freshness_ok(manifest, self.MAX_MANIFEST_AGE_SECONDS):
                return False

        # Step 6: upstream-newer check (if configured).
        upstreams = list(self.upstream_manifests())
        if upstreams:
            newer = upstream_newer_than(
                manifest,
                upstreams,
                tolerance_seconds=self.UPSTREAM_TOLERANCE_SECONDS,
            )
            if newer:
                return False

        return True
