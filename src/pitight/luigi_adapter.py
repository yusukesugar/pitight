"""Luigi adapter — bridge between Luigi tasks and pitight's Artifact Registry.

Usage:
    from pitight.luigi_adapter import artifact_from_luigi_task, register_luigi_output

    # After a Luigi task completes:
    artifact = artifact_from_luigi_task(my_task, config_hash="abc123...")
    registry.register(artifact)

    # Or as a one-liner:
    register_luigi_output(my_task, registry, config_hash="abc123...")

Requires: pip install pitight[luigi]
"""

from __future__ import annotations

from typing import Any

from pitight.artifact import Artifact, ArtifactRegistry

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
