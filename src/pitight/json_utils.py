"""JSON utilities for atomic file writing.

Provides a safe, atomic JSON write that prevents partial/corrupt files
on crash or concurrent access. Used internally by ``write_manifest``
and available for downstream projects.

Usage:
    from pitight.json_utils import write_json_atomically, json_serial

    write_json_atomically(path, {"key": "value"})
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any


def json_serial(obj: Any) -> str:
    """Default JSON serializer for datetime objects.

    Converts ``datetime.datetime`` and ``datetime.date`` to ISO format strings.

    Raises:
        TypeError: If *obj* is not a supported type.
    """
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def write_json_atomically(
    path: Path,
    payload: dict[str, Any],
    *,
    sort_keys: bool = False,
    default: Any = None,
) -> None:
    """Write a JSON file atomically via temp-file + rename.

    Creates parent directories if they don't exist.

    Args:
        path: Target file path.
        payload: Dict to serialize as JSON.
        sort_keys: Sort dict keys in output.
        default: Custom serializer for non-standard types
            (falls back to ``json_serial`` if None).
    """
    effective_default = default if default is not None else json_serial
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=sort_keys,
            default=effective_default,
        )
    tmp_path.rename(path)
