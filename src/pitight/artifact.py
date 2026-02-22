"""Artifact registry — track what was built, with which config, and when.

Core concept: every pipeline output (parquet, model, CSV) is an Artifact
with a known config_hash. The registry answers:
  - "What's stale?" (config changed but artifact wasn't rebuilt)
  - "What's missing?" (upstream exists but downstream doesn't)
  - "What's orphaned?" (config deleted but artifact remains)

Usage:
    from pitight.artifact import Artifact, ArtifactRegistry

    registry = ArtifactRegistry("./artifacts.json")
    registry.register(Artifact(
        name="s50_probs",
        path="data/s50_probs/2025-01/part.parquet",
        config_hash="abc123...",
    ))
    stale = registry.find_stale()
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Artifact:
    """A tracked pipeline output."""

    name: str
    path: str
    config_hash: str
    created_at: str = field(default_factory=lambda: _now_iso())
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def path_obj(self) -> Path:
        return Path(self.path)

    @property
    def exists(self) -> bool:
        return self.path_obj.exists()


@dataclass
class ArtifactRegistry:
    """JSON-file-backed artifact registry.

    Lightweight on purpose — no database, no server. Just a JSON file
    that tracks what was built and with which config.
    """

    registry_path: str = ".pitight/artifacts.json"

    def __post_init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}
        self._load()

    def _load(self) -> None:
        p = Path(self.registry_path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            for key, rec in data.items():
                self._artifacts[key] = Artifact(**rec)

    def _save(self) -> None:
        p = Path(self.registry_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {k: asdict(v) for k, v in self._artifacts.items()}
        p.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def register(self, artifact: Artifact) -> None:
        """Register or update an artifact."""
        key = f"{artifact.name}::{artifact.path}"
        self._artifacts[key] = artifact
        self._save()
        logger.info("Registered: %s (hash=%s)", key, artifact.config_hash[:10])

    def list_all(self) -> list[Artifact]:
        """Return all registered artifacts."""
        return list(self._artifacts.values())

    def find_stale(self, current_configs: dict[str, str]) -> list[Artifact]:
        """Find artifacts whose config_hash doesn't match the current config.

        Args:
            current_configs: mapping of artifact name → current config_hash.

        Returns:
            List of artifacts that need rebuilding.
        """
        stale = []
        for art in self._artifacts.values():
            if art.name in current_configs:
                if art.config_hash != current_configs[art.name]:
                    stale.append(art)
        return stale

    def find_missing(self) -> list[Artifact]:
        """Find artifacts that are registered but whose files don't exist on disk."""
        return [a for a in self._artifacts.values() if not a.exists]

    def find_orphaned(self, known_names: set[str]) -> list[Artifact]:
        """Find artifacts whose name is not in the known set (config was deleted)."""
        return [a for a in self._artifacts.values() if a.name not in known_names]

    def get(self, name: str) -> list[Artifact]:
        """Get all artifacts with a given name."""
        return [a for a in self._artifacts.values() if a.name == name]

    def remove(self, name: str, path: str) -> bool:
        """Remove an artifact from the registry (does not delete the file)."""
        key = f"{name}::{path}"
        if key in self._artifacts:
            del self._artifacts[key]
            self._save()
            return True
        return False

    def summary(self) -> dict[str, Any]:
        """Return a summary of the registry state."""
        total = len(self._artifacts)
        existing = sum(1 for a in self._artifacts.values() if a.exists)
        missing = total - existing
        names = sorted({a.name for a in self._artifacts.values()})
        return {
            "total_artifacts": total,
            "existing_on_disk": existing,
            "missing_on_disk": missing,
            "artifact_names": names,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
