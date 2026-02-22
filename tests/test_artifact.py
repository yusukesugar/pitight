import json
from pathlib import Path

import pytest

from pitight.artifact import Artifact, ArtifactRegistry


@pytest.fixture()
def tmp_registry(tmp_path):
    return ArtifactRegistry(str(tmp_path / "artifacts.json"))


class TestArtifact:
    def test_exists_false(self):
        art = Artifact(name="test", path="/nonexistent/path", config_hash="abc")
        assert not art.exists

    def test_exists_true(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.write_text("fake")
        art = Artifact(name="test", path=str(p), config_hash="abc")
        assert art.exists


class TestArtifactRegistry:
    def test_register_and_list(self, tmp_registry):
        art = Artifact(name="s50", path="/tmp/s50.parquet", config_hash="aaa")
        tmp_registry.register(art)
        assert len(tmp_registry.list_all()) == 1

    def test_find_missing(self, tmp_registry):
        art = Artifact(name="s50", path="/nonexistent.parquet", config_hash="aaa")
        tmp_registry.register(art)
        missing = tmp_registry.find_missing()
        assert len(missing) == 1
        assert missing[0].name == "s50"

    def test_find_stale(self, tmp_registry):
        art = Artifact(name="s50", path="/tmp/s50.parquet", config_hash="old_hash")
        tmp_registry.register(art)
        stale = tmp_registry.find_stale({"s50": "new_hash"})
        assert len(stale) == 1

    def test_find_not_stale(self, tmp_registry):
        art = Artifact(name="s50", path="/tmp/s50.parquet", config_hash="same_hash")
        tmp_registry.register(art)
        stale = tmp_registry.find_stale({"s50": "same_hash"})
        assert len(stale) == 0

    def test_find_orphaned(self, tmp_registry):
        art = Artifact(name="deleted_stage", path="/tmp/x.parquet", config_hash="aaa")
        tmp_registry.register(art)
        orphaned = tmp_registry.find_orphaned({"s50", "s60"})
        assert len(orphaned) == 1

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "artifacts.json")
        reg1 = ArtifactRegistry(path)
        reg1.register(Artifact(name="s50", path="/tmp/s50.parquet", config_hash="aaa"))

        reg2 = ArtifactRegistry(path)
        assert len(reg2.list_all()) == 1

    def test_summary(self, tmp_registry):
        tmp_registry.register(
            Artifact(name="s50", path="/nonexistent.parquet", config_hash="aaa")
        )
        s = tmp_registry.summary()
        assert s["total_artifacts"] == 1
        assert s["missing_on_disk"] == 1

    def test_remove(self, tmp_registry):
        tmp_registry.register(
            Artifact(name="s50", path="/tmp/s50.parquet", config_hash="aaa")
        )
        assert tmp_registry.remove("s50", "/tmp/s50.parquet")
        assert len(tmp_registry.list_all()) == 0
