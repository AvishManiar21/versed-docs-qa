import httpx

from versed.ingest.manifest import build_manifest, resolve_latest_langchain_version


class _FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"info": {"version": "1.9.9"}}


def test_resolve_latest_langchain_version(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse())
    assert resolve_latest_langchain_version() == "1.9.9"


def test_build_manifest_has_four_versions(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse())
    manifest = build_manifest()
    assert [s.version for s in manifest] == ["0.1", "0.2", "0.3", "1.9.9"]
    assert manifest[0].ref == "v0.1.0"
    assert manifest[2].ref == "langchain==0.3.0"
    assert manifest[3].repo_url == "https://github.com/langchain-ai/docs.git"
    assert manifest[3].pip_spec == "langchain==1.9.9"
