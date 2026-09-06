import httpx

from versed.ingest.manifest import build_manifest, resolve_latest_langchain_version


class _FakeResponse:
    def __init__(self, version: str):
        self._version = version

    def raise_for_status(self):
        pass

    def json(self):
        return {"info": {"version": self._version}}


def _fake_get(url, *a, **k):
    if "langchain-classic" in url:
        return _FakeResponse("1.0.8")
    return _FakeResponse("1.9.9")


def test_resolve_latest_langchain_version(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get)
    assert resolve_latest_langchain_version() == "1.9.9"


def test_build_manifest_has_four_versions(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get)
    manifest = build_manifest()
    assert [s.version for s in manifest] == ["0.1", "0.2", "0.3", "1.9.9"]
    assert manifest[0].ref == "v0.1.0"
    assert manifest[2].ref == "langchain==0.3.0"
    assert manifest[3].repo_url == "https://github.com/langchain-ai/docs.git"
    assert manifest[3].pip_spec == "langchain==1.9.9"


def test_only_current_version_has_extra_pip_specs(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get)
    manifest = build_manifest()
    assert manifest[0].extra_pip_specs == []
    assert manifest[1].extra_pip_specs == []
    assert manifest[2].extra_pip_specs == []
    assert manifest[3].extra_pip_specs == ["langchain-classic==1.0.8"]
