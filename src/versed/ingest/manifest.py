from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class VersionSource:
    version: str
    repo_url: str
    ref: str
    docs_subpath: str
    pip_spec: str


def resolve_latest_langchain_version() -> str:
    response = httpx.get("https://pypi.org/pypi/langchain/json", timeout=10.0)
    response.raise_for_status()
    return response.json()["info"]["version"]


def build_manifest() -> list[VersionSource]:
    latest = resolve_latest_langchain_version()
    langchain_repo = "https://github.com/langchain-ai/langchain.git"
    return [
        VersionSource(
            version="0.1",
            repo_url=langchain_repo,
            ref="v0.1.0",
            docs_subpath="docs/docs",
            pip_spec="langchain==0.1.0",
        ),
        VersionSource(
            version="0.2",
            repo_url=langchain_repo,
            ref="langchain==0.2.0",
            docs_subpath="docs/docs",
            pip_spec="langchain==0.2.0",
        ),
        VersionSource(
            version="0.3",
            repo_url=langchain_repo,
            ref="langchain==0.3.0",
            docs_subpath="docs/docs",
            pip_spec="langchain==0.3.0",
        ),
        VersionSource(
            version=latest,
            repo_url="https://github.com/langchain-ai/docs.git",
            ref="main",
            docs_subpath="src/oss/python",
            pip_spec=f"langchain=={latest}",
        ),
    ]
