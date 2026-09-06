from dataclasses import dataclass, field

import httpx


@dataclass(frozen=True)
class VersionSource:
    version: str
    repo_url: str
    ref: str
    docs_subpath: str
    pip_spec: str
    extra_pip_specs: list[str] = field(default_factory=list)


def resolve_latest_pypi_version(package: str) -> str:
    response = httpx.get(f"https://pypi.org/pypi/{package}/json", timeout=10.0)
    response.raise_for_status()
    return response.json()["info"]["version"]


def resolve_latest_langchain_version() -> str:
    return resolve_latest_pypi_version("langchain")


def build_manifest() -> list[VersionSource]:
    latest = resolve_latest_langchain_version()
    latest_classic = resolve_latest_pypi_version("langchain-classic")
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
            extra_pip_specs=[f"langchain-classic=={latest_classic}"],
        ),
    ]
