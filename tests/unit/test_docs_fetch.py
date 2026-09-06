import subprocess
from pathlib import Path

import pytest

from versed.ingest.docs_fetch import fetch_ref, iter_doc_files


@pytest.fixture
def local_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source_repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)

    docs_dir = repo / "docs" / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "intro.md").write_text("# Intro\n\nHello.")
    (docs_dir / "guide.mdx").write_text("# Guide\n\nHi.")
    (repo / "README.md").write_text("not in subpath")

    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    subprocess.run(["git", "tag", "v0.1.0"], cwd=repo, check=True)
    return repo


def test_fetch_ref_checks_out_tag(local_repo: Path, tmp_path: Path):
    dest = tmp_path / "checkout"
    fetch_ref(f"file://{local_repo}", "v0.1.0", dest)
    assert (dest / "docs" / "docs" / "intro.md").exists()


def test_iter_doc_files_only_returns_subpath_md_and_mdx(local_repo: Path, tmp_path: Path):
    dest = tmp_path / "checkout2"
    fetch_ref(f"file://{local_repo}", "v0.1.0", dest)
    files = iter_doc_files(dest, "docs/docs")
    names = sorted(p.name for p in files)
    assert names == ["guide.mdx", "intro.md"]
