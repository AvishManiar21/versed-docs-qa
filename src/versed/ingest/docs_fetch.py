import subprocess
from pathlib import Path


def fetch_ref(repo_url: str, ref: str, dest: Path) -> None:
    """Shallow-clone a single ref (tag or branch) of repo_url into dest."""
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; remove it before fetching")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", ref, "--", repo_url, str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )


def iter_doc_files(root: Path, subpath: str) -> list[Path]:
    """Return every .md/.mdx file under root/subpath, sorted for determinism."""
    base = root / subpath
    return sorted(p for p in base.rglob("*") if p.suffix in {".md", ".mdx"})
