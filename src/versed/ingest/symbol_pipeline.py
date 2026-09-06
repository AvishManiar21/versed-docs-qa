import json
import subprocess
from pathlib import Path

from sqlalchemy.orm import Session

from versed.db.models import Symbol
from versed.ingest.manifest import VersionSource

WORKER_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "introspect_worker.py"
INTROSPECT_PYTHON = "3.11"


def create_isolated_python(venv_dir: Path) -> Path:
    subprocess.run(
        ["uv", "venv", str(venv_dir), "--python", INTROSPECT_PYTHON, "--allow-existing"],
        check=True,
        capture_output=True,
        text=True,
    )
    return venv_dir / "bin" / "python"


def install_into(python_path: Path, pip_spec: str) -> None:
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python_path), "--", pip_spec],
        check=True,
        capture_output=True,
        text=True,
    )


def run_introspection(python_path: Path, package_name: str) -> list[dict]:
    result = subprocess.run(
        [str(python_path), str(WORKER_SCRIPT), package_name],
        check=True,
        capture_output=True,
        text=True,
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def _pip_spec_to_import_name(pip_spec: str) -> str:
    return pip_spec.split("==")[0].replace("-", "_")


def build_symbols_for_version(source: VersionSource, venv_root: Path, session: Session) -> int:
    venv_dir = venv_root / source.version.replace(".", "_")
    python_path = create_isolated_python(venv_dir)
    install_into(python_path, source.pip_spec)

    records = run_introspection(python_path, _pip_spec_to_import_name(source.pip_spec))

    for extra_spec in source.extra_pip_specs:
        install_into(python_path, extra_spec)
        records.extend(run_introspection(python_path, _pip_spec_to_import_name(extra_spec)))

    session.query(Symbol).filter_by(version=source.version).delete()
    for record in records:
        session.add(
            Symbol(
                version=source.version,
                qualified_name=record["qualified_name"],
                kind=record["kind"],
                signature=record["signature"],
                docstring=record["docstring"] or None,
                deprecated_since=record["deprecated_since"],
                alternative=record["alternative"],
            )
        )
    session.commit()
    return len(records)
