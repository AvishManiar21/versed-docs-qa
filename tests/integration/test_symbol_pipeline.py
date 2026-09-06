import pytest
from sqlalchemy import select

from versed.db.models import Symbol
from versed.db.session import get_session
from versed.ingest.manifest import VersionSource
from versed.ingest.symbol_pipeline import build_symbols_for_version

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_build_symbols_for_version_finds_known_deprecated_class(tmp_path):
    source = VersionSource(
        version="0.2",
        repo_url="https://github.com/langchain-ai/langchain.git",
        ref="langchain==0.2.0",
        docs_subpath="docs/docs",
        pip_spec="langchain==0.2.0",
    )
    with get_session() as session:
        session.query(Symbol).filter_by(version="0.2").delete()
        session.commit()

        count = build_symbols_for_version(source, tmp_path / "venvs", session)
        assert count > 0

        retrieval_qa = session.scalars(
            select(Symbol).where(
                Symbol.version == "0.2",
                Symbol.qualified_name.like("%.RetrievalQA"),
            )
        ).first()
        assert retrieval_qa is not None
        assert retrieval_qa.deprecated_since is not None
