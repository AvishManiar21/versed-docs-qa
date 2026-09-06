import pytest
from sqlalchemy import select

from versed.db.models import Chunk
from versed.db.session import get_session
from versed.ingest.docs_pipeline import ingest_docs_for_version
from versed.ingest.manifest import VersionSource

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_ingest_docs_for_version_persists_chunks():
    source = VersionSource(
        version="0.1",
        repo_url="https://github.com/langchain-ai/langchain.git",
        ref="v0.1.0",
        docs_subpath="docs/docs",
        pip_spec="langchain==0.1.0",
    )
    with get_session() as session:
        session.query(Chunk).filter_by(version="0.1").delete()
        session.commit()

        count = ingest_docs_for_version(source, session)
        assert count > 0

        stored = session.scalars(select(Chunk).where(Chunk.version == "0.1")).all()
        assert len(stored) == count
        assert all(c.embedding is not None for c in stored)
