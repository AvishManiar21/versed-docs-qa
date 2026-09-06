import tempfile
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from sqlalchemy.orm import Session

from versed.db.models import Chunk
from versed.ingest.chunker import chunk_markdown
from versed.ingest.docs_fetch import fetch_ref, iter_doc_files
from versed.ingest.manifest import VersionSource
from versed.ingest.text_features import count_tokens, extract_symbols_mentioned

EMBED_BATCH_SIZE = 100


def ingest_docs_for_version(source: VersionSource, session: Session) -> int:
    """Fetch, chunk, embed, and persist one version's docs. Returns chunk count."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "repo"
        fetch_ref(source.repo_url, source.ref, dest)
        doc_files = iter_doc_files(dest, source.docs_subpath)

        raw_chunks = []
        for path in doc_files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for raw in chunk_markdown(text):
                raw_chunks.append((path.relative_to(dest), raw))

        session.query(Chunk).filter_by(version=source.version).delete()

        for batch_start in range(0, len(raw_chunks), EMBED_BATCH_SIZE):
            batch = raw_chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
            # ponytail: no retry on embedding calls, add tenacity if a transient
            # failure ever kills a real ingestion run
            vectors = embeddings.embed_documents([raw.content for _, raw in batch])
            for (rel_path, raw), vector in zip(batch, vectors):
                session.add(
                    Chunk(
                        version=source.version,
                        source_path=str(rel_path),
                        heading_path=" > ".join(raw.heading_path),
                        content=raw.content,
                        embedding=vector,
                        has_code=raw.has_code,
                        symbols_mentioned=extract_symbols_mentioned(raw.content),
                        token_count=count_tokens(raw.content),
                    )
                )

        session.commit()
        return len(raw_chunks)
