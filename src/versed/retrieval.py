from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaEmbeddings
from sqlalchemy import select

from versed.db.models import Chunk
from versed.db.session import get_session


class VersedRetriever(BaseRetriever):
    """Dense-only retrieval across every ingested version, no filtering.

    Deliberately naive: this is the Milestone 3 baseline's retrieval step.
    Milestone 5 extends this same interface with a version filter, BM25, and
    reranking rather than replacing it.
    """

    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector = embeddings.embed_query(query)
        with get_session() as session:
            stmt = select(Chunk).order_by(Chunk.embedding.cosine_distance(vector)).limit(self.k)
            chunks = session.scalars(stmt).all()
            return [
                Document(
                    page_content=chunk.content,
                    metadata={
                        "version": chunk.version,
                        "source_path": chunk.source_path,
                        "heading_path": chunk.heading_path,
                    },
                )
                for chunk in chunks
            ]
