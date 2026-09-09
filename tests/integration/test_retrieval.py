import pytest

from versed.retrieval import VersedRetriever

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_versed_retriever_returns_k_results_spanning_versions():
    retriever = VersedRetriever(k=5)
    docs = retriever.invoke("Is ConversationChain still valid?")
    assert len(docs) == 5
    versions = {d.metadata["version"] for d in docs}
    assert len(versions) > 1  # proves there's no version filter


def test_versed_retriever_respects_k():
    retriever = VersedRetriever(k=2)
    docs = retriever.invoke("What is a retriever?")
    assert len(docs) == 2
