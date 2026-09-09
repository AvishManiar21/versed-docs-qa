from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from versed.rag import build_rag_chain, format_docs


class _FakeRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return [
            Document(page_content="fake chunk", metadata={"version": "0.1"}),
        ]


def test_format_docs_prefixes_each_chunk_with_its_version():
    docs = [
        Document(page_content="a", metadata={"version": "0.1"}),
        Document(page_content="b", metadata={"version": "0.2"}),
    ]
    result = format_docs(docs)
    assert "[0.1] a" in result
    assert "[0.2] b" in result


def test_build_rag_chain_wires_retriever_into_prompt_and_returns_llm_text():
    fake_llm = RunnableLambda(lambda _: AIMessage(content="canned answer"))
    chain = build_rag_chain(retriever=_FakeRetriever(), llm=fake_llm)
    result = chain.invoke("Is X still valid?")
    assert result == "canned answer"
