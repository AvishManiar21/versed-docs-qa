import pytest

from versed.rag import build_rag_chain

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_rag_chain_answers_a_real_question_via_local_ollama():
    chain = build_rag_chain()
    answer = chain.invoke("Is ConversationChain still valid?")
    assert isinstance(answer, str)
    assert len(answer) > 0
