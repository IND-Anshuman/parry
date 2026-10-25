from __future__ import annotations

import pytest

import parry
from parry import Parry

pytest.importorskip("llama_index.core")
li_openai = pytest.importorskip("llama_index.llms.openai")

from llama_index.core import Document, Settings, VectorStoreIndex  # noqa: E402
from llama_index.core.callbacks import CallbackManager  # noqa: E402
from llama_index.core.embeddings import MockEmbedding  # noqa: E402
from llama_index.core.llms import ChatMessage  # noqa: E402

from parry.integrations.llamaindex import ParryHandler  # noqa: E402

GPT4O_MINI_CALL = 10 * 0.15 / 1e6 + 5 * 0.60 / 1e6


@pytest.fixture
def llm(mock):
    return li_openai.OpenAI(
        model="gpt-4o-mini", api_base=f"{mock.url}/v1", api_key="test-key", max_retries=0
    )


@pytest.fixture
def callback_settings():
    previous = Settings.callback_manager
    yield
    Settings.callback_manager = previous


def _query_engine(llm):
    docs = [
        Document(text="parry is a guard rail for LLM calls."),
        Document(text="The token bucket refills one slot per second."),
    ]
    index = VectorStoreIndex.from_documents(docs, embed_model=MockEmbedding(embed_dim=8))
    return index.as_query_engine(llm=llm)


def test_install_caches_llamaindex_chat(mock, llm):
    guard = parry.install()
    msgs = [ChatMessage(role="user", content="hello")]
    a = llm.chat(msgs)
    b = llm.chat(msgs)
    assert a.message.content == b.message.content == "echo: hello"
    assert mock.count() == 1
    assert guard.stats.cache_hits == 1


def test_handler_tracks_cost_on_query_engine(mock, llm, callback_settings):
    guard = Parry()
    Settings.callback_manager = CallbackManager([ParryHandler(guard)])
    engine = _query_engine(llm)
    response = engine.query("What is parry?")
    assert str(response).startswith("echo:")
    s = guard.stats
    assert s.requests == mock.count("/chat/completions") >= 1
    assert s.total_cost_usd == pytest.approx(s.requests * GPT4O_MINI_CALL)


def test_handler_enforces_budget(mock, llm, callback_settings):
    Settings.callback_manager = CallbackManager([ParryHandler(Parry(max_cost_usd=0))])
    engine = _query_engine(llm)
    with pytest.raises(parry.BudgetExceeded):
        engine.query("What is parry?")
    assert mock.count("/chat/completions") == 0


def test_handler_and_install_do_not_double_count(mock, llm, callback_settings):
    guard = parry.install()
    Settings.callback_manager = CallbackManager([ParryHandler(guard)])
    engine = _query_engine(llm)
    engine.query("What is parry?")
    engine.query("What is parry?")
    s = guard.stats
    assert mock.count("/chat/completions") == 1
    assert s.cache_hits == 1
    assert s.total_cost_usd == pytest.approx(GPT4O_MINI_CALL)
