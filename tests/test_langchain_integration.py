from __future__ import annotations

import pytest

import parry
from parry import Parry

langchain_openai = pytest.importorskip("langchain_openai")
from parry.integrations.langchain import ParryCallback  # noqa: E402

GPT4O_MINI_CALL = 10 * 0.15 / 1e6 + 5 * 0.60 / 1e6


@pytest.fixture
def llm_factory(mock):
    def make(**kwargs):
        return langchain_openai.ChatOpenAI(
            model="gpt-4o-mini",
            base_url=f"{mock.url}/v1",
            api_key="test-key",
            max_retries=0,
            **kwargs,
        )

    return make


def test_install_caches_chatopenai(mock, llm_factory):
    guard = parry.install()
    llm = llm_factory()
    a = llm.invoke("What is a token bucket?")
    b = llm.invoke("What is a token bucket?")
    assert a.content == b.content == "echo: What is a token bucket?"
    assert mock.count() == 1
    assert guard.stats.cache_hits == 1
    assert guard.total_cost_usd == pytest.approx(GPT4O_MINI_CALL)


async def test_install_caches_async_chatopenai(mock, llm_factory):
    parry.install()
    llm = llm_factory()
    await llm.ainvoke("hi")
    await llm.ainvoke("hi")
    assert mock.count() == 1


def test_install_with_streaming(mock, llm_factory):
    guard = parry.install()
    llm = llm_factory()
    text = "".join(chunk.content for chunk in llm.stream("stream me"))
    assert text.strip() == "echo: stream me"
    assert guard.stats.requests == 1 and guard.stats.output_tokens > 0


def test_callback_without_install(mock, llm_factory):
    guard = Parry()
    llm = llm_factory(callbacks=[ParryCallback(guard)])
    llm.invoke("one")
    llm.invoke("one")  # no hooks → no cache, but cost is tracked
    assert mock.count() == 2
    s = guard.stats
    assert s.requests == 2
    assert (s.input_tokens, s.output_tokens) == (20, 10)
    assert s.total_cost_usd == pytest.approx(2 * GPT4O_MINI_CALL)


def test_callback_enforces_budget(mock, llm_factory):
    guard = Parry(max_cost_usd=0)
    llm = llm_factory(callbacks=[ParryCallback(guard)])
    with pytest.raises(parry.BudgetExceeded):
        llm.invoke("hi")
    assert mock.count() == 0


def test_callback_enforces_rate_limit(mock, llm_factory):
    guard = Parry(on_rate_limit="raise")
    guard.set_rate_limit("gpt-4o-mini", requests_per_minute=1)
    llm = llm_factory(callbacks=[ParryCallback(guard)])
    llm.invoke("a")
    with pytest.raises(parry.RateLimited):
        llm.invoke("b")


def test_callback_and_install_do_not_double_count(mock, llm_factory):
    guard = parry.install()
    llm = llm_factory(callbacks=[ParryCallback(guard)])
    llm.invoke("count me once")
    s = guard.stats
    assert s.requests == 1
    assert s.total_cost_usd == pytest.approx(GPT4O_MINI_CALL)


def test_install_caches_chatanthropic(mock):
    langchain_anthropic = pytest.importorskip("langchain_anthropic")
    guard = parry.install()
    llm = langchain_anthropic.ChatAnthropic(
        model="claude-haiku-5-5", base_url=mock.url, api_key="test-key", max_retries=0
    )
    a = llm.invoke("hello")
    b = llm.invoke("hello")
    assert a.content == b.content == "echo: hello"
    assert mock.count() == 1
    assert guard.stats.cache_hits == 1
