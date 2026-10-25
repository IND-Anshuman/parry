"""End-to-end tests of parry.install() against the real SDKs and a mock server."""

from __future__ import annotations

import inspect

import pytest
from conftest import chat, message

import parry
from parry import Parry

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")


# --- install / uninstall ------------------------------------------------------


def test_install_is_idempotent(mock, openai_client):
    from openai.resources.chat.completions import Completions

    g1 = parry.install()
    hooked = Completions.create
    g2 = parry.install()
    assert g1 is g2
    assert Completions.create is hooked
    assert hooked.__parry_original__.__name__ == "create"
    chat(openai_client)
    assert g1.stats.requests == 1  # not double counted


def test_install_with_new_options_replaces_guard():
    g1 = parry.install()
    g2 = parry.install(max_cost_usd=5)
    assert g2 is not g1 and parry.get_guard() is g2
    with pytest.raises(TypeError):
        parry.install(Parry(), cache=False)


def test_uninstall_restores_originals(mock, openai_client):
    from openai.resources.chat.completions import Completions

    original = Completions.create
    parry.install()
    assert Completions.create is not original
    parry.uninstall()
    assert Completions.create is original
    assert not parry.is_installed()
    chat(openai_client)
    chat(openai_client)
    assert mock.count() == 2


def test_signature_is_preserved():
    import inspect

    from openai.resources.chat.completions import Completions

    before = inspect.signature(Completions.create)
    parry.install()
    assert inspect.signature(Completions.create) == before


def test_selective_install(mock, openai_client, anthropic_client):
    parry.install(anthropic=False)
    chat(openai_client)
    chat(openai_client)
    message(anthropic_client)
    message(anthropic_client)
    assert mock.count("/chat/completions") == 1
    assert mock.count("/messages") == 2


# --- scoping -------------------------------------------------------------------


def test_context_manager_scopes_the_guard(mock, openai_client):
    with Parry() as guard:
        chat(openai_client)
        chat(openai_client)
        assert parry.get_guard() is guard
    chat(openai_client)  # outside the block: no guard → straight to the SDK
    assert mock.count() == 2
    assert guard.stats.cache_hits == 1


def test_nested_context_guards(mock, openai_client):
    outer = parry.install()
    with Parry(cache=False) as inner:
        chat(openai_client)
        chat(openai_client)
    assert inner.stats.requests == 2 and outer.stats.requests == 0
    chat(openai_client)
    assert outer.stats.requests == 1


def test_wrap_guards_one_client_only(mock):
    guarded = parry.wrap(openai.OpenAI(base_url=f"{mock.url}/v1", api_key="k", max_retries=0))
    plain = openai.OpenAI(base_url=f"{mock.url}/v1", api_key="k", max_retries=0)
    chat(guarded)
    chat(guarded)
    chat(plain)
    chat(plain)
    assert mock.count() == 3
    parry.unwrap(guarded)
    chat(guarded)
    assert mock.count() == 4


def test_wrap_wins_over_global_guard(mock):
    global_guard = parry.install()
    own = Parry()
    client = own.wrap(openai.OpenAI(base_url=f"{mock.url}/v1", api_key="k", max_retries=0))
    chat(client)
    assert own.stats.requests == 1 and global_guard.stats.requests == 0
    assert own.is_intercepting and global_guard.is_intercepting


def test_bypass(mock, openai_client):
    parry.install()
    chat(openai_client)
    with parry.bypass():
        chat(openai_client)
    assert mock.count() == 2


def test_parry_errors_are_not_retried(mock, openai_client):
    guard = parry.install(max_cost_usd=0)
    with pytest.raises(parry.BudgetExceeded):
        chat(openai_client)
    assert guard.stats.retries == 0


# --- OpenAI --------------------------------------------------------------------


async def test_openai_async_client(mock, async_openai_client):
    guard = parry.install()
    r1 = await async_openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
    )
    r2 = await async_openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
    )
    assert mock.count() == 1
    assert r1.choices[0].message.content == r2.choices[0].message.content
    assert guard.stats.total_cost_usd == pytest.approx(10 * 0.15 / 1e6 + 5 * 0.60 / 1e6)


async def test_openai_async_raw_response(mock, async_openai_client):
    parry.install()
    for _ in range(2):
        raw = await async_openai_client.chat.completions.with_raw_response.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )
        parsed = raw.parse()  # openai keeps a sync parse() on async raw responses
        assert parsed.choices[0].message.content == "echo: hi"
    assert mock.count() == 1


def test_install_async_only(mock, openai_client, async_openai_client):
    parry.install_async()
    chat(openai_client)
    chat(openai_client)
    assert mock.count() == 2


def test_openai_responses_api(mock, openai_client):
    guard = parry.install()
    r1 = openai_client.responses.create(model="gpt-4.1-mini", input="hello")
    r2 = openai_client.responses.create(model="gpt-4.1-mini", input="hello")
    assert mock.count() == 1
    assert r1.output_text == r2.output_text == "echo: hello"
    assert guard.stats.cache_hits == 1


def test_responses_background_not_cached(mock, openai_client):
    parry.install()
    openai_client.responses.create(model="gpt-4.1", input="x", background=True)
    openai_client.responses.create(model="gpt-4.1", input="x", background=True)
    assert mock.count() == 2


# --- streaming ------------------------------------------------------------------


def test_openai_stream_passthrough_and_cache_fill(mock, openai_client):
    guard = parry.install()
    stream = chat(openai_client, stream=True)
    assert type(stream) is openai.Stream  # the SDK's own object, instrumented in place
    text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    assert text.strip() == "echo: hello"
    s = guard.stats
    # No usage in the stream (include_usage not set) → estimated from tokens.
    assert s.estimated_requests == 1 and s.output_tokens > 0
    # A streaming request never reads the cache...
    list(chat(openai_client, stream=True))
    assert mock.count() == 2
    # ...but the assembled result serves the next non-streaming call.
    r = chat(openai_client)
    assert mock.count() == 2
    assert r.choices[0].message.content.strip() == "echo: hello"


def test_openai_stream_with_usage(mock, openai_client):
    guard = parry.install(cache=False)
    with chat(openai_client, stream=True, stream_options={"include_usage": True}) as stream:
        chunks = list(stream)
    assert chunks[-1].usage.prompt_tokens == 10
    s = guard.stats
    assert (s.input_tokens, s.output_tokens, s.estimated_requests) == (10, 5, 0)


def test_partially_consumed_stream_is_not_cached(mock, openai_client):
    parry.install()
    stream = chat(openai_client, stream=True)
    next(iter(stream))
    stream.close()
    chat(openai_client)
    assert mock.count() == 2


async def test_openai_async_stream(mock, async_openai_client):
    guard = parry.install()
    stream = await async_openai_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    assert isinstance(stream, openai.AsyncStream)
    parts = [c.choices[0].delta.content or "" async for c in stream if c.choices]
    assert "".join(parts).strip() == "echo: hi"
    assert guard.stats.requests == 1 and guard.stats.output_tokens > 0


def test_responses_stream(mock, openai_client):
    guard = parry.install()
    events = list(openai_client.responses.create(model="gpt-4.1", input="hello", stream=True))
    assert events[-1].type == "response.completed"
    assert (guard.stats.input_tokens, guard.stats.output_tokens) == (10, 5)
    openai_client.responses.create(model="gpt-4.1", input="hello")
    assert mock.count() == 1


# --- Anthropic ------------------------------------------------------------------


def test_anthropic_sync(mock, anthropic_client):
    guard = parry.install()
    m1 = message(anthropic_client)
    m2 = message(anthropic_client)
    assert mock.count() == 1
    assert isinstance(m2, anthropic.types.Message)
    assert m2.content[0].text == m1.content[0].text == "echo: hello"
    assert guard.stats.cache_hits == 1


def test_anthropic_system_prompt_is_part_of_the_key(mock, anthropic_client):
    parry.install()
    message(anthropic_client, system="be terse")
    message(anthropic_client, system="be verbose")
    assert mock.count() == 2


async def test_anthropic_async_and_raw(mock, async_anthropic_client):
    parry.install()
    for _ in range(2):
        raw = await async_anthropic_client.messages.with_raw_response.create(
            model="claude-opus-5-5", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )
        # anthropic >= 1.0 makes async raw parse() a coroutine; older SDKs keep it
        # synchronous. parry's cached stand-in must match whichever is installed.
        parsed = raw.parse()
        if inspect.isawaitable(parsed):
            parsed = await parsed
        assert parsed.content[0].text == "echo: hi"
    assert mock.count() == 1


def test_anthropic_sync_raw(mock, anthropic_client):
    parry.install()
    for _ in range(2):
        raw = anthropic_client.messages.with_raw_response.create(
            model="claude-opus-5-5", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )
        assert raw.parse().content[0].text == "echo: hi"
    assert mock.count() == 1


def test_anthropic_stream(mock, anthropic_client):
    guard = parry.install()
    stream = message(anthropic_client, stream=True)
    assert type(stream) is anthropic.Stream
    text = "".join(e.delta.text for e in stream if e.type == "content_block_delta")
    assert text.strip() == "echo: hello"
    s = guard.stats
    assert (s.input_tokens, s.output_tokens, s.estimated_requests) == (10, 5, 0)
    m = message(anthropic_client)  # served from the assembled stream
    assert mock.count() == 1
    assert m.content[0].text.strip() == "echo: hello"
    assert m.usage.output_tokens == 5


async def test_anthropic_async_stream(mock, async_anthropic_client):
    guard = parry.install()
    stream = await async_anthropic_client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=10,
        messages=[{"role": "user", "content": "x"}],
        stream=True,
    )
    async with stream:
        events = [e async for e in stream]
    assert events[-1].type == "message_stop"
    assert guard.stats.output_tokens == 5


def test_anthropic_beta_messages(mock, anthropic_client):
    parry.install()
    for _ in range(2):
        anthropic_client.beta.messages.create(
            model="claude-opus-5-5", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )
    assert mock.count() == 1


def test_anthropic_cache_tokens_weighting(mock, anthropic_client):
    from parry._adapters import AnthropicMessages

    usage = {
        "input_tokens": 100,
        "cache_creation_input_tokens": 1000,
        "cache_read_input_tokens": 10_000,
    }
    assert AnthropicMessages.billable_input(usage) == 100 + 1250 + 1000


def test_bookkeeping_failure_never_breaks_the_call(mock, openai_client, monkeypatch, caplog):
    import parry._core

    guard = parry.install()

    def explode(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(guard, "_record", explode)
    r = chat(openai_client)
    assert r.choices[0].message.content == "echo: hello"
    assert any("bookkeeping failed" in rec.getMessage() for rec in caplog.records)
    assert parry._core is not None


def test_stream_without_iterator_falls_back_to_proxy():
    from parry._adapters import OPENAI_CHAT
    from parry._stream import StreamWrapper, instrument_stream

    class Odd:
        def __init__(self):
            self.items = iter(
                [{"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]}]
            )

        def __next__(self):
            return next(self.items)

        def close(self):
            pass

    seen = []
    wrapped = instrument_stream(Odd(), OPENAI_CHAT.accumulator(), seen.append)
    assert isinstance(wrapped, StreamWrapper) and isinstance(wrapped, Odd)
    assert [c["choices"][0]["delta"]["content"] for c in wrapped] == ["hi"]
    assert seen and seen[0].complete and seen[0].output_text() == "hi"


async def test_async_stream_closed_early_is_metered_not_cached(mock, async_anthropic_client):
    guard = parry.install()
    stream = await async_anthropic_client.messages.create(
        model="claude-opus-5-5",
        max_tokens=10,
        messages=[{"role": "user", "content": "a b c"}],
        stream=True,
    )
    async for _ in stream:
        break
    await stream.close()
    assert guard.stats.requests == 1 and guard.stats.input_tokens == 10
    await async_anthropic_client.messages.create(
        model="claude-opus-5-5", max_tokens=10, messages=[{"role": "user", "content": "a b c"}]
    )
    assert mock.count() == 2
