from __future__ import annotations

import time

import pytest
from conftest import chat, message

import parry
from parry import Parry, RateLimit


def test_standalone_token_bucket():
    rl = RateLimit()
    rl.set_limit("gpt-4o", requests_per_minute=60)
    assert all(rl.try_acquire("openai", "gpt-4o") == 0.0 for _ in range(60))
    wait = rl.try_acquire("openai", "gpt-4o")
    assert 0 < wait <= 1.05


def test_standalone_blocking_acquire():
    rl = RateLimit()
    rl.set_limit("m", requests_per_minute=600, burst=1)  # one slot every 0.1 s
    assert rl.acquire("p", "m")
    start = time.perf_counter()
    assert rl.acquire("p", "m", timeout=1.0)
    assert time.perf_counter() - start >= 0.05
    assert not rl.acquire("p", "m", timeout=0.0)


def test_invalid_limits_rejected():
    with pytest.raises(ValueError):
        RateLimit().set_limit("m", requests_per_minute=0)


def test_defaults_follow_provider_tables():
    guard = Parry()
    assert guard.rate_limit_for("openai", "gpt-4o")["requests_per_minute"] == 500
    assert guard.rate_limit_for("anthropic", "claude-opus-5-5")["requests_per_minute"] == 50
    guard.set_rate_limit("gpt-4o", requests_per_minute=60)
    assert guard.rate_limit_for("openai", "gpt-4o")["requests_per_minute"] == 60
    assert guard.rate_limit_for("openai", "gpt-4o-mini")["requests_per_minute"] == 500


def test_61st_request_is_refused(mock, openai_client):
    guard = Parry(cache=False, on_rate_limit="raise")
    guard.set_rate_limit("gpt-4o", requests_per_minute=60)
    parry.install(guard)
    start = time.monotonic()
    with pytest.raises(parry.RateLimited) as info:
        for i in range(100):
            chat(openai_client, content=f"q{i}")
    elapsed = time.monotonic() - start
    assert info.value.model == "gpt-4o" and info.value.retry_after > 0
    # The bucket holds 60 and refills one slot per second, so exactly 60 get
    # through plus one per second the loop itself took.
    sent = mock.count()
    assert 60 <= sent <= 60 + int(elapsed) + 1, (sent, elapsed)
    s = guard.stats
    assert s.requests_blocked == 1 and s.rate_limit_rejections == 1
    # Other models have their own bucket.
    chat(openai_client, model="gpt-4o-mini")
    assert mock.count() == sent + 1


def test_wait_mode_delays_instead_of_failing(mock, openai_client):
    guard = Parry(cache=False)
    guard.set_rate_limit("gpt-4o", requests_per_minute=600, burst=1)
    parry.install(guard)
    start = time.perf_counter()
    for i in range(3):
        chat(openai_client, content=f"q{i}")
    elapsed = time.perf_counter() - start
    assert mock.count() == 3
    assert elapsed >= 0.15, elapsed
    assert guard.stats.requests_blocked == 2


def test_wait_mode_gives_up_after_max_wait(mock, openai_client):
    guard = Parry(cache=False, max_wait=0.01)
    guard.set_rate_limit("gpt-4o", requests_per_minute=1)
    parry.install(guard)
    chat(openai_client)
    with pytest.raises(parry.RateLimited):
        chat(openai_client, content="again")


def test_cache_hits_do_not_consume_rate_limit(mock, openai_client):
    guard = Parry(on_rate_limit="raise")
    guard.set_rate_limit("gpt-4o", requests_per_minute=1)
    parry.install(guard)
    for _ in range(10):
        chat(openai_client)
    assert mock.count() == 1


def test_unlimited_override(mock, openai_client):
    guard = Parry(cache=False, on_rate_limit="raise")
    guard.set_rate_limit("*", requests_per_minute=1)
    guard.set_rate_limit("gpt-4o")  # no limits → unlimited for this model
    parry.install(guard)
    for i in range(5):
        chat(openai_client, content=str(i))
    assert mock.count() == 5


def test_tokens_per_minute(mock, openai_client):
    guard = Parry(cache=False, on_rate_limit="raise")
    guard.set_rate_limit("gpt-4o", tokens_per_minute=60)
    parry.install(guard)
    chat(openai_client, content="short", max_tokens=20)
    with pytest.raises(parry.RateLimited):
        chat(openai_client, content="short again", max_tokens=40)


def test_anthropic_limits(mock, anthropic_client):
    guard = Parry(cache=False, on_rate_limit="raise")
    guard.set_rate_limit("anthropic:*", requests_per_minute=2)
    parry.install(guard)
    message(anthropic_client, "a")
    message(anthropic_client, "b")
    with pytest.raises(parry.RateLimited):
        message(anthropic_client, "c")


async def test_async_wait(mock, async_openai_client):
    guard = Parry(cache=False)
    guard.set_rate_limit("gpt-4o", requests_per_minute=600, burst=1)
    parry.install(guard)
    start = time.perf_counter()
    for i in range(3):
        await async_openai_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": str(i)}]
        )
    assert time.perf_counter() - start >= 0.15
    assert mock.count() == 3
