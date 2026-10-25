from __future__ import annotations

import time

import pytest
from conftest import chat, message

import parry
from parry import Parry, Retry


class FakeHTTPError(Exception):
    def __init__(self, status_code: int, headers: dict[str, str] | None = None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.response = type("R", (), {"headers": headers or {}})()


def test_backoff_schedule():
    r = Retry(max_attempts=5, base_delay=0.5, max_delay=4.0, jitter=False)
    assert [r.delay(i) for i in range(1, 6)] == [0.5, 1.0, 2.0, 4.0, 4.0]
    assert r.delay(1, retry_after=3.0) == 3.0
    assert r.delay(1, retry_after=100.0) == 4.0


def test_jitter_is_bounded():
    r = Retry(base_delay=1.0, max_delay=30.0)
    for _ in range(200):
        assert 1.0 <= r.delay(2) <= 2.0


def test_classification():
    assert Retry.is_retryable("RateLimitError", 429)
    assert Retry.is_retryable("APITimeoutError")
    assert Retry.is_retryable("InternalServerError", 503)
    assert Retry.is_retryable("OverloadedError", 529)
    assert not Retry.is_retryable("BadRequestError", 400)
    assert not Retry.is_retryable("AuthenticationError", 401)
    assert not Retry.is_retryable("ValueError")
    r = Retry()
    assert r.classify(FakeHTTPError(429, {"retry-after": "2"})) == (True, 2.0)
    assert r.classify(FakeHTTPError(429, {"retry-after-ms": "250"})) == (True, 0.25)
    assert r.classify(FakeHTTPError(401)) == (False, None)


def test_standalone_call():
    calls = []

    def flaky(x, *, y):
        calls.append(1)
        if len(calls) < 3:
            raise FakeHTTPError(503)
        return x + y

    r = Retry(max_attempts=3, base_delay=0.001, jitter=False)
    assert r.call(flaky, 1, y=2) == 3
    assert len(calls) == 3 and r.retries == 2


def test_standalone_call_does_not_retry_client_errors():
    calls = []

    def bad():
        calls.append(1)
        raise FakeHTTPError(401)

    with pytest.raises(FakeHTTPError):
        Retry(base_delay=0.001).call(bad)
    assert len(calls) == 1


def test_invalid_policy():
    with pytest.raises(ValueError):
        Retry(max_attempts=0)
    with pytest.raises(ValueError):
        Retry(multiplier=0.5)


def test_429_is_retried_and_succeeds(mock, openai_client):
    mock.state.failures.extend([429])
    guard = parry.install(Parry(retry_base_delay=0.01))
    response = chat(openai_client)
    assert response.choices[0].message.content == "echo: hello"
    assert mock.count() == 2
    assert guard.stats.retries == 1


def test_retry_after_header_is_honoured(mock, openai_client):
    mock.state.failures.extend([429])
    mock.state.retry_after = "0.3"
    parry.install(Parry(retry_base_delay=0.001))
    start = time.perf_counter()
    chat(openai_client)
    assert time.perf_counter() - start >= 0.25


def test_gives_up_after_max_attempts(mock, openai_client):
    import openai

    mock.state.failures.extend([500, 502, 503, 504])
    guard = parry.install(Parry(max_attempts=3, retry_base_delay=0.001))
    with pytest.raises(openai.InternalServerError):
        chat(openai_client)
    assert mock.count() == 3
    assert guard.stats.errors == 1 and guard.stats.retries == 2


def test_auth_errors_are_not_retried(mock, openai_client):
    import openai

    mock.state.failures.extend([401, 401])
    parry.install(Parry(retry_base_delay=0.001))
    with pytest.raises(openai.AuthenticationError):
        chat(openai_client)
    assert mock.count() == 1


def test_bad_request_not_retried(mock, openai_client):
    import openai

    mock.state.failures.extend([400])
    parry.install(Parry(retry_base_delay=0.001))
    with pytest.raises(openai.BadRequestError):
        chat(openai_client)
    assert mock.count() == 1


def test_retry_disabled(mock, openai_client):
    import openai

    mock.state.failures.extend([429])
    parry.install(retry=False)
    with pytest.raises(openai.RateLimitError):
        chat(openai_client)


def test_connection_errors_are_retried(openai_client):
    import openai

    client = openai.OpenAI(base_url="http://127.0.0.1:9/v1", api_key="k", max_retries=0)
    guard = parry.install(Parry(max_attempts=2, retry_base_delay=0.001))
    with pytest.raises(openai.APIConnectionError):
        chat(client)
    assert guard.stats.retries == 1


def test_anthropic_overload_is_retried(mock, anthropic_client):
    mock.state.failures.extend([529])
    guard = parry.install(Parry(retry_base_delay=0.001))
    message(anthropic_client)
    assert guard.stats.retries == 1 and mock.count() == 2


async def test_async_retry(mock, async_anthropic_client):
    mock.state.failures.extend([429, 503])
    guard = parry.install(Parry(retry_base_delay=0.001))
    await async_anthropic_client.messages.create(
        model="claude-haiku-5-5", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
    )
    assert guard.stats.retries == 2 and mock.count() == 3
