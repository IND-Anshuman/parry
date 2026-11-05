"""Multi-key rotation (guard.set_keys)."""

from __future__ import annotations

from collections import deque

import pytest
from conftest import chat

import parry
from parry import Parry
from parry._core import _Keyring


def _auth(request) -> str | None:
    for name, value in request["headers"].items():
        if name.lower() == "authorization":
            return value
    return None


# --- _Keyring unit ------------------------------------------------------------


def test_keyring_round_robin():
    ring = _Keyring("openai", ["a", "b", "c"], ("Authorization", "Bearer {key}"))
    assert [ring.next_key() for _ in range(4)] == ["a", "b", "c", "a"]


def test_keyring_skips_a_resting_key():
    ring = _Keyring("openai", ["a", "b"], ("Authorization", "Bearer {key}"))
    assert ring.next_key() == "a"
    ring.penalize("b", 60)  # b rests; rotation should avoid it
    assert ring.next_key() == "a"
    assert ring.next_key() == "a"


def test_keyring_all_resting_returns_soonest():
    ring = _Keyring("openai", ["a", "b"], ("Authorization", "Bearer {key}"))
    ring.penalize("a", 60)
    ring.penalize("b", 1)
    assert ring.next_key() == "b"  # frees up soonest


def test_keyring_header_template():
    ring = _Keyring("anthropic", ["secret"], ("x-api-key", "{key}"))
    assert ring.header_for("secret") == ("x-api-key", "secret")


# --- set_keys validation ------------------------------------------------------


def test_set_keys_validation():
    guard = Parry()
    with pytest.raises(ValueError):
        guard.set_keys("openai", [])
    with pytest.raises(ValueError):
        guard.set_keys("openai", ["k"], header=("X", "no-placeholder"))
    assert guard.set_keys("openai", ["k"]) is guard
    assert guard.remove_keys("openai") is True
    assert guard.remove_keys("openai") is False


# --- end to end over the mock -------------------------------------------------


def test_requests_rotate_across_keys(mock, openai_client):
    guard = parry.install(cache=False)
    guard.set_keys("openai", ["key-one", "key-two", "key-three"])
    for _ in range(3):
        chat(openai_client)
    auths = [_auth(r) for r in mock.requests]
    assert auths == ["Bearer key-one", "Bearer key-two", "Bearer key-three"]


def test_rate_limited_key_is_skipped_on_retry(mock, openai_client):
    guard = parry.install(cache=False)
    guard.set_keys("openai", ["key-A", "key-B"])
    mock.state.failures = deque([429])  # the first request (key-A) gets limited
    chat(openai_client)
    assert mock.count() == 2
    assert _auth(mock.requests[0]) == "Bearer key-A"
    assert _auth(mock.requests[1]) == "Bearer key-B"  # retry rotated off the limited key


def test_rotation_does_not_change_the_cache_key(mock, openai_client):
    guard = parry.install()  # cache on
    guard.set_keys("openai", ["key-one", "key-two"])
    chat(openai_client)
    chat(openai_client)  # identical request → served from cache despite a new key
    assert mock.count() == 1
    assert guard.stats.cache_hits == 1


def test_remove_keys_restores_the_client_key(mock, openai_client):
    guard = parry.install(cache=False)
    guard.set_keys("openai", ["rotated"])
    chat(openai_client)
    guard.remove_keys("openai")
    chat(openai_client)
    assert _auth(mock.requests[0]) == "Bearer rotated"
    assert _auth(mock.requests[1]) == "Bearer test-key"  # the client's own key


def test_rotation_preserves_user_extra_headers(mock, openai_client):
    guard = parry.install(cache=False)
    guard.set_keys("openai", ["rotated"])
    chat(openai_client, extra_headers={"x-trace": "abc"})
    assert _auth(mock.requests[0]) == "Bearer rotated"
    assert any(
        k.lower() == "x-trace" and v == "abc" for k, v in mock.requests[0]["headers"].items()
    )
