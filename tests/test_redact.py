"""PII redaction: the Redactor and the keep-PII-off-disk cache policy."""

from __future__ import annotations

import pytest
from conftest import chat

import parry
from parry import Redactor

# --- Redactor -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "reach me at alice@example.com",
        "ssn 123-45-6789",
        "card 4111 1111 1111 1111",
        "call 415-555-0137",
        "host 10.0.0.42",
        "token sk-abcdefghijklmnop0123",
    ],
)
def test_detects_each_default_pattern(text):
    assert Redactor().detect(text) is True


def test_clean_text_is_not_detected():
    assert Redactor().detect("the quick brown fox jumped over it") is False


def test_redact_replaces_and_counts():
    r = Redactor(["email"])
    out, hits = r.redact("a@b.com and c@d.org")
    assert out == "[REDACTED:email] and [REDACTED:email]" and hits == 2


def test_pattern_selection_and_extra():
    r = Redactor(["email"], extra={"ticket": r"TICK-\d+"})
    assert r.names == ("email", "ticket")
    assert r.detect("TICK-42") and not r.detect("123-45-6789")  # ssn not selected


def test_unknown_pattern_and_empty_raise():
    with pytest.raises(ValueError):
        Redactor(["not-a-thing"])
    with pytest.raises(ValueError):
        Redactor([])


# --- the cache policy ---------------------------------------------------------


def _fresh_guard_misses(db, mock, openai_client, content) -> bool:
    """A new guard on the same DB: True if it had to call the provider (i.e. the
    entry was not persisted)."""
    parry.uninstall()
    parry.install(storage=db)
    before = mock.count()
    chat(openai_client, content=content)
    return mock.count() > before


def test_pii_response_kept_in_memory_but_off_disk(tmp_path, mock, openai_client):
    db = tmp_path / "p.db"
    guard = parry.install(storage=db)
    guard.set_redaction()
    chat(openai_client, content="contact me at alice@example.com")  # echoed → PII in response
    assert guard.stats.pii_redacted == 1

    chat(openai_client, content="contact me at alice@example.com")  # same process → memory hit
    assert mock.count() == 1 and guard.stats.cache_hits == 1

    # A fresh guard on the same DB must miss: the entry was never persisted.
    assert _fresh_guard_misses(db, mock, openai_client, "contact me at alice@example.com")


def test_pii_in_request_blocks_persist(tmp_path, mock, openai_client):
    mock.state.reply = "nothing sensitive here"  # clean response, PII only in the prompt
    db = tmp_path / "p.db"
    guard = parry.install(storage=db)
    guard.set_redaction()
    chat(openai_client, content="my ssn is 123-45-6789")
    assert guard.stats.pii_redacted == 1


def test_clean_response_is_persisted(tmp_path, mock, openai_client):
    db = tmp_path / "p.db"
    guard = parry.install(storage=db)
    guard.set_redaction()
    chat(openai_client, content="hello world")
    assert guard.stats.pii_redacted == 0
    # A fresh guard on the same DB hits: the entry was persisted.
    assert not _fresh_guard_misses(db, mock, openai_client, "hello world")


def test_redaction_off_by_default(tmp_path, mock, openai_client):
    db = tmp_path / "p.db"
    guard = parry.install(storage=db)  # no set_redaction
    chat(openai_client, content="email bob@example.com")
    assert guard.stats.pii_redacted == 0
    assert not _fresh_guard_misses(db, mock, openai_client, "email bob@example.com")


def test_set_redaction_false_disables(tmp_path, mock, openai_client):
    db = tmp_path / "p.db"
    guard = parry.install(storage=db)
    guard.set_redaction()
    guard.set_redaction(False)
    chat(openai_client, content="email bob@example.com")
    assert guard.stats.pii_redacted == 0


def test_custom_redactor_can_be_passed(tmp_path, mock, openai_client):
    db = tmp_path / "p.db"
    guard = parry.install(storage=db)
    guard.set_redaction(redactor=Redactor(extra={"codename": r"PROJECT-\w+"}))
    mock.state.reply = "status for PROJECT-falcon is green"
    chat(openai_client, content="status?")
    assert guard.stats.pii_redacted == 1
