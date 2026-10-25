from __future__ import annotations

import json
import sqlite3
import time

import pytest
from conftest import chat

import parry
from parry import Cache, Parry

# --- native primitive --------------------------------------------------------


def test_cache_standalone_roundtrip():
    cache = Cache(max_entries=100, ttl_seconds=60)
    key = Cache.make_key("ns", json.dumps({"model": "gpt-4o", "messages": []}))
    assert cache.get(key) is None
    cache.set(key, "value", tokens=12, cost_usd=0.5)
    assert cache.get(key) == "value"
    assert cache.lookup(key) == ("value", 12, 0.5)
    assert key in cache and len(cache) == 1
    assert cache.hits == 2 and cache.misses == 1
    assert cache.invalidate(key) and key not in cache


def test_cache_key_is_canonical():
    a = Cache.make_key("ns", '{"b": 1, "a": "  hi  "}')
    b = Cache.make_key("ns", '{"a": "hi", "b": 1}')
    assert a == b
    assert a != Cache.make_key("other", '{"a": "hi", "b": 1}')
    with pytest.raises(ValueError):
        Cache.make_key("ns", "{not json")


def test_cache_is_always_bounded():
    with pytest.raises(ValueError):
        Cache(max_entries=0)
    cache = Cache(max_entries=10, ttl_seconds=None)
    for i in range(500):
        cache.set(f"k{i}", "v")
    assert len(cache) <= 10


def test_cache_ttl_expires():
    cache = Cache(max_entries=10, ttl_seconds=0.05)
    cache.set("k", "v")
    time.sleep(0.15)
    assert cache.get("k") is None


def test_rejects_unlimited_cache_size():
    with pytest.raises(ValueError):
        Parry(cache_size=None)  # type: ignore[arg-type]


# --- through install() ------------------------------------------------------


def test_second_identical_call_hits_cache_without_network(mock, openai_client):
    guard = parry.install()
    first = chat(openai_client)
    start = time.perf_counter()
    second = chat(openai_client)
    elapsed = time.perf_counter() - start
    assert mock.count() == 1
    assert elapsed < 0.005, f"cache hit took {elapsed * 1000:.2f} ms"
    assert type(second) is type(first)
    assert second.choices[0].message.content == first.choices[0].message.content == "echo: hello"
    s = guard.stats
    assert (s.cache_hits, s.cache_misses) == (1, 1)
    assert s.tokens_cached == 15
    assert s.cost_saved_usd == pytest.approx(s.total_cost_usd)


def test_different_parameters_miss(mock, openai_client):
    parry.install()
    chat(openai_client, temperature=0)
    chat(openai_client, temperature=1)
    chat(openai_client, temperature=0, response_format={"type": "json_object"})
    chat(openai_client, model="gpt-4o-mini", temperature=0)
    assert mock.count() == 4


def test_non_semantic_fields_do_not_change_the_key(mock, openai_client):
    parry.install()
    chat(openai_client, user="alice")
    chat(openai_client, user="bob", timeout=30, extra_headers={"x-trace": "1"})
    assert mock.count() == 1


def test_base_url_separates_entries(mock):
    import openai

    parry.install()
    a = openai.OpenAI(base_url=f"{mock.url}/v1", api_key="k", max_retries=0)
    b = openai.OpenAI(base_url=f"{mock.url}/v1/", api_key="k", max_retries=0)
    c = openai.OpenAI(
        base_url=f"{mock.url.replace('127.0.0.1', 'localhost')}/v1", api_key="k", max_retries=0
    )
    chat(a)
    chat(b)  # same normalised base URL → hit
    chat(c)  # different host → miss
    assert mock.count() == 2


def test_cache_disabled(mock, openai_client):
    parry.install(cache=False)
    chat(openai_client)
    chat(openai_client)
    assert mock.count() == 2


def test_hits_are_served_even_over_budget(mock, openai_client):
    guard = parry.install(max_cost_usd=0.0001)
    chat(openai_client)  # spends 10*2.5e-6 + 5*1e-5 = 7.5e-5
    chat(openai_client, content="other")  # pushes past the limit
    with pytest.raises(parry.BudgetExceeded):
        chat(openai_client, content="third")
    chat(openai_client)  # cached → free → allowed
    assert guard.stats.cache_hits == 1


def test_with_raw_response_hit(mock, openai_client):
    parry.install()
    raw = openai_client.chat.completions.with_raw_response.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hello"}]
    )
    first = raw.parse()
    raw2 = openai_client.chat.completions.with_raw_response.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hello"}]
    )
    assert mock.count() == 1
    assert isinstance(raw2, parry.CachedRawResponse)
    assert raw2.parse().choices[0].message.content == first.choices[0].message.content
    assert dict(raw2.headers)["x-parry-cache"] == "hit"
    # A plain create() shares the entry.
    chat(openai_client)
    assert mock.count() == 1


def test_unreadable_entry_is_dropped(mock, openai_client, monkeypatch):
    import parry._core

    guard = parry.install()
    chat(openai_client)

    def broken(blob):
        raise ValueError("SDK changed shape")

    monkeypatch.setattr(parry._core, "decode_response", broken)
    chat(openai_client)
    assert mock.count() == 2
    assert guard.stats.cache_hits == 0


# --- persistence --------------------------------------------------------------


def test_sqlite_cache_survives_restart(mock, openai_client, tmp_path):
    db = tmp_path / "parry.db"
    parry.install(Parry(storage=db))
    chat(openai_client)
    parry.uninstall()

    guard = parry.install(Parry(storage=db))
    chat(openai_client)
    assert mock.count() == 1
    assert guard.stats.cache_hits == 1
    assert guard.stats.lifetime_cost_usd == pytest.approx(10 * 2.5e-6 + 5 * 1e-5)


def test_corrupt_database_falls_back_to_memory(mock, openai_client, tmp_path, caplog):
    db = tmp_path / "broken.db"
    db.write_bytes(b"this is not a sqlite database" * 100)
    guard = parry.install(Parry(storage=db))
    chat(openai_client)
    chat(openai_client)
    assert mock.count() == 1
    assert guard._store is not None and not guard._store.ok
    assert any("continuing in memory only" in r.getMessage() for r in caplog.records)


def test_storage_ledger_schema(tmp_path, mock, openai_client):
    db = tmp_path / "ledger.db"
    parry.install(storage=db)
    chat(openai_client)
    chat(openai_client, content="again")
    rows = (
        sqlite3.connect(db)
        .execute("SELECT provider, model, input_tokens, output_tokens FROM ledger")
        .fetchall()
    )
    assert rows == [("openai", "gpt-4o", 10, 5), ("openai", "gpt-4o", 10, 5)]
