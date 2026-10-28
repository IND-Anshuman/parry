from __future__ import annotations

import logging

import pytest
from conftest import chat, message

import parry
from parry import Cost, Parry

GPT4O_CALL = 10 * 2.50 / 1e6 + 5 * 10.00 / 1e6  # 10 prompt + 5 completion tokens


def test_standalone_pricing():
    cost = Cost()
    assert cost.compute("gpt-4o", 1_000_000, 0) == pytest.approx(2.50)
    assert cost.compute("gpt-4o-mini", 0, 1_000_000) == pytest.approx(0.60)
    assert cost.compute("claude-opus-5-5", 1_000_000, 1_000_000) == pytest.approx(24.0)
    assert cost.compute("gpt-4o-2024-08-06", 1_000_000) == pytest.approx(2.50)
    assert cost.compute("not-a-model", 1, 1) is None
    assert cost.price("o3") == (2.0, 8.0)
    table = Cost.pricing_table()
    assert {"openai", "anthropic"} <= {row["provider"] for row in table}


def test_standalone_ledger():
    cost = Cost()
    cost.record("openai", "gpt-4o", 10, 5)
    cost.record("anthropic", "mystery", 10, 5)
    totals = cost.totals()
    assert totals["requests"] == 2 and totals["unpriced_requests"] == 1
    assert cost.total_cost_usd == pytest.approx(GPT4O_CALL)
    assert list(cost.by_model()) == ["openai:gpt-4o", "anthropic:mystery"]


def test_set_pricing_validates():
    with pytest.raises(ValueError):
        Cost().set_pricing("m", -1, 1)


def test_five_calls_match_hand_calculation(mock, openai_client):
    guard = parry.install(cache=False)
    for i in range(5):
        chat(openai_client, content=str(i))
    s = guard.stats
    assert s.total_cost_usd == pytest.approx(5 * GPT4O_CALL)
    assert (s.input_tokens, s.output_tokens) == (50, 25)
    assert s.by_model["openai:gpt-4o"]["requests"] == 5


def test_anthropic_cost(mock, anthropic_client):
    guard = parry.install(cache=False)
    message(anthropic_client, model="claude-sonnet-5-5")
    assert guard.total_cost_usd == pytest.approx(10 * 2 / 1e6 + 5 * 10 / 1e6)


def test_budget_exceeded_before_sending(mock, openai_client):
    guard = parry.install(cache=False, max_cost_usd=0.0002)
    chat(openai_client, content="1")
    chat(openai_client, content="2")
    assert guard.total_cost_usd == pytest.approx(2 * GPT4O_CALL)
    with pytest.raises(parry.BudgetExceeded) as info:
        for i in range(10):
            chat(openai_client, content=f"more {i}")
    # The third call still fits (spend 1.5e-4 + small prompt estimate < 2e-4);
    # it pushes spend to 2.25e-4, so the fourth is refused before sending.
    assert mock.count() == 3
    assert info.value.limit_usd == 0.0002
    assert guard.stats.budget_rejections == 1
    assert isinstance(info.value, parry.ParryError)


def test_budget_pre_flight_uses_prompt_estimate(mock, openai_client):
    # Nothing spent yet, but the prompt alone would cost more than the budget.
    parry.install(max_cost_usd=0.000001)
    with pytest.raises(parry.BudgetExceeded):
        chat(openai_client, content="word " * 2000)
    assert mock.count() == 0


def test_zero_budget_blocks_everything(mock, openai_client):
    parry.install(max_cost_usd=0)
    with pytest.raises(parry.BudgetExceeded):
        chat(openai_client)
    assert mock.count() == 0


def test_unpriced_model_warns_once(mock, openai_client, caplog):
    caplog.set_level(logging.WARNING, logger="parry")
    guard = parry.install(cache=False)
    chat(openai_client, model="my-local-llama")
    chat(openai_client, model="my-local-llama", content="again")
    warnings = [r for r in caplog.records if "no pricing" in r.getMessage()]
    assert len(warnings) == 1
    assert guard.stats.unpriced_requests == 2
    guard.set_pricing("my-local-llama", input_per_million=1.0, output_per_million=1.0)
    chat(openai_client, model="my-local-llama", content="priced")
    assert guard.total_cost_usd == pytest.approx(15 / 1e6)


def test_unpriced_model_raises_under_budget_by_default(mock, openai_client):
    # Fail closed: with a budget set, a model parry cannot price must not slip
    # through at $0 and silently bypass the limit.
    parry.install(cache=False, max_cost_usd=1.0)
    with pytest.raises(parry.UnpricedModel) as info:
        chat(openai_client, model="my-local-llama")
    assert mock.count() == 0
    assert info.value.model == "my-local-llama"
    assert info.value.limit_usd == 1.0
    # Subclasses BudgetExceeded so existing budget handlers still catch it.
    assert isinstance(info.value, parry.BudgetExceeded)


def test_unpriced_model_allowed_when_on_unpriced_ignore(mock, openai_client):
    guard = parry.install(cache=False, max_cost_usd=1.0, on_unpriced="ignore")
    chat(openai_client, model="my-local-llama")
    assert mock.count() == 1
    assert guard.stats.unpriced_requests == 1


def test_fail_closed_only_applies_with_a_budget(mock, openai_client):
    # No budget set: unpriced models pass through regardless of on_unpriced.
    guard = parry.install(cache=False)
    chat(openai_client, model="my-local-llama")
    assert mock.count() == 1
    assert guard.stats.unpriced_requests == 1


def test_priced_model_unaffected_by_fail_closed(mock, openai_client):
    guard = parry.install(cache=False, max_cost_usd=1.0)
    chat(openai_client)  # gpt-4o is in the pricing table
    assert mock.count() == 1
    assert guard.total_cost_usd == pytest.approx(GPT4O_CALL)


def test_on_unpriced_validates():
    with pytest.raises(ValueError):
        Parry(on_unpriced="nope")  # type: ignore[arg-type]


def test_cost_tracking_off(mock, openai_client):
    guard = parry.install(cost_tracking=False, cache=False)
    chat(openai_client)
    assert guard.total_cost_usd == 0


def test_responses_api_cost(mock, openai_client):
    guard = parry.install(cache=False)
    openai_client.responses.create(model="gpt-4.1", input="hello")
    assert guard.total_cost_usd == pytest.approx(10 * 2 / 1e6 + 5 * 8 / 1e6)


def test_reset_stats_restarts_budget(mock, openai_client):
    guard = parry.install(cache=False, max_cost_usd=GPT4O_CALL * 1.5)
    chat(openai_client, "a")
    chat(openai_client, "b")
    with pytest.raises(parry.BudgetExceeded):
        chat(openai_client, "c")
    guard.reset_stats()
    chat(openai_client, "c")
    assert guard.stats.requests == 1


def test_stats_summary_renders(mock, openai_client):
    guard = parry.install()
    chat(openai_client)
    chat(openai_client)
    text = guard.stats.summary()
    assert "1 hits / 1 misses" in text and "openai:gpt-4o" in text
    assert parry.stats().cache_hits == 1
    assert Parry().stats.requests == 0
