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


def test_adapter_usage_detail_splits_cache_tokens():
    from parry._adapters import ANTHROPIC_MESSAGES, OPENAI_CHAT

    # OpenAI prompt_tokens includes cached tokens; they must be split out.
    oa = OPENAI_CHAT.usage_detailed(
        {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "prompt_tokens_details": {"cached_tokens": 400},
            }
        }
    )
    assert oa == (600, 200, 400, 0, 0)
    # Anthropic reports cache tokens separately from the base input.
    an = ANTHROPIC_MESSAGES.usage_detailed(
        {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 1000,
                "cache_creation_input_tokens": 200,
            }
        }
    )
    assert an == (100, 50, 1000, 200, 0)
    # The newer per-TTL cache-creation breakdown is honoured when present.
    an2 = ANTHROPIC_MESSAGES.usage_detailed(
        {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 30,
                    "ephemeral_1h_input_tokens": 70,
                },
            }
        }
    )
    assert an2 == (100, 50, 0, 30, 70)


def test_openai_cache_read_billed_at_discount():
    cost = Cost()
    # 400 of 1000 prompt tokens were cache hits; the cached part bills at 10%.
    c = cost.record_tiered("openai", "gpt-4o", 600, 200, 400)
    assert c == pytest.approx((600 * 2.5 + 400 * 0.25 + 200 * 10.0) / 1e6)
    totals = cost.totals()
    assert totals["input_tokens"] == 1000  # cached folded into the input total


def test_priority_service_tier_bills_fast(mock, openai_client):
    guard = parry.install(cache=False)
    chat(openai_client, service_tier="priority")
    assert guard.total_cost_usd == pytest.approx(GPT4O_CALL * 1.75)


def test_tiered_cost_cache_batch_fast():
    cost = Cost()
    std = cost.compute("gpt-4o", 1000, 500)
    assert cost.compute_tiered("gpt-4o", 1000, 500, tier="batch") == pytest.approx(std * 0.5)
    assert cost.compute_tiered("gpt-4o", 1000, 500, tier="fast") == pytest.approx(std * 1.75)
    # Cache-read input is 10% of the standard input rate on Anthropic.
    read = cost.compute_tiered("claude-sonnet-5-5", 0, 0, cached_input_tokens=1_000_000)
    assert read == pytest.approx(0.20)  # 2.00 * 0.10
    assert cost.compute_tiered("not-a-model", 1, 1) is None


def test_tier_rates_exposed():
    cost = Cost()
    rates = cost.tier_rates("claude-sonnet-5-5")
    assert rates is not None
    assert rates["cached_input_per_million"] == pytest.approx(0.20)
    assert rates["cache_write_5m_per_million"] == pytest.approx(2.50)
    assert rates["cache_write_1h_per_million"] == pytest.approx(4.00)
    assert rates["fast_multiplier"] == pytest.approx(2.0)
    assert cost.tier_rates("not-a-model") is None


def test_tier_validation():
    with pytest.raises(ValueError):
        Cost().compute_tiered("gpt-4o", 1, tier="bogus")


def test_pricing_table_carries_tier_fields():
    row = next(r for r in Cost.pricing_table() if r["model"] == "gpt-4o")
    assert row["cached_input_per_million"] == pytest.approx(0.25)  # 2.50 * 0.10
    assert row["batch_multiplier"] == pytest.approx(0.5)
    assert row["fast_multiplier"] == pytest.approx(1.75)
    assert row["cache_write_5m_per_million"] is None  # OpenAI auto-caches


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
