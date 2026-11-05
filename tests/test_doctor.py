"""Tests for ``parry doctor`` — the diagnostic over a guard or a ledger."""

from __future__ import annotations

import json

from conftest import chat

import parry
from parry.__main__ import main
from parry._core import ParryStats
from parry._doctor import Report, diagnose_ledger, diagnose_stats
from parry._storage import LedgerHealth, SpendSummary, SQLiteStore


def _codes(findings) -> set[str]:
    return {f.code for f in findings}


# --- diagnose_stats -----------------------------------------------------------


def test_clean_guard_reports_ok():
    findings = diagnose_stats(ParryStats())
    assert _codes(findings) == {"ok"}
    assert Report(findings).ok


def test_low_cache_hit_rate_warns():
    stats = ParryStats(requests=100, cache_hits=2, cache_misses=98)
    report = Report(diagnose_stats(stats))
    assert "cache.low_hit_rate" in _codes(report.findings)
    assert not report.ok and report.worst == "warn"


def test_few_lookups_do_not_trip_cache_check():
    # Below the minimum sample, a miss-heavy start is not flagged.
    findings = diagnose_stats(ParryStats(requests=3, cache_hits=0, cache_misses=3))
    assert "cache.low_hit_rate" not in _codes(findings)


def test_estimated_and_unpriced_warn():
    stats = ParryStats(requests=10, estimated_requests=5, unpriced_requests=3)
    codes = _codes(diagnose_stats(stats))
    assert "cost.estimated" in codes and "cost.unpriced" in codes


def test_rate_limit_rejections_warn_blocked_is_info():
    assert "ratelimit.rejections" in _codes(
        diagnose_stats(ParryStats(requests=5, rate_limit_rejections=2))
    )
    findings = diagnose_stats(ParryStats(requests=5, requests_blocked=2))
    assert "ratelimit.throttled" in _codes(findings)
    assert Report(findings).ok  # throttling alone is informational, not a warning


def test_spend_concentration_is_info():
    stats = ParryStats(
        requests=10,
        total_cost_usd=10.0,
        by_model={
            "openai:gpt-4o": {"requests": 9, "cost_usd": 9.5},
            "openai:gpt-4o-mini": {"requests": 1, "cost_usd": 0.5},
        },
    )
    findings = diagnose_stats(stats)
    assert "cost.concentration" in _codes(findings)
    assert Report(findings).ok


# --- Report formatting --------------------------------------------------------


def test_report_summary_and_as_dict():
    report = Report(diagnose_stats(ParryStats(requests=10, unpriced_requests=4)))
    text = report.summary()
    assert "no known price" in text and "set_pricing" in text
    d = report.as_dict()
    assert d["ok"] is False and d["worst"] == "warn"
    assert any(f["code"] == "cost.unpriced" for f in d["findings"])


# --- diagnose_ledger ----------------------------------------------------------


def test_empty_ledger_is_info():
    findings = diagnose_ledger(LedgerHealth(0, 0, 0), [])
    assert _codes(findings) == {"ledger.empty"}


def test_ledger_flags_estimated_and_unpriced():
    health = LedgerHealth(requests=20, estimated=10, unpriced=4)
    codes = _codes(diagnose_ledger(health, []))
    assert "cost.estimated" in codes and "cost.unpriced" in codes


def test_ledger_spend_concentration():
    health = LedgerHealth(requests=10, estimated=0, unpriced=0)
    by_model = [
        ("openai:gpt-4o", SpendSummary(9, 900, 90, 9.5)),
        ("openai:gpt-4o-mini", SpendSummary(1, 10, 5, 0.5)),
    ]
    assert "cost.concentration" in _codes(diagnose_ledger(health, by_model))


# --- guard + module-level entry points ----------------------------------------


def test_guard_doctor_after_real_calls(mock, openai_client):
    guard = parry.install()
    chat(openai_client)
    chat(openai_client)  # identical → cache hit
    report = guard.doctor()
    assert isinstance(report, Report)
    assert parry.doctor().ok == report.ok  # module-level reads the active guard


# --- CLI ----------------------------------------------------------------------


def test_doctor_cli_missing_db(tmp_path):
    assert main(["doctor", "--db", str(tmp_path / "nope.db")]) == 1


def test_doctor_cli_reports_and_exits_two_on_problems(tmp_path, capsys):
    db = tmp_path / "parry.db"
    store = SQLiteStore(db)
    # Four of five records are unpriced (cost None) and estimated.
    store.record("x", "mystery", 100, 50, None, True)
    store.record("x", "mystery", 100, 50, None, True)
    store.record("x", "mystery", 100, 50, None, True)
    store.record("x", "mystery", 100, 50, None, True)
    store.record("openai", "gpt-4o", 10, 5, 0.001, False)
    store.close()

    code = main(["doctor", "--db", str(db), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 2 and report["ok"] is False
    assert {"cost.estimated", "cost.unpriced"} <= {f["code"] for f in report["findings"]}


def test_doctor_cli_clean_ledger_exits_zero(tmp_path, capsys):
    db = tmp_path / "parry.db"
    store = SQLiteStore(db)
    store.record("openai", "gpt-4o", 10, 5, 0.001, False)
    store.close()
    assert main(["doctor", "--db", str(db)]) == 0
    assert "No problems detected" in capsys.readouterr().out
