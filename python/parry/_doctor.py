"""``parry doctor``: turn the guard's counters into plain-language findings.

The checks are pure functions over primitive inputs so they can run against a
live guard (:meth:`parry.Parry.doctor`) or a persisted SQLite ledger (the
``parry doctor`` CLI), and are easy to test without either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._core import ParryStats
    from ._storage import LedgerHealth, SpendSummary

# Levels, worst last. A "warn" is actionable; "info" is worth knowing; "ok"
# means a check passed with nothing to say.
_RANK = {"ok": 0, "info": 1, "warn": 2}
# ASCII tags, so the report prints on any console (Windows cp1252 included).
_SYMBOL = {"ok": "[ok]  ", "info": "[info]", "warn": "[warn]"}

# Thresholds. Kept conservative so the doctor stays quiet until a signal is real.
MIN_LOOKUPS_FOR_CACHE = 20
LOW_CACHE_HIT_RATE = 0.20
HIGH_ESTIMATED_FRACTION = 0.25
SPEND_CONCENTRATION = 0.90


@dataclass(frozen=True)
class Finding:
    level: str  # "ok" | "info" | "warn"
    code: str  # stable dotted id, e.g. "cost.unpriced"
    message: str
    hint: str = ""


@dataclass(frozen=True)
class Report:
    findings: list[Finding]

    @property
    def ok(self) -> bool:
        """True when no finding is a warning."""
        return all(f.level != "warn" for f in self.findings)

    @property
    def worst(self) -> str:
        return max((f.level for f in self.findings), key=lambda level: _RANK[level], default="ok")

    def summary(self) -> str:
        lines: list[str] = []
        for f in self.findings:
            lines.append(f"{_SYMBOL[f.level]} {f.message}")
            if f.hint:
                lines.append(f"       -> {f.hint}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "worst": self.worst,
            "findings": [
                {"level": f.level, "code": f.code, "message": f.message, "hint": f.hint}
                for f in self.findings
            ],
        }


def _finalise(findings: list[Finding]) -> list[Finding]:
    return findings or [Finding("ok", "ok", "No problems detected.")]


def diagnose_stats(stats: ParryStats) -> list[Finding]:
    """Findings from a live guard's session counters."""
    out: list[Finding] = []

    lookups = stats.cache_hits + stats.cache_misses
    if lookups >= MIN_LOOKUPS_FOR_CACHE and stats.cache_hit_rate < LOW_CACHE_HIT_RATE:
        out.append(
            Finding(
                "warn",
                "cache.low_hit_rate",
                f"Cache hit rate is {stats.cache_hit_rate:.0%} over {lookups} lookups.",
                "Most calls miss the cache. Requests must repeat verbatim to hit, and "
                "streaming calls never read the cache.",
            )
        )

    if stats.requests:
        estimated_fraction = stats.estimated_requests / stats.requests
        if estimated_fraction >= HIGH_ESTIMATED_FRACTION:
            out.append(
                Finding(
                    "warn",
                    "cost.estimated",
                    f"{estimated_fraction:.0%} of requests used estimated token counts "
                    f"({stats.estimated_requests}/{stats.requests}).",
                    "The response carried no usage (e.g. a stream without "
                    "stream_options={'include_usage': True}); those costs are approximate.",
                )
            )

    if stats.unpriced_requests:
        out.append(
            Finding(
                "warn",
                "cost.unpriced",
                f"{stats.unpriced_requests} requests used a model with no known price.",
                "Their cost is recorded as $0 and ignored by max_cost_usd. Add a price with "
                "guard.set_pricing(...), or sync the table (scripts/sync_pricing.py).",
            )
        )

    if stats.rate_limit_rejections:
        out.append(
            Finding(
                "warn",
                "ratelimit.rejections",
                f"{stats.rate_limit_rejections} requests were rejected by the rate limiter.",
                "Raise the limit with guard.set_rate_limit(...), or use on_rate_limit='wait' "
                "with a higher max_wait.",
            )
        )
    elif stats.requests_blocked:
        out.append(
            Finding(
                "info",
                "ratelimit.throttled",
                f"The limiter delayed {stats.requests_blocked} requests (none rejected).",
            )
        )

    if stats.budget_rejections:
        out.append(
            Finding(
                "info",
                "budget.stops",
                f"The budget stopped {stats.budget_rejections} requests.",
            )
        )

    if stats.total_cost_usd > 0 and len(stats.by_model) > 1:
        top_model, top = max(stats.by_model.items(), key=lambda kv: kv[1].get("cost_usd", 0.0))
        share = top.get("cost_usd", 0.0) / stats.total_cost_usd
        if share >= SPEND_CONCENTRATION:
            out.append(
                Finding(
                    "info",
                    "cost.concentration",
                    f"{share:.0%} of spend is on {top_model}.",
                )
            )

    return _finalise(out)


def diagnose_ledger(
    health: LedgerHealth, by_model: list[tuple[str, SpendSummary]]
) -> list[Finding]:
    """Findings from a persisted ledger. Session-only signals (cache hit rate,
    rate-limit triggers) are not stored, so this covers the subset the ledger
    keeps: estimated usage, unpriced models and spend concentration."""
    out: list[Finding] = []

    if health.requests == 0:
        return [Finding("info", "ledger.empty", "The ledger has no recorded requests yet.")]

    estimated_fraction = health.estimated / health.requests
    if estimated_fraction >= HIGH_ESTIMATED_FRACTION:
        out.append(
            Finding(
                "warn",
                "cost.estimated",
                f"{estimated_fraction:.0%} of recorded requests used estimated token counts "
                f"({health.estimated}/{health.requests}).",
                "Estimated costs are approximate; prefer non-streaming calls or "
                "stream_options={'include_usage': True} where exactness matters.",
            )
        )

    if health.unpriced:
        out.append(
            Finding(
                "warn",
                "cost.unpriced",
                f"{health.unpriced} recorded requests used a model with no known price.",
                "Those requests contributed $0 to spend. Add prices with guard.set_pricing(...) "
                "or sync the table (scripts/sync_pricing.py).",
            )
        )

    total = sum(s.cost_usd for _, s in by_model)
    if total > 0 and len(by_model) > 1:
        top_model, top = max(by_model, key=lambda kv: kv[1].cost_usd)
        share = top.cost_usd / total
        if share >= SPEND_CONCENTRATION:
            out.append(
                Finding(
                    "info",
                    "cost.concentration",
                    f"{share:.0%} of recorded spend is on {top_model}.",
                )
            )

    return _finalise(out)
