"""Exceptions raised by parry."""

from __future__ import annotations


class ParryError(Exception):
    """Base class for every error parry raises on its own behalf."""


class BudgetExceeded(ParryError):
    """Raised *before* a request is sent when it would push spend past ``max_cost_usd``."""

    def __init__(
        self, spent_usd: float, limit_usd: float, estimate_usd: float = 0.0, model: str = ""
    ):
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
        self.estimate_usd = estimate_usd
        self.model = model
        detail = f" (next call to {model} estimated at ${estimate_usd:.6f})" if model else ""
        super().__init__(
            f"parry budget exceeded: spent ${spent_usd:.6f} of ${limit_usd:.6f}{detail}"
        )


class UnpricedModel(BudgetExceeded):
    """Raised *before* a request when ``max_cost_usd`` is set but the model has
    no known price, so parry cannot tell whether the call fits the budget.

    parry fails closed rather than let an unbudgetable call through: an unpriced
    model would otherwise be charged $0 and silently bypass the limit. Register
    a price with ``guard.set_pricing(model, ...)``, or pass ``on_unpriced="ignore"``
    to allow these calls (they are still tracked as unpriced and charged $0).

    It subclasses :class:`BudgetExceeded` so existing ``except BudgetExceeded``
    handlers catch it too.
    """

    def __init__(self, model: str, spent_usd: float, limit_usd: float):
        self.model = model
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
        self.estimate_usd = 0.0
        # Skip BudgetExceeded's "spent/limit" phrasing; this is a different cause.
        ParryError.__init__(
            self,
            f"parry: no pricing for model {model!r}, so its cost cannot be counted "
            f"toward the ${limit_usd:.6f} budget (${spent_usd:.6f} spent so far). "
            f"Register it with set_pricing(), or pass on_unpriced='ignore' to allow it.",
        )


class RateLimited(ParryError):
    """Raised when parry's own rate limiter refuses a request.

    With ``on_rate_limit="wait"`` (the default) this only happens when no slot
    frees up within ``max_wait`` seconds; with ``on_rate_limit="raise"`` it
    happens as soon as the bucket is empty.
    """

    def __init__(self, provider: str, model: str, retry_after: float):
        self.provider = provider
        self.model = model
        self.retry_after = retry_after
        super().__init__(
            f"parry rate limit reached for {provider}:{model}; next slot in {retry_after:.3f}s"
        )
