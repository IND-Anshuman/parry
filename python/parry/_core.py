"""The guard: one object that owns cache, rate limits, spend and retry policy."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from . import _state
from ._adapters import (
    Adapter,
    StreamAccumulator,
    Uncacheable,
    _get,
    decode_response,
    encode_response,
    raw_mode,
    request_json,
)
from ._errors import BudgetExceeded, ParryError, RateLimited
from ._parry_native import Cache, Cost, RateLimit, Retry, count_tokens, estimate_tokens
from ._storage import SQLiteStore, start_of_today
from ._stream import instrument_async_stream, instrument_stream

log = logging.getLogger("parry")

_COUNTERS = (
    "requests",
    "cache_hits",
    "cache_misses",
    "tokens_cached",
    "requests_blocked",
    "rate_limit_rejections",
    "budget_rejections",
    "retries",
    "errors",
)


def usd(amount: float) -> str:
    """Dollars, with enough precision that small spends stay visible."""
    if amount == 0 or abs(amount) >= 0.01:
        return f"${amount:.4f}"
    return f"${amount:.6f}"


@dataclass(frozen=True)
class ParryStats:
    """A snapshot of what a guard has done. Cheap to take; safe to keep."""

    requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    tokens_cached: int = 0
    cost_saved_usd: float = 0.0
    total_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    requests_blocked: int = 0
    rate_limit_rejections: int = 0
    budget_rejections: int = 0
    retries: int = 0
    errors: int = 0
    unpriced_requests: int = 0
    estimated_requests: int = 0
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)
    today_cost_usd: float | None = None
    lifetime_cost_usd: float | None = None

    @property
    def cache_hit_rate(self) -> float:
        lookups = self.cache_hits + self.cache_misses
        return self.cache_hits / lookups if lookups else 0.0

    def __repr__(self) -> str:
        return (
            f"ParryStats(cache_hits={self.cache_hits}, cache_misses={self.cache_misses}, "
            f"tokens_cached={self.tokens_cached}, total_cost_usd={self.total_cost_usd:.6f}, "
            f"cost_saved_usd={self.cost_saved_usd:.6f}, requests={self.requests}, "
            f"requests_blocked={self.requests_blocked}, retries={self.retries}, "
            f"budget_rejections={self.budget_rejections})"
        )

    def summary(self) -> str:
        """A short human-readable report."""
        lines = [
            f"requests        {self.requests}",
            f"cache           {self.cache_hits} hits / {self.cache_misses} misses"
            f" ({self.cache_hit_rate:.0%}), {self.tokens_cached} tokens and"
            f" {usd(self.cost_saved_usd)} saved",
            f"spend           {usd(self.total_cost_usd)} this session"
            f" ({self.input_tokens} in / {self.output_tokens} out tokens)",
        ]
        if self.today_cost_usd is not None:
            lines.append(
                f"                {usd(self.today_cost_usd)} today, {usd(self.lifetime_cost_usd or 0)} lifetime"
            )
        lines.append(
            f"rate limiting   {self.requests_blocked} delayed, {self.rate_limit_rejections} rejected"
        )
        lines.append(f"retries         {self.retries}   budget stops {self.budget_rejections}")
        if self.unpriced_requests:
            lines.append(
                f"unpriced        {self.unpriced_requests} requests (model not in pricing table)"
            )
        for model, t in self.by_model.items():
            lines.append(f"  {model:<40} {t['requests']:>6} req  {usd(t['cost_usd'])}")
        return "\n".join(lines)


class CachedRawResponse:
    """Stands in for ``LegacyAPIResponse`` / ``APIResponse`` on a cache hit made
    through ``.with_raw_response``. Exposes the parts callers actually use."""

    status_code = 200
    http_response = None
    retries_taken = 0

    def __init__(self, parsed: Any) -> None:
        self._parsed = parsed
        self.headers: dict[str, str] = {"x-parry-cache": "hit"}

    def parse(self, *, to: Any = None) -> Any:
        return self._parsed

    def json(self) -> Any:
        return self._parsed.model_dump(mode="json")

    @property
    def text(self) -> str:
        return self._parsed.model_dump_json()

    @property
    def content(self) -> bytes:
        return self.text.encode()

    def read(self) -> bytes:
        return self.content

    def __repr__(self) -> str:
        return f"<CachedRawResponse {type(self._parsed).__name__}>"


class AsyncCachedRawResponse(CachedRawResponse):
    """Async flavour: ``parse()`` / ``json()`` / ``text()`` / ``read()`` are awaitable."""

    async def parse(self, *, to: Any = None) -> Any:  # type: ignore[override]
        return self._parsed

    async def json(self) -> Any:  # type: ignore[override]
        return self._parsed.model_dump(mode="json")

    async def text(self) -> str:  # type: ignore[override]
        return self._parsed.model_dump_json()

    async def read(self) -> bytes:  # type: ignore[override]
        return self._parsed.model_dump_json().encode()


_MISS = object()


class _Call:
    """Per-request state while a call moves through the guard."""

    __slots__ = (
        "guard",
        "adapter",
        "resource",
        "kwargs",
        "is_async",
        "async_raw_parse",
        "provider",
        "model",
        "mode",
        "streaming",
        "key",
        "_json",
        "_json_done",
        "_prompt_tokens",
    )

    def __init__(
        self,
        guard: Parry,
        adapter: Adapter,
        resource: Any,
        kwargs: dict[str, Any],
        is_async: bool,
        async_raw_parse: bool,
    ):
        self.guard = guard
        self.adapter = adapter
        self.resource = resource
        self.kwargs = kwargs
        self.is_async = is_async
        self.async_raw_parse = async_raw_parse
        self.provider = adapter.provider
        self.model = adapter.model(kwargs)
        self.mode = raw_mode(kwargs)
        self.streaming = kwargs.get("stream") is True
        self._json: str | None = None
        self._json_done = False
        self._prompt_tokens: int | None = None
        self.key: str | None = None
        if guard.cache and self.mode != "stream" and adapter.cacheable_request(kwargs):
            payload = self.payload_json()
            if payload is not None:
                client = getattr(resource, "_client", None)
                base_url = str(getattr(client, "base_url", ""))
                namespace = f"{self.provider}:{adapter.endpoint}:{base_url}"
                self.key = Cache.make_key(namespace, payload)

    def payload_json(self) -> str | None:
        if not self._json_done:
            self._json_done = True
            try:
                self._json = request_json(self.kwargs)
            except Uncacheable as exc:
                log.debug("parry: request not hashable (%s); skipping cache", exc)
        return self._json

    def prompt_tokens(self) -> int:
        if self._prompt_tokens is None:
            payload = self.payload_json()
            try:
                self._prompt_tokens = estimate_tokens(payload, self.model) if payload else 0
            except ValueError:
                self._prompt_tokens = 0
        return self._prompt_tokens

    # --- cache -------------------------------------------------------------

    def lookup(self) -> Any:
        if self.key is None or self.streaming:
            return _MISS
        g = self.guard
        found = g._cache.lookup(self.key)
        if found is None and g._store is not None:
            found = g._store.get(self.key)
            if found is not None:
                g._cache.set(self.key, *found)
        if found is None:
            g._count("cache_misses")
            return _MISS
        blob, tokens, cost = found
        try:
            obj = decode_response(blob)
        except Exception as exc:
            log.debug("parry: dropping unreadable cache entry (%s)", exc)
            g._cache.invalidate(self.key)
            if g._store is not None:
                g._store.delete(self.key)
            g._count("cache_misses")
            return _MISS
        g._on_hit(tokens, cost)
        log.debug("parry: cache hit %s:%s", self.provider, self.model)
        if self.mode == "raw":
            return AsyncCachedRawResponse(obj) if self.async_raw_parse else CachedRawResponse(obj)
        return obj

    # --- budget ------------------------------------------------------------

    def check_budget(self) -> None:
        g = self.guard
        if g.max_cost_usd is None:
            return
        tokens = self.prompt_tokens() if g._cost.total_cost_usd < g.max_cost_usd else 0
        g.check_budget(self.model, tokens)

    def tpm_tokens(self) -> int:
        if not self.guard._tpm_configured:
            return 0
        max_out = self.kwargs.get("max_tokens") or self.kwargs.get("max_completion_tokens") or 0
        if not isinstance(max_out, int):
            max_out = 0
        return min(self.prompt_tokens() + max_out, 2**32 - 1)

    # --- completion --------------------------------------------------------

    def complete(self, result: Any) -> Any:
        if self.mode == "stream":
            return result
        if self.mode == "raw":
            if not self.streaming:
                self.account(result.parse())
            return result
        if self.streaming:
            return instrument_stream(result, self.adapter.accumulator(), self.on_stream_done)
        self.account(result)
        return result

    async def complete_async(self, result: Any) -> Any:
        if self.mode == "stream":
            return result
        if self.mode == "raw":
            if not self.streaming:
                parsed = result.parse()
                if inspect.isawaitable(parsed):
                    parsed = await parsed
                self.account(parsed)
            return result
        if self.streaming:
            return instrument_async_stream(result, self.adapter.accumulator(), self.on_stream_done)
        self.account(result)
        return result

    def account(self, response: Any) -> None:
        try:
            usage = self.adapter.usage(response)
            estimated = usage is None
            if usage is None:
                usage = (self.prompt_tokens(), 0)
            model = _get(response, "model") or self.model
            cost = self.guard._record(self.provider, str(model), usage[0], usage[1], estimated)
            if self.key is not None and self.adapter.cacheable_response(response):
                self.guard._store_entry(self.key, encode_response(response), sum(usage), cost)
        except Exception as exc:  # bookkeeping must never break the user's call
            log.warning("parry: bookkeeping failed for %s:%s (%s)", self.provider, self.model, exc)

    def on_stream_done(self, acc: StreamAccumulator) -> None:
        try:
            usage = acc.usage
            estimated = usage is None
            if usage is None:
                usage = (self.prompt_tokens(), count_tokens(acc.output_text(), self.model))
            self.guard._record(self.provider, self.model, usage[0], usage[1], estimated)
            if self.key is None:
                return
            final = acc.result()
            if final is not None:
                cost = self.guard._cost.compute(self.model, usage[0], usage[1]) or 0.0
                self.guard._store_entry(self.key, encode_response(final), sum(usage), cost)
        except Exception as exc:
            log.warning(
                "parry: stream bookkeeping failed for %s:%s (%s)", self.provider, self.model, exc
            )


class Parry:
    """A guard rail for LLM calls: cache, rate limit, cost tracking and retry.

    Use it globally with ``parry.install()``, for a block of code with
    ``with Parry(...):``, or for one client with ``guard.wrap(client)``.
    """

    def __init__(
        self,
        *,
        cache: bool = True,
        rate_limit: bool = True,
        cost_tracking: bool = True,
        retry: bool = True,
        max_cost_usd: float | None = None,
        cache_ttl: float | None = 3600,
        cache_size: int = 10_000,
        on_rate_limit: Literal["wait", "raise"] = "wait",
        max_wait: float = 60.0,
        max_attempts: int = 3,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 30.0,
        storage: str | Path | None = None,
    ) -> None:
        if on_rate_limit not in ("wait", "raise"):
            raise ValueError("on_rate_limit must be 'wait' or 'raise'")
        if max_cost_usd is not None and max_cost_usd < 0:
            raise ValueError("max_cost_usd must be >= 0")
        if cache_size is None or cache_size < 1:
            raise ValueError(
                "cache_size must be a positive integer; use cache=False to disable caching"
            )
        self.cache = cache
        self.rate_limit = rate_limit
        self.cost_tracking = cost_tracking
        self.retry = retry
        self.max_cost_usd = max_cost_usd
        self.on_rate_limit = on_rate_limit
        self.max_wait = max_wait
        self.cache_ttl = cache_ttl
        self._cache = Cache(max_entries=cache_size, ttl_seconds=cache_ttl)
        self._limiter = RateLimit()
        self._cost = Cost()
        self._retry = Retry(
            max_attempts=max_attempts, base_delay=retry_base_delay, max_delay=retry_max_delay
        )
        self._store: SQLiteStore | None = SQLiteStore(storage) if storage else None
        self._tpm_configured = False
        self._lock = threading.Lock()
        self._counts = dict.fromkeys(_COUNTERS, 0)
        self._cost_saved = 0.0
        self._warned_unpriced: set[str] = set()
        self._context_tokens: list[Any] = []

    # --- configuration -----------------------------------------------------

    def set_rate_limit(
        self,
        model: str,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        *,
        burst: int | None = None,
        provider: str | None = None,
    ) -> Parry:
        """Limit a model. ``model`` may also be ``"provider:model"``, ``"provider:*"`` or ``"*"``.

        Passing neither limit makes the model unlimited (overriding defaults).
        """
        key = f"{provider}:{model}" if provider else model
        self._limiter.set_limit(key, requests_per_minute, tokens_per_minute, burst)
        if tokens_per_minute:
            self._tpm_configured = True
        return self

    def remove_rate_limit(self, model: str, *, provider: str | None = None) -> bool:
        return self._limiter.remove_limit(f"{provider}:{model}" if provider else model)

    def rate_limit_for(self, provider: str, model: str) -> dict[str, int | None]:
        return dict(self._limiter.limit_for(provider, model))

    def set_pricing(self, model: str, input_per_million: float, output_per_million: float) -> Parry:
        """Price a model the built-in table does not know (or override it)."""
        self._cost.set_pricing(model, input_per_million, output_per_million)
        self._warned_unpriced.discard(model)
        return self

    def set_cache(self, backend: str = "memory", *, path: str | Path | None = None) -> Parry:
        """``"memory"`` (default) or ``"sqlite"`` with ``path=``. Redis is planned for a later release."""
        if backend == "memory":
            if self._store is not None:
                self._store.close()
            self._store = None
        elif backend == "sqlite":
            if path is None:
                raise ValueError("set_cache('sqlite') needs path=")
            self.storage(path)
        else:
            raise ValueError(
                f"unknown cache backend {backend!r}; v0.1 supports 'memory' and 'sqlite'"
            )
        return self

    def storage(self, path: str | Path) -> Parry:
        """Persist the cache and the spend ledger to a SQLite file."""
        if self._store is not None:
            self._store.close()
        self._store = SQLiteStore(path)
        return self

    def clear_cache(self) -> None:
        self._cache.clear()
        if self._store is not None:
            self._store.clear_cache()

    def reset_stats(self) -> None:
        """Zero the session counters and session spend (the budget restarts too)."""
        with self._lock:
            self._counts = dict.fromkeys(_COUNTERS, 0)
            self._cost_saved = 0.0
        self._cost.reset()

    # --- building blocks for integrations and custom clients ----------------

    def acquire(self, provider: str, model: str, tokens: int = 0) -> None:
        """Block until the rate limiter admits one request (or raise ``RateLimited``)."""
        waited: float | None = None
        while True:
            wait, waited = self._rate_wait_for(provider, model, tokens, waited)
            if wait <= 0:
                return
            time.sleep(wait)

    async def acquire_async(self, provider: str, model: str, tokens: int = 0) -> None:
        """Async version of :meth:`acquire`."""
        waited: float | None = None
        while True:
            wait, waited = self._rate_wait_for(provider, model, tokens, waited)
            if wait <= 0:
                return
            await asyncio.sleep(wait)

    def check_budget(self, model: str = "", prompt_tokens: int = 0) -> None:
        """Raise ``BudgetExceeded`` if a call to ``model`` would cross ``max_cost_usd``."""
        limit = self.max_cost_usd
        if limit is None:
            return
        spent = self._cost.total_cost_usd
        estimate = (
            (self._cost.compute(model, prompt_tokens, 0) or 0.0) if model and spent < limit else 0.0
        )
        if spent >= limit or spent + estimate > limit:
            self._count("budget_rejections")
            raise BudgetExceeded(spent, limit, estimate, model)

    def record_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        estimated: bool = False,
    ) -> float:
        """Record a call made outside parry's hooks. Returns its cost in USD."""
        self._count("requests")
        return self._record(provider, model, int(input_tokens), int(output_tokens), estimated)

    # --- introspection -----------------------------------------------------

    @property
    def total_cost_usd(self) -> float:
        return self._cost.total_cost_usd

    @property
    def stats(self) -> ParryStats:
        with self._lock:
            counts = dict(self._counts)
            saved = self._cost_saved
        totals = self._cost.totals()
        today = lifetime = None
        if self._store is not None and self._store.ok:
            t = self._store.spend(start_of_today())
            life = self._store.spend()
            today = t.cost_usd if t else None
            lifetime = life.cost_usd if life else None
        return ParryStats(
            requests=counts["requests"],
            cache_hits=counts["cache_hits"],
            cache_misses=counts["cache_misses"],
            tokens_cached=counts["tokens_cached"],
            cost_saved_usd=saved,
            total_cost_usd=totals["cost_usd"],
            input_tokens=totals["input_tokens"],
            output_tokens=totals["output_tokens"],
            requests_blocked=counts["requests_blocked"],
            rate_limit_rejections=counts["rate_limit_rejections"],
            budget_rejections=counts["budget_rejections"],
            retries=counts["retries"],
            errors=counts["errors"],
            unpriced_requests=totals["unpriced_requests"],
            estimated_requests=totals["estimated_requests"],
            by_model={k: dict(v) for k, v in self._cost.by_model().items()},
            today_cost_usd=today,
            lifetime_cost_usd=lifetime,
        )

    @property
    def is_intercepting(self) -> bool:
        """True when this guard is receiving SDK calls through parry's hooks."""
        from . import _patch

        return _patch.is_installed() and (
            _state.global_guard is self
            or _state.context_guard.get() is self
            or _patch.has_bound_clients(self)
        )

    # --- activation --------------------------------------------------------

    def install(self, **kwargs: Any) -> Parry:
        """Make this the process-wide guard (same as ``parry.install(guard)``)."""
        from . import install

        return install(self, **kwargs)

    def wrap(self, client: Any) -> Any:
        """Guard one SDK client only (same as ``parry.wrap(client, guard)``)."""
        from . import wrap

        return wrap(client, self)

    def __enter__(self) -> Parry:
        from . import _patch

        _patch.apply()
        self._context_tokens.append(_state.context_guard.set(self))
        return self

    def __exit__(self, *exc: object) -> None:
        _state.context_guard.reset(self._context_tokens.pop())

    async def __aenter__(self) -> Parry:
        return self.__enter__()

    async def __aexit__(self, *exc: object) -> None:
        self.__exit__(*exc)

    def __repr__(self) -> str:
        return (
            f"Parry(cache={self.cache}, rate_limit={self.rate_limit}, cost_tracking={self.cost_tracking}, "
            f"retry={self.retry}, max_cost_usd={self.max_cost_usd})"
        )

    # --- internals used by _Call / patches ----------------------------------

    def _count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counts[name] += n

    def _on_hit(self, tokens: int, cost: float) -> None:
        with self._lock:
            self._counts["cache_hits"] += 1
            self._counts["tokens_cached"] += tokens
            self._cost_saved += cost

    def _record(
        self, provider: str, model: str, input_tokens: int, output_tokens: int, estimated: bool
    ) -> float:
        if not self.cost_tracking:
            return self._cost.compute(model, input_tokens, output_tokens) or 0.0
        cost = self._cost.record(provider, model, input_tokens, output_tokens, estimated)
        if cost is None and model not in self._warned_unpriced:
            self._warned_unpriced.add(model)
            log.warning(
                "parry: no pricing for model %r; its cost is recorded as $0 and does not count "
                "toward max_cost_usd. Use guard.set_pricing(%r, input_per_million=..., "
                "output_per_million=...) to fix.",
                model,
                model,
            )
        if self._store is not None:
            self._store.record(provider, model, input_tokens, output_tokens, cost, estimated)
        log.debug(
            "parry: %s:%s %d in / %d out tokens, $%.6f%s",
            provider,
            model,
            input_tokens,
            output_tokens,
            cost or 0.0,
            " (estimated)" if estimated else "",
        )
        return cost or 0.0

    def _store_entry(self, key: str, blob: str, tokens: int, cost: float) -> None:
        self._cache.set(key, blob, tokens, cost)
        if self._store is not None:
            self._store.put(key, blob, tokens, cost, self.cache_ttl)

    def _rate_wait(self, call: _Call, waited_since: float | None) -> tuple[float, float | None]:
        return self._rate_wait_for(call.provider, call.model, call.tpm_tokens(), waited_since)

    def _rate_wait_for(
        self, provider: str, model: str, tokens: int, waited_since: float | None
    ) -> tuple[float, float | None]:
        """Returns ``(seconds_to_sleep, waited_since)``; 0 means go."""
        if not self.rate_limit:
            return 0.0, waited_since
        wait = self._limiter.try_acquire(provider, model, tokens)
        if wait <= 0:
            return 0.0, waited_since
        now = time.monotonic()
        if waited_since is None:
            self._count("requests_blocked")
            waited_since = now
        if self.on_rate_limit == "raise" or (now - waited_since) + wait > self.max_wait:
            self._count("rate_limit_rejections")
            raise RateLimited(provider, model, wait)
        return wait, waited_since

    def _retry_delay(self, exc: BaseException, attempt: int) -> float | None:
        if not self.retry or isinstance(exc, ParryError) or not isinstance(exc, Exception):
            return None
        retryable, retry_after = self._retry.classify(exc)
        if not retryable or attempt >= self._retry.max_attempts:
            return None
        self._retry.note_retry()
        self._count("retries")
        delay = self._retry.delay(attempt, retry_after)
        log.debug(
            "parry: retrying after %s in %.2fs (attempt %d)", type(exc).__name__, delay, attempt + 1
        )
        return delay

    def _invoke(
        self,
        adapter: Adapter,
        original: Callable[..., Any],
        resource: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if args:  # SDK create() methods are keyword-only; leave odd calls alone
            return original(resource, *args, **kwargs)
        self._count("requests")
        _state.note_intercept()
        call = _Call(self, adapter, resource, kwargs, is_async=False, async_raw_parse=False)
        hit = call.lookup()
        if hit is not _MISS:
            return hit
        call.check_budget()
        attempt = 1
        while True:
            waited: float | None = None
            while True:
                wait, waited = self._rate_wait(call, waited)
                if wait <= 0:
                    break
                time.sleep(wait)
            try:
                result = original(resource, **kwargs)
                break
            except Exception as exc:
                delay = self._retry_delay(exc, attempt)
                if delay is None:
                    self._count("errors")
                    raise
                time.sleep(delay)
                attempt += 1
        return call.complete(result)

    async def _invoke_async(
        self,
        adapter: Adapter,
        original: Callable[..., Any],
        resource: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        async_raw_parse: bool,
    ) -> Any:
        if args:
            return await original(resource, *args, **kwargs)
        self._count("requests")
        _state.note_intercept()
        call = _Call(
            self, adapter, resource, kwargs, is_async=True, async_raw_parse=async_raw_parse
        )
        hit = call.lookup()
        if hit is not _MISS:
            return hit
        call.check_budget()
        attempt = 1
        while True:
            waited: float | None = None
            while True:
                wait, waited = self._rate_wait(call, waited)
                if wait <= 0:
                    break
                await asyncio.sleep(wait)
            try:
                result = await original(resource, **kwargs)
                break
            except Exception as exc:
                delay = self._retry_delay(exc, attempt)
                if delay is None:
                    self._count("errors")
                    raise
                await asyncio.sleep(delay)
                attempt += 1
        return await call.complete_async(result)
