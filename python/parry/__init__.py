"""parry — zero-config rate limiting, caching, and cost control for LLM calls.

    import parry
    parry.install()   # every OpenAI / Anthropic call is now guarded

See https://github.com/IND-Anshuman/parry for the full documentation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from . import _patch, _state
from ._core import AsyncCachedRawResponse, CachedRawResponse, Parry, ParryStats
from ._errors import BudgetExceeded, ParryError, RateLimited
from ._parry_native import Cache, Cost, RateLimit, Retry, count_tokens, estimate_tokens

__version__ = "0.1.0"

__all__ = [
    "AsyncCachedRawResponse",
    "BudgetExceeded",
    "Cache",
    "CachedRawResponse",
    "Cost",
    "Parry",
    "ParryError",
    "ParryStats",
    "RateLimit",
    "RateLimited",
    "Retry",
    "bypass",
    "count_tokens",
    "estimate_tokens",
    "get_guard",
    "install",
    "install_async",
    "is_installed",
    "stats",
    "storage",
    "uninstall",
    "unwrap",
    "wrap",
]

logging.getLogger("parry").addHandler(logging.NullHandler())

_C = TypeVar("_C")


def install(
    guard: Parry | None = None,
    *,
    sync: bool = True,
    async_: bool = True,
    openai: bool = True,
    anthropic: bool = True,
    **options: Any,
) -> Parry:
    """Guard every OpenAI and Anthropic call in this process.

    Safe to call more than once: methods are never wrapped twice. With no
    arguments a second call returns the guard that is already installed.
    Keyword ``options`` are passed to :class:`Parry` when a new guard is made.
    """
    with _state.lock:
        if guard is None:
            if _state.global_guard is not None and not options:
                guard = _state.global_guard
            else:
                guard = Parry(**options)
        elif options:
            raise TypeError("pass either a Parry instance or Parry options, not both")
        _state.global_guard = guard
        providers = tuple(p for p, on in (("openai", openai), ("anthropic", anthropic)) if on)
        _patch.apply(sync=sync, async_=async_, providers=providers)
        return guard


def install_async(guard: Parry | None = None, **options: Any) -> Parry:
    """Like :func:`install`, but only hooks the async clients."""
    return install(guard, sync=False, async_=True, **options)


def uninstall() -> None:
    """Remove every hook and forget the global guard."""
    with _state.lock:
        _patch.restore()
        _state.global_guard = None


def is_installed() -> bool:
    return _patch.is_installed() and _state.global_guard is not None


def get_guard() -> Parry | None:
    """The guard that applies right now: ``with Parry()`` block first, then ``install()``."""
    return _state.context_guard.get() or _state.global_guard


def _ensure_global() -> Parry:
    with _state.lock:
        if _state.global_guard is None:
            _state.global_guard = Parry()
        return _state.global_guard


def stats() -> ParryStats:
    """Stats of the active guard (an empty snapshot if nothing is installed)."""
    guard = get_guard()
    return guard.stats if guard is not None else ParryStats()


def storage(path: str | Path) -> Parry:
    """Persist the global guard's cache and spend ledger to a SQLite file."""
    return _ensure_global().storage(path)


def wrap(client: _C, guard: Parry | None = None) -> _C:
    """Guard a single client instead of the whole process. Returns the client.

    Other clients are unaffected. Note that ``client.with_options(...)``
    returns a fresh client, which needs wrapping again.
    """
    guard = guard or Parry()
    _patch.bind(client, guard)
    _patch.apply()
    return client


def unwrap(client: _C) -> _C:
    _patch.unbind(client)
    return client


@contextmanager
def bypass() -> Iterator[None]:
    """Run the enclosed calls straight against the SDK, with no parry involvement."""
    token = _state.bypassed.set(True)
    try:
        yield
    finally:
        _state.bypassed.reset(token)
