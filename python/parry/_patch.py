"""Runtime hooks on the OpenAI and Anthropic SDKs.

Each hooked ``create`` method resolves the active guard on every call and
falls straight through to the original method when there is none, so the
hooks are inert until a guard is installed, entered or bound to a client.
Applying is idempotent: hooks are recognised by ``__parry_original__``.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import logging
import sys
import weakref
from typing import Any

from . import _state
from ._adapters import (
    ANTHROPIC_BETA_MESSAGES,
    ANTHROPIC_MESSAGES,
    OPENAI_CHAT,
    OPENAI_RESPONSES,
    Adapter,
)

log = logging.getLogger("parry")

# (provider, module, class, adapter, is_async)
TARGETS: tuple[tuple[str, str, str, Adapter, bool], ...] = (
    ("openai", "openai.resources.chat.completions", "Completions", OPENAI_CHAT, False),
    ("openai", "openai.resources.chat.completions", "AsyncCompletions", OPENAI_CHAT, True),
    ("openai", "openai.resources.responses", "Responses", OPENAI_RESPONSES, False),
    ("openai", "openai.resources.responses", "AsyncResponses", OPENAI_RESPONSES, True),
    ("anthropic", "anthropic.resources.messages", "Messages", ANTHROPIC_MESSAGES, False),
    ("anthropic", "anthropic.resources.messages", "AsyncMessages", ANTHROPIC_MESSAGES, True),
    ("anthropic", "anthropic.resources.beta.messages", "Messages", ANTHROPIC_BETA_MESSAGES, False),
    (
        "anthropic",
        "anthropic.resources.beta.messages",
        "AsyncMessages",
        ANTHROPIC_BETA_MESSAGES,
        True,
    ),
)

_applied: dict[type, Any] = {}  # class -> original create
_bound: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


def _make_sync(original: Any, adapter: Adapter) -> Any:
    @functools.wraps(original)
    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        guard = _state.active_guard(self)
        if guard is None:
            return original(self, *args, **kwargs)
        return guard._invoke(adapter, original, self, args, kwargs)

    create.__parry_original__ = original  # type: ignore[attr-defined]
    return create


def _make_async(original: Any, adapter: Adapter, async_raw_parse: bool) -> Any:
    @functools.wraps(original)
    async def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        guard = _state.active_guard(self)
        if guard is None:
            return await original(self, *args, **kwargs)
        return await guard._invoke_async(adapter, original, self, args, kwargs, async_raw_parse)

    create.__parry_original__ = original  # type: ignore[attr-defined]
    return create


def _load(module: str, cls_name: str) -> type | None:
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return None
    cls = getattr(mod, cls_name, None)
    return cls if isinstance(cls, type) and hasattr(cls, "create") else None


def apply(
    *, sync: bool = True, async_: bool = True, providers: tuple[str, ...] = ("openai", "anthropic")
) -> list[str]:
    """Hook every available SDK method. Returns the names that are hooked."""
    hooked: list[str] = []
    with _state.lock:
        for provider, module, cls_name, adapter, is_async in TARGETS:
            if (
                provider not in providers
                or (is_async and not async_)
                or (not is_async and not sync)
            ):
                continue
            cls = _load(module, cls_name)
            if cls is None:
                continue
            current = cls.__dict__.get("create")
            if current is None:
                continue
            if getattr(current, "__parry_original__", None) is None:
                if is_async:
                    # Stainless SDKs on the legacy raw-response module keep a
                    # synchronous .parse() even on async clients.
                    legacy = hasattr(sys.modules.get(cls.__module__), "_legacy_response")
                    patched = _make_async(current, adapter, async_raw_parse=not legacy)
                else:
                    patched = _make_sync(current, adapter)
                cls.create = patched  # type: ignore[attr-defined]
                _applied[cls] = current
                log.debug("parry: hooked %s.%s.create", cls.__module__, cls.__qualname__)
            hooked.append(f"{module}.{cls_name}.create")
    return hooked


def restore() -> None:
    """Remove every hook and put the original methods back."""
    with _state.lock:
        for cls, original in list(_applied.items()):
            if getattr(cls.__dict__.get("create"), "__parry_original__", None) is original:
                cls.create = original  # type: ignore[attr-defined]
            del _applied[cls]


def is_installed() -> bool:
    return bool(_applied)


def hooked_methods() -> list[str]:
    return sorted(f"{c.__module__}.{c.__qualname__}.create" for c in _applied)


def bind(client: Any, guard: Any) -> None:
    setattr(client, _state.CLIENT_ATTR, guard)
    # Not every object is weak-referenceable; the attribute alone still works.
    with contextlib.suppress(TypeError):
        _bound[client] = guard


def unbind(client: Any) -> None:
    if hasattr(client, _state.CLIENT_ATTR):
        delattr(client, _state.CLIENT_ATTR)
    _bound.pop(client, None)


def has_bound_clients(guard: Any) -> bool:
    return any(g is guard for g in list(_bound.values()))
