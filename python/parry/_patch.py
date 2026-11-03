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
import inspect
import logging
import sys
import weakref
from typing import Any, NamedTuple

from . import _state
from ._adapters import (
    ANTHROPIC_BETA_MESSAGES,
    ANTHROPIC_MESSAGES,
    COHERE_CHAT,
    GOOGLE_GENAI,
    MISTRAL_CHAT,
    OLLAMA_CHAT,
    OPENAI_CHAT,
    OPENAI_RESPONSES,
    OPENAI_SHAPED,
    Adapter,
    openai_shaped_adapter,
)

log = logging.getLogger("parry")


class _Target(NamedTuple):
    """One hookable SDK method. ``method`` is usually ``create``; Google uses
    ``generate_content`` / ``generate_content_stream``. ``is_stream`` marks a
    method that returns a stream directly rather than via a ``stream=True``
    flag, so the guard knows to instrument the result."""

    provider: str
    module: str
    cls_name: str
    adapter: Adapter
    is_async: bool
    method: str = "create"
    is_stream: bool = False


def _build_targets() -> tuple[_Target, ...]:
    targets: list[_Target] = [
        _Target("openai", "openai.resources.chat.completions", "Completions", OPENAI_CHAT, False),
        _Target(
            "openai", "openai.resources.chat.completions", "AsyncCompletions", OPENAI_CHAT, True
        ),
        _Target("openai", "openai.resources.responses", "Responses", OPENAI_RESPONSES, False),
        _Target("openai", "openai.resources.responses", "AsyncResponses", OPENAI_RESPONSES, True),
        _Target("anthropic", "anthropic.resources.messages", "Messages", ANTHROPIC_MESSAGES, False),
        _Target(
            "anthropic", "anthropic.resources.messages", "AsyncMessages", ANTHROPIC_MESSAGES, True
        ),
        _Target(
            "anthropic",
            "anthropic.resources.beta.messages",
            "Messages",
            ANTHROPIC_BETA_MESSAGES,
            False,
        ),
        _Target(
            "anthropic",
            "anthropic.resources.beta.messages",
            "AsyncMessages",
            ANTHROPIC_BETA_MESSAGES,
            True,
        ),
    ]
    # OpenAI-shaped SDKs (groq, together, ...): the same create hook, bound to
    # each SDK's own resource classes by duck typing in _load().
    for provider, module, sync_cls, async_cls, completion_type in OPENAI_SHAPED:
        adapter = openai_shaped_adapter(provider, completion_type)
        targets.append(_Target(provider, module, sync_cls, adapter, False))
        targets.append(_Target(provider, module, async_cls, adapter, True))
    # Google Gen AI: a different method name, and a dedicated streaming method.
    for cls_name, is_async in (("Models", False), ("AsyncModels", True)):
        targets.append(
            _Target(
                "google",
                "google.genai.models",
                cls_name,
                GOOGLE_GENAI,
                is_async,
                method="generate_content",
            )
        )
        targets.append(
            _Target(
                "google",
                "google.genai.models",
                cls_name,
                GOOGLE_GENAI,
                is_async,
                method="generate_content_stream",
                is_stream=True,
            )
        )
    # Ollama: one ``chat`` method, streaming via a ``stream=True`` flag.
    targets.append(_Target("ollama", "ollama._client", "Client", OLLAMA_CHAT, False, "chat"))
    targets.append(_Target("ollama", "ollama._client", "AsyncClient", OLLAMA_CHAT, True, "chat"))
    # Mistral: complete()/stream() instead of create(), all on one Chat class.
    # The module path has moved across releases, so both known homes are tried;
    # whichever is not installed simply no-ops in _load().
    for module in ("mistralai.chat", "mistralai.client.chat"):
        for method, is_async, is_stream in (
            ("complete", False, False),
            ("complete_async", True, False),
            ("stream", False, True),
            ("stream_async", True, True),
        ):
            targets.append(
                _Target("mistral", module, "Chat", MISTRAL_CHAT, is_async, method, is_stream)
            )
    # Cohere: chat()/chat_stream() on the sync and async base clients.
    for cls_name, is_async in (("BaseCohere", False), ("AsyncBaseCohere", True)):
        targets.append(
            _Target("cohere", "cohere.base_client", cls_name, COHERE_CHAT, is_async, "chat")
        )
        targets.append(
            _Target(
                "cohere", "cohere.base_client", cls_name, COHERE_CHAT, is_async, "chat_stream", True
            )
        )
    return tuple(targets)


TARGETS: tuple[_Target, ...] = _build_targets()

# The providers parry guards by default: every one it knows a target for. The
# SDKs that are not installed simply no-op (their classes don't import).
ALL_PROVIDERS: tuple[str, ...] = tuple(dict.fromkeys(t.provider for t in TARGETS))

_applied: dict[tuple[type, str], Any] = {}  # (class, method name) -> original
_bound: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


def _make_sync(original: Any, adapter: Adapter, is_stream: bool) -> Any:
    @functools.wraps(original)
    def hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        guard = _state.active_guard(self)
        if guard is None:
            return original(self, *args, **kwargs)
        return guard._invoke(adapter, original, self, args, kwargs, is_stream)

    hook.__parry_original__ = original  # type: ignore[attr-defined]
    return hook


def _make_async(original: Any, adapter: Adapter, async_raw_parse: bool, is_stream: bool) -> Any:
    @functools.wraps(original)
    async def hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        guard = _state.active_guard(self)
        if guard is None:
            result = original(self, *args, **kwargs)
            return await result if inspect.isawaitable(result) else result
        return await guard._invoke_async(
            adapter, original, self, args, kwargs, async_raw_parse, is_stream
        )

    hook.__parry_original__ = original  # type: ignore[attr-defined]
    return hook


def _load(module: str, cls_name: str, method: str) -> type | None:
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return None
    cls = getattr(mod, cls_name, None)
    return cls if isinstance(cls, type) and hasattr(cls, method) else None


def apply(
    *, sync: bool = True, async_: bool = True, providers: tuple[str, ...] | None = None
) -> list[str]:
    """Hook every available SDK method. Returns the names that are hooked.

    ``providers=None`` means every provider parry knows (:data:`ALL_PROVIDERS`).
    """
    if providers is None:
        providers = ALL_PROVIDERS
    hooked: list[str] = []
    with _state.lock:
        for t in TARGETS:
            if (
                t.provider not in providers
                or (t.is_async and not async_)
                or (not t.is_async and not sync)
            ):
                continue
            cls = _load(t.module, t.cls_name, t.method)
            if cls is None:
                continue
            current = cls.__dict__.get(t.method)
            if current is None:
                continue
            if getattr(current, "__parry_original__", None) is None:
                if t.is_async:
                    # Stainless SDKs on the legacy raw-response module keep a
                    # synchronous .parse() even on async clients.
                    legacy = hasattr(sys.modules.get(cls.__module__), "_legacy_response")
                    patched = _make_async(
                        current, t.adapter, async_raw_parse=not legacy, is_stream=t.is_stream
                    )
                else:
                    patched = _make_sync(current, t.adapter, is_stream=t.is_stream)
                setattr(cls, t.method, patched)
                _applied[(cls, t.method)] = current
                log.debug("parry: hooked %s.%s.%s", cls.__module__, cls.__qualname__, t.method)
            hooked.append(f"{t.module}.{t.cls_name}.{t.method}")
    return hooked


def restore() -> None:
    """Remove every hook and put the original methods back."""
    with _state.lock:
        for (cls, method), original in list(_applied.items()):
            if getattr(cls.__dict__.get(method), "__parry_original__", None) is original:
                setattr(cls, method, original)
            del _applied[(cls, method)]


def is_installed() -> bool:
    return bool(_applied)


def hooked_methods() -> list[str]:
    return sorted(f"{c.__module__}.{c.__qualname__}.{m}" for (c, m) in _applied)


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
