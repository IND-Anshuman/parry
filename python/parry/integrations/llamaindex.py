"""LlamaIndex callback handler.

    from llama_index.core import Settings
    from llama_index.core.callbacks import CallbackManager
    from parry.integrations.llamaindex import ParryHandler

    Settings.callback_manager = CallbackManager([ParryHandler(guard)])

Like the LangChain handler, it enforces budget and rate limits before each
LLM event and records usage after it; caching comes from ``parry.install()``
(LlamaIndex's OpenAI and Anthropic LLMs call the official SDKs). Calls the
SDK hooks already handled are not counted twice.
"""

from __future__ import annotations

import threading
from typing import Any

from .. import _state
from .._core import Parry
from .._parry_native import count_tokens

try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    from llama_index.core.callbacks.schema import CBEventType, EventPayload
except ImportError as exc:  # pragma: no cover - exercised only without llama-index
    raise ImportError(
        "parry.integrations.llamaindex needs llama-index-core: pip install llama-index-core"
    ) from exc

__all__ = ["ParryHandler"]


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _usage(response: Any) -> tuple[int, int] | None:
    raw = _get(response, "raw")
    usage = _get(raw, "usage")
    if usage is not None:
        inp = _get(usage, "prompt_tokens")
        if inp is None:
            inp = _get(usage, "input_tokens")
        out = _get(usage, "completion_tokens")
        if out is None:
            out = _get(usage, "output_tokens")
        if inp is not None or out is not None:
            return int(inp or 0), int(out or 0)
    extra = _get(response, "additional_kwargs") or {}
    if "prompt_tokens" in extra or "completion_tokens" in extra:
        return int(extra.get("prompt_tokens", 0)), int(extra.get("completion_tokens", 0))
    return None


def _text(response: Any) -> str:
    message = _get(response, "message")
    content = _get(message, "content") if message is not None else _get(response, "text")
    return content if isinstance(content, str) else ""


class ParryHandler(BaseCallbackHandler):
    """Budget, rate limiting and cost tracking for LlamaIndex LLM calls."""

    def __init__(self, guard: Parry | None = None) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.guard = guard or _state.global_guard or Parry()
        self._events: dict[str, tuple[str, str, int, int]] = {}
        self._lock = threading.Lock()

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if event_type != CBEventType.LLM:
            return event_id
        payload = payload or {}
        serialized = payload.get(EventPayload.SERIALIZED) or {}
        model = str(serialized.get("model") or serialized.get("model_name") or "unknown")
        class_name = str(serialized.get("class_name") or "").lower()
        provider = "anthropic" if "anthropic" in class_name else "openai"
        messages = payload.get(EventPayload.MESSAGES) or []
        prompt = payload.get(EventPayload.PROMPT) or "\n".join(
            str(getattr(m, "content", "") or "") for m in messages
        )
        prompt_tokens = count_tokens(str(prompt), model) if prompt else 0
        self.guard.check_budget(model, prompt_tokens)
        if not self.guard.is_intercepting:
            self.guard.acquire(provider, model)
        with self._lock:
            self._events[event_id] = (provider, model, prompt_tokens, _state.intercepted.get())
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_type != CBEventType.LLM:
            return
        with self._lock:
            event = self._events.pop(event_id, None)
        if event is None:
            return
        provider, model, prompt_tokens, intercepts_before = event
        if _state.intercepted.get() > intercepts_before:
            return  # the SDK hook already recorded this call
        payload = payload or {}
        response = payload.get(EventPayload.RESPONSE) or payload.get(EventPayload.COMPLETION)
        usage = _usage(response)
        estimated = usage is None
        if usage is None:
            usage = (prompt_tokens, count_tokens(_text(response), model))
        self.guard.record_usage(provider, model, usage[0], usage[1], estimated=estimated)

    def start_trace(self, trace_id: str | None = None) -> None:
        pass

    def end_trace(
        self, trace_id: str | None = None, trace_map: dict[str, list[str]] | None = None
    ) -> None:
        pass
