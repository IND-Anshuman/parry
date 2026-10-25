"""LangChain callback handler.

``parry.install()`` already covers ``ChatOpenAI`` / ``ChatAnthropic``
(caching included), because they call the official SDKs. Use this handler
when you do not want global hooks, or for chat models from other providers:

    from parry import Parry
    from parry.integrations.langchain import ParryCallback

    guard = Parry(max_cost_usd=5)
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[ParryCallback(guard)])

The handler enforces the budget and rate limits before each call and records
token usage after it. Callbacks cannot return a response, so caching only
happens through ``parry.install()`` / ``parry.wrap()``. When parry's SDK hooks
already handled a call, the handler does not count it a second time.
"""

from __future__ import annotations

import json
import threading
from typing import Any
from uuid import UUID

from .. import _state
from .._core import Parry
from .._parry_native import count_tokens

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError as exc:  # pragma: no cover - exercised only without langchain
    raise ImportError(
        "parry.integrations.langchain needs langchain-core: pip install langchain-core"
    ) from exc

__all__ = ["ParryCallback"]


def _provider(serialized: dict[str, Any] | None, metadata: dict[str, Any] | None) -> str:
    provider = (metadata or {}).get("ls_provider")
    if provider:
        return str(provider).lower()
    ids = " ".join(str(x) for x in (serialized or {}).get("id") or []).lower()
    return "anthropic" if "anthropic" in ids else "openai"


def _model(
    serialized: dict[str, Any] | None, metadata: dict[str, Any] | None, kwargs: dict[str, Any]
) -> str:
    params = kwargs.get("invocation_params") or {}
    for value in (
        (metadata or {}).get("ls_model_name"),
        params.get("model"),
        params.get("model_name"),
        ((serialized or {}).get("kwargs") or {}).get("model"),
        ((serialized or {}).get("kwargs") or {}).get("model_name"),
    ):
        if value:
            return str(value)
    return "unknown"


def _usage(response: LLMResult) -> tuple[int, int] | None:
    total_in = total_out = 0
    found = False
    for generations in response.generations:
        for gen in generations:
            meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
            if meta:
                total_in += int(meta.get("input_tokens", 0) or 0)
                total_out += int(meta.get("output_tokens", 0) or 0)
                found = True
    if found:
        return total_in, total_out
    usage = (response.llm_output or {}).get("token_usage") or (response.llm_output or {}).get(
        "usage"
    )
    if usage:
        return (
            int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
            int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0),
        )
    return None


class ParryCallback(BaseCallbackHandler):
    """Budget, rate limiting and cost tracking for any LangChain chat model / LLM."""

    raise_error = True  # let BudgetExceeded / RateLimited stop the run
    run_inline = True

    def __init__(self, guard: Parry | None = None) -> None:
        super().__init__()
        self.guard = guard or _state.global_guard or Parry()
        self._runs: dict[UUID, tuple[str, str, int, int]] = {}
        self._lock = threading.Lock()

    def _start(
        self,
        serialized: dict[str, Any] | None,
        prompt_text: str,
        run_id: UUID,
        metadata: dict[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> None:
        provider = _provider(serialized, metadata)
        model = _model(serialized, metadata, kwargs)
        prompt_tokens = count_tokens(prompt_text, model) if prompt_text else 0
        self.guard.check_budget(model, prompt_tokens)
        if not self.guard.is_intercepting:
            self.guard.acquire(provider, model)
        with self._lock:
            self._runs[run_id] = (provider, model, prompt_tokens, _state.intercepted.get())

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        text = "\n".join(
            m.content if isinstance(m.content, str) else json.dumps(m.content, default=str)
            for batch in messages
            for m in batch
        )
        self._start(serialized, text, run_id, metadata, kwargs)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(serialized, "\n".join(prompts), run_id, metadata, kwargs)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            run = self._runs.pop(run_id, None)
        if run is None:
            return
        provider, model, prompt_tokens, intercepts_before = run
        if _state.intercepted.get() > intercepts_before:
            return  # the SDK hook already recorded this call
        usage = _usage(response)
        estimated = usage is None
        if usage is None:
            text = "".join(g.text for gens in response.generations for g in gens)
            usage = (prompt_tokens, count_tokens(text, model))
        self.guard.record_usage(provider, model, usage[0], usage[1], estimated=estimated)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            self._runs.pop(run_id, None)
