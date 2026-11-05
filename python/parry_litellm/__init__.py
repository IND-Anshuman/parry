"""A drop-in parry guard for LiteLLM users.

    from parry_litellm import completion     # was: from litellm import completion
    completion(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

Every call runs through a parry guard — cache, rate limiting, budget, cost
tracking, retry and any OpenTelemetry export you've set up — while LiteLLM still
does the actual provider call and returns its usual ``ModelResponse``.
``import parry_litellm as litellm`` also works: any name parry_litellm doesn't
define proxies to the real ``litellm``.

The guard used is the one active via ``parry.install()`` or a ``with Parry():``
block; with none installed, parry_litellm keeps its own default guard (set your
own with :func:`configure`). LiteLLM is imported lazily, so importing this module
never requires it to be installed.
"""

from __future__ import annotations

import importlib
from typing import Any

from parry import Parry, get_guard
from parry._adapters import OpenAIChat

# An unimportable completion type, so a streamed response is metered but not
# rebuilt into the wrong type for the cache (non-streaming responses still cache
# as LiteLLM's own ModelResponse, which round-trips through model_dump_json).
_NO_STREAM_CACHE = "parry_litellm._absent:Nope"

# A stand-in "resource" for the guard; LiteLLM has no client object to carry.
_RESOURCE = type("_LiteLLMResource", (), {"_client": None})()

_default_guard: Parry | None = None


def configure(guard: Parry) -> None:
    """Use ``guard`` for parry_litellm calls when none is installed globally."""
    global _default_guard
    _default_guard = guard


def _guard() -> Parry:
    active = get_guard()
    if active is not None:
        return active
    global _default_guard
    if _default_guard is None:
        _default_guard = Parry()
    return _default_guard


def _litellm() -> Any:
    return importlib.import_module("litellm")


def _provider(model: str) -> str:
    try:
        _, provider, _, _ = _litellm().get_llm_provider(model)
        return provider or "litellm"
    except Exception:
        return "litellm"


def _adapter(model: str) -> OpenAIChat:
    # LiteLLM's ModelResponse is OpenAI-chat-shaped, so OpenAIChat parses its
    # usage; the provider label drives rate-limit buckets, cost and spans.
    return OpenAIChat(
        provider=_provider(model), endpoint="completion", completion_type=_NO_STREAM_CACHE
    )


def completion(**kwargs: Any) -> Any:
    """``litellm.completion``, guarded by parry."""
    lm = _litellm()

    def original(_resource: Any, **kw: Any) -> Any:
        return lm.completion(**kw)

    return _guard()._invoke(
        _adapter(str(kwargs.get("model") or "")),
        original,
        _RESOURCE,
        (),
        kwargs,
        kwargs.get("stream") is True,
    )


async def acompletion(**kwargs: Any) -> Any:
    """``litellm.acompletion``, guarded by parry."""
    lm = _litellm()

    def original(_resource: Any, **kw: Any) -> Any:
        return lm.acompletion(**kw)

    return await _guard()._invoke_async(
        _adapter(str(kwargs.get("model") or "")),
        original,
        _RESOURCE,
        (),
        kwargs,
        async_raw_parse=False,
        is_stream=kwargs.get("stream") is True,
    )


def __getattr__(name: str) -> Any:
    # Anything parry_litellm doesn't define proxies to the real litellm, so
    # `import parry_litellm as litellm` is a near drop-in replacement.
    return getattr(_litellm(), name)
