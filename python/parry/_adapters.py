"""Provider adapters: everything parry needs to know about one SDK endpoint.

An adapter answers: which model is this call for, what part of the request
identifies it for caching, how many tokens did the response use, and how do
we rebuild a response (or a whole stream) from cached data.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
from typing import Any, NamedTuple

log = logging.getLogger("parry")


class UsageDetail(NamedTuple):
    """A token breakdown for one call: standard input, output, and the three
    prompt-cache dimensions. Cache fields default to 0, so a plain call behaves
    exactly like the old ``(input, output)`` tuple."""

    input: int
    output: int
    cached_input: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0

    @property
    def total_input(self) -> int:
        return self.input + self.cached_input + self.cache_write_5m + self.cache_write_1h

    @property
    def total_tokens(self) -> int:
        return self.total_input + self.output


# Request fields that never change what the model returns. Everything else
# is part of the cache key (see PLAN.md, "cache key").
NON_SEMANTIC_KEYS = frozenset(
    {
        "stream",
        "stream_options",
        "timeout",
        "extra_headers",
        "extra_query",
        "extra_body",
        "user",
        "metadata",
        "store",
        "service_tier",
        "safety_identifier",
        "prompt_cache_key",
        "prompt_cache_retention",
        "background",
        "betas",
    }
)

RAW_RESPONSE_HEADER = "X-Stainless-Raw-Response"

# Only classes from these packages may be re-created from a cache entry. Every
# SDK parry hooks ships its own response types, so their top-level packages are
# allow-listed here too.
_ALLOWED_MODULE_PREFIXES = (
    "openai.",
    "anthropic.",
    "groq.",
    "together.",
    "google.",
    "ollama.",
    "mistralai.",
    "cohere.",
)
_class_cache: dict[str, type] = {}


def _import_type(path: str) -> type:
    """Resolve a ``"module:Qual.Name"`` path to the class it names."""
    module_name, _, qualname = path.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


class Uncacheable(Exception):
    """The request contains something parry cannot hash deterministically."""


def is_given(value: Any) -> bool:
    """False for the SDKs' ``NOT_GIVEN`` / ``omit`` sentinels."""
    return type(value).__name__ not in ("NotGiven", "Omit")


def _json_default(obj: Any) -> Any:
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump(mode="json", exclude_unset=True)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return {"__bytes_blake2b__": hashlib.blake2b(bytes(obj), digest_size=16).hexdigest()}
    if isinstance(obj, (tuple, set, frozenset)):
        return sorted(obj, key=repr) if isinstance(obj, (set, frozenset)) else list(obj)
    raise Uncacheable(type(obj).__name__)


def request_json(kwargs: dict[str, Any]) -> str:
    """Serialise the semantic part of a request deterministically."""
    payload = {
        k: v
        for k, v in kwargs.items()
        if k not in NON_SEMANTIC_KEYS and is_given(v) and v is not None
    }
    try:
        return json.dumps(payload, default=_json_default, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise Uncacheable(str(exc)) from exc


def raw_mode(kwargs: dict[str, Any]) -> str | None:
    """``"raw"`` for ``.with_raw_response``, ``"stream"`` for ``.with_streaming_response``."""
    headers = kwargs.get("extra_headers")
    if not headers or not is_given(headers):
        return None
    for k, v in headers.items():
        if k.lower() == RAW_RESPONSE_HEADER.lower():
            return "stream" if v == "stream" else "raw"
    return None


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def encode_response(response: Any) -> str:
    cls = type(response)
    return f"{cls.__module__}:{cls.__qualname__}\n{response.model_dump_json()}"


def decode_response(blob: str) -> Any:
    header, _, body = blob.partition("\n")
    cls = _class_cache.get(header)
    if cls is None:
        module_name, _, qualname = header.partition(":")
        if not module_name.startswith(_ALLOWED_MODULE_PREFIXES):
            raise ValueError(f"refusing to load cached type from {module_name!r}")
        obj: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        cls = obj
        _class_cache[header] = cls
    try:
        return cls.model_validate_json(body)
    except Exception:
        # Fall back to the SDKs' lenient construction (what they do for live
        # responses) when strict validation rejects an older payload.
        construct = getattr(cls, "construct", None)
        if construct is None:
            raise
        return construct(**json.loads(body))


class Adapter:
    provider: str = ""
    endpoint: str = ""

    def model(self, kwargs: dict[str, Any]) -> str:
        model = kwargs.get("model")
        return str(model) if model and is_given(model) else "unknown"

    def cacheable_request(self, kwargs: dict[str, Any]) -> bool:
        return True

    def usage(self, response: Any) -> tuple[int, int] | None:
        raise NotImplementedError

    def usage_detailed(self, response: Any) -> UsageDetail | None:
        """Full token breakdown, including prompt-cache dimensions. Defaults to
        the plain ``usage()`` with no cache tokens broken out."""
        u = self.usage(response)
        return None if u is None else UsageDetail(u[0], u[1])

    def tier(self, kwargs: dict[str, Any]) -> str:
        """The pricing tier this request runs at: "standard", "batch" or "fast"."""
        return "standard"

    def cacheable_response(self, response: Any) -> bool:
        return True

    def accumulator(self) -> StreamAccumulator:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<parry adapter {self.provider}.{self.endpoint}>"


class StreamAccumulator:
    """Watches stream events go by and rebuilds usage and the final response."""

    def __init__(self) -> None:
        self.usage: tuple[int, int] | None = None
        self.usage_detail: UsageDetail | None = None
        self.text_only = True
        self.complete = False

    def feed(self, event: Any) -> None:
        raise NotImplementedError

    def result(self) -> Any | None:
        """The assembled response, or None if it cannot be rebuilt faithfully."""
        return None

    def output_text(self) -> str:
        return ""


# --- OpenAI chat completions --------------------------------------------------


def _cached_from(u: Any, key: str) -> int:
    """Cached input tokens from a usage object's ``*_tokens_details``."""
    details = _get(u, key)
    return int(_get(details, "cached_tokens", 0) or 0)


class OpenAIChat(Adapter):
    """The OpenAI chat-completions surface — and every Stainless-generated SDK
    that mirrors it (groq, together, ...). The wire shape is identical; only the
    provider label and the response type used to rebuild a cached stream differ,
    so one adapter covers them all when parameterised."""

    def __init__(
        self,
        provider: str = "openai",
        endpoint: str = "chat.completions",
        completion_type: str = "openai.types.chat:ChatCompletion",
    ) -> None:
        self.provider = provider
        self.endpoint = endpoint
        self.completion_type = completion_type

    @staticmethod
    def usage_detail_from(u: Any) -> UsageDetail:
        # OpenAI's prompt_tokens *includes* cached tokens, so split them out:
        # the cached part is billed at the cheaper cache-read rate.
        prompt = int(_get(u, "prompt_tokens", 0) or 0)
        cached = _cached_from(u, "prompt_tokens_details")
        output = int(_get(u, "completion_tokens", 0) or 0)
        return UsageDetail(max(prompt - cached, 0), output, cached_input=cached)

    def usage(self, response: Any) -> tuple[int, int] | None:
        u = _get(response, "usage")
        if u is None:
            return None
        return int(_get(u, "prompt_tokens", 0) or 0), int(_get(u, "completion_tokens", 0) or 0)

    def usage_detailed(self, response: Any) -> UsageDetail | None:
        u = _get(response, "usage")
        return None if u is None else self.usage_detail_from(u)

    def tier(self, kwargs: dict[str, Any]) -> str:
        return "fast" if kwargs.get("service_tier") == "priority" else "standard"

    def accumulator(self) -> StreamAccumulator:
        return _OpenAIChatAccumulator(self.completion_type)


class _OpenAIChatAccumulator(StreamAccumulator):
    def __init__(self, completion_type: str = "openai.types.chat:ChatCompletion") -> None:
        super().__init__()
        self.completion_type = completion_type
        self.meta: dict[str, Any] = {}
        self.choices: dict[int, dict[str, Any]] = {}

    def feed(self, chunk: Any) -> None:
        if not self.meta:
            self.meta = {
                "id": _get(chunk, "id"),
                "created": _get(chunk, "created"),
                "model": _get(chunk, "model"),
                "system_fingerprint": _get(chunk, "system_fingerprint"),
            }
        u = _get(chunk, "usage")
        if u is not None:
            self.usage = (
                int(_get(u, "prompt_tokens", 0) or 0),
                int(_get(u, "completion_tokens", 0) or 0),
            )
            self.usage_detail = OpenAIChat.usage_detail_from(u)
        for choice in _get(chunk, "choices", None) or []:
            idx = int(_get(choice, "index", 0) or 0)
            slot = self.choices.setdefault(idx, {"text": [], "finish_reason": None})
            delta = _get(choice, "delta")
            content = _get(delta, "content")
            if content:
                slot["text"].append(content)
            if _get(delta, "tool_calls") or _get(delta, "function_call") or _get(delta, "refusal"):
                self.text_only = False
            if _get(choice, "logprobs") is not None:
                self.text_only = False
            finish = _get(choice, "finish_reason")
            if finish:
                slot["finish_reason"] = finish

    def output_text(self) -> str:
        return "".join("".join(c["text"]) for c in self.choices.values())

    def result(self) -> Any | None:
        if not (self.complete and self.text_only and self.choices):
            return None
        if any(c["finish_reason"] is None for c in self.choices.values()):
            return None
        try:
            completion_cls = _import_type(self.completion_type)
        except Exception:
            # The SDK's response type isn't importable (or moved); meter the
            # stream but don't risk rebuilding it into the wrong shape.
            return None

        data: dict[str, Any] = {
            **{k: v for k, v in self.meta.items() if v is not None},
            "object": "chat.completion",
            "choices": [
                {
                    "index": i,
                    "message": {"role": "assistant", "content": "".join(c["text"])},
                    "finish_reason": c["finish_reason"],
                    "logprobs": None,
                }
                for i, c in sorted(self.choices.items())
            ],
        }
        if self.usage is not None:
            data["usage"] = {
                "prompt_tokens": self.usage[0],
                "completion_tokens": self.usage[1],
                "total_tokens": sum(self.usage),
            }
        return completion_cls.model_validate(data)


# --- OpenAI responses -----------------------------------------------------------


class OpenAIResponses(Adapter):
    provider = "openai"
    endpoint = "responses"

    def cacheable_request(self, kwargs: dict[str, Any]) -> bool:
        # Background responses are polled later; caching the queued stub is wrong.
        return kwargs.get("background") is not True

    @staticmethod
    def usage_detail_from(u: Any) -> UsageDetail:
        # The Responses API's input_tokens also includes cached tokens.
        inp = int(_get(u, "input_tokens", 0) or 0)
        cached = _cached_from(u, "input_tokens_details")
        output = int(_get(u, "output_tokens", 0) or 0)
        return UsageDetail(max(inp - cached, 0), output, cached_input=cached)

    def usage(self, response: Any) -> tuple[int, int] | None:
        u = _get(response, "usage")
        if u is None:
            return None
        return int(_get(u, "input_tokens", 0) or 0), int(_get(u, "output_tokens", 0) or 0)

    def usage_detailed(self, response: Any) -> UsageDetail | None:
        u = _get(response, "usage")
        return None if u is None else self.usage_detail_from(u)

    def tier(self, kwargs: dict[str, Any]) -> str:
        return "fast" if kwargs.get("service_tier") == "priority" else "standard"

    def cacheable_response(self, response: Any) -> bool:
        return _get(response, "status", "completed") == "completed"

    def accumulator(self) -> StreamAccumulator:
        return _OpenAIResponsesAccumulator(self)


class _OpenAIResponsesAccumulator(StreamAccumulator):
    def __init__(self, adapter: OpenAIResponses) -> None:
        super().__init__()
        self.adapter = adapter
        self.final: Any = None
        self.text: list[str] = []

    def feed(self, event: Any) -> None:
        kind = _get(event, "type", "")
        if kind == "response.output_text.delta":
            self.text.append(_get(event, "delta", "") or "")
        elif kind in ("response.completed", "response.incomplete", "response.failed"):
            self.final = _get(event, "response")
            self.usage = self.adapter.usage(self.final)
            self.usage_detail = self.adapter.usage_detailed(self.final)

    def output_text(self) -> str:
        return "".join(self.text)

    def result(self) -> Any | None:
        if self.complete and self.final is not None and self.adapter.cacheable_response(self.final):
            return self.final
        return None


# --- Anthropic messages ---------------------------------------------------------


class AnthropicMessages(Adapter):
    provider = "anthropic"

    def __init__(self, endpoint: str = "messages", message_type: str = "anthropic.types:Message"):
        self.endpoint = endpoint
        self.message_type = message_type

    @staticmethod
    def billable_input(u: Any) -> int:
        # Cache writes bill at 1.25x and cache reads at 0.1x the base input
        # price, so fold them into a billing-equivalent input count.
        base = int(_get(u, "input_tokens", 0) or 0)
        write = int(_get(u, "cache_creation_input_tokens", 0) or 0)
        read = int(_get(u, "cache_read_input_tokens", 0) or 0)
        return base + round(write * 1.25) + round(read * 0.1)

    @staticmethod
    def usage_detail_from(u: Any) -> UsageDetail:
        # Anthropic reports cache tokens separately from input_tokens (which is
        # already the uncached base). Cache writes split by TTL when the newer
        # `cache_creation` breakdown is present; otherwise the default 5m TTL.
        base = int(_get(u, "input_tokens", 0) or 0)
        read = int(_get(u, "cache_read_input_tokens", 0) or 0)
        creation = _get(u, "cache_creation")
        write_5m = int(_get(creation, "ephemeral_5m_input_tokens", 0) or 0)
        write_1h = int(_get(creation, "ephemeral_1h_input_tokens", 0) or 0)
        if write_5m == 0 and write_1h == 0:
            write_5m = int(_get(u, "cache_creation_input_tokens", 0) or 0)
        return UsageDetail(
            base,
            int(_get(u, "output_tokens", 0) or 0),
            cached_input=read,
            cache_write_5m=write_5m,
            cache_write_1h=write_1h,
        )

    def usage(self, response: Any) -> tuple[int, int] | None:
        u = _get(response, "usage")
        if u is None:
            return None
        return self.billable_input(u), int(_get(u, "output_tokens", 0) or 0)

    def usage_detailed(self, response: Any) -> UsageDetail | None:
        u = _get(response, "usage")
        return None if u is None else self.usage_detail_from(u)

    def cacheable_response(self, response: Any) -> bool:
        return _get(response, "stop_reason") != "pause_turn"

    def accumulator(self) -> StreamAccumulator:
        return _AnthropicAccumulator(self)


class _AnthropicAccumulator(StreamAccumulator):
    def __init__(self, adapter: AnthropicMessages) -> None:
        super().__init__()
        self.adapter = adapter
        self.message: dict[str, Any] | None = None
        self.blocks: dict[int, list[str]] = {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.stop_reason: str | None = None
        self.stop_sequence: str | None = None

    def feed(self, event: Any) -> None:
        kind = _get(event, "type", "")
        if kind == "message_start":
            msg = _get(event, "message")
            dump = getattr(msg, "model_dump", None)
            self.message = dump(mode="json") if callable(dump) else dict(msg or {})
            u = _get(msg, "usage")
            if u is not None:
                self.input_tokens = self.adapter.billable_input(u)
                self.output_tokens = int(_get(u, "output_tokens", 0) or 0)
                self.usage_detail = self.adapter.usage_detail_from(u)
        elif kind == "content_block_start":
            block = _get(event, "content_block")
            if _get(block, "type") != "text":
                self.text_only = False
            self.blocks[int(_get(event, "index", 0) or 0)] = [_get(block, "text", "") or ""]
        elif kind == "content_block_delta":
            delta = _get(event, "delta")
            if _get(delta, "type") == "text_delta":
                self.blocks.setdefault(int(_get(event, "index", 0) or 0), []).append(
                    _get(delta, "text", "")
                )
            else:
                self.text_only = False
        elif kind == "message_delta":
            delta = _get(event, "delta")
            self.stop_reason = _get(delta, "stop_reason") or self.stop_reason
            self.stop_sequence = _get(delta, "stop_sequence") or self.stop_sequence
            u = _get(event, "usage")
            if u is not None:
                self.output_tokens = int(_get(u, "output_tokens", 0) or 0)
                if _get(u, "input_tokens") is not None:
                    self.input_tokens = self.adapter.billable_input(u)
                    self.usage_detail = self.adapter.usage_detail_from(u)
        if self.message is not None:
            self.usage = (self.input_tokens, self.output_tokens)
            if self.usage_detail is not None:
                self.usage_detail = self.usage_detail._replace(output=self.output_tokens)

    def output_text(self) -> str:
        return "".join("".join(parts) for parts in self.blocks.values())

    def result(self) -> Any | None:
        if not (self.complete and self.text_only and self.message is not None and self.stop_reason):
            return None
        if self.stop_reason == "pause_turn":
            return None
        module_name, _, qualname = self.adapter.message_type.partition(":")
        cls = getattr(importlib.import_module(module_name), qualname)
        data = dict(self.message)
        data["content"] = [
            {"type": "text", "text": "".join(p)} for _, p in sorted(self.blocks.items())
        ]
        data["stop_reason"] = self.stop_reason
        data["stop_sequence"] = self.stop_sequence
        usage = dict(data.get("usage") or {})
        usage["output_tokens"] = self.output_tokens
        data["usage"] = usage
        return cls.model_validate(data)


# --- Google Gen AI (google-genai) -----------------------------------------------


class GoogleGenAI(Adapter):
    """Google's ``google-genai`` SDK. Unlike the OpenAI-shaped clients, this is
    a genuinely different surface: the method is ``generate_content`` (not
    ``create``), streaming is a separate ``generate_content_stream`` method
    rather than a ``stream=True`` flag, and usage lives in ``usage_metadata``.
    """

    provider = "google"
    endpoint = "generate_content"

    @staticmethod
    def usage_detail_from(u: Any) -> UsageDetail:
        # Gemini's prompt_token_count already includes cached context, so split
        # the cached part out to bill it at the cheaper cache-read rate. Thinking
        # tokens (thoughts_token_count) bill as output, like the candidates.
        prompt = int(_get(u, "prompt_token_count", 0) or 0)
        cached = int(_get(u, "cached_content_token_count", 0) or 0)
        candidates = int(_get(u, "candidates_token_count", 0) or 0)
        thoughts = int(_get(u, "thoughts_token_count", 0) or 0)
        return UsageDetail(max(prompt - cached, 0), candidates + thoughts, cached_input=cached)

    def usage(self, response: Any) -> tuple[int, int] | None:
        u = _get(response, "usage_metadata")
        if u is None:
            return None
        d = self.usage_detail_from(u)
        return d.total_input, d.output

    def usage_detailed(self, response: Any) -> UsageDetail | None:
        u = _get(response, "usage_metadata")
        return None if u is None else self.usage_detail_from(u)

    def accumulator(self) -> StreamAccumulator:
        return _GoogleAccumulator()


class _GoogleAccumulator(StreamAccumulator):
    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []

    def feed(self, chunk: Any) -> None:
        try:
            text = _get(chunk, "text")  # a property on GenerateContentResponse
        except Exception:
            text = None
        if text:
            self.text_parts.append(text)
        u = _get(chunk, "usage_metadata")
        if u is not None:
            detail = GoogleGenAI.usage_detail_from(u)
            self.usage_detail = detail
            self.usage = (detail.total_input, detail.output)

    def output_text(self) -> str:
        return "".join(self.text_parts)

    def result(self) -> Any | None:
        # Metering is faithful; rebuilding a merged GenerateContentResponse for
        # the cache is not worth the risk in this first cut, so streamed Gemini
        # responses are metered but not cached.
        return None


# --- Ollama ----------------------------------------------------------------------


class Ollama(Adapter):
    """Ollama's ``Client.chat`` / ``AsyncClient.chat``. Token counts live at the
    top level of the response (``prompt_eval_count`` / ``eval_count``), not under
    a ``usage`` object, and streaming is a ``stream=True`` flag whose final chunk
    carries the counts. Local models are usually unpriced, so the value here is
    rate limiting, caching and request metering."""

    provider = "ollama"
    endpoint = "chat"

    @staticmethod
    def usage_detail_from(resp: Any) -> UsageDetail:
        return UsageDetail(
            int(_get(resp, "prompt_eval_count", 0) or 0),
            int(_get(resp, "eval_count", 0) or 0),
        )

    def usage(self, response: Any) -> tuple[int, int] | None:
        if _get(response, "prompt_eval_count") is None and _get(response, "eval_count") is None:
            return None
        d = self.usage_detail_from(response)
        return d.input, d.output

    def usage_detailed(self, response: Any) -> UsageDetail | None:
        u = self.usage(response)
        return None if u is None else UsageDetail(u[0], u[1])

    def accumulator(self) -> StreamAccumulator:
        return _OllamaAccumulator()


class _OllamaAccumulator(StreamAccumulator):
    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []

    def feed(self, chunk: Any) -> None:
        content = _get(_get(chunk, "message"), "content")
        if content:
            self.text_parts.append(content)
        if _get(chunk, "eval_count") is not None or _get(chunk, "prompt_eval_count") is not None:
            d = Ollama.usage_detail_from(chunk)
            self.usage_detail = d
            self.usage = (d.input, d.output)

    def output_text(self) -> str:
        return "".join(self.text_parts)

    def result(self) -> Any | None:
        return None


# --- Mistral ---------------------------------------------------------------------


class MistralChat(OpenAIChat):
    """Mistral's chat surface. Usage is OpenAI-shaped (``.usage.prompt_tokens`` /
    ``.completion_tokens``) so the parsing is inherited; only the method names
    (``complete`` / ``stream``) and the stream chunk wrapper differ."""

    def __init__(self) -> None:
        super().__init__(provider="mistral", endpoint="chat.completions")

    def accumulator(self) -> StreamAccumulator:
        return _MistralAccumulator()


class _MistralAccumulator(StreamAccumulator):
    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []

    def feed(self, event: Any) -> None:
        # Mistral wraps each streamed chunk in a CompletionEvent with a ``.data``
        # payload shaped like an OpenAI chunk.
        data = _get(event, "data", event)
        for choice in _get(data, "choices", None) or []:
            content = _get(_get(choice, "delta"), "content")
            if content:
                self.text_parts.append(content)
        u = _get(data, "usage")
        if u is not None:
            self.usage_detail = OpenAIChat.usage_detail_from(u)
            self.usage = (self.usage_detail.total_input, self.usage_detail.output)

    def output_text(self) -> str:
        return "".join(self.text_parts)

    def result(self) -> Any | None:
        return None


# --- Cohere ----------------------------------------------------------------------


class CohereChat(Adapter):
    """Cohere's ``BaseCohere.chat`` / ``chat_stream``. Usage lives under
    ``.usage.tokens`` (v2) or ``.meta.tokens`` / ``.meta.billed_units`` (v1); both
    are tried so the one adapter covers either client generation."""

    provider = "cohere"
    endpoint = "chat"

    @staticmethod
    def usage_detail_from(response: Any) -> UsageDetail | None:
        meta = _get(response, "usage") or _get(response, "meta")
        tokens = _get(meta, "tokens") or _get(meta, "billed_units")
        if tokens is None:
            return None
        return UsageDetail(
            int(_get(tokens, "input_tokens", 0) or 0),
            int(_get(tokens, "output_tokens", 0) or 0),
        )

    def usage(self, response: Any) -> tuple[int, int] | None:
        d = self.usage_detail_from(response)
        return None if d is None else (d.input, d.output)

    def usage_detailed(self, response: Any) -> UsageDetail | None:
        return self.usage_detail_from(response)

    def accumulator(self) -> StreamAccumulator:
        return _CohereAccumulator()


class _CohereAccumulator(StreamAccumulator):
    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []

    def feed(self, event: Any) -> None:
        delta = _get(event, "delta")
        # v2 content-delta events: delta.message.content.text
        text = _get(_get(_get(delta, "message"), "content"), "text") or _get(event, "text")
        if text:
            self.text_parts.append(text)
        detail = CohereChat.usage_detail_from(delta) or CohereChat.usage_detail_from(event)
        if detail is not None:
            self.usage_detail = detail
            self.usage = (detail.input, detail.output)

    def output_text(self) -> str:
        return "".join(self.text_parts)

    def result(self) -> Any | None:
        return None


# --- AWS Bedrock (Converse API) --------------------------------------------------


class BedrockConverse(Adapter):
    """AWS Bedrock's Converse API (boto3 ``bedrock-runtime``). boto3 builds its
    methods dynamically per client, so this is not auto-hooked by
    :func:`parry.install`; guard a client with
    :func:`parry.integrations.bedrock.guard`. Responses are plain dicts, so they
    are metered and rate-limited but not cached."""

    provider = "bedrock"
    endpoint = "converse"

    def model(self, kwargs: dict[str, Any]) -> str:
        model = kwargs.get("modelId") or kwargs.get("model")
        return str(model) if model and is_given(model) else "unknown"

    def cacheable_request(self, kwargs: dict[str, Any]) -> bool:
        return False  # dict responses aren't re-creatable from the cache encoder

    @staticmethod
    def usage_detail_from(u: Any) -> UsageDetail:
        return UsageDetail(
            int(_get(u, "inputTokens", 0) or 0),
            int(_get(u, "outputTokens", 0) or 0),
            cached_input=int(_get(u, "cacheReadInputTokens", 0) or 0),
            cache_write_5m=int(_get(u, "cacheWriteInputTokens", 0) or 0),
        )

    def usage(self, response: Any) -> tuple[int, int] | None:
        u = _get(response, "usage")
        if u is None:
            return None
        d = self.usage_detail_from(u)
        return d.total_input, d.output

    def usage_detailed(self, response: Any) -> UsageDetail | None:
        u = _get(response, "usage")
        return None if u is None else self.usage_detail_from(u)

    def accumulator(self) -> StreamAccumulator:
        return _BedrockAccumulator()


class _BedrockAccumulator(StreamAccumulator):
    """Bedrock ``converse_stream`` yields a dict with an event iterator; parry
    instruments the events. Usage arrives in a ``metadata`` event."""

    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []

    def feed(self, event: Any) -> None:
        delta = _get(_get(event, "contentBlockDelta"), "delta")
        text = _get(delta, "text")
        if text:
            self.text_parts.append(text)
        meta = _get(event, "metadata")
        u = _get(meta, "usage")
        if u is not None:
            d = BedrockConverse.usage_detail_from(u)
            self.usage_detail = d
            self.usage = (d.total_input, d.output)

    def output_text(self) -> str:
        return "".join(self.text_parts)

    def result(self) -> Any | None:
        return None


OPENAI_CHAT = OpenAIChat()
OPENAI_RESPONSES = OpenAIResponses()
ANTHROPIC_MESSAGES = AnthropicMessages()
ANTHROPIC_BETA_MESSAGES = AnthropicMessages("beta.messages", "anthropic.types.beta:BetaMessage")
GOOGLE_GENAI = GoogleGenAI()
OLLAMA_CHAT = Ollama()
MISTRAL_CHAT = MistralChat()
COHERE_CHAT = CohereChat()
BEDROCK_CONVERSE = BedrockConverse()

# OpenAI-shaped SDKs: Stainless-generated clients whose chat-completions surface
# mirrors OpenAI's exactly, so the generic patcher binds them by duck-typing the
# (resource class, create method) pair. Each becomes a config row here rather
# than a hand-written adapter.
#   (provider, module, sync_class, async_class, completion_type)
OPENAI_SHAPED: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "groq",
        "groq.resources.chat.completions",
        "Completions",
        "AsyncCompletions",
        "groq.types.chat:ChatCompletion",
    ),
    (
        "together",
        "together.resources.chat.completions",
        "CompletionsResource",
        "AsyncCompletionsResource",
        "together.types:ChatCompletionResponse",
    ),
)


def openai_shaped_adapter(provider: str, completion_type: str) -> OpenAIChat:
    """An OpenAIChat adapter labelled for one Stainless-generated SDK."""
    return OpenAIChat(provider=provider, completion_type=completion_type)
