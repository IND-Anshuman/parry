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
from typing import Any

log = logging.getLogger("parry")

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

# Only classes from these packages may be re-created from a cache entry.
_ALLOWED_MODULE_PREFIXES = ("openai.", "anthropic.")
_class_cache: dict[str, type] = {}


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


class OpenAIChat(Adapter):
    provider = "openai"
    endpoint = "chat.completions"

    def usage(self, response: Any) -> tuple[int, int] | None:
        u = _get(response, "usage")
        if u is None:
            return None
        return int(_get(u, "prompt_tokens", 0) or 0), int(_get(u, "completion_tokens", 0) or 0)

    def accumulator(self) -> StreamAccumulator:
        return _OpenAIChatAccumulator()


class _OpenAIChatAccumulator(StreamAccumulator):
    def __init__(self) -> None:
        super().__init__()
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
        from openai.types.chat import ChatCompletion

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
        return ChatCompletion.model_validate(data)


# --- OpenAI responses -----------------------------------------------------------


class OpenAIResponses(Adapter):
    provider = "openai"
    endpoint = "responses"

    def cacheable_request(self, kwargs: dict[str, Any]) -> bool:
        # Background responses are polled later; caching the queued stub is wrong.
        return kwargs.get("background") is not True

    def usage(self, response: Any) -> tuple[int, int] | None:
        u = _get(response, "usage")
        if u is None:
            return None
        return int(_get(u, "input_tokens", 0) or 0), int(_get(u, "output_tokens", 0) or 0)

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

    def usage(self, response: Any) -> tuple[int, int] | None:
        u = _get(response, "usage")
        if u is None:
            return None
        return self.billable_input(u), int(_get(u, "output_tokens", 0) or 0)

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
        if self.message is not None:
            self.usage = (self.input_tokens, self.output_tokens)

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


OPENAI_CHAT = OpenAIChat()
OPENAI_RESPONSES = OpenAIResponses()
ANTHROPIC_MESSAGES = AnthropicMessages()
ANTHROPIC_BETA_MESSAGES = AnthropicMessages("beta.messages", "anthropic.types.beta:BetaMessage")
