"""Phase 1 — provider reach.

The groq / together / google-genai SDKs are not installed in CI's default
environment, so these tests exercise the generic patcher and the new adapters
with synthesised fakes. OpenAI and Anthropic keep their full end-to-end
coverage in ``test_patch.py``; here we prove the *generalisation* is sound:

* an OpenAI-shaped SDK is hooked by duck typing its own resource class, and
  calls route through the guard with the right provider label;
* Google's different surface (``generate_content``, a dedicated streaming
  method, ``usage_metadata``) parses and meters correctly.
"""

from __future__ import annotations

import sys
import types

import parry
from parry import Parry, _patch
from parry._adapters import (
    BEDROCK_CONVERSE,
    COHERE_CHAT,
    GOOGLE_GENAI,
    MISTRAL_CHAT,
    OLLAMA_CHAT,
    OPENAI_SHAPED,
    GoogleGenAI,
    OpenAIChat,
    openai_shaped_adapter,
)

# --- OpenAI-shaped SDKs (groq, together) --------------------------------------


def test_openai_shaped_adapter_parses_like_openai():
    groq = openai_shaped_adapter("groq", "groq.types.chat:ChatCompletion")
    assert groq.provider == "groq"
    # Identical wire shape to OpenAI: prompt_tokens includes the cached part.
    detail = groq.usage_detailed(
        {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "prompt_tokens_details": {"cached_tokens": 400},
            }
        }
    )
    assert detail == (600, 200, 400, 0, 0)


def test_openai_shaped_providers_are_registered():
    providers = {row[0] for row in OPENAI_SHAPED}
    assert {"groq", "together"} <= providers
    # Every shaped SDK contributes a sync and an async target.
    for provider in ("groq", "together"):
        hits = [t for t in _patch.TARGETS if t.provider == provider]
        assert len(hits) == 2
        assert {t.is_async for t in hits} == {False, True}
        assert all(t.method == "create" and not t.is_stream for t in hits)


def test_generic_patcher_hooks_an_openai_shaped_sdk(monkeypatch):
    # A stand-in for a Stainless-generated SDK whose resource class is named
    # differently from OpenAI's but still exposes .create(**kwargs).
    mod = types.ModuleType("fakeshaped.resources")

    class CompletionsResource:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                model=kwargs["model"],
                usage=types.SimpleNamespace(
                    prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None
                ),
            )

    mod.CompletionsResource = CompletionsResource
    monkeypatch.setitem(sys.modules, "fakeshaped.resources", mod)

    adapter = openai_shaped_adapter("fakeshaped", "fakeshaped.types:ChatCompletion")
    target = _patch._Target(
        "fakeshaped", "fakeshaped.resources", "CompletionsResource", adapter, False
    )
    monkeypatch.setattr(_patch, "TARGETS", _patch.TARGETS + (target,))

    with Parry(cache=False) as guard:
        _patch.apply(providers=("fakeshaped",))
        # The resource class was hooked in place (its display name uses the
        # class's real __module__, so check the hook marker directly).
        assert getattr(CompletionsResource.create, "__parry_original__", None) is not None
        resource = CompletionsResource()
        resource._client = types.SimpleNamespace(base_url="https://fakeshaped.test")
        resource.create(model="llama-3.3-70b", messages=[{"role": "user", "content": "hi"}])
        assert guard.stats.requests == 1
        assert "fakeshaped:llama-3.3-70b" in guard.stats.by_model
    _patch.restore()
    # restore() put the plain method back.
    assert not hasattr(CompletionsResource.__dict__["create"], "__parry_original__")


# --- Google Gen AI ------------------------------------------------------------


def test_google_usage_splits_cache_and_bills_thinking_as_output():
    detail = GOOGLE_GENAI.usage_detailed(
        types.SimpleNamespace(
            usage_metadata=types.SimpleNamespace(
                prompt_token_count=1000,
                cached_content_token_count=400,
                candidates_token_count=200,
                thoughts_token_count=50,
            )
        )
    )
    # 1000 prompt includes 400 cached → 600 fresh input; 200 + 50 thinking out.
    assert detail == (600, 250, 400, 0, 0)
    assert detail.total_input == 1000


def test_google_usage_absent_is_none():
    assert GOOGLE_GENAI.usage_detailed(types.SimpleNamespace(usage_metadata=None)) is None


def test_google_targets_cover_both_methods():
    google = [t for t in _patch.TARGETS if t.provider == "google"]
    methods = {(t.method, t.is_stream, t.is_async) for t in google}
    assert methods == {
        ("generate_content", False, False),
        ("generate_content", False, True),
        ("generate_content_stream", True, False),
        ("generate_content_stream", True, True),
    }
    assert GOOGLE_GENAI.provider == "google"


def test_google_stream_accumulator_meters_without_caching():
    acc = GoogleGenAI().accumulator()
    acc.feed(types.SimpleNamespace(text="hello ", usage_metadata=None))
    acc.feed(
        types.SimpleNamespace(
            text="world",
            usage_metadata=types.SimpleNamespace(
                prompt_token_count=10,
                cached_content_token_count=0,
                candidates_token_count=5,
                thoughts_token_count=0,
            ),
        )
    )
    acc.complete = True
    assert acc.output_text() == "hello world"
    assert acc.usage == (10, 5)
    assert acc.result() is None  # streamed Gemini responses are metered, not cached


# --- install() provider flags -------------------------------------------------


def test_install_can_opt_out_of_every_provider():
    parry.install(openai=False, anthropic=False, groq=False, together=False, google=False)
    assert not _patch.is_installed()


def test_install_default_covers_openai_shaped_and_google():
    # openai/anthropic are installed so they hook; the rest simply no-op, but
    # the call must not raise when they are absent.
    parry.install()
    assert _patch.is_installed()
    assert _patch.ALL_PROVIDERS == (
        "openai",
        "anthropic",
        "groq",
        "together",
        "google",
        "ollama",
        "mistral",
        "cohere",
    )


def test_shaped_adapter_rebuild_type_degrades_gracefully():
    # An un-importable completion type must not crash stream rebuild; it just
    # declines to cache (returns None).
    acc = OpenAIChat("groq", completion_type="nonexistent.module:Nope").accumulator()
    acc.feed({"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]})
    acc.complete = True
    assert acc.result() is None


# --- bespoke adapters: ollama / mistral / cohere ------------------------------


def test_ollama_usage_from_top_level_counts():
    resp = types.SimpleNamespace(
        model="llama3.3", prompt_eval_count=120, eval_count=45, message=None
    )
    assert OLLAMA_CHAT.usage_detailed(resp) == (120, 45, 0, 0, 0)
    # A mid-stream chunk with no counts yet reports no usage.
    assert OLLAMA_CHAT.usage(types.SimpleNamespace(prompt_eval_count=None, eval_count=None)) is None


def test_ollama_stream_accumulator_reads_done_chunk():
    acc = OLLAMA_CHAT.accumulator()
    acc.feed(types.SimpleNamespace(message=types.SimpleNamespace(content="hi "), eval_count=None))
    acc.feed(
        types.SimpleNamespace(
            message=types.SimpleNamespace(content="there"),
            prompt_eval_count=10,
            eval_count=7,
        )
    )
    acc.complete = True
    assert acc.output_text() == "hi there"
    assert acc.usage == (10, 7)
    assert acc.result() is None


def test_mistral_usage_is_openai_shaped():
    assert MISTRAL_CHAT.provider == "mistral"
    detail = MISTRAL_CHAT.usage_detailed({"usage": {"prompt_tokens": 30, "completion_tokens": 12}})
    assert detail == (30, 12, 0, 0, 0)


def test_mistral_stream_accumulator_unwraps_data():
    acc = MISTRAL_CHAT.accumulator()
    acc.feed(types.SimpleNamespace(data={"choices": [{"delta": {"content": "yo"}}]}))
    acc.feed(
        types.SimpleNamespace(
            data={
                "choices": [{"delta": {"content": "!"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }
        )
    )
    acc.complete = True
    assert acc.output_text() == "yo!"
    assert acc.usage == (5, 2)


def test_cohere_usage_v2_and_v1_shapes():
    v2 = COHERE_CHAT.usage_detailed({"usage": {"tokens": {"input_tokens": 40, "output_tokens": 9}}})
    assert v2 == (40, 9, 0, 0, 0)
    v1 = COHERE_CHAT.usage_detailed(
        {"meta": {"billed_units": {"input_tokens": 5, "output_tokens": 3}}}
    )
    assert v1 == (5, 3, 0, 0, 0)
    assert COHERE_CHAT.usage_detailed({"something_else": 1}) is None


def test_bespoke_providers_registered_with_right_methods():
    by_provider = {}
    for t in _patch.TARGETS:
        by_provider.setdefault(t.provider, set()).add((t.cls_name, t.method, t.is_stream))
    assert ("Client", "chat", False) in by_provider["ollama"]
    assert ("AsyncClient", "chat", False) in by_provider["ollama"]
    assert ("Chat", "complete", False) in by_provider["mistral"]
    assert ("Chat", "stream", True) in by_provider["mistral"]
    assert ("BaseCohere", "chat", False) in by_provider["cohere"]
    assert ("BaseCohere", "chat_stream", True) in by_provider["cohere"]


# --- AWS Bedrock (per-client guard) -------------------------------------------


def test_bedrock_usage_splits_cache_tokens():
    detail = BEDROCK_CONVERSE.usage_detailed(
        {"usage": {"inputTokens": 100, "outputTokens": 40, "cacheReadInputTokens": 20}}
    )
    assert detail == (100, 40, 20, 0, 0)
    assert detail.total_input == 120
    assert (
        BEDROCK_CONVERSE.model({"modelId": "anthropic.claude-sonnet"}) == "anthropic.claude-sonnet"
    )
    assert BEDROCK_CONVERSE.cacheable_request({}) is False


def test_bedrock_guard_meters_converse():
    from parry.integrations import bedrock

    class FakeBedrockClient:
        def __init__(self):
            self.calls = 0

        def converse(self, **kwargs):
            self.calls += 1
            return {"usage": {"inputTokens": 100, "outputTokens": 40}, "stopReason": "end_turn"}

    client = FakeBedrockClient()
    with Parry(cache=False) as guard:
        bedrock.guard(client)
        client.converse(modelId="anthropic.claude-sonnet", messages=[{"role": "user"}])
        assert client.calls == 1
        assert guard.stats.requests == 1
        assert "bedrock:anthropic.claude-sonnet" in guard.stats.by_model
    # Guarding is idempotent and reversible.
    marker = client.converse
    bedrock.guard(client)
    assert client.converse is marker
    bedrock.unguard(client)
    assert client.converse is not marker
