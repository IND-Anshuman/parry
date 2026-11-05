"""The LiteLLM drop-in shim (parry_litellm).

LiteLLM isn't installed in CI's default environment, so a fake ``litellm``
module is injected. Its ``completion`` returns a real ``openai`` ChatCompletion
(LiteLLM's ModelResponse is the same shape), which lets the cache round-trip.
"""

from __future__ import annotations

import sys
import types

import pytest

import parry

openai = pytest.importorskip("openai")
from openai.types.chat import ChatCompletion  # noqa: E402


def _response(model: str, content: str = "echo", prompt: int = 10, completion: int = 5):
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            },
        }
    )


class _FakeStream:
    def __init__(self, chunks):
        self._it = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def close(self):
        pass


def _stream_chunks(model: str):
    base = {"id": "c", "created": 1, "model": model}
    return [
        {**base, "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            **base,
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]


@pytest.fixture
def fake_litellm(monkeypatch):
    mod = types.ModuleType("litellm")
    mod.calls = []

    def get_llm_provider(model):
        provider = "anthropic" if str(model).startswith("claude") else "openai"
        return (model, provider, None, None)

    def completion(**kw):
        mod.calls.append(kw)
        if kw.get("stream"):
            return _FakeStream(_stream_chunks(kw["model"]))
        return _response(kw["model"])

    async def acompletion(**kw):
        mod.calls.append(kw)
        return _response(kw["model"])

    mod.get_llm_provider = get_llm_provider
    mod.completion = completion
    mod.acompletion = acompletion
    mod.answer_to_everything = 42
    monkeypatch.setitem(sys.modules, "litellm", mod)

    import parry_litellm

    monkeypatch.setattr(parry_litellm, "_default_guard", None, raising=False)
    return mod


def _ask(fn, **extra):
    return fn(
        model=extra.pop("model", "gpt-4o"), messages=[{"role": "user", "content": "hi"}], **extra
    )


def test_completion_is_guarded(fake_litellm):
    import parry_litellm

    guard = parry.install(cache=False)
    r = _ask(parry_litellm.completion)
    assert r.choices[0].message.content == "echo"
    assert guard.stats.requests == 1
    assert "openai:gpt-4o" in guard.stats.by_model
    assert len(fake_litellm.calls) == 1


def test_completion_caches(fake_litellm):
    import parry_litellm

    guard = parry.install()  # cache on
    _ask(parry_litellm.completion)
    _ask(parry_litellm.completion)  # identical → served from cache
    assert len(fake_litellm.calls) == 1
    assert guard.stats.cache_hits == 1


def test_provider_label_comes_from_litellm(fake_litellm):
    import parry_litellm

    guard = parry.install(cache=False)
    _ask(parry_litellm.completion, model="claude-sonnet-5-5")
    assert "anthropic:claude-sonnet-5-5" in guard.stats.by_model


def test_uses_a_default_guard_without_install(fake_litellm):
    import parry_litellm

    assert parry.get_guard() is None
    _ask(parry_litellm.completion)
    assert parry_litellm._guard().stats.requests == 1


def test_configure_sets_the_guard(fake_litellm):
    import parry_litellm

    g = parry.Parry(cache=False)
    parry_litellm.configure(g)
    _ask(parry_litellm.completion)
    assert g.stats.requests == 1


def test_streaming_is_metered(fake_litellm):
    import parry_litellm

    guard = parry.install(cache=False)
    stream = _ask(parry_litellm.completion, stream=True)
    chunks = list(stream)
    assert chunks[-1]["usage"]["completion_tokens"] == 5
    assert (guard.stats.input_tokens, guard.stats.output_tokens) == (10, 5)


async def test_acompletion_is_guarded(fake_litellm):
    import parry_litellm

    guard = parry.install(cache=False)
    r = await _ask(parry_litellm.acompletion)
    assert r.choices[0].message.content == "echo"
    assert guard.stats.requests == 1


def test_unknown_attribute_proxies_to_litellm(fake_litellm):
    import parry_litellm

    assert parry_litellm.answer_to_everything == 42


def test_importable_without_litellm_installed():
    # Importing the shim must not require litellm; only a call does.
    import parry_litellm

    assert hasattr(parry_litellm, "completion")
