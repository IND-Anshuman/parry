"""OpenTelemetry GenAI span export (parry.integrations.otel)."""

from __future__ import annotations

import pytest
from conftest import chat, message

import parry
from parry import Parry

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from parry.integrations import otel  # noqa: E402


@pytest.fixture
def tracer_and_spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def _attrs(span):
    return dict(span.attributes)


def test_span_emitted_with_genai_attributes(mock, openai_client, tracer_and_spans):
    tracer, exporter = tracer_and_spans
    guard = parry.install(cache=False)
    handle = otel.instrument(guard, tracer=tracer)
    try:
        chat(openai_client)
    finally:
        handle.uninstrument()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat gpt-4o"
    a = _attrs(span)
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.provider.name"] == "openai"
    assert a["gen_ai.request.model"] == "gpt-4o"
    assert a["gen_ai.usage.input_tokens"] == 10
    assert a["gen_ai.usage.output_tokens"] == 5
    assert a["parry.pricing.tier"] == "standard"
    assert a["parry.cache.hit"] is False
    assert a["parry.cost.usd"] > 0
    assert span.start_time < span.end_time


def test_cache_hit_emits_a_cached_span(mock, openai_client, tracer_and_spans):
    tracer, exporter = tracer_and_spans
    guard = parry.install()
    handle = otel.instrument(guard, tracer=tracer)
    try:
        chat(openai_client)  # miss → provider call
        chat(openai_client)  # identical → cache hit
    finally:
        handle.uninstrument()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    hits = [s for s in spans if _attrs(s)["parry.cache.hit"]]
    assert len(hits) == 1
    a = _attrs(hits[0])
    assert a["parry.cost.usd"] == 0.0
    assert a["parry.cost.saved_usd"] > 0


def test_anthropic_cache_tokens_land_on_span(mock, anthropic_client, tracer_and_spans):
    tracer, exporter = tracer_and_spans
    guard = parry.install(cache=False)
    handle = otel.instrument(guard, tracer=tracer)
    try:
        mock.state.prompt_tokens = 100
        message(anthropic_client)
    finally:
        handle.uninstrument()

    a = _attrs(exporter.get_finished_spans()[0])
    assert a["gen_ai.provider.name"] == "anthropic"
    assert a["gen_ai.usage.input_tokens"] == 100


def test_streaming_call_emits_one_span(mock, openai_client, tracer_and_spans):
    tracer, exporter = tracer_and_spans
    guard = parry.install(cache=False)
    handle = otel.instrument(guard, tracer=tracer)
    try:
        list(chat(openai_client, stream=True, stream_options={"include_usage": True}))
    finally:
        handle.uninstrument()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    a = _attrs(spans[0])
    assert a["gen_ai.usage.input_tokens"] == 10
    assert a["gen_ai.usage.output_tokens"] == 5


def test_uninstrument_stops_spans(mock, openai_client, tracer_and_spans):
    tracer, exporter = tracer_and_spans
    guard = parry.install(cache=False)
    otel.instrument(guard, tracer=tracer).uninstrument()
    chat(openai_client)
    assert exporter.get_finished_spans() == ()


def test_observer_error_never_breaks_the_call(mock, openai_client, caplog):
    guard = parry.install(cache=False)

    def boom(_event):
        raise RuntimeError("observer kaboom")

    guard.add_observer(boom)
    r = chat(openai_client)
    assert r.choices[0].message.content == "echo: hello"
    assert any("observer raised" in rec.getMessage() for rec in caplog.records)


def test_instrument_without_explicit_guard_uses_global(mock, openai_client, tracer_and_spans):
    tracer, exporter = tracer_and_spans
    parry.install(cache=False)
    handle = otel.instrument(tracer=tracer)  # no guard → the installed one
    try:
        chat(openai_client)
    finally:
        handle.uninstrument()
    assert len(exporter.get_finished_spans()) == 1


def test_max_tokens_attribute(mock, openai_client, tracer_and_spans):
    tracer, exporter = tracer_and_spans
    guard = Parry(cache=False)
    handle = otel.instrument(guard, tracer=tracer)
    with guard:
        try:
            chat(openai_client, max_completion_tokens=256)
        finally:
            handle.uninstrument()
    a = _attrs(exporter.get_finished_spans()[0])
    assert a["gen_ai.request.max_tokens"] == 256
