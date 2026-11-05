"""Export a span per guarded LLM call using the OpenTelemetry GenAI semantic
conventions.

parry does not configure OpenTelemetry — you set up your own ``TracerProvider``
and exporter — it only emits spans. Turn it on once the guard exists::

    import parry
    from parry.integrations import otel

    parry.install()
    otel.instrument()            # spans now flow to your configured provider
    # ... your OpenAI / Anthropic / ... calls ...

Each accounted call (cache hits included) becomes one CLIENT span named
``"{operation} {model}"`` carrying ``gen_ai.*`` usage and request/response
attributes, plus a few ``parry.*`` extensions (cost, pricing tier, cache hit).

Only ``opentelemetry-api`` is needed to import this module; install
``opentelemetry-sdk`` (the ``parry[otel]`` extra) to actually record spans.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import __version__

if TYPE_CHECKING:
    from .._core import CallEvent, Parry

# GenAI operation-name values for the endpoints parry hooks.
_OPERATION = {
    "chat.completions": "chat",
    "responses": "chat",
    "messages": "chat",
    "beta.messages": "chat",
    "generate_content": "generate_content",
    "chat": "chat",
    "converse": "chat",
}


def _operation_name(endpoint: str) -> str:
    return _OPERATION.get(endpoint, "chat")


def _make_observer(tracer: Any) -> Any:
    from opentelemetry.trace import SpanKind, Status, StatusCode

    def observe(event: CallEvent) -> None:
        span = tracer.start_span(
            f"{_operation_name(event.operation)} {event.request_model}",
            kind=SpanKind.CLIENT,
            start_time=event.start_time_ns,
        )
        try:
            u = event.usage
            attrs: dict[str, Any] = {
                "gen_ai.operation.name": _operation_name(event.operation),
                "gen_ai.provider.name": event.provider,
                "gen_ai.request.model": event.request_model,
                "gen_ai.response.model": event.response_model,
                "gen_ai.usage.input_tokens": u.input,
                "gen_ai.usage.output_tokens": u.output,
                # parry extensions, namespaced so they never clash with GenAI.
                "parry.cost.usd": event.cost_usd,
                "parry.cost.estimated": event.estimated,
                "parry.pricing.tier": event.tier,
                "parry.cache.hit": event.cached,
            }
            if event.response_id:
                attrs["gen_ai.response.id"] = event.response_id
            if event.max_tokens is not None:
                attrs["gen_ai.request.max_tokens"] = event.max_tokens
            if u.cached_input:
                attrs["gen_ai.usage.cache_read.input_tokens"] = u.cached_input
            cache_creation = u.cache_write_5m + u.cache_write_1h
            if cache_creation:
                attrs["gen_ai.usage.cache_creation.input_tokens"] = cache_creation
            if event.saved_usd:
                attrs["parry.cost.saved_usd"] = event.saved_usd
            span.set_attributes(attrs)
            if event.error:
                span.set_status(Status(StatusCode.ERROR, event.error))
        finally:
            span.end(end_time=event.end_time_ns)

    return observe


class Instrumentation:
    """Handle returned by :func:`instrument`; call :meth:`uninstrument` to stop."""

    def __init__(self, guard: Parry, observer: Any) -> None:
        self._guard = guard
        self._observer = observer

    def uninstrument(self) -> None:
        self._guard.remove_observer(self._observer)


def instrument(guard: Parry | None = None, *, tracer: Any | None = None) -> Instrumentation:
    """Emit an OpenTelemetry span per guarded call.

    ``guard`` defaults to the process-wide guard (creating one if needed), so it
    pairs with ``parry.install()``. Pass ``tracer`` to use a specific tracer;
    otherwise parry fetches one from the globally configured provider.
    """
    if tracer is None:
        from opentelemetry import trace

        tracer = trace.get_tracer("parry", __version__)
    if guard is None:
        from .. import _ensure_global

        guard = _ensure_global()
    observer = guard.add_observer(_make_observer(tracer))
    return Instrumentation(guard, observer)
