"""Guard an AWS Bedrock (boto3 ``bedrock-runtime``) client.

boto3 builds a client's methods dynamically per instance, so ``parry.install()``
cannot hook them the way it hooks the SDK classes. Guard a client explicitly
instead::

    import boto3, parry
    from parry.integrations import bedrock

    client = bedrock.guard(boto3.client("bedrock-runtime"))
    client.converse(modelId="...", messages=[...])   # metered + rate-limited

The non-streaming ``converse`` call is routed through the guard for metering,
rate limiting, the budget and retry. Responses are plain dicts, so they are not
cached, and ``converse_stream`` is left untouched in this release.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .._adapters import BEDROCK_CONVERSE


def _drop_self(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Adapt an already-bound client method to the ``(resource, **kwargs)``
    shape ``Parry._invoke`` calls originals with."""

    def original(_resource: Any, **kwargs: Any) -> Any:
        return fn(**kwargs)

    return original


def guard(client: Any, guard: Any | None = None) -> Any:
    """Route ``client.converse`` through a parry guard. Returns the client.

    ``guard`` may be a :class:`~parry.Parry`; with ``None`` the guard active at
    call time (``parry.install()`` or a ``with Parry():`` block) is used, so a
    client guarded once keeps working as the active guard changes. Calling this
    twice on the same client is a no-op.
    """
    from .. import get_guard

    bound = getattr(client, "converse", None)
    if bound is None or getattr(bound, "__parry_original__", None) is not None:
        return client

    def converse(**kwargs: Any) -> Any:
        active = guard or get_guard()
        if active is None:
            return bound(**kwargs)
        return active._invoke(BEDROCK_CONVERSE, _drop_self(bound), client, (), kwargs)

    converse.__parry_original__ = bound  # type: ignore[attr-defined]
    client.converse = converse
    return client


def unguard(client: Any) -> Any:
    """Undo :func:`guard` on a client."""
    bound = getattr(client, "converse", None)
    original = getattr(bound, "__parry_original__", None)
    if original is not None:
        client.converse = original
    return client
