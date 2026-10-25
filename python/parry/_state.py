"""Process-wide state shared by the guard and the patches."""

from __future__ import annotations

import threading
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._core import Parry

lock = threading.RLock()

# Guard installed with parry.install(); used when no narrower guard applies.
global_guard: Parry | None = None

# Guard entered with `with Parry(...):` in the current thread / task.
context_guard: ContextVar[Parry | None] = ContextVar("parry_context_guard", default=None)

# `with parry.bypass():` turns parry off for the enclosed calls.
bypassed: ContextVar[bool] = ContextVar("parry_bypassed", default=False)

CLIENT_ATTR = "_parry_guard"


def active_guard(resource: Any) -> Parry | None:
    """Resolve the guard for a call on an SDK resource.

    Order: bypass → client bound with ``parry.wrap`` → ``with Parry():`` → ``install()``.
    """
    if bypassed.get():
        return None
    client = getattr(resource, "_client", None)
    bound = getattr(client, CLIENT_ATTR, None) if client is not None else None
    if bound is not None:
        return bound
    return context_guard.get() or global_guard


# Number of SDK calls the hooks have handled in this context. Framework
# callbacks compare it before/after a run to avoid double counting.
intercepted: ContextVar[int] = ContextVar("parry_intercepted", default=0)


def note_intercept() -> None:
    intercepted.set(intercepted.get() + 1)
