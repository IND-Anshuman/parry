"""PII detection and redaction.

parry uses :meth:`Redactor.detect` to keep personal data out of the *persistent*
cache: a response (or its request) that matches is cached in memory for the
process but never written to the SQLite store. :meth:`Redactor.redact` is a
public utility for scrubbing text yourself; parry never rewrites what is sent to
the provider or returned to the caller.

Patterns are deliberately broad — a false positive only means "don't persist to
disk", which is the safe direction. The default set covers common identifiers;
add your own with ``extra={"name": r"regex"}``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

# Linear-time patterns (no nested quantifiers over overlapping classes), so they
# are safe to run on untrusted text.
_DEFAULT_PATTERNS: dict[str, str] = {
    "email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[ \-]\d{4}[ \-]\d{4}[ \-]\d{1,4}\b|\b\d{13,16}\b",
    # Separators are required, so bare digit runs (unix timestamps, ids) don't
    # read as phone numbers — a formatted number is the realistic PII case.
    "phone": r"(?:\+\d{1,3}[ \-.])?\(?\d{3}\)?[ \-.]\d{3}[ \-.]\d{4}\b",
    "ipv4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "api_key": r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{16,}\b",
}


class Redactor:
    """Detects and redacts PII with a configurable set of named regex patterns.

    ``patterns`` selects which defaults to use (``None`` = all of them); ``extra``
    adds or overrides patterns by name. Each match redacts to ``[REDACTED:name]``.
    """

    DEFAULTS = tuple(_DEFAULT_PATTERNS)

    def __init__(
        self,
        patterns: Iterable[str] | None = None,
        *,
        extra: Mapping[str, str] | None = None,
    ) -> None:
        names = list(_DEFAULT_PATTERNS) if patterns is None else list(patterns)
        compiled: dict[str, re.Pattern[str]] = {}
        for name in names:
            if name not in _DEFAULT_PATTERNS:
                raise ValueError(f"unknown PII pattern {name!r}; add it via extra=")
            compiled[name] = re.compile(_DEFAULT_PATTERNS[name])
        for name, pattern in (extra or {}).items():
            compiled[name] = re.compile(pattern)
        if not compiled:
            raise ValueError("Redactor needs at least one pattern")
        self._patterns = compiled

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._patterns)

    def detect(self, text: str) -> bool:
        """True if any pattern matches ``text``."""
        return any(p.search(text) for p in self._patterns.values())

    def redact(self, text: str) -> tuple[str, int]:
        """Replace every match with ``[REDACTED:name]``; returns (text, hits)."""
        hits = 0
        for name, pattern in self._patterns.items():
            text, n = pattern.subn(f"[REDACTED:{name}]", text)
            hits += n
        return text, hits
