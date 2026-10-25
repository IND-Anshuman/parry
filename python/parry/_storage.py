"""Optional SQLite persistence for the response cache and the spend ledger.

Persistence is opt-in (``Parry(storage="parry.db")``). Any SQLite failure
(corrupt file, read-only disk, locked database) disables the store with a
warning; parry keeps working from memory.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("parry")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    tokens     INTEGER NOT NULL DEFAULT 0,
    cost_usd   REAL NOT NULL DEFAULT 0,
    expires_at REAL
);
CREATE TABLE IF NOT EXISTS ledger (
    ts            REAL NOT NULL,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd      REAL,
    estimated     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ledger_ts ON ledger (ts);
"""


@dataclass(frozen=True)
class SpendSummary:
    requests: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class SQLiteStore:
    def __init__(self, path: str | Path, *, max_entries: int = 100_000) -> None:
        self.path = str(path)
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._writes = 0
        try:
            conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
            conn.executescript(_SCHEMA)
            with contextlib.suppress(sqlite3.DatabaseError):
                conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()
            self._conn = conn
        except sqlite3.DatabaseError as exc:
            log.warning(
                "parry: storage %s unavailable (%s); continuing in memory only", self.path, exc
            )

    @property
    def ok(self) -> bool:
        return self._conn is not None

    def _fail(self, exc: Exception) -> None:
        log.warning("parry: storage %s failed (%s); continuing in memory only", self.path, exc)
        conn, self._conn = self._conn, None
        if conn is not None:
            with contextlib.suppress(sqlite3.Error):
                conn.close()

    # --- cache -------------------------------------------------------------

    def get(self, key: str) -> tuple[str, int, float] | None:
        with self._lock:
            if self._conn is None:
                return None
            try:
                row = self._conn.execute(
                    "SELECT value, tokens, cost_usd, expires_at FROM cache WHERE key = ?", (key,)
                ).fetchone()
                if row is None:
                    return None
                if row[3] is not None and row[3] < time.time():
                    self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    self._conn.commit()
                    return None
                return row[0], int(row[1]), float(row[2])
            except sqlite3.Error as exc:
                self._fail(exc)
                return None

    def put(self, key: str, value: str, tokens: int, cost_usd: float, ttl: float | None) -> None:
        expires = time.time() + ttl if ttl else None
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, tokens, cost_usd, expires_at) VALUES (?, ?, ?, ?, ?)",
                    (key, value, tokens, cost_usd, expires),
                )
                self._writes += 1
                if self._writes % 256 == 0:
                    self._evict()
                self._conn.commit()
            except sqlite3.Error as exc:
                self._fail(exc)

    def delete(self, key: str) -> None:
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._conn.commit()
            except sqlite3.Error as exc:
                self._fail(exc)

    def clear_cache(self) -> None:
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.execute("DELETE FROM cache")
                self._conn.commit()
            except sqlite3.Error as exc:
                self._fail(exc)

    def _evict(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?", (time.time(),)
        )
        (count,) = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
        excess = count - self.max_entries
        if excess > 0:
            self._conn.execute(
                "DELETE FROM cache WHERE key IN (SELECT key FROM cache ORDER BY expires_at ASC LIMIT ?)",
                (excess,),
            )

    # --- ledger ------------------------------------------------------------

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None,
        estimated: bool,
    ) -> None:
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.execute(
                    "INSERT INTO ledger VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        time.time(),
                        provider,
                        model,
                        input_tokens,
                        output_tokens,
                        cost_usd,
                        int(estimated),
                    ),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                self._fail(exc)

    def spend(self, since: float | None = None) -> SpendSummary | None:
        with self._lock:
            if self._conn is None:
                return None
            try:
                row = self._conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0),"
                    " COALESCE(SUM(cost_usd), 0) FROM ledger WHERE ts >= ?",
                    (since or 0.0,),
                ).fetchone()
            except sqlite3.Error as exc:
                self._fail(exc)
                return None
        return SpendSummary(int(row[0]), int(row[1]), int(row[2]), float(row[3]))

    def spend_by_model(self, since: float | None = None) -> list[tuple[str, SpendSummary]]:
        with self._lock:
            if self._conn is None:
                return []
            try:
                rows = self._conn.execute(
                    "SELECT provider || ':' || model, COUNT(*), SUM(input_tokens), SUM(output_tokens),"
                    " COALESCE(SUM(cost_usd), 0) FROM ledger WHERE ts >= ?"
                    " GROUP BY provider, model ORDER BY 5 DESC",
                    (since or 0.0,),
                ).fetchall()
            except sqlite3.Error as exc:
                self._fail(exc)
                return []
        return [(r[0], SpendSummary(int(r[1]), int(r[2]), int(r[3]), float(r[4]))) for r in rows]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


def start_of_today() -> float:
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
