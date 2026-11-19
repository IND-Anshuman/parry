"""The local stats dashboard (parry stats --serve)."""

from __future__ import annotations

import json
import threading
import urllib.request

from parry._dashboard import doctor_payload, make_handler, serve, stats_payload
from parry._storage import SQLiteStore


def _seed(path):
    store = SQLiteStore(path)
    store.record("openai", "gpt-4o", 100, 50, 0.004, False)
    store.record("openai", "gpt-4o", 100, 50, 0.004, False)
    store.record("anthropic", "mystery", 80, 40, None, True)  # unpriced + estimated
    return store


# --- payloads -----------------------------------------------------------------


def test_stats_payload_aggregates(tmp_path):
    store = _seed(tmp_path / "p.db")
    payload = stats_payload(store)
    store.close()
    assert payload["lifetime"]["requests"] == 3
    models = {row["model"]: row for row in payload["by_model"]}
    assert models["openai:gpt-4o"]["requests"] == 2
    assert models["openai:gpt-4o"]["cost_usd"] == 0.008
    assert "anthropic:mystery" in models


def test_doctor_payload_flags_unpriced(tmp_path):
    store = _seed(tmp_path / "p.db")
    payload = doctor_payload(store)
    store.close()
    codes = {f["code"] for f in payload["findings"]}
    assert "cost.unpriced" in codes
    assert payload["ok"] is False


def test_doctor_payload_empty_ledger(tmp_path):
    store = SQLiteStore(tmp_path / "empty.db")
    payload = doctor_payload(store)
    store.close()
    assert payload["findings"][0]["code"] == "ledger.empty"


# --- HTTP ---------------------------------------------------------------------


def test_dashboard_serves_html_and_json(tmp_path):
    db = tmp_path / "p.db"
    _seed(db).close()
    server = serve(str(db), "127.0.0.1", 0)  # ephemeral port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        base = f"http://{host}:{port}"

        with urllib.request.urlopen(f"{base}/") as r:
            assert r.status == 200
            body = r.read().decode()
            assert "<title>parry stats</title>" in body and "api/stats" in body

        with urllib.request.urlopen(f"{base}/api/stats") as r:
            stats = json.loads(r.read())
        assert stats["lifetime"]["requests"] == 3

        with urllib.request.urlopen(f"{base}/api/doctor") as r:
            doc = json.loads(r.read())
        assert doc["ok"] is False
    finally:
        server.shutdown()
        server.server_close()


def test_dashboard_unknown_path_is_404(tmp_path):
    db = tmp_path / "p.db"
    _seed(db).close()
    server = serve(str(db), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        try:
            urllib.request.urlopen(f"http://{host}:{port}/nope")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_make_handler_is_a_handler_class(tmp_path):
    from http.server import BaseHTTPRequestHandler

    handler = make_handler(str(tmp_path / "p.db"))
    assert issubclass(handler, BaseHTTPRequestHandler)
