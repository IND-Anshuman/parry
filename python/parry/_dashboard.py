"""A tiny local dashboard for a parry SQLite ledger (``parry stats --serve``).

Read-only, standard-library only, and bound to localhost by default. It serves
aggregates — spend, per-model totals and the doctor's findings — never request
or response content. This is a developer convenience next to ``parry stats`` and
``parry doctor``, not a hosted observability service.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ._doctor import Report, diagnose_ledger
from ._storage import SQLiteStore, start_of_today


def stats_payload(store: SQLiteStore) -> dict[str, Any]:
    life = store.spend()
    today = store.spend(start_of_today())
    by_model = store.spend_by_model()
    return {
        "lifetime": life.__dict__ if life else None,
        "today": today.__dict__ if today else None,
        "by_model": [{"model": name, **summary.__dict__} for name, summary in by_model],
    }


def doctor_payload(store: SQLiteStore) -> dict[str, Any]:
    health = store.ledger_health()
    by_model = store.spend_by_model()
    if health is None:
        return {"ok": True, "worst": "ok", "findings": []}
    return Report(diagnose_ledger(health, by_model)).as_dict()


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>parry stats</title>
<style>
  :root { color-scheme: light dark; --bg:#fff; --fg:#111; --muted:#666; --line:#e3e3e3;
          --card:#fafafa; --warn:#b4532a; --info:#2a6db4; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#141414; --fg:#eee; --muted:#9a9a9a; --line:#2a2a2a; --card:#1d1d1d;
            --warn:#e08a5a; --info:#6ab0ff; } }
  body { font:15px/1.5 system-ui,-apple-system,sans-serif; margin:0; background:var(--bg); color:var(--fg); }
  main { max-width:860px; margin:0 auto; padding:24px 16px 64px; }
  h1 { font-size:20px; margin:0 0 2px; } .sub { color:var(--muted); font-size:13px; margin-bottom:20px; }
  .cards { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:24px; }
  .card { flex:1 1 160px; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
  .card .k { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
  .card .v { font-size:24px; font-weight:600; margin-top:4px; }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  th,td { text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); }
  td.n,th.n { text-align:right; font-variant-numeric:tabular-nums; }
  h2 { font-size:15px; margin:28px 0 8px; }
  .finding { padding:9px 12px; border:1px solid var(--line); border-radius:8px; margin-bottom:8px; }
  .finding.warn { border-color:var(--warn); } .finding .tag { font-weight:600; }
  .finding.warn .tag { color:var(--warn); } .finding.info .tag { color:var(--info); }
  .finding .hint { color:var(--muted); font-size:13px; margin-top:3px; }
  button { font:inherit; padding:5px 12px; border:1px solid var(--line); border-radius:7px;
           background:var(--card); color:var(--fg); cursor:pointer; }
  .muted { color:var(--muted); }
</style>
</head>
<body><main>
  <h1>parry</h1>
  <div class="sub">local ledger &middot; <span id="db"></span> &middot;
    <button onclick="load()">Refresh</button></div>
  <div class="cards" id="cards"></div>
  <h2>By model</h2>
  <table><thead><tr><th>Model</th><th class="n">Requests</th>
    <th class="n">Input</th><th class="n">Output</th><th class="n">Cost</th></tr></thead>
    <tbody id="bymodel"></tbody></table>
  <h2>Doctor</h2>
  <div id="doctor"></div>
</main>
<script>
const $ = (id) => document.getElementById(id);
const money = (n) => "$" + (n || 0).toFixed(4);
const num = (n) => (n || 0).toLocaleString();
async function load() {
  const [s, d] = await Promise.all([
    fetch("api/stats").then(r => r.json()),
    fetch("api/doctor").then(r => r.json()),
  ]);
  const life = s.lifetime || {requests:0,input_tokens:0,output_tokens:0,cost_usd:0};
  const today = s.today || {cost_usd:0,requests:0};
  $("cards").innerHTML = [
    ["Lifetime spend", money(life.cost_usd)],
    ["Today", money(today.cost_usd)],
    ["Requests", num(life.requests)],
    ["Tokens", num(life.input_tokens + life.output_tokens)],
  ].map(([k,v]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  $("bymodel").innerHTML = (s.by_model || []).map(m =>
    `<tr><td>${m.model}</td><td class="n">${num(m.requests)}</td>
     <td class="n">${num(m.input_tokens)}</td><td class="n">${num(m.output_tokens)}</td>
     <td class="n">${money(m.cost_usd)}</td></tr>`).join("")
    || `<tr><td colspan="5" class="muted">No requests recorded yet.</td></tr>`;
  $("doctor").innerHTML = (d.findings || []).map(f =>
    `<div class="finding ${f.level}"><span class="tag">[${f.level}]</span> ${f.message}
     ${f.hint ? `<div class="hint">${f.hint}</div>` : ""}</div>`).join("");
}
load();
setInterval(load, 10000);
</script>
</body></html>"""


def make_handler(db_path: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:
            pass  # keep the terminal quiet

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict[str, Any]) -> None:
            self._send(200, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/":
                self._send(200, DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
                return
            if path in ("/api/stats", "/api/doctor"):
                store = SQLiteStore(db_path)
                if not store.ok:
                    self._send(503, b'{"error":"ledger unavailable"}', "application/json")
                    return
                try:
                    payload = (
                        stats_payload(store) if path == "/api/stats" else doctor_payload(store)
                    )
                finally:
                    store.close()
                self._json(payload)
                return
            self._send(404, b'{"error":"not found"}', "application/json")

    return Handler


def serve(db_path: str, host: str = "127.0.0.1", port: int = 8787) -> ThreadingHTTPServer:
    """Build (but do not start) a dashboard server. Caller runs ``serve_forever``."""
    server = ThreadingHTTPServer((host, port), make_handler(db_path))
    server.daemon_threads = True
    return server
