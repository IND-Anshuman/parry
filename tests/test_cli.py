from __future__ import annotations

import json

from conftest import chat

import parry
from parry.__main__ import main


def test_pricing_table(capsys):
    assert main(["pricing", "--provider", "anthropic"]) == 0
    out = capsys.readouterr().out
    assert "claude-opus-5-5" in out and "gpt-4o" not in out


def test_stats_from_database(tmp_path, capsys, mock, openai_client):
    db = tmp_path / "parry.db"
    parry.install(storage=db)
    chat(openai_client)
    chat(openai_client, content="two")
    parry.uninstall()

    assert main(["stats", "--db", str(db), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["lifetime"]["requests"] == 2
    assert report["by_model"]["openai:gpt-4o"]["input_tokens"] == 20

    assert main(["stats", "--db", str(db)]) == 0
    assert "openai:gpt-4o" in capsys.readouterr().out


def test_stats_missing_database(tmp_path, capsys):
    assert main(["stats", "--db", str(tmp_path / "nope.db")]) == 1
