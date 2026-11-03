from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import sync_pricing as sp  # noqa: E402  (path set above)

SAMPLE = ROOT / "scripts" / "testdata" / "litellm_sample.json"


def _rows() -> dict[str, dict]:
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    return {r["model"]: r for r in sp.build_rows(data)}


def test_fail_closed_and_mode_filtering():
    # The embedding model (off-scope mode), the half-priced model (no output
    # cost) and LiteLLM's sample_spec are all dropped.
    assert set(_rows()) == {"gpt-4o", "gemini-3-pro"}


def test_openai_mapping_with_cache_and_batch():
    g = _rows()["gpt-4o"]
    assert g["provider"] == "openai"
    assert g["input"] == pytest.approx(2.5)
    assert g["output"] == pytest.approx(10.0)
    assert g["cached_input"] == pytest.approx(1.25)
    assert g["batch"] == pytest.approx(0.5)  # 1.25e-6 / 2.5e-6
    assert g["long_threshold"] is None


def test_gemini_long_context_and_provider_prefix_strip():
    g = _rows()["gemini-3-pro"]  # the "gemini/" prefix is stripped
    assert g["input"] == pytest.approx(2.0)
    assert g["output"] == pytest.approx(12.0)
    assert g["long_threshold"] == 200_000
    assert g["long_input"] == pytest.approx(4.0)
    assert g["long_output"] == pytest.approx(24.0)


def test_rendered_block_is_valid_rust():
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    text = sp.render_block(sp.build_rows(data), "sample")
    assert "// SYNC:START" in text and "// SYNC:END" in text
    assert "crate::cost::Pricing::full(" in text
    assert '"gpt-4o"' in text
    assert "Some(200000)" in text  # gemini long-context threshold


def test_check_mode_detects_staleness(tmp_path):
    stub = "// head\n    // SYNC:START\n    // SYNC:END\n// tail\n"
    out = tmp_path / "gen.rs"
    out.write_text(stub, encoding="utf-8")
    # Write, then --check passes; reset to the stub, then --check fails.
    assert sp.main(["--file", str(SAMPLE), "--out", str(out)]) == 0
    assert sp.main(["--file", str(SAMPLE), "--out", str(out), "--check"]) == 0
    out.write_text(stub, encoding="utf-8")
    assert sp.main(["--file", str(SAMPLE), "--out", str(out), "--check"]) == 1
