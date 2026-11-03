#!/usr/bin/env python3
"""Sync parry's pricing table from LiteLLM's model-price database.

LiteLLM publishes ``model_prices_and_context_window.json`` (thousands of models,
with cache/batch tiers and per-model provenance). This tool maps those entries
into parry's pricing schema and writes the Rust source at
``src/pricing_generated.rs``. Hand-curated rows in ``src/cost.rs`` win on an
equal-length prefix, so a sync only ever *adds* breadth — it never overwrites a
model parry prices more accurately.

Fail closed: a model is skipped unless it has a finite, positive input *and*
output price. A half-priced model would silently undercharge, so we refuse it
(the rule LiteLLM-derived guards like floe-guard adopt).

    python scripts/sync_pricing.py                 # fetch the live LiteLLM table
    python scripts/sync_pricing.py --file x.json   # map a local snapshot
    python scripts/sync_pricing.py --check         # CI: fail if the file is stale
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

# parry prices text generation only; skip embeddings, images, audio, rerank, etc.
TEXT_MODES = {"chat", "completion", "responses"}

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "src" / "pricing_generated.rs"

_ABOVE_RE = re.compile(r"_above_(\d+)k_tokens$")
_PER_MILLION = 1_000_000.0


def _num(value: Any) -> float | None:
    """A finite, non-negative float, or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) and f >= 0.0 else None


def _per_million(value: Any) -> float | None:
    n = _num(value)
    return None if n is None else n * _PER_MILLION


def canonical_name(key: str) -> str:
    """Drop a single leading ``provider/`` segment so names match parry's
    lookup (``gemini/gemini-3-pro`` -> ``gemini-3-pro``). Bedrock's
    ``anthropic.*`` form is left for parry's own canonicalisation."""
    return key.split("/", 1)[1] if "/" in key else key


def map_entry(key: str, entry: dict[str, Any]) -> dict[str, Any] | None:
    """One LiteLLM entry -> a parry pricing row, or None if unpriceable / off-scope."""
    if not isinstance(entry, dict) or key == "sample_spec":
        return None
    mode = entry.get("mode")
    if mode is not None and mode not in TEXT_MODES:
        return None

    inp = _per_million(entry.get("input_cost_per_token"))
    out = _per_million(entry.get("output_cost_per_token"))
    if inp is None or out is None:  # fail closed
        return None

    row: dict[str, Any] = {
        "provider": str(entry.get("litellm_provider") or "unknown"),
        "model": canonical_name(key),
        "input": inp,
        "output": out,
        "long_threshold": None,
        "long_input": inp,
        "long_output": out,
        "cached_input": _per_million(entry.get("cache_read_input_token_cost")),
        "cache_write_5m": _per_million(entry.get("cache_creation_input_token_cost")),
        "cache_write_1h": None,
        "batch": None,
        "fast": None,
    }

    # Long-context tier: smallest "_above_Nk_tokens" override present.
    thresholds = sorted(
        int(m.group(1)) for k in entry if (m := _ABOVE_RE.search(k)) and k.startswith("input_cost")
    )
    if thresholds:
        n = thresholds[0]
        li = _per_million(entry.get(f"input_cost_per_token_above_{n}k_tokens"))
        lo = _per_million(entry.get(f"output_cost_per_token_above_{n}k_tokens"))
        if li is not None and lo is not None:
            row["long_threshold"] = n * 1000
            row["long_input"] = li
            row["long_output"] = lo

    # Batch tier -> a multiplier off the base input rate.
    batch_in = _num(entry.get("input_cost_per_token_batches"))
    base_in = _num(entry.get("input_cost_per_token"))
    if batch_in is not None and base_in:
        row["batch"] = round(batch_in / base_in, 6)

    return row


def _rust_f64(x: float) -> str:
    r = repr(float(x))
    # Rust needs a decimal point or exponent; repr() of a float always has one.
    return r


def _rust_opt_f64(x: float | None) -> str:
    return "None" if x is None else f"Some({_rust_f64(x)})"


def _rust_opt_u64(x: int | None) -> str:
    return "None" if x is None else f"Some({x})"


def render_row(row: dict[str, Any]) -> str:
    return (
        f'    crate::cost::ModelPricing {{ provider: {json.dumps(row["provider"])}, '
        f'model: {json.dumps(row["model"])}, pricing: crate::cost::Pricing::full('
        f'{_rust_f64(row["input"])}, {_rust_f64(row["output"])}, '
        f'{_rust_opt_u64(row["long_threshold"])}, {_rust_f64(row["long_input"])}, '
        f'{_rust_f64(row["long_output"])}, {_rust_opt_f64(row["cached_input"])}, '
        f'{_rust_opt_f64(row["cache_write_5m"])}, {_rust_opt_f64(row["cache_write_1h"])}, '
        f'{_rust_opt_f64(row["batch"])}, {_rust_opt_f64(row["fast"])}) }},'
    )


def build_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for key, entry in data.items():
        row = map_entry(key, entry)
        if row is not None:
            rows[row["model"]] = row  # dedup by canonical model name
    return sorted(rows.values(), key=lambda r: (r["provider"], r["model"]))


def render_block(rows: list[dict[str, Any]], source: str) -> str:
    body = "\n".join(render_row(r) for r in rows)
    note = f"    // {len(rows)} models synced from {source}"
    inner = f"{note}\n{body}" if body else note
    return f"    // SYNC:START\n{inner}\n    // SYNC:END"


def splice(existing: str, block: str) -> str:
    # Function replacement so model names with regex-special characters are
    # inserted literally.
    return re.sub(
        r" *// SYNC:START.*? *// SYNC:END",
        lambda _: block,
        existing,
        count=1,
        flags=re.DOTALL,
    )


def load(url: str | None, file: str | None) -> dict[str, Any]:
    if file:
        return json.loads(Path(file).read_text(encoding="utf-8"))
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (fixed https host)
        return json.loads(resp.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--url", default=LITELLM_URL, help="LiteLLM JSON URL (default: canonical)")
    ap.add_argument("--file", help="map a local JSON snapshot instead of fetching")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="generated Rust file to write")
    ap.add_argument("--check", action="store_true", help="exit 1 if --out is stale (CI)")
    args = ap.parse_args(argv)

    data = load(args.url, args.file)
    rows = build_rows(data)
    out = Path(args.out)
    block = render_block(rows, args.file or args.url)
    updated = splice(out.read_text(encoding="utf-8"), block)

    if args.check:
        if out.read_text(encoding="utf-8") != updated:
            print(f"parry: {out} is out of sync; run scripts/sync_pricing.py", file=sys.stderr)
            return 1
        print(f"parry: {out} is up to date ({len(rows)} synced models)")
        return 0

    out.write_text(updated, encoding="utf-8")
    print(f"parry: wrote {len(rows)} synced models to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
