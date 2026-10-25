"""``parry`` command line: inspect persisted spend and the pricing table.

parry stats --db parry.db
parry pricing --provider anthropic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from ._parry_native import Cost
from ._storage import SQLiteStore, start_of_today


def _stats(args: argparse.Namespace) -> int:
    if not Path(args.db).exists():
        print(f"parry: no database at {args.db}", file=sys.stderr)
        return 1
    store = SQLiteStore(args.db)
    if not store.ok:
        return 1
    life = store.spend()
    today = store.spend(start_of_today())
    rows = store.spend_by_model()
    store.close()
    if life is None or today is None:
        return 1
    if args.json:
        print(
            json.dumps(
                {
                    "lifetime": life.__dict__,
                    "today": today.__dict__,
                    "by_model": {k: v.__dict__ for k, v in rows},
                },
                indent=2,
            )
        )
        return 0
    print(
        f"lifetime  ${life.cost_usd:.4f}  {life.requests} requests  {life.input_tokens} in / {life.output_tokens} out"
    )
    print(f"today     ${today.cost_usd:.4f}  {today.requests} requests")
    if rows:
        print()
        for name, s in rows:
            print(f"  {name:<44} {s.requests:>7} req  ${s.cost_usd:>10.4f}")
    return 0


def _pricing(args: argparse.Namespace) -> int:
    rows = [r for r in Cost.pricing_table() if not args.provider or r["provider"] == args.provider]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'provider':<10} {'model':<28} {'$/M input':>10} {'$/M output':>11}")
    for r in rows:
        print(
            f"{r['provider']:<10} {r['model']:<28} {r['input_per_million']:>10.2f} {r['output_per_million']:>11.2f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="parry", description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument("--version", action="version", version=f"parry {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_stats = sub.add_parser("stats", help="show spend recorded in a parry SQLite database")
    p_stats.add_argument(
        "--db", default="parry.db", help="path to the database (default: parry.db)"
    )
    p_stats.add_argument("--json", action="store_true", help="machine-readable output")
    p_stats.set_defaults(func=_stats)

    p_price = sub.add_parser("pricing", help="show the built-in pricing table")
    p_price.add_argument("--provider", choices=["openai", "anthropic"])
    p_price.add_argument("--json", action="store_true")
    p_price.set_defaults(func=_pricing)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
