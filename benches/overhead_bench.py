"""Measure what parry adds to an LLM call.

The real OpenAI SDK is used with an in-memory HTTP transport, so no network
is involved and the numbers isolate SDK + parry work:

- **unguarded**: a plain SDK call (baseline)
- **guarded, cache miss**: hashing, rate limiter, cost ledger, response
  serialisation into the cache, on top of the SDK call
- **guarded, cache hit**: the whole call when parry answers from cache

Run:  python benches/overhead_bench.py [--n 2000] [--update-readme]
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
from pathlib import Path

import httpx
import openai

import parry
from parry import Parry

BODY = json.dumps(
    {
        "id": "chatcmpl-bench",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "A token bucket refills at a fixed rate.",
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 42, "completion_tokens": 9, "total_tokens": 51},
    }
).encode()

MESSAGES = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Explain token buckets in one sentence."},
]


def make_client() -> openai.OpenAI:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=BODY, headers={"content-type": "application/json"}
        )
    )
    return openai.OpenAI(
        api_key="bench",
        base_url="http://bench.local/v1",
        max_retries=0,
        http_client=httpx.Client(transport=transport),
    )


def timed(fn, n: int) -> float:
    """Median per-call time in microseconds over 5 batches."""
    for _ in range(min(200, n)):  # warm up
        fn(0)
    batches = []
    per = max(1, n // 5)
    for b in range(5):
        start = time.perf_counter()
        for i in range(per):
            fn(b * per + i)
        batches.append((time.perf_counter() - start) / per * 1e6)
    return statistics.median(batches)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args()

    client = make_client()

    def unguarded(i: int) -> None:
        client.chat.completions.create(
            model="gpt-4o", messages=[*MESSAGES, {"role": "user", "content": str(i)}]
        )

    base = timed(unguarded, args.n)

    guard = Parry()
    guard.set_rate_limit("gpt-4o", requests_per_minute=4_000_000_000)
    parry.install(guard)

    counter = iter(range(10**9))

    def miss(i: int) -> None:
        client.chat.completions.create(
            model="gpt-4o", messages=[*MESSAGES, {"role": "user", "content": f"q{next(counter)}"}]
        )

    def hit(i: int) -> None:
        client.chat.completions.create(model="gpt-4o", messages=MESSAGES)

    miss_t = timed(miss, args.n)
    hit(0)
    hit_t = timed(hit, args.n)
    parry.uninstall()

    rows = [
        ("Unguarded SDK call (baseline, in-memory transport)", f"{base:.0f} µs", "—"),
        (
            "Guarded, cache miss",
            f"{miss_t:.0f} µs",
            f"+{miss_t - base:.0f} µs ({(miss_t - base) / 1000:.3f} ms)",
        ),
        (
            "Guarded, cache hit (no HTTP at all)",
            f"{hit_t:.0f} µs",
            f"{hit_t / base:.2f}× the baseline",
        ),
    ]
    env = f"{platform.system()} {platform.machine()}, Python {platform.python_version()}, openai {openai.__version__}"
    table = "\n".join(
        ["| Path | Time per call | parry overhead |", "|---|---|---|"]
        + [f"| {a} | {b} | {c} |" for a, b, c in rows]
        + ["", f"_{env}, median of 5 batches of {args.n // 5} calls._"]
    )
    print(table)

    if args.update_readme:
        readme = Path(__file__).resolve().parents[1] / "README.md"
        text = readme.read_text(encoding="utf-8")
        new = re.sub(
            r"<!-- BENCH:START -->.*<!-- BENCH:END -->",
            f"<!-- BENCH:START -->\n{table}\n<!-- BENCH:END -->",
            text,
            flags=re.S,
        )
        readme.write_text(new, encoding="utf-8")
        print(f"\nupdated {readme}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
