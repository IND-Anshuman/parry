# parry

**Zero-config rate limiting, caching, and cost control for LLM calls.**

One line to install. No rewrites. Works with OpenAI, Anthropic, LangChain and LlamaIndex.

[![CI](https://github.com/IND-Anshuman/parry/actions/workflows/ci.yml/badge.svg)](https://github.com/IND-Anshuman/parry/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Documentation: [www.anshumanbuilds.dev/parry](https://www.anshumanbuilds.dev/parry/)**

Drop parry in front of any OpenAI or Anthropic client and get response caching,
per-model rate limiting, cost tracking, budget limits and retry with backoff.
You don't configure anything or change your call sites, and it doesn't lock you
into its own SDK. The hot path (hashing, LRU cache, token buckets, the pricing
ledger) is Rust, exposed through [PyO3](https://pyo3.rs).

## Install

```bash
pip install llm-parry
```

The package is published as `llm-parry`; you import it as `parry`. Wheels are `abi3`, so one wheel per platform covers Python 3.10 – 3.14.

From source (needs a Rust toolchain):

```bash
pip install "git+https://github.com/IND-Anshuman/parry.git"
```

## Quickstart

```python
import parry
import openai

parry.install()  # ← this is the only line you add

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
# ↑ cached, rate-limited, cost-tracked, retried on transient failure

print(parry.stats())
# ParryStats(cache_hits=0, cache_misses=1, tokens_cached=0, total_cost_usd=0.000043, ...)
```

Run the same request again and it comes back from memory: no network round
trip and no cost.

## What you get for free

- ✅ **Response caching**: identical requests are served from an in-memory LRU (10,000 entries, 1 h TTL by default).
- ✅ **Per-model rate limiting**: a token bucket per `provider:model`. The defaults match the providers' published tier-1 limits.
- ✅ **Cost tracking**: per session and per model, and per day and lifetime when you turn on SQLite storage.
- ✅ **Hard budget limits**: `max_cost_usd=10` raises `parry.BudgetExceeded` *before* the request that would cross the limit is sent.
- ✅ **Retry with exponential backoff and jitter** on 429s, timeouts, connection errors, 5xx and overload. It honours `Retry-After` and never retries 400/401/403/404.
- ✅ **Streaming**: streams pass through untouched, are metered, and the assembled result is cached for the next non-streaming call.
- ✅ **Sync and async clients**, `.with_raw_response`, the OpenAI Responses API, and Anthropic `beta.messages`.
- ✅ **LangChain and LlamaIndex**: both work through `install()`, and both also have callback handlers.

## Before / after

Before, every script reinvents the same thing:

```python
for attempt in range(5):
    try:
        if prompt in my_cache:
            return my_cache[prompt]
        time.sleep(my_rate_limiter.wait_time("gpt-4o"))
        r = client.chat.completions.create(model="gpt-4o", messages=msgs)
        my_spend += r.usage.prompt_tokens * 2.5e-6 + r.usage.completion_tokens * 1e-5
        if my_spend > BUDGET:
            raise RuntimeError("over budget")
        my_cache[prompt] = r
        return r
    except openai.RateLimitError:
        time.sleep(2 ** attempt)
```

After:

```python
parry.install(max_cost_usd=10)
r = client.chat.completions.create(model="gpt-4o", messages=msgs)
```

## When to use parry

1. **Iterating on prompts and pipelines.** Re-running a notebook or an eval
   script stops re-paying for every unchanged call.
2. **Batch jobs that hit 429s.** The token bucket spaces requests out *before*
   the provider rejects them, and retries absorb the 429s that still happen.
3. **Agents and RAG apps with a spend ceiling.** `max_cost_usd` is a hard stop,
   and `parry.stats()` tells you where the money went, model by model.

## When NOT to use parry

- **Multi-process or multi-host deployments.** The cache and rate limiter
  live in one process. Use a shared cache and an API gateway instead.
- **High-concurrency distributed systems.** Limits are per process, not
  coordinated across replicas.
- **When you need sampled outputs to vary.** Identical requests return the
  same cached answer even at `temperature=1`. Turn caching off with
  `parry.install(cache=False)` or `with parry.bypass():`.

## Configuration

```python
from parry import Parry

guard = Parry(
    cache=True,              # response cache
    rate_limit=True,         # per-model token buckets
    cost_tracking=True,      # accumulate spend
    retry=True,              # retry transient failures
    max_cost_usd=10.0,       # hard stop (default: None)
    cache_ttl=3600,          # seconds (None = no expiry, still size-bounded)
    cache_size=10_000,       # entries; the cache is always bounded
    on_rate_limit="wait",    # or "raise" → parry.RateLimited
    max_wait=60.0,           # longest parry will wait for a rate-limit slot
    max_attempts=3,          # total attempts per request, including the first
    storage="parry.db",      # optional SQLite persistence (cache + ledger)
)
guard.set_rate_limit("gpt-4o", requests_per_minute=60)
guard.set_rate_limit("gpt-4o-mini", requests_per_minute=500, tokens_per_minute=200_000)
guard.set_pricing("my-finetune", input_per_million=3.0, output_per_million=12.0)

parry.install(guard)        # whole process
with guard:                 # ...or just this block
    ...
guard.wrap(client)          # ...or just this client

print(guard.stats.summary())
```

Every option, default and failure mode is documented in
[docs/configuration.md](docs/configuration.md).

## Integrations

**LangChain.** `ChatOpenAI` and `ChatAnthropic` call the official SDKs
underneath, so `parry.install()` covers them, caching included:

```python
import parry
from langchain_openai import ChatOpenAI

parry.install()
llm = ChatOpenAI(model="gpt-4o-mini")
llm.invoke("What is a token bucket?")   # second identical call is a cache hit
```

If you'd rather not hook the SDK globally, add the callback. It does rate
limiting, budget enforcement and cost tracking, but not caching, because
LangChain callbacks can't return a response:

```python
from parry import Parry
from parry.integrations.langchain import ParryCallback

guard = Parry(max_cost_usd=5)
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[ParryCallback(guard)])
```

**LlamaIndex.**

```python
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from parry.integrations.llamaindex import ParryHandler

Settings.callback_manager = CallbackManager([ParryHandler(guard)])
```

More detail is in [docs/integrations.md](docs/integrations.md).

## Benchmarks

Measured with [`benches/overhead_bench.py`](benches/overhead_bench.py), which drives the
real OpenAI SDK through an in-memory HTTP transport. No network is involved,
so the difference between rows is parry's own cost. Release build:

<!-- BENCH:START -->
| Path | Time per call | parry overhead |
|---|---|---|
| Unguarded SDK call (baseline, in-memory transport) | 604 µs | — |
| Guarded, cache miss | 669 µs | +65 µs (0.065 ms) |
| Guarded, cache hit (no HTTP at all) | 10 µs | 0.02× the baseline |

_Linux x86_64, Python 3.13.5, openai 2.54.0, median of 5 batches of 1000 calls._
<!-- BENCH:END -->

## How it works

`parry.install()` replaces `create` on the SDK resource classes
(`openai.resources.chat.completions.Completions`, `...Responses`,
`anthropic.resources.messages.Messages` and their async twins) with a thin
hook. On each call the hook finds the active guard and runs:

`cache lookup → budget pre-check → rate-limit acquire → original create (with retry) → record cost → store in cache`

With no active guard, the hook calls the original method directly. See
[docs/architecture.md](docs/architecture.md).

## Links

- Documentation: <https://www.anshumanbuilds.dev/parry/>
- Source: <https://github.com/IND-Anshuman/parry>
- Issues and pricing updates: <https://github.com/IND-Anshuman/parry/issues>
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT. See [LICENSE](LICENSE).
