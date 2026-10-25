# Configuration

Everything has a default. You only need this page when the defaults don't
fit.

## Ways to activate a guard

```python
import parry
from parry import Parry

parry.install()                       # global guard with defaults
parry.install(max_cost_usd=10)        # global guard with options
parry.install(Parry(...))             # global guard you built yourself

with Parry(cache=False) as guard:     # only inside this block (thread/task-local)
    ...

client = parry.wrap(openai.OpenAI(), Parry(...))   # only this client
```

When more than one applies, the most specific one wins: a client bound with
`wrap()`, then the innermost `with Parry():` block, then `install()`.
`with parry.bypass():` turns parry off for everything inside it.

`install()` can be called any number of times. Methods are never wrapped
twice, and calling it again with no arguments returns the guard already
installed. Calling it with options creates and installs a new guard.

| `install()` argument | Default | Meaning |
|---|---|---|
| `guard` | `None` | install this `Parry` instead of creating one |
| `sync` / `async_` | `True` / `True` | hook sync / async clients (`parry.install_async()` = async only) |
| `openai` / `anthropic` | `True` / `True` | which SDKs to hook |
| `**options` | | passed to `Parry(...)` |

## `Parry(...)` options

| Option | Default | Meaning |
|---|---|---|
| `cache` | `True` | serve identical requests from the cache |
| `cache_ttl` | `3600` | seconds an entry lives; `None` = until evicted by size |
| `cache_size` | `10_000` | maximum entries. The cache is **always** bounded; `None`/`0` is rejected |
| `rate_limit` | `True` | apply per-model token buckets |
| `on_rate_limit` | `"wait"` | `"wait"`: sleep until a slot frees; `"raise"`: raise `parry.RateLimited` immediately |
| `max_wait` | `60.0` | in `"wait"` mode, give up and raise `RateLimited` after this many seconds |
| `cost_tracking` | `True` | record spend per model |
| `max_cost_usd` | `None` | hard budget for this guard's session |
| `retry` | `True` | retry transient failures |
| `max_attempts` | `3` | total attempts, including the first |
| `retry_base_delay` | `0.5` | first backoff delay in seconds (doubles each attempt, ±50% jitter) |
| `retry_max_delay` | `30.0` | cap for any single delay, including server `Retry-After` values |
| `storage` | `None` | path to a SQLite file for a persistent cache and spend ledger |

All boolean switches and `max_cost_usd` are plain attributes, so you can
change them at runtime (`guard.cache = False`).

## Rate limits

```python
guard.set_rate_limit("gpt-4o", requests_per_minute=60)
guard.set_rate_limit("gpt-4o-mini", requests_per_minute=500, tokens_per_minute=200_000)
guard.set_rate_limit("claude-opus-5-5", requests_per_minute=50, burst=5)
guard.set_rate_limit("openai:*", requests_per_minute=3_000)   # every OpenAI model
guard.set_rate_limit("*", requests_per_minute=100)            # everything else
guard.set_rate_limit("gpt-4o")                                # no limits: unlimited
guard.set_rate_limit("gpt-4o", requests_per_minute=60, provider="azure")  # one provider only
```

Each `provider:model` pair gets its **own** bucket, because providers enforce
limits per model. When a call comes in, parry uses the first rule that
matches, in this order: `provider:model`, `model`, `provider:*`, `*`, then
the built-in default.

Built-in defaults are the providers' published **tier-1** request limits:
500 RPM for OpenAI models and 50 RPM for Anthropic models. If your account is
on a higher tier, raise them.

A bucket holds `burst` requests (default: the per-minute rate) and refills
continuously. With `requests_per_minute=60`, 60 back-to-back calls go through
and the 61st waits about one second.

`tokens_per_minute` counts the estimated prompt tokens plus `max_tokens` of
each request.

Cache hits never consume rate-limit capacity.

## Pricing

The built-in table covers current OpenAI and Anthropic models (`parry pricing`
prints it). Dated snapshots and fine-tunes resolve to their family:
`gpt-4o-2024-08-06` uses `gpt-4o` pricing, and `ft:gpt-4o-mini:acme::x` uses
`gpt-4o-mini`. Bedrock-style ids such as `anthropic.claude-opus-5-5` work too.

```python
guard.set_pricing("my-finetune", input_per_million=3.0, output_per_million=12.0)
guard.set_pricing("gpt-4o", input_per_million=2.0, output_per_million=8.0)  # negotiated rate
```

An override only beats a more specific built-in row if it is at least as
specific. Overriding `gpt-4o` does not reprice `gpt-4o-mini`.

What the recorded cost includes:
- OpenAI: `prompt_tokens` and `completion_tokens` at standard short-context
  rates. Cached-input discounts, batch and flex tiers are not modelled, so
  the recorded cost is an upper bound.
- Anthropic: prompt-cache writes count at 1.25× and cache reads at 0.1× the
  input price, matching Anthropic's billing.
- Streams that don't report usage (OpenAI without
  `stream_options={"include_usage": True}`) are estimated with tiktoken and
  counted in `stats.estimated_requests`.

## Budget

`max_cost_usd` is checked **before** every call that isn't cached. parry
estimates the prompt's input cost with tiktoken and refuses the call if
`spent + estimate` would exceed the limit. The call that crosses the line
(because of its output) is allowed to finish; every call after it raises
`parry.BudgetExceeded`. Cache hits are free and are always served.

The budget covers the guard's session. `guard.reset_stats()` restarts it.
Concurrent threads can each have one call in flight when the line is
crossed, so the limit can be overshot by that much.

## Retry

parry retries `RateLimitError` (429), timeouts, connection errors, 408, 409,
425, overload (529) and every 5xx. It never retries 400, 401, 403, 404 or 422.
It waits `retry_base_delay × 2^(attempt-1)` with equal jitter, capped at
`retry_max_delay`. When the server sends `retry-after-ms` or `retry-after`,
that value is used instead, capped at the same limit.

**SDK retries stack with parry's.** The OpenAI and Anthropic SDKs retry twice
by default (`max_retries=2`). With parry on top, one request can be tried
`3 × (2 + 1) = 9` times in the worst case. To let parry be the only retry
layer, create clients with `max_retries=0`. To keep the SDK's retries and
turn parry's off, set `retry=False`.

## Caching

The cache key is a blake3 hash of:

- the provider, the endpoint and the client's `base_url`, so OpenAI, Azure
  and a local server never share entries
- every request field **except** `stream`, `stream_options`, `timeout`,
  `extra_headers`, `extra_query`, `extra_body`, `user`, `metadata`, `store`,
  `service_tier`, `safety_identifier`, `prompt_cache_key`,
  `prompt_cache_retention`, `background` and `betas`

The request JSON is canonicalised first: keys are sorted, CRLF becomes LF and
leading/trailing whitespace of strings is trimmed. Internal whitespace and
everything else are kept exactly. Changing `temperature`, `response_format`,
`system`, `tools`, `max_tokens` or anything else semantic is a cache miss.

Not cached: streaming requests (they never *read* the cache), Responses API
calls with `background=True`, responses that are not `completed`, and
Anthropic `pause_turn` responses.

When a text-only stream runs to completion, parry rebuilds the full response
and caches it, so the next identical **non-streaming** call is a hit.

## Persistence

```python
parry.install(storage="parry.db")     # or guard.storage("parry.db")
guard.set_cache("sqlite", path="parry.db")
guard.set_cache("memory")             # back to memory only
```

The SQLite file holds cache entries (with their TTL) and one ledger row per
call. `guard.stats.today_cost_usd` and `lifetime_cost_usd` come from it. Run
`parry stats --db parry.db` for a report.

## Stats

```python
s = guard.stats          # or parry.stats() for the active guard
s.cache_hits, s.cache_misses, s.cache_hit_rate, s.tokens_cached, s.cost_saved_usd
s.total_cost_usd, s.input_tokens, s.output_tokens, s.by_model
s.requests_blocked, s.rate_limit_rejections, s.budget_rejections, s.retries, s.errors
s.unpriced_requests, s.estimated_requests, s.today_cost_usd, s.lifetime_cost_usd
print(s.summary())
```

## Logging

parry logs to the `parry` logger. At `DEBUG` it logs one line per call with
the model, token counts, cost and cache hits. Request and response bodies are
never logged at any level.

```python
logging.getLogger("parry").setLevel(logging.DEBUG)
```

## Failure modes

| Situation | What parry does |
|---|---|
| Model missing from the pricing table | records the call at $0, logs one warning per model, counts `unpriced_requests`. The budget can't see that spend; use `set_pricing` |
| Rate limit exhausted | `"wait"`: sleeps, then raises `parry.RateLimited` after `max_wait`. `"raise"`: raises at once |
| Budget reached | raises `parry.BudgetExceeded` before sending |
| SQLite file corrupt, locked or read-only | logs a warning and continues in memory |
| Cached entry no longer matches the SDK's types (after an SDK upgrade) | drops the entry and treats the call as a miss |
| Request contains something that can't be hashed | skips the cache for that call; everything else still applies |
| parry's own bookkeeping raises | logs a warning; your response is still returned |
| SDK not installed | `install()` skips it |
| Retries exhausted | re-raises the SDK's last exception unchanged |

## What isn't intercepted (v0.1)

- `client.chat.completions.parse(...)` and `client.responses.parse(...)`
- `client.messages.stream(...)`, `client.beta.chat.completions.stream(...)` and other stream *helpers*. `create(stream=True)` is covered
- `.with_streaming_response` (rate limit and budget apply; no caching or cost)
- `create(stream=True)` through `.with_raw_response` (rate limit and budget apply; no caching or cost)
- embeddings, batches, files, images and audio
- `client.with_options(...)` returns a *new* client. If you used `wrap()`,
  wrap that one too. Global `install()` covers it automatically.
