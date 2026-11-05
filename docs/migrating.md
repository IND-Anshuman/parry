# Migrating to parry

parry is an **in-process** guard: it hooks the SDK classes you already call and
adds caching, rate limiting, a budget, cost tracking and retry. There is no
server, no proxy and no network hop, and nothing leaves your process. That makes
the move from a gateway or a single-purpose library small.

## From LiteLLM

LiteLLM gives you a unified `completion()` across providers (and, separately, a
proxy server). parry doesn't replace that — it guards it. Change one import:

```python
from parry_litellm import completion     # was: from litellm import completion

completion(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
```

Every call now runs through a parry guard (cache, rate limit, budget, cost,
retry, OpenTelemetry) while LiteLLM still makes the call and returns its usual
`ModelResponse`. `import parry_litellm as litellm` works too — any name the shim
doesn't define proxies to the real `litellm`. Configure the guard as usual:

```python
import parry
parry.install(max_cost_usd=25)                       # the shim uses the active guard
parry.get_guard().set_rate_limit("openai:*", requests_per_minute=5_000)
```

If instead you call the official SDKs directly (LiteLLM is optional), just
`parry.install()` — no shim needed.

## From Helicone (or any logging proxy)

Helicone and similar tools sit in front of your calls as a hosted proxy: you
point `base_url` at them and they log and dashboard your traffic. parry runs
inside your process instead, so:

- **Remove the proxy `base_url`.** Point your client back at the provider.
- **`parry.install()`** for caching, rate limiting, the budget and cost tracking.
- **Export traces yourself** if you want them, with the OpenTelemetry integration
  — spans go to *your* collector, not a third party:

  ```python
  import parry
  from parry.integrations import otel

  parry.install()
  otel.instrument()
  ```

What you give up is the hosted dashboard and long-term log storage — parry is not
an observability backend (see below). What you gain is no network hop, no data
leaving your process, and enforcement (a budget that actually blocks a call)
rather than after-the-fact logging.

## How parry compares

| | parry | LiteLLM | Helicone | tenacity / cachier |
|---|---|---|---|---|
| Runs | in your process | library + optional proxy | hosted proxy | in your process |
| Response cache | ✅ | proxy only | ✅ (proxy) | cachier: cache only |
| Rate limiting | ✅ per model/provider | proxy only | ✅ (proxy) | ❌ |
| Budget enforcement | ✅ pre-call, fail-closed | proxy only | spend alerts | ❌ |
| Cost tracking | ✅ Rust table + tiers | ✅ | ✅ | ❌ |
| Retry/backoff | ✅ | ✅ | ❌ | tenacity: retry only |
| Provider breadth | OpenAI, Anthropic, Google, groq, together, ollama, mistral, cohere, Bedrock | very broad | broad | n/a |
| Dashboards / hosted logs | ❌ (export via OTel) | partial | ✅ | ❌ |
| Data leaves your process | no | only via the proxy | yes | no |

### When parry is the right tool

- You want **enforcement** (a hard budget, real rate limits) in the same process
  that makes the calls, with no proxy to run or trust.
- You want cost numbers that match the bill — tiers for cache, batch and fast,
  and a table that fails closed rather than silently billing $0.
- You want retries, caching and metering with near-zero overhead and no new
  infrastructure.

### When it isn't

- You need a **hosted dashboard**, long-term log storage, or multi-tenant spend
  reporting at a shared gateway — that's an observability product; run one and,
  if you like, feed it parry's OpenTelemetry spans.
- You need **routing, fallback or load balancing** across providers — use a
  gateway (LiteLLM proxy, a router) and keep parry in-process behind it.
