# Recipes

Short, copy-pasteable patterns. Each uses only the public API; see
[configuration](configuration.md) and [integrations](integrations.md) for the
details.

## Multi-tenant SaaS: a budget per tenant

parry's budget and ledger live on a guard, so give each tenant its own guard and
run their calls inside it. A `with guard:` block is thread- and task-local, so
concurrent requests don't bleed into each other.

```python
import parry
from parry import Parry, BudgetExceeded

_guards: dict[str, Parry] = {}

def guard_for(tenant_id: str) -> Parry:
    g = _guards.get(tenant_id)
    if g is None:
        g = Parry(max_cost_usd=10.0, storage=f"spend/{tenant_id}.db")
        _guards[tenant_id] = g
    return g

def handle(tenant_id: str, prompt: str):
    with guard_for(tenant_id):               # this tenant's cache + budget + ledger
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
            )
        except BudgetExceeded:
            return "This month's budget is used up."
```

Per-tenant SQLite storage keeps each tenant's spend and cache separate and
durable across restarts. `guard_for(tid).stats` or `parry doctor --db
spend/<tid>.db` reports one tenant.

## FastAPI: guard every request

Call `parry.install()` once at startup so the SDK hooks are in place, then enter
the tenant's guard **inside the handler** around the call. Keeping the `with`
block in the same frame as the SDK call makes it correct whether the endpoint is
sync (threadpool) or async — the guard is thread- and task-local.

```python
from fastapi import FastAPI, HTTPException
import parry
from parry import RateLimited, BudgetExceeded

app = FastAPI()
parry.install()                              # hooks in place for the whole app

@app.post("/chat")
def chat(prompt: str, tenant_id: str):
    try:
        with guard_for(tenant_id):           # guard_for from the recipe above
            r = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
            )
    except BudgetExceeded:
        raise HTTPException(402, "budget exceeded")
    except RateLimited:
        raise HTTPException(429, "slow down")
    return r.choices[0].message.content
```

## Stop an agent's cost runaway

A hard budget turns a runaway loop into one caught exception instead of a
surprise bill. `on_unpriced="raise"` (the default) means an unknown model can't
slip past the budget at $0.

```python
import parry
from parry import BudgetExceeded

parry.install(max_cost_usd=5.0)              # also guards the OpenAI Agents SDK

try:
    run_my_agent()                           # many tool + model calls
except BudgetExceeded as e:
    print(f"stopped at {e}")                 # the next call that would cross $5 is blocked
print(parry.stats().summary())
```

For streaming-heavy agents, keep `stream_options={"include_usage": True}` on so
cost is exact rather than estimated — `parry doctor` will flag it if it isn't.

## OpenAI-compatible providers

Anything you reach with the `openai` SDK and a custom `base_url` is already
hooked by `install()` — the hook is on the SDK class, whatever the URL:

```python
import parry, openai
parry.install()
client = openai.OpenAI(base_url="https://my-gateway.example/v1", api_key="...")
client.chat.completions.create(model="my-model", messages=[...])
parry.get_guard().set_pricing("my-model", input_per_million=0.5, output_per_million=1.5)
```

groq, Together, Mistral and others that ship their *own* SDK are hooked natively
— see [integrations](integrations.md#more-providers). Models not in the pricing
table record at $0 and count as `unpriced_requests` until you `set_pricing` them.

## Pricing a fine-tune

Fine-tunes resolve to their base family by longest-prefix match, so
`ft:gpt-4o-mini:acme::abc` already prices as `gpt-4o-mini`. Override only when
your negotiated or custom rate differs:

```python
guard = parry.get_guard()
guard.set_pricing("ft:gpt-4o-mini:acme::abc", input_per_million=3.0, output_per_million=12.0)
guard.set_pricing("gpt-4o", input_per_million=2.0, output_per_million=8.0)  # negotiated rate
```

## A large offline / batch job

For a big ETL run, cap concurrency with a rate limit, keep a durable ledger, and
cache so a re-run after a crash is cheap:

```python
from parry import Parry

guard = Parry(max_cost_usd=200.0, storage="etl.db")
guard.set_rate_limit("openai:*", requests_per_minute=3_000, tokens_per_minute=2_000_000)

with guard:
    for row in rows:
        summarize(row)                       # identical rows hit the cache on re-run

print(guard.stats.summary())
```

The batch *tier* (50% off on OpenAI/Anthropic) isn't auto-detected from a hooked
call; compute batch cost directly when you need it:

```python
from parry import Cost
Cost().compute_tiered("gpt-4o", input_tokens=1_000_000, output_tokens=200_000, tier="batch")
```

## Audit what happened

```python
print(parry.stats().summary())      # requests, cache, spend, limiter, retries
print(parry.doctor().summary())     # plain-language findings + one-line fixes
```

Persisted runs: `parry stats --db etl.db` and `parry doctor --db etl.db` (exit
code 2 when the doctor finds something actionable, so it can gate a pipeline).
