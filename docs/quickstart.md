# Quickstart (5 minutes)

## 1. Install

```bash
pip install llm-parry openai    # or anthropic, or both
```

The distribution is called `llm-parry`; the module you import is `parry`. It
has no required dependencies. It hooks whichever SDKs you already have.

## 2. Add one line

```python
import parry
parry.install()
```

Put it once at startup, before or after creating clients. Nothing else in your
code changes.

## 3. See it work

```python
import time
import openai
import parry

parry.install()
client = openai.OpenAI()

for i in range(3):
    t = time.perf_counter()
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hi in five words."}],
    )
    print(f"{(time.perf_counter() - t) * 1000:.1f} ms  {r.choices[0].message.content}")

print(parry.stats().summary())
```

```
612.4 ms  Hi there, how are you?
  0.4 ms  Hi there, how are you?
  0.1 ms  Hi there, how are you?
requests        3
cache           2 hits / 1 misses (67%), 46 tokens and $0.000011 saved
spend           $0.000006 this session (18 in / 5 out tokens)
...
```

The first call went to OpenAI. The next two came from parry's cache and cost
nothing.

## 4. Add a budget (optional)

```python
parry.install(max_cost_usd=5.00)
```

Once this guard has spent $5, any call that isn't already cached raises
`parry.BudgetExceeded` *before* it reaches the provider.

## 5. Keep the cache and the spend across restarts (optional)

```python
parry.install(storage="parry.db")
```

```bash
parry stats --db parry.db
```

## Turning it off

| You want | Do |
|---|---|
| no caching for one block | `with parry.bypass(): ...` (turns parry off entirely for the block) |
| no caching at all | `parry.install(cache=False)` |
| parry only in one place | `with parry.Parry() as guard: ...` instead of `install()` |
| parry on one client only | `client = parry.wrap(openai.OpenAI())` |
| undo everything | `parry.uninstall()` |

Next: [configuration](configuration.md) · [integrations](integrations.md) · [architecture](architecture.md)
