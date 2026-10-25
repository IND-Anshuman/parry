# parry v0.1 — implementation plan

This is the working plan the code in this repository was built against. It
follows the original project brief (zero-config rate limiting, caching, cost
control for LLM calls, Rust core via PyO3, `parry.install()` as the headline
feature) and records every place where the brief was changed and why.

## Architecture in one picture

```
user code ──► openai / anthropic SDK method (patched by parry.install())
                │
                ▼
        parry._core.Parry._call(adapter, original_fn, self, kwargs)
                │  1. resolve active guard (client-bound → context → global)
                │  2. cache lookup            (Rust: blake3 key + moka LRU/TTL)
                │  3. budget pre-check        (Rust: cost ledger + pricing table)
                │  4. rate-limit acquire      (Rust: governor token bucket per provider:model)
                │  5. call original w/ retry  (Rust: backoff + jitter + classification)
                │  6. record usage/cost, store in cache
                ▼
        original SDK response object (or a cached copy re-validated into the same type)
```

Rust (`src/*.rs`, compiled to `parry._parry_native`) owns the hot, stateful,
thread-safe primitives. Python (`python/parry/`) owns everything that touches
SDK objects: patching, (de)serialising responses, exceptions, sleeping.
Sleeping happens in Python so the same primitives serve both sync and async
code (`time.sleep` vs `await asyncio.sleep`).

## Phases and status

| Phase | Deliverable | Status |
|---|---|---|
| 0 | maturin + PyO3 scaffold, `import parry` works | done |
| 1 | Rust primitives: `Cache`, `RateLimit`, `Cost`, `Retry`, tokenizer | done |
| 2 | `parry.install()` for OpenAI chat completions + responses, Anthropic messages (sync + async + streaming), retry, budget | done |
| 3 | LangChain callback, LlamaIndex callback handler | done |
| 4 | README, docs, CHANGELOG, CONTRIBUTING, CI, issue templates, bench | done |
| 5 | Real-world validation | yours — ongoing |

## Deviations from the original brief (and why)

1. **Python source lives in `python/parry/`, not `src/python/`.** That is
   maturin's standard mixed Rust/Python layout (`python-source = "python"`,
   `module-name = "parry._parry_native"`). Keeping Python out of `src/` stops
   Cargo and maturin from fighting over the directory.

2. **Python 3.10–3.14, not 3.9–3.13.** Python 3.9 reached end of life in
   October 2025, current LangChain requires 3.10+, and the Anthropic SDK 1.x
   requires 3.10+. The wheel is built with `abi3-py310`, so one wheel per
   platform covers every supported interpreter.

3. **No `extension-module` Cargo feature.** PyO3 has deprecated it in favour of
   maturin ≥ 1.9.4 handling linkage. This also means `cargo test` works
   without feature juggling.

4. **Crate versions are whatever was current when this was built**, not the
   brief's numbers (several had moved on). `Cargo.lock` is committed and the
   toolchain is pinned in `rust-toolchain.toml`.

5. **Cache key = everything semantic, minus a denylist** — not the brief's
   `(model, messages, temperature, top_p, tools)` allowlist. An allowlist
   silently collides requests that differ in `response_format`, `max_tokens`,
   `system` (Anthropic), `stop`, `seed`, `tool_choice`, etc., and would serve
   the wrong answer. The denylist drops only transport/bookkeeping fields
   (`stream`, `stream_options`, `timeout`, `extra_headers`, `user`,
   `metadata`, `store`, ...). The client's `base_url` is part of the key so
   OpenAI, Azure and a local OpenAI-compatible server never share entries.

6. **Key normalisation is conservative.** JSON is canonicalised (sorted keys),
   CRLF → LF, and leading/trailing whitespace of strings is trimmed. Internal
   whitespace is *not* collapsed (it matters for code) and timestamps are
   *not* stripped (that would change the prompt's meaning and return answers
   computed for a different prompt).

7. **`guard.set_cache("redis", ...)` is not implemented.** The brief lists
   Redis as out of scope for v0.1 while also using it in the API sketch; the
   out-of-scope list wins. Optional persistence is SQLite via
   `Parry(storage="parry.db")` / `parry.storage("parry.db")`.

8. **Framework callbacks cannot serve cached responses.** LangChain and
   LlamaIndex callbacks are observers: they can block (rate limit), raise
   (budget) and record (cost), but cannot return a response. So the
   integration story is: `parry.install()` does everything (both frameworks
   call the patched OpenAI/Anthropic SDKs underneath); the callback handlers
   are for users who do not want global patching, and they do rate limit +
   budget + cost only. When the guard is already intercepting at the SDK
   level, the callbacks skip accounting so nothing is double counted.

9. **LlamaIndex `ServiceContext` is gone upstream.** The handler is used via
   `Settings.callback_manager = CallbackManager([ParryHandler(guard)])`.

10. **`parry.wrap(client)` binds a guard to one client** (Strategy B) instead
    of being a separate patch-free proxy. A proxy would have to reimplement
    every access path the SDK offers (`with_raw_response`, which LangChain
    always uses, `with_options`, sub-resources), so it would cover less and
    break more often. The class hooks are inert for clients that are not
    bound, so only the wrapped client is affected. The trade-off: a client
    returned by `client.with_options(...)` needs wrapping again.

11. **Also covers the OpenAI Responses API** (`client.responses.create`). It is
    OpenAI's primary API now; leaving it out would make "zero-config" a lie
    for most new code. Same adapter machinery, ~60 lines.

12. **Retries do not multiply silently.** The OpenAI and Anthropic SDKs already
    retry twice by default. parry's retry layer wraps the SDK call, honours
    `Retry-After`, and defaults to 3 attempts; the interaction is documented in
    `docs/configuration.md`. Set `max_retries=0` on the client if you want
    parry to be the only retry layer.

13. **Pricing table updated** to models that exist today (GPT-5 family,
    GPT-4.1, GPT-4o, o-series; Claude 5.x/4.x), longest-prefix matched so
    dated snapshots (`gpt-4o-2024-08-06`) and fine-tunes (`ft:gpt-4o-mini:...`)
    resolve. `guard.set_pricing(model, ...)` exists for models missing from the
    table — a single override, not a billing engine.

14. **Budget enforcement is "pre-flight + latch".** Before each uncached call,
    parry estimates the prompt cost with the tokenizer and refuses the call if
    spend + estimate would exceed `max_cost_usd`. A call that is already in
    flight is never cancelled; once the limit is crossed every further
    uncached call raises `BudgetExceeded`. Cache hits are free and are still
    served. Concurrent threads may overshoot by at most one call each — this
    is documented.

15. **Streaming:** streams are never served from cache (Q2 in the brief).
    OpenAI chat and Anthropic message streams are passed through untouched;
    when a text-only stream completes, the assembled response is stored so the
    next identical *non-streaming* call is a hit. Cost comes from usage in the
    stream when the provider sends it (Anthropic always, OpenAI with
    `stream_options={"include_usage": True}`), otherwise it is estimated with
    the tokenizer and flagged as estimated in stats. parry never injects
    `stream_options` because that changes the chunks user code sees.

16. **Not intercepted in v0.1** (documented): `client.chat.completions.parse`,
    `client.messages.stream(...)` helper, `with_streaming_response`, batches,
    embeddings. These either bypass `create` inside the SDK or return
    lazily-read HTTP responses that cannot be cached safely.

## Failure modes (defined behaviour)

| Situation | Behaviour |
|---|---|
| Model missing from pricing table | cost recorded as 0, `stats.unpriced_requests` += 1, one warning per model, budget cannot account for it |
| Rate limit exhausted | `on_rate_limit="wait"` (default) sleeps until a token is free, up to `max_wait` seconds, then raises `parry.RateLimited`; `"raise"` raises immediately |
| Budget exceeded | `parry.BudgetExceeded` before the network call |
| SQLite file corrupted / unwritable | warning logged, parry continues in memory only |
| Cached payload no longer validates (SDK upgraded) | entry dropped, treated as a miss |
| Response cannot be serialised | not cached, call still returns normally |
| parry internal error during bookkeeping | logged at warning, the user's response is still returned |
| SDK not installed | that SDK is skipped by `install()` |

## Test plan

All tests run against an in-process mock HTTP server speaking the OpenAI and
Anthropic wire formats (JSON and SSE). The real `openai`, `anthropic`,
`langchain-openai` and `llama-index` packages are used — no SDK mocking — so
SDK upgrades that break the patch show up as test failures. No test makes a
network call outside localhost.

## Open items before the first PyPI release

- **Distribution name: `llm-parry`** (`parry` is taken on PyPI by an
  unrelated project). The import name, the CLI command and the Rust module
  stay `parry`.
- Configure PyPI trusted publishing for `.github/workflows/release.yml`
  (environment `pypi`).
- CI has not run on GitHub yet. Everything above was verified locally on
  Linux (Python 3.10, 3.11, 3.13; latest SDKs plus openai 1.82 / anthropic
  0.52). macOS and Windows have only been covered by the CI matrix definition.
