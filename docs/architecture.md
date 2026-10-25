# Architecture

```
python/parry/                     src/ (Rust, compiled to parry._parry_native)
├── __init__.py   public API      ├── lib.rs         module definition
├── _patch.py     SDK hooks       ├── cache.rs       blake3 key + moka LRU/TTL
├── _core.py      Parry guard     ├── rate_limit.rs  governor token buckets
├── _adapters.py  per-endpoint    ├── cost.rs        pricing table + ledger
├── _stream.py    stream proxies  ├── retry.rs       backoff + classification
├── _storage.py   SQLite          └── tokenize.rs    canonical JSON, tiktoken
└── integrations/ LangChain, LlamaIndex
```

## The split

**Rust** holds the state that has to be fast and thread-safe and never needs
a Python object. That's the cache (values are strings), the rate-limit
buckets, the spend ledger, the retry policy, request canonicalisation and
token counting. Every type is `Sync` and frozen, so one instance can be shared
across threads and asyncio tasks with no Python-level locking. Hashing and
token counting release the GIL.

**Python** handles everything that touches SDK objects: replacing methods,
serialising pydantic responses, wrapping streams, raising exceptions, and
sleeping. Sleeping lives in Python because sync code has to `time.sleep` and
async code has to `await asyncio.sleep`. The rate limiter's `try_acquire`
therefore returns "seconds to wait" instead of blocking.

## A call, step by step

1. **Hook.** `Completions.create` (and the other seven targets) is replaced
   by a function that calls `_state.active_guard(self)`. With no guard, it
   calls the original method and nothing else happens.
2. **`_Call`.** Collects the model, whether this is a stream, whether it came
   through `.with_raw_response` (detected by the `X-Stainless-Raw-Response`
   header the SDK adds), and the cache key. The key is
   `blake3(provider:endpoint:base_url ∥ canonical_json(semantic kwargs))`.
3. **Cache lookup** (non-streaming only): moka, then SQLite if enabled. On a
   hit, the stored JSON is validated back into the original SDK type
   (`ChatCompletion`, `Message`, `Response`, ...). Raw-response calls get a
   `CachedRawResponse` with `.parse()` and `.headers`.
4. **Budget pre-check**, if `max_cost_usd` is set: the tiktoken estimate of
   the prompt, priced with the model's input rate.
5. **Rate limit:** non-blocking `try_acquire`, and the caller sleeps for as
   long as it says.
6. **Original `create`**, inside the retry loop. Each retry reacquires a
   rate-limit slot.
7. **Completion:**
   - non-streaming: read `usage`, record the cost, store the response
   - streaming: return a transparent proxy that feeds an accumulator. When
     the stream ends, it records usage (from the stream, or estimated) and
     caches the rebuilt response if the stream was text only and completed
   - Bookkeeping errors are logged and never reach the caller.

## Why patch the class instead of the client?

Users create clients everywhere: inside LangChain, inside LlamaIndex, inside
their own helpers. Patching the resource *class* catches all of them,
including clients created before `install()` ran. `with_raw_response`
wrappers bind `self.create` lazily, so they see the hook too. The hooks are
inert without a guard, so importing parry by itself changes nothing.

`parry.wrap(client)` uses the same hooks but binds a guard to one client
object, so other clients stay untouched.

## SDK compatibility

The hooks rely on:

- `openai.resources.chat.completions.Completions.create` / `AsyncCompletions`
- `openai.resources.responses.Responses.create` / `AsyncResponses`
- `anthropic.resources.messages.Messages.create` / `AsyncMessages`
- `anthropic.resources.beta.messages.Messages.create` / `AsyncMessages`
- the Stainless `X-Stainless-Raw-Response` header convention
- pydantic v2 `model_dump_json` / `model_validate_json` on response types

Missing targets are skipped. CI runs the suite against the newest SDKs and
against a pinned older pair so that SDK changes that break the hooks show up
as test failures.
