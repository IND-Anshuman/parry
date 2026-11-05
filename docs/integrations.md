# Integrations

## Which option should I use?

| | `parry.install()` | `ParryCallback` / `ParryHandler` |
|---|---|---|
| Caching | ✅ | ❌ (callbacks can't return a response) |
| Rate limiting | ✅ | ✅ |
| Budget | ✅ | ✅ |
| Cost tracking | ✅ exact usage from the SDK | ✅ usage reported by the framework |
| Works for | anything built on the `openai` / `anthropic` SDKs | any chat model / LLM the framework supports |
| Global side effects | hooks the SDK classes | none |

You can use both with the same guard. When the SDK hook has already handled a
call, the callback doesn't count it again.

## LangChain

`langchain-openai` and `langchain-anthropic` call the official SDKs, so
`install()` covers them, caching included. That holds for `invoke`, `ainvoke`,
`stream` and batch calls:

```python
import parry
from langchain_openai import ChatOpenAI

parry.install()
llm = ChatOpenAI(model="gpt-4o-mini")
llm.invoke("What is a token bucket?")
llm.invoke("What is a token bucket?")   # cache hit, no network
```

For chat models from other providers, or to avoid global hooks, use the
callback:

```python
from parry import Parry
from parry.integrations.langchain import ParryCallback

guard = Parry(max_cost_usd=5)
guard.set_rate_limit("gpt-4o-mini", requests_per_minute=100)

llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[ParryCallback(guard)])
# or per call:  chain.invoke(inputs, config={"callbacks": [ParryCallback(guard)]})
```

The callback sets `raise_error = True`, so `BudgetExceeded` and `RateLimited`
stop the run instead of being swallowed by LangChain's callback manager. It
reads token usage from `AIMessage.usage_metadata`, falls back to
`llm_output["token_usage"]`, and estimates with tiktoken as a last resort.

## LlamaIndex

LlamaIndex's `OpenAI` and `Anthropic` LLMs call the official SDKs too, so
`install()` covers them:

```python
import parry
from llama_index.llms.openai import OpenAI

parry.install()
llm = OpenAI(model="gpt-4o-mini")
```

The callback handler adds rate limiting, budget and cost tracking at the
LlamaIndex event level:

```python
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from parry import Parry
from parry.integrations.llamaindex import ParryHandler

guard = Parry(max_cost_usd=1)
Settings.callback_manager = CallbackManager([ParryHandler(guard)])
engine = index.as_query_engine()
engine.query("...")
print(guard.stats.summary())
```

(`ServiceContext` has been removed from LlamaIndex. `Settings` replaces it.)

## More providers

`install()` also guards several non-OpenAI/Anthropic SDKs when they are
installed — no extra arguments, same guard:

| Provider | SDK | Hooked surface |
|---|---|---|
| Groq | `groq` | `chat.completions.create` (OpenAI-shaped) |
| Together | `together` | `chat.completions.create` (OpenAI-shaped) |
| Google | `google-genai` | `Models.generate_content` / `generate_content_stream` |
| Ollama | `ollama` | `Client.chat` / `AsyncClient.chat` |
| Mistral | `mistralai` | `Chat.complete` / `complete_async` / `stream` / `stream_async` |
| Cohere | `cohere` | `BaseCohere.chat` / `chat_stream` |

```python
import parry
from groq import Groq

parry.install()                      # groq, together, google, ollama, mistral,
client = Groq()                      # cohere are all covered when installed
client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[...])
```

Opt a provider out with its flag, e.g. `parry.install(google=False)`. Groq and
Together are Stainless-generated clients whose chat surface mirrors OpenAI's, so
one code path covers them and any future SDK of the same shape. Models these
providers serve are often not in parry's pricing table; cost is then recorded as
`$0` (and estimated for streams) until you add a price with
`guard.set_pricing(...)` — rate limiting, caching, the budget and retry all work
regardless.

### AWS Bedrock (Converse)

boto3 builds its client methods dynamically, so `install()` can't hook them.
Guard a client explicitly:

```python
import boto3, parry
from parry.integrations import bedrock

parry.install()                                   # or pass a Parry to guard(...)
client = bedrock.guard(boto3.client("bedrock-runtime"))
client.converse(modelId="anthropic.claude-...", messages=[...])
```

The non-streaming `converse` call is metered, rate-limited, budgeted and
retried. Responses are plain dicts, so they are not cached, and
`converse_stream` is left untouched in this release.

### OpenAI Agents SDK

The `openai-agents` SDK calls the `openai` SDK under the hood, so
`parry.install()` already guards every agent and tool-model call — no extra
code. Install the guard once at startup, before you run an agent:

```python
import parry
from agents import Agent, Runner

parry.install(max_cost_usd=5)
result = Runner.run_sync(Agent(name="Assistant", model="gpt-4o"), "Hello")
print(parry.stats().summary())
```

### Pydantic AI

Pydantic AI already tracks cost itself, so parry's role there is rate limiting
and the budget. When a Pydantic AI model runs on the OpenAI or Anthropic SDK,
`install()` guards it automatically:

```python
import parry
from pydantic_ai import Agent

parry.install()                         # rate limit + budget around every call
agent = Agent("openai:gpt-4o")
agent.run_sync("What is a token bucket?")
```

For a model backed by a provider parry doesn't hook, wrap the call with the
building blocks shown below.

## OpenTelemetry

parry can emit one span per guarded call using the OpenTelemetry GenAI semantic
conventions. You configure OpenTelemetry (provider + exporter); parry only emits
spans. Install the SDK with `pip install "llm-parry[otel]"`, then:

```python
import parry
from parry.integrations import otel

parry.install()
otel.instrument()          # spans now flow to your configured provider
```

Each accounted call (cache hits included) becomes a `CLIENT` span named
`"{operation} {model}"` with `gen_ai.*` attributes — `gen_ai.request.model`,
`gen_ai.response.model`, `gen_ai.usage.input_tokens` / `output_tokens`,
`gen_ai.usage.cache_read.input_tokens` / `cache_creation.input_tokens` — plus
`parry.*` extensions: `parry.cost.usd`, `parry.cost.estimated`,
`parry.pricing.tier`, `parry.cache.hit` and `parry.cost.saved_usd` on a hit.

`instrument()` returns a handle; call `handle.uninstrument()` to stop. Pass a
specific `guard=` or `tracer=` if you don't want the process-wide ones. The
underlying hook is `guard.add_observer(fn)`, which calls `fn(event)` with a
`CallEvent` per call — use it directly to feed metrics or any other sink.

## LiteLLM

If you call models through LiteLLM, change one import and every call is guarded
— cache, rate limiting, budget, cost tracking, retry and any OpenTelemetry
export you've set up — while LiteLLM still makes the actual call and returns its
usual `ModelResponse`:

```python
from parry_litellm import completion       # was: from litellm import completion

completion(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
completion(model="claude-sonnet-5-5", messages=[...])   # any model LiteLLM routes
```

`import parry_litellm as litellm` works too: any name the shim doesn't define
proxies to the real `litellm`. The guard used is whatever `parry.install()` or a
`with Parry():` block has active; with none installed the shim keeps its own
default guard (set yours with `parry_litellm.configure(guard)`). The provider
label for rate limiting and cost comes from LiteLLM's own routing, so a
`set_rate_limit` / `set_pricing` keyed on that provider applies. Install the
extra with `pip install "llm-parry[litellm]"`.

## Anything else

If a library calls `client.chat.completions.create`, `client.responses.create`
or `client.messages.create` on the official SDKs, `install()` already covers
it. For any other client, call the guard's building blocks around your own
call:

```python
guard.check_budget("my-model", prompt_tokens=parry.count_tokens(prompt, "my-model"))
guard.acquire("myprovider", "my-model")            # or: await guard.acquire_async(...)
response = my_client.generate(prompt)
guard.record_usage("myprovider", "my-model", response.in_tokens, response.out_tokens)
```

The Rust primitives also work on their own:

```python
from parry import Cache, RateLimit, Cost, Retry

Retry(max_attempts=5).call(flaky_function, arg, kw=1)
RateLimit().acquire("p", "m", timeout=10)
Cost().compute("gpt-4o", input_tokens=1200, output_tokens=300)
```
