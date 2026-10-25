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
