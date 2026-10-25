"""parry with the Anthropic SDK: sync, streaming and async calls.

export ANTHROPIC_API_KEY=sk-ant-...
python examples/basic_anthropic.py
"""

import asyncio

import anthropic

import parry

guard = parry.install()
guard.set_rate_limit("claude-haiku-5-5", requests_per_minute=50)

client = anthropic.Anthropic()
MODEL = "claude-haiku-5-5"
PROMPT = [{"role": "user", "content": "Name three uses of exponential backoff."}]

# 1. Plain call: priced from response.usage.
reply = client.messages.create(model=MODEL, max_tokens=300, messages=PROMPT)
print(reply.content[0].text, "\n")

# 2. Streaming passes through untouched; usage comes from the stream events.
with client.messages.create(model=MODEL, max_tokens=300, messages=PROMPT, stream=True) as stream:
    for event in stream:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            print(event.delta.text, end="", flush=True)
print("\n")


# 3. Async clients are covered too; the identical request is a cache hit.
async def main() -> None:
    aclient = anthropic.AsyncAnthropic()
    again = await aclient.messages.create(model=MODEL, max_tokens=300, messages=PROMPT)
    print("async (cached):", again.content[0].text[:60], "...")


asyncio.run(main())
print()
print(guard.stats.summary())
