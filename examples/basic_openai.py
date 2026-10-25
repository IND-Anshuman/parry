"""The whole pitch in one file: add parry.install(), change nothing else.

export OPENAI_API_KEY=sk-...
python examples/basic_openai.py
"""

import time

import openai

import parry

parry.install(max_cost_usd=1.00)  # ← the only parry line; the budget is optional

client = openai.OpenAI()
question = [{"role": "user", "content": "In one sentence, what is a token bucket?"}]

for attempt in range(1, 4):
    start = time.perf_counter()
    response = client.chat.completions.create(model="gpt-4o-mini", messages=question)
    ms = (time.perf_counter() - start) * 1000
    print(f"call {attempt}: {ms:7.1f} ms  {response.choices[0].message.content!r}")

# Calls 2 and 3 come back from the cache in well under a millisecond.
print()
print(parry.stats().summary())
