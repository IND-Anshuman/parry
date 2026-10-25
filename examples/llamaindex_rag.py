"""LlamaIndex query engine with parry's handler for cost tracking and a budget.

pip install llama-index-core llama-index-llms-openai llama-index-embeddings-openai
export OPENAI_API_KEY=sk-...
python examples/llamaindex_rag.py
"""

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.openai import OpenAI

import parry
from parry import Parry
from parry.integrations.llamaindex import ParryHandler

guard = Parry(max_cost_usd=0.50)
parry.install(guard)  # caching for the OpenAI calls LlamaIndex makes
Settings.callback_manager = CallbackManager(
    [ParryHandler(guard)]
)  # rate limit + budget per LLM event
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)

documents = [
    Document(text="parry wraps LLM SDK calls with caching, rate limiting and cost tracking."),
    Document(text="parry's rate limiter is a token bucket per provider and model."),
    Document(text="max_cost_usd stops requests before they would exceed the budget."),
]
engine = VectorStoreIndex.from_documents(documents).as_query_engine()

for q in ["How does parry rate limit?", "How does parry rate limit?", "What does max_cost_usd do?"]:
    print(f"Q: {q}\nA: {engine.query(q)}\n")

print(guard.stats.summary())
