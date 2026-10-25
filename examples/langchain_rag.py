"""A tiny LangChain retrieval chain; parry caches the LLM calls underneath.

    pip install langchain-openai langchain-core
    export OPENAI_API_KEY=sk-...
    python examples/langchain_rag.py

Run it twice in a row with storage enabled and the second run costs nothing.
"""

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import parry

logging.basicConfig(format="%(name)s %(message)s")
logging.getLogger("parry").setLevel(logging.DEBUG)  # shows "cache hit" lines (metadata only)

parry.install(storage="parry.db")  # cache + spend survive restarts

DOCS = {
    "rate limits": "OpenAI tier 1 allows 500 requests per minute for most models.",
    "backoff": "Exponential backoff doubles the delay after each failed attempt.",
    "caching": "Identical requests can be answered from a cache at zero cost.",
}


def retrieve(question: str) -> str:
    """Keyword 'retriever' so the example needs no vector store."""
    hits = [text for key, text in DOCS.items() if any(w in question.lower() for w in key.split())]
    return "\n".join(hits) or "\n".join(DOCS.values())


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer using only this context:\n{context}"),
        ("human", "{question}"),
    ]
)
chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()

for question in ["How does backoff work?", "What are the rate limits?", "How does backoff work?"]:
    print(
        f"Q: {question}\nA: {chain.invoke({'question': question, 'context': retrieve(question)})}\n"
    )

print(parry.stats().summary())
