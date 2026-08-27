from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.core.models import HYBRID, LOCAL, VALID_ROUTES, WEB


class SourceRoute(BaseModel):
    """Select the retrieval strategy for a user query."""

    source: Literal["LOCAL", "WEB", "HYBRID"] = Field(
        description="Best retrieval strategy for the user's query."
    )


SOURCE_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a retrieval router, not an answer generator.

Choose the knowledge source strategy for the user's question.

Available routes:

- LOCAL:
  Use the local vector index / RAPTOR corpus for stable, project-specific,
  or already-indexed knowledge.

- WEB:
  Use Tavily and Wikipedia for current, live, latest, today, recent,
  or time-sensitive information, and for facts that need an external lookup.

- HYBRID:
  Use both local retrieval and web retrieval when the question may need
  both indexed documents and external confirmation.

Do not answer the question. Return only the selected route.
""",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)

CURRENT_QUERY_KEYWORDS = (
    "today",
    "current",
    "latest",
    "now",
    "live",
    "recent",
    "yesterday",
    "tomorrow",
)


def create_source_router(llm: Optional[object] = None):
    """Create a structured-output source router."""
    if llm is None:
        llm = ChatOllama(
            model="llama3:latest",
            temperature=0,
        )

    structured_llm = llm.with_structured_output(SourceRoute)
    return SOURCE_ROUTER_PROMPT | structured_llm


def parse_route(raw) -> str:
    """Normalize a router response to LOCAL, WEB, or HYBRID."""
    if raw is None:
        return HYBRID

    source = ""
    try:
        source = str(getattr(raw, "source", raw)).upper()
    except Exception:
        source = str(raw).upper()

    if "HYBRID" in source:
        return HYBRID
    if "WEB" in source or "TAVILY" in source or "WIKIPEDIA" in source:
        return WEB
    if "LOCAL" in source or "CHROMA" in source or "RAPTOR" in source:
        return LOCAL
    return HYBRID


class SourceRouter:
    """Ollama router that selects LOCAL / WEB / HYBRID retrieval."""

    def __init__(self, llm: Optional[object] = None):
        self.llm = llm
        self._chain = None
        try:
            self._chain = create_source_router(llm=llm)
        except Exception as exc:
            print(f"Structured router init failed: {exc}")

    def route(self, query: str) -> str:
        if any(keyword in query.lower() for keyword in CURRENT_QUERY_KEYWORDS):
            keyword_route = WEB
        else:
            keyword_route = None

        if self._chain is None:
            return keyword_route or HYBRID

        try:
            raw = self._chain.invoke({"question": query})
            decision = parse_route(raw)
            if decision in VALID_ROUTES:
                return decision
        except Exception as exc:
            print(f"Router LLM failed: {exc}")
            if self.llm is not None:
                try:
                    fallback = self.llm.invoke(
                        "Route this query as LOCAL, WEB, or HYBRID only.\n"
                        f"Query: {query}"
                    )
                    content = getattr(fallback, "content", fallback)
                    return parse_route(content)
                except Exception as inner:
                    print(f"Router fallback failed: {inner}")

        return keyword_route or HYBRID
