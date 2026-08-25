from typing import Any, Callable, Dict, Literal, Optional, Type, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """Schema for routing queries to appropriate datasources."""

    datasource: Literal[
        "python_docs",
        "js_docs",
        "golang_docs",
    ] = Field(description="The most relevant datasource for the question.")


DEFAULT_LOGICAL_ROUTER_SYSTEM = """You are an expert at routing user questions.

Choose the most appropriate datasource based on the programming language mentioned in the question.

Available datasources:
- python_docs
- js_docs
- golang_docs

Return only the datasource.
"""

DEFAULT_LOGICAL_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", DEFAULT_LOGICAL_ROUTER_SYSTEM),
        ("human", "{question}"),
    ]
)


def create_logical_router(
    llm: Optional[Any] = None,
    schema: Type[BaseModel] = RouteQuery,
    prompt: Optional[ChatPromptTemplate] = None,
):
    """Create a structured output logical router chain."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if prompt is None:
        prompt = DEFAULT_LOGICAL_ROUTER_PROMPT

    structured_llm = llm.with_structured_output(schema)
    return prompt | structured_llm


def default_choose_route(result: RouteQuery) -> str:
    """Default route selector mapping RouteQuery to target chain identifiers."""
    datasource = result.datasource.lower() if hasattr(result, "datasource") else str(result).lower()
    if "python_docs" in datasource:
        return "chain for python_docs"
    elif "js_docs" in datasource:
        return "chain for js_docs"
    else:
        return "chain for golang_docs"


def create_routed_chain(
    router: Any,
    route_mapping: Optional[Union[Dict[str, Any], Callable[[Any], Any]]] = None,
):
    """Create a full routing pipeline linking router decision to destination chains or functions."""
    if route_mapping is None:
        route_fn = default_choose_route
    elif isinstance(route_mapping, dict):
        def route_fn(result: Any):
            ds = getattr(result, "datasource", str(result))
            return route_mapping.get(ds, route_mapping.get("default", f"chain for {ds}"))
    else:
        route_fn = route_mapping

    return router | RunnableLambda(route_fn)
