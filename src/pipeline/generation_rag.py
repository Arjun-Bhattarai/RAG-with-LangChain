from typing import Any, List, Optional
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama


def format_docs(docs: List[Document]) -> str:
    """Format a list of documents into a single string context."""
    if not docs:
        return ""
    parts = []
    for doc in docs:
        content = getattr(doc, "page_content", None)
        if content is None:
            content = str(doc)
        text = str(content).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


RAG_PROMPT_TEMPLATE = """You are an assistant for question-answering tasks.

Use the following retrieved context to answer the question.

If you don't know the answer, just say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

DEFAULT_GENERATION_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

CONTEXT_ONLY_PROMPT_TEMPLATE = """Answer the question based only on the following context.

Context:
{context}

Question:
{question}
"""

CONTEXT_ONLY_PROMPT = ChatPromptTemplate.from_template(CONTEXT_ONLY_PROMPT_TEMPLATE)


def create_generation_chain(
    retriever: Any,
    llm: Optional[Any] = None,
    prompt: Optional[ChatPromptTemplate] = None,
):
    """Create a RAG generation chain that formats retrieved documents and invokes LLM."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if prompt is None:
        prompt = DEFAULT_GENERATION_PROMPT

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def create_answer_chain(
    llm: Optional[Any] = None,
    prompt: Optional[ChatPromptTemplate] = None,
):
    """Create a generation chain that answers from already-built context."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if prompt is None:
        prompt = DEFAULT_GENERATION_PROMPT
    return prompt | llm | StrOutputParser()
