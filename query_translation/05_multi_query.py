from operator import itemgetter
from typing import Any, List, Optional
from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

MULTI_QUERY_PROMPT_TEMPLATE = """You are an AI language model assistant.

Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database.

By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search.

Provide these alternative questions separated by new lines.

Original question: {question}
"""

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT_TEMPLATE)

MULTI_QUERY_RAG_PROMPT_TEMPLATE = """Answer the following question based only on the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

MULTI_QUERY_RAG_PROMPT = ChatPromptTemplate.from_template(MULTI_QUERY_RAG_PROMPT_TEMPLATE)


def get_unique_union(documents: List[List[Document]]) -> List[Document]:
    """Deduplicate a list of lists of documents while preserving Document objects."""
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]


def create_multi_query_generator(
    llm: Optional[Any] = None,
    prompt: Optional[ChatPromptTemplate] = None,
):
    """Create a chain that generates multiple query perspectives from a single question."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if prompt is None:
        prompt = MULTI_QUERY_PROMPT

    return (
        prompt
        | llm
        | StrOutputParser()
        | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    )


def create_multi_query_retrieval_chain(
    retriever: Any,
    llm: Optional[Any] = None,
    prompt: Optional[ChatPromptTemplate] = None,
):
    """Create a multi-query retrieval chain returning unique deduplicated documents."""
    query_generator = create_multi_query_generator(llm=llm, prompt=prompt)
    return query_generator | retriever.map() | get_unique_union


def create_multi_query_rag_chain(
    retriever: Any,
    llm: Optional[Any] = None,
    multi_query_prompt: Optional[ChatPromptTemplate] = None,
    rag_prompt: Optional[ChatPromptTemplate] = None,
):
    """Create a full Multi-Query RAG generation chain."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if rag_prompt is None:
        rag_prompt = MULTI_QUERY_RAG_PROMPT

    retrieval_chain = create_multi_query_retrieval_chain(
        retriever=retriever,
        llm=llm,
        prompt=multi_query_prompt,
    )

    final_rag_chain = (
        {
            "context": retrieval_chain,
            "question": itemgetter("question"),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return final_rag_chain
