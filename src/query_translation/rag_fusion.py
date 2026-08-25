from operator import itemgetter
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

RAG_FUSION_PROMPT_TEMPLATE = """You are a helpful assistant that generates multiple search queries based on a single input query.

Generate multiple search queries related to the following question:

{question}

Output (4 queries):
"""

RAG_FUSION_PROMPT = ChatPromptTemplate.from_template(RAG_FUSION_PROMPT_TEMPLATE)

RAG_FUSION_ANSWER_PROMPT_TEMPLATE = """Answer the following question based only on the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

RAG_FUSION_ANSWER_PROMPT = ChatPromptTemplate.from_template(RAG_FUSION_ANSWER_PROMPT_TEMPLATE)


def reciprocal_rank_fusion(
    results: List[List[Document]],
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """Rerank retrieved document lists using Reciprocal Rank Fusion (RRF)."""
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0.0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc_str), score)
        for doc_str, score in sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    ]
    return reranked_results


def create_rag_fusion_generator(
    llm: Optional[Any] = None,
    prompt: Optional[ChatPromptTemplate] = None,
):
    """Create a chain that generates multiple queries for RAG-Fusion."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if prompt is None:
        prompt = RAG_FUSION_PROMPT

    return (
        prompt
        | llm
        | StrOutputParser()
        | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    )


def create_rag_fusion_retrieval_chain(
    retriever: Any,
    llm: Optional[Any] = None,
    prompt: Optional[ChatPromptTemplate] = None,
    k: int = 60,
):
    """Create a retrieval chain combining multiple query results using RRF."""
    generator = create_rag_fusion_generator(llm=llm, prompt=prompt)
    return generator | retriever.map() | (lambda results: reciprocal_rank_fusion(results, k=k))


def create_rag_fusion_rag_chain(
    retriever: Any,
    llm: Optional[Any] = None,
    fusion_prompt: Optional[ChatPromptTemplate] = None,
    answer_prompt: Optional[ChatPromptTemplate] = None,
    k: int = 60,
):
    """Create a full RAG-Fusion generation chain."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if answer_prompt is None:
        answer_prompt = RAG_FUSION_ANSWER_PROMPT

    retrieval_chain = create_rag_fusion_retrieval_chain(
        retriever=retriever,
        llm=llm,
        prompt=fusion_prompt,
        k=k,
    )

    final_rag_chain = (
        {
            "context": retrieval_chain,
            "question": itemgetter("question"),
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )
    return final_rag_chain
