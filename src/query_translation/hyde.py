from typing import Any, Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

HYDE_PROMPT_TEMPLATE = """Please write a scientific paper passage that answers the following question.

Question:
{question}

Passage:
"""

HYDE_PROMPT = ChatPromptTemplate.from_template(HYDE_PROMPT_TEMPLATE)

HYDE_ANSWER_PROMPT_TEMPLATE = """Answer the following question based on the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

HYDE_ANSWER_PROMPT = ChatPromptTemplate.from_template(HYDE_ANSWER_PROMPT_TEMPLATE)


def create_hyde_generator(
    llm: Optional[Any] = None,
    prompt: Optional[ChatPromptTemplate] = None,
):
    """Create a chain that generates a hypothetical document answering the query."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if prompt is None:
        prompt = HYDE_PROMPT

    return prompt | llm | StrOutputParser()


def create_hyde_retrieval_chain(
    retriever: Any,
    llm: Optional[Any] = None,
    prompt: Optional[ChatPromptTemplate] = None,
):
    """Create a HyDE retrieval chain: query -> hypothetical doc -> retriever."""
    generator = create_hyde_generator(llm=llm, prompt=prompt)
    return generator | retriever


def create_hyde_rag_chain(
    retriever: Any,
    llm: Optional[Any] = None,
    hyde_prompt: Optional[ChatPromptTemplate] = None,
    answer_prompt: Optional[ChatPromptTemplate] = None,
):
    """Create a complete HyDE RAG pipeline."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if answer_prompt is None:
        answer_prompt = HYDE_ANSWER_PROMPT

    retrieval_chain = create_hyde_retrieval_chain(
        retriever=retriever,
        llm=llm,
        prompt=hyde_prompt,
    )

    return (
        {
            "context": retrieval_chain,
            "question": lambda x: x["question"] if isinstance(x, dict) else x,
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )
