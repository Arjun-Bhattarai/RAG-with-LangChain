from operator import itemgetter
from typing import Any, List, Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

DECOMPOSITION_PROMPT_TEMPLATE = """You are a helpful assistant that generates multiple sub-questions related to an input question.

The goal is to break down the input into a set of sub-problems or sub-questions that can be answered independently.

Generate 3 sub-questions related to:

{question}

Output (3 queries):
"""

DECOMPOSITION_PROMPT = ChatPromptTemplate.from_template(DECOMPOSITION_PROMPT_TEMPLATE)

SEQUENTIAL_QA_PROMPT_TEMPLATE = """Here is the question you need to answer:

--------------------
{question}
--------------------

Here are the available background question and answer pairs:

--------------------
{q_a_pairs}
--------------------

Here is additional context relevant to the question:

--------------------
{context}
--------------------

Use the retrieved context and the background question-answer pairs to answer the original question.

Answer:
"""

SEQUENTIAL_QA_PROMPT = ChatPromptTemplate.from_template(SEQUENTIAL_QA_PROMPT_TEMPLATE)

PARALLEL_SUB_QUESTION_PROMPT_TEMPLATE = """Answer the following question based only on the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

PARALLEL_SUB_QUESTION_PROMPT = ChatPromptTemplate.from_template(PARALLEL_SUB_QUESTION_PROMPT_TEMPLATE)

SYNTHESIS_PROMPT_TEMPLATE = """Here is a set of Question and Answer pairs:

{context}

Use these Question-Answer pairs to generate the final answer to the following question.

Question:
{question}

Answer:
"""

SYNTHESIS_PROMPT = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT_TEMPLATE)


def format_qa_pair(question: str, answer: str) -> str:
    """Format a single question and answer pair."""
    return f"Question: {question}\nAnswer: {answer}".strip()


def format_qa_pairs(questions: List[str], answers: List[str]) -> str:
    """Format multiple question and answer pairs into a single context string."""
    formatted = []
    for i, (q, a) in enumerate(zip(questions, answers), start=1):
        formatted.append(f"Question {i}: {q}\nAnswer {i}: {a}")
    return "\n\n".join(formatted).strip()


def create_sub_question_generator(
    llm: Optional[Any] = None,
    prompt: Optional[ChatPromptTemplate] = None,
):
    """Create a chain that generates sub-questions from a complex question."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if prompt is None:
        prompt = DECOMPOSITION_PROMPT

    return (
        prompt
        | llm
        | StrOutputParser()
        | (lambda x: [q.strip() for q in x.split("\n") if q.strip() and not q.strip().startswith("Here are")])
    )


def retrieve_and_rag(
    question: str,
    retriever: Any,
    llm: Optional[Any] = None,
    prompt_rag: Optional[ChatPromptTemplate] = None,
    decomposition_prompt: Optional[ChatPromptTemplate] = None,
    sub_question_generator: Optional[Any] = None,
):
    """Answer each decomposed sub-question independently and return answers plus sub-questions."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if prompt_rag is None:
        prompt_rag = PARALLEL_SUB_QUESTION_PROMPT
    if sub_question_generator is None:
        sub_question_generator = create_sub_question_generator(
            llm=llm,
            prompt=decomposition_prompt,
        )

    sub_questions = sub_question_generator.invoke({"question": question})
    rag_results = []

    for sub_question in sub_questions:
        retrieved_docs = retriever.invoke(sub_question)
        answer = (prompt_rag | llm | StrOutputParser()).invoke(
            {
                "context": retrieved_docs,
                "question": sub_question,
            }
        )
        rag_results.append(answer)

    return rag_results, sub_questions


def sequential_query_decomposition(
    question: str,
    retriever: Any,
    llm: Optional[Any] = None,
    decomposition_prompt: Optional[ChatPromptTemplate] = None,
    qa_prompt: Optional[ChatPromptTemplate] = None,
) -> str:
    """Execute Sequential Query Decomposition where each sub-answer informs subsequent ones."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if qa_prompt is None:
        qa_prompt = SEQUENTIAL_QA_PROMPT

    generator = create_sub_question_generator(llm=llm, prompt=decomposition_prompt)
    sub_questions = generator.invoke({"question": question})

    q_a_pairs = ""
    last_answer = ""

    for sub_q in sub_questions:
        rag_chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
                "q_a_pairs": itemgetter("q_a_pairs"),
            }
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        last_answer = rag_chain.invoke(
            {
                "question": sub_q,
                "q_a_pairs": q_a_pairs,
            }
        )

        pair_str = format_qa_pair(sub_q, last_answer)
        q_a_pairs = f"{q_a_pairs}\n---\n{pair_str}" if q_a_pairs else pair_str

    return last_answer


def parallel_query_decomposition(
    question: str,
    retriever: Any,
    llm: Optional[Any] = None,
    decomposition_prompt: Optional[ChatPromptTemplate] = None,
    sub_qa_prompt: Optional[ChatPromptTemplate] = None,
    synthesis_prompt: Optional[ChatPromptTemplate] = None,
) -> str:
    """Execute Parallel Query Decomposition where sub-questions are answered independently and synthesized."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if sub_qa_prompt is None:
        sub_qa_prompt = PARALLEL_SUB_QUESTION_PROMPT
    if synthesis_prompt is None:
        synthesis_prompt = SYNTHESIS_PROMPT

    sub_answers, sub_questions = retrieve_and_rag(
        question=question,
        retriever=retriever,
        llm=llm,
        prompt_rag=sub_qa_prompt,
        decomposition_prompt=decomposition_prompt,
    )

    context = format_qa_pairs(sub_questions, sub_answers)

    final_chain = synthesis_prompt | llm | StrOutputParser()
    return final_chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )
