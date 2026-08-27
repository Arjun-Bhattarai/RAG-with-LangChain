from typing import Any, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.core.models import EvidenceEvaluation


grader_prompt = ChatPromptTemplate.from_template(
    """
You are a retrieval relevance grader.

Your task is to determine whether the following
document is relevant to the user's question.

Question:
{question}

Document:
{document}

If the document contains information that can help
answer the question, respond with:

YES

Otherwise respond with:

NO

Respond with only YES or NO.
"""
)

generation_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful RAG assistant.

Answer the question using only the provided context.

Context:
{context}

Question:
{question}

If the context does not contain enough information
to answer the question, say that you do not have
enough information.

Answer:
"""
)


def _normalize_yes_no(value: str) -> str:
    text = str(value).strip().upper()
    if text.startswith("YES"):
        return "YES"
    if text.startswith("NO"):
        return "NO"
    return "UNKNOWN"


class CRAGEvaluator:
    """CRAG as an evidence evaluation / corrective-retrieval controller."""

    def __init__(self, llm: Optional[Any] = None, min_relevant: int = 1):
        self.llm = llm or ChatOllama(model="llama3:latest", temperature=0)
        self.min_relevant = min_relevant
        self.retrieval_grader = grader_prompt | self.llm | StrOutputParser()
        self.generation_chain = generation_prompt | self.llm | StrOutputParser()

    def grade_documents(self, question: str, documents: List[Any]) -> List[Any]:
        relevant_docs = []
        for i, doc in enumerate(documents, start=1):
            try:
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                grade = self.retrieval_grader.invoke(
                    {"question": question, "document": content}
                )
                grade = _normalize_yes_no(grade)
            except Exception as exc:
                print(f"CRAG grading failed for document {i}: {exc}")
                continue

            print(f"Document {i} grade: {grade}")
            if grade == "YES":
                relevant_docs.append(doc)
        return relevant_docs

    def evaluate(self, query: str, documents: List[Any]) -> EvidenceEvaluation:
        if not documents:
            return EvidenceEvaluation(
                sufficient=False,
                requires_more_retrieval=True,
                relevant_documents=[],
                reason="No documents retrieved.",
            )

        relevant_docs = self.grade_documents(query, documents)
        sufficient = len(relevant_docs) >= self.min_relevant
        return EvidenceEvaluation(
            sufficient=sufficient,
            requires_more_retrieval=not sufficient,
            relevant_documents=relevant_docs,
            reason=(
                f"Found {len(relevant_docs)} relevant documents."
                if sufficient
                else "Insufficient relevant evidence."
            ),
        )


def crag(question, retriever, vectorstore, llm=None):
    """Standalone CRAG pipeline kept for independent testing."""
    evaluator = CRAGEvaluator(llm=llm)

    retrieved_docs = retriever.invoke(question)
    print(f"Initial retrieved documents: {len(retrieved_docs)}")

    evaluation = evaluator.evaluate(question, retrieved_docs)
    relevant_docs = evaluation.relevant_documents

    if evaluation.requires_more_retrieval:
        print("No relevant documents found.")
        print("Applying corrective retrieval...")
        corrective_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        relevant_docs = corrective_retriever.invoke(question)
    else:
        print(f"Found {len(relevant_docs)} relevant documents.")

    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    return evaluator.generation_chain.invoke(
        {"context": context, "question": question}
    )
