from typing import Any, List, Optional
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama


def format_docs(docs: List[Document]) -> str:
    """Format a list of documents into a single string context."""
    return "\n\n".join(doc.page_content for doc in docs)


DEFAULT_BASIC_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{question}"),
    ]
)


def create_basic_rag_chain(
    retriever: Any,
    llm: Optional[Any] = None,
    prompt: Optional[ChatPromptTemplate] = None,
):
    """Create a standard RAG chain using the provided retriever, LLM, and prompt."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if prompt is None:
        prompt = DEFAULT_BASIC_RAG_PROMPT

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


class BasicRAG:
    """Encapsulates the basic RAG workflow."""

    def __init__(
        self,
        retriever: Any,
        llm: Optional[Any] = None,
        prompt: Optional[ChatPromptTemplate] = None,
    ):
        self.retriever = retriever
        self.llm = llm or ChatOllama(model="llama3:latest", temperature=0)
        self.prompt = prompt or DEFAULT_BASIC_RAG_PROMPT
        self.chain = create_basic_rag_chain(
            retriever=self.retriever,
            llm=self.llm,
            prompt=self.prompt,
        )

    def invoke(self, question: str) -> str:
        """Run question through the RAG chain and return response."""
        return self.chain.invoke(question)
