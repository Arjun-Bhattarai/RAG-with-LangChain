from typing import Any, List, Optional

from langchain_core.documents import Document

from src.pipeline.retrieval_rag import retrieve_documents


class ChromaRetriever:
    """Thin wrapper around the existing Chroma retriever helpers."""

    def __init__(self, retriever: Optional[Any] = None):
        self.retriever = retriever

    def set_retriever(self, retriever: Any) -> None:
        self.retriever = retriever

    def retrieve(self, query: str) -> List[Document]:
        if self.retriever is None:
            return []
        try:
            documents = retrieve_documents(self.retriever, query)
            return documents or []
        except Exception as exc:
            print(f"Local Chroma retrieval failed: {exc}")
            return []
