from importlib import import_module
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document


def reciprocal_rank_fusion(results: List[List[Document]], k: int = 60):
    """Rerank multiple retrieval lists using Reciprocal Rank Fusion from RAG-Fusion."""
    fusion_mod = import_module("query_translation.06_Rag_fusion")
    return fusion_mod.reciprocal_rank_fusion(results, k=k)


class CrossEncoderReranker:
    """Reranker using a HuggingFace CrossEncoder model to rescore and order retrieved documents."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self._model: Optional[Any] = None

    @property
    def model(self):
        """Lazy load the CrossEncoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """Score each document against the query and return sorted (Document, score) pairs."""
        if not documents:
            return []

        k = top_k if top_k is not None else self.top_k
        pairs = [
            (query, doc.page_content if hasattr(doc, "page_content") else str(doc))
            for doc in documents
        ]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """Return the top-k reordered Document objects."""
        ranked_pairs = self.rerank(query, documents, top_k=top_k)
        return [doc for doc, _ in ranked_pairs]
