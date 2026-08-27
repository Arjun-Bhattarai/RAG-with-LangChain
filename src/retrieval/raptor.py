from typing import List, Optional

from langchain_core.documents import Document

from src.advanced_indexing.raptor import RaptorIndexer


class RaptorRetriever:
    """Use RAPTOR as an alternative local retrieval strategy, not a post-rerank stage."""

    def __init__(self, indexer: Optional[RaptorIndexer] = None, k: int = 3):
        self.indexer = indexer
        self.k = k

    def is_ready(self) -> bool:
        return bool(self.indexer and getattr(self.indexer, "leaf_summaries", None))

    def retrieve(self, query: str) -> List[Document]:
        if not self.is_ready():
            return []

        try:
            results = self.indexer.retrieve(query=query, k=self.k)
        except Exception as exc:
            print(f"RAPTOR retrieval failed: {exc}")
            return []

        documents: List[Document] = []
        for text in results.get("leaf", []):
            documents.append(
                Document(page_content=str(text), metadata={"source": "raptor_leaf"})
            )
        for text in results.get("clusters", []):
            documents.append(
                Document(page_content=str(text), metadata={"source": "raptor_cluster"})
            )
        return documents
