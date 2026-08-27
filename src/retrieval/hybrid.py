from typing import List, Optional

from langchain_core.documents import Document

from src.retrieval.chroma import ChromaRetriever
from src.retrieval.raptor import RaptorRetriever
from src.retrieval.web import WebRetriever


class HybridRetriever:
    """Combine local Chroma/RAPTOR retrieval with web retrieval."""

    def __init__(
        self,
        local: Optional[ChromaRetriever] = None,
        web: Optional[WebRetriever] = None,
        raptor: Optional[RaptorRetriever] = None,
        use_raptor: bool = True,
    ):
        self.local = local or ChromaRetriever()
        self.web = web or WebRetriever()
        self.raptor = raptor
        self.use_raptor = use_raptor

    def retrieve_local(self, query: str) -> List[Document]:
        documents = self.local.retrieve(query)
        if self.use_raptor and self.raptor is not None:
            documents.extend(self.raptor.retrieve(query))
        return documents

    def retrieve_web(self, query: str) -> List[Document]:
        return self.web.retrieve(query)

    def retrieve(self, query: str) -> List[Document]:
        return self.retrieve_local(query) + self.retrieve_web(query)
