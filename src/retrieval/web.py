from typing import List, Optional

from langchain_core.documents import Document

from data_source.web_search import WebSearchSource
from data_source.wikipedia import WikipediaSource


class WebRetriever:
    """Retrieve from Tavily and Wikipedia using the existing source classes."""

    def __init__(
        self,
        wikipedia: Optional[WikipediaSource] = None,
        web_search: Optional[WebSearchSource] = None,
        wikipedia_top_k: int = 5,
        web_top_k: int = 5,
    ):
        self.wikipedia = wikipedia
        self.web_search = web_search
        self.wikipedia_top_k = wikipedia_top_k
        self.web_top_k = web_top_k

        if self.wikipedia is None:
            try:
                self.wikipedia = WikipediaSource(top_k=self.wikipedia_top_k)
            except Exception as exc:
                print(f"Wikipedia source init failed: {exc}")
                self.wikipedia = None

        if self.web_search is None:
            try:
                self.web_search = WebSearchSource(top_k=self.web_top_k)
            except Exception as exc:
                print(f"Tavily source init failed: {exc}")
                self.web_search = None

    def retrieve(self, query: str) -> List[Document]:
        documents: List[Document] = []

        if self.web_search is not None:
            try:
                documents.extend(self.web_search.retrieve(query) or [])
            except Exception as exc:
                print(f"Tavily retrieval failed: {exc}")

        if self.wikipedia is not None:
            try:
                documents.extend(self.wikipedia.retrieve(query) or [])
            except Exception as exc:
                print(f"Wikipedia retrieval failed: {exc}")

        return documents
