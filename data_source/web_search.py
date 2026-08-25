import os
from typing import List, Optional

from langchain_core.documents import Document
from tavily import TavilyClient


class WebSearchSource:
    """Search the web and return results as LangChain Documents."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        top_k: int = 5,
        search_depth: str = "basic",
    ):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

        if not self.api_key:
            raise ValueError(
                "TAVILY_API_KEY environment variable is not set."
            )

        self.top_k = top_k
        self.search_depth = search_depth

        self.client = TavilyClient(
            api_key=self.api_key
        )

    def retrieve(self, query: str) -> List[Document]:
        """Search the web and return results as LangChain Documents."""

        response = self.client.search(
            query=query,
            max_results=self.top_k,
            search_depth=self.search_depth,
        )

        documents: List[Document] = []

        for result in response.get("results", []):
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")

            if not content.strip():
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": "web_search",
                        "title": title,
                        "url": url,
                    },
                )
            )

        return documents