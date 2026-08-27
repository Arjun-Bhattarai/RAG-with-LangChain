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
        self.top_k = top_k
        self.search_depth = search_depth
        self.client = None

        if not self.api_key:
            print("TAVILY_API_KEY is not set; web search will return no results.")
            return

        try:
            self.client = TavilyClient(api_key=self.api_key)
        except Exception as exc:
            print(f"Tavily client init failed: {exc}")

    def retrieve(self, query: str) -> List[Document]:
        """Search the web and return results as LangChain Documents."""

        if self.client is None:
            return []

        try:
            response = self.client.search(
                query=query,
                max_results=self.top_k,
                search_depth=self.search_depth,
            )
        except Exception as exc:
            print(f"Tavily request failed: {exc}")
            return []

        if not isinstance(response, dict):
            return []

        documents: List[Document] = []

        for result in response.get("results", []) or []:
            if not isinstance(result, dict):
                continue
            title = result.get("title", "")
            content = result.get("content") or ""
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
                        "published_date": result.get("published_date"),
                    },
                )
            )

        return documents