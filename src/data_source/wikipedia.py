from typing import List

import requests
from langchain_core.documents import Document


class WikipediaSource:
    """Retrieve Wikipedia articles and return LangChain Documents."""

    def __init__(
        self,
        language: str = "en",
        top_k: int = 5,
    ):
        self.language = language
        self.top_k = top_k

        self.base_url = (
            f"https://{self.language}.wikipedia.org"
        )

        self.headers = {
            "User-Agent": (
                "RAG-from-scratch/1.0 "
                "(educational RAG project)"
            ),
            "Accept": "application/json",
        }

    def search_titles(self, query: str) -> List[str]:
        """Search Wikipedia and return relevant article titles."""

        url = f"{self.base_url}/w/rest.php/v1/search/page"

        params = {
            "q": query,
            "limit": self.top_k,
        }

        response = requests.get(
            url,
            params=params,
            headers=self.headers,
            timeout=15,
        )

        if not response.ok:
            raise requests.HTTPError(
                f"Wikipedia search failed "
                f"({response.status_code}): "
                f"{response.text[:300]}",
                response=response,
            )

        data = response.json()

        pages = data.get("pages", [])

        return [
            page["title"]
            for page in pages
            if page.get("title")
        ]

    def get_article(self, title: str) -> Document:
        """Retrieve a Wikipedia article as a LangChain Document."""

        encoded_title = requests.utils.quote(
            title.replace(" ", "_"),
            safe="_()",
        )

        url = (
            f"{self.base_url}/w/rest.php/v1/page/"
            f"{encoded_title}"
        )

        response = requests.get(
            url,
            headers=self.headers,
            timeout=15,
        )

        if not response.ok:
            raise requests.HTTPError(
                f"Wikipedia article request failed "
                f"({response.status_code}) for '{title}': "
                f"{response.text[:300]}",
                response=response,
            )

        data = response.json()

        content = data.get("source", "")

        return Document(
            page_content=content,
            metadata={
                "source": "wikipedia",
                "title": title,
                "url": (
                    f"{self.base_url}/wiki/"
                    f"{encoded_title}"
                ),
            },
        )

    def retrieve(self, query: str) -> List[Document]:
        """Search Wikipedia and retrieve matching articles."""

        titles = self.search_titles(query)

        documents: List[Document] = []

        for title in titles:
            try:
                document = self.get_article(title)

                if document.page_content.strip():
                    documents.append(document)

            except requests.RequestException as exc:
                print(
                    f"Failed to retrieve '{title}': {exc}"
                )

        return documents