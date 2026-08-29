from __future__ import annotations

import os

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


class PersonalWebsiteSource:
    """Loads personal information from Arjun Bhattarai's website."""

    def __init__(self, timeout: int = 10):
        self.url = os.getenv("PERSONAL_WEBSITE_URL")
        self.timeout = timeout

        if not self.url:
            raise ValueError(
                "PERSONAL_WEBSITE_URL is not configured in .env"
            )

    def retrieve(self, query: str | None = None) -> list[Document]:
        try:
            response = requests.get(
                self.url,
                timeout=self.timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()

        except requests.RequestException as exc:
            print(f"Personal website retrieval failed: {exc}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        for element in soup(
            ["script", "style", "noscript", "nav", "footer"]
        ):
            element.decompose()

        text = soup.get_text(
            separator="\n",
            strip=True,
        )

        if not text:
            return []

        return [
            Document(
                page_content=text,
                metadata={
                    "source": "personal_website",
                    "title": "Arjun Bhattarai",
                    "url": self.url,
                },
            )
        ]