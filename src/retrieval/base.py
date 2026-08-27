from typing import List, Protocol

from langchain_core.documents import Document


class Retriever(Protocol):
    def retrieve(self, query: str) -> List[Document]:
        ...
