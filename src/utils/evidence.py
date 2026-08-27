from typing import List, Optional

from langchain_core.documents import Document


def select_evidence(
    documents: List[Document],
    top_k: int = 5,
) -> List[Document]:
    """Keep the top-k ranked documents as generation evidence."""
    if not documents:
        return []
    k = max(int(top_k), 0)
    return documents[:k]
