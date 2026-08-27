from typing import List, Sequence, Union

from langchain_core.documents import Document

from src.query_translation.multi_query import get_unique_union


def remove_duplicates(documents: Sequence[Union[Document, Sequence[Document]]]) -> List[Document]:
    """Remove duplicate documents by URL, then by title + content prefix."""
    flattened: List[Document] = []
    nested: List[List[Document]] = []

    for item in documents:
        if isinstance(item, Document):
            flattened.append(item)
        elif isinstance(item, (list, tuple)):
            nested.append(list(item))
        else:
            continue

    unique_by_key: dict[str, Document] = {}
    for document in flattened:
        try:
            metadata = document.metadata if isinstance(document.metadata, dict) else {}
            url = metadata.get("url")
            title = metadata.get("title", "")
            content = document.page_content or ""
            key = str(url) if url else f"{title}{content[:200]}"
        except Exception:
            key = str(getattr(document, "page_content", document))[:200]

        if key not in unique_by_key:
            unique_by_key[key] = document

    deduped = list(unique_by_key.values())

    if nested:
        try:
            extra = get_unique_union(nested + [deduped])
            return remove_duplicates(extra)
        except Exception:
            pass

    return deduped
