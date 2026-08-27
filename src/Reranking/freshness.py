import re
from datetime import datetime
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from src.Reranking.reranking import CrossEncoderReranker


def extract_document_datetime(document: Document) -> Optional[datetime]:
    """Safely extract a datetime from document metadata or content."""
    try:
        metadata = document.metadata if isinstance(getattr(document, "metadata", None), dict) else {}
    except Exception:
        metadata = {}

    for key in ("published_date", "date", "published_at"):
        metadata_date = metadata.get(key)
        if metadata_date in (None, "", "None"):
            continue
        parsed = _parse_datetime_text(str(metadata_date))
        if parsed is not None:
            return parsed

    try:
        text = f"{getattr(document, 'page_content', '')} {metadata}"
    except Exception:
        return None

    patterns = [
        (
            r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+"
            r"\d{1,2}\s+"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"\s+\d{4}"
            r"(?:\s*\|\s*\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM))?"
        ),
        (
            r"\b(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},\s+\d{4}"
        ),
        r"\b\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?)?",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        parsed = _parse_datetime_text(match.group(0).strip())
        if parsed is not None:
            return parsed

    return None


def _parse_datetime_text(date_text: str) -> Optional[datetime]:
    if not date_text:
        return None

    formats = [
        "%a, %d %b %Y | %I:%M:%S %p",
        "%a %d %b %Y | %I:%M:%S %p",
        "%a, %d %b %Y",
        "%a %d %b %Y",
        "%B %d, %Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
    ]

    cleaned = str(date_text).replace("Z", "").strip()
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except (ValueError, TypeError):
            continue
    return None


def freshness_score(document: Document, now: Optional[datetime] = None) -> float:
    """Return 0.0 when no usable date exists."""
    published = extract_document_datetime(document)
    if published is None:
        return 0.0

    try:
        if published.tzinfo is not None:
            published = published.replace(tzinfo=None)
        current = now or datetime.now()
        age_days = max((current - published).total_seconds() / 86400.0, 0.0)
        return 1.0 / (1.0 + age_days)
    except Exception:
        return 0.0


def apply_freshness_ranking(
    documents: List[Document],
    relevance_scores: Optional[List[float]] = None,
    relevance_weight: float = 0.7,
    freshness_weight: float = 0.3,
    top_k: Optional[int] = None,
) -> List[Document]:
    """Combine relevance scores with freshness. Missing dates do not crash ranking."""
    if not documents:
        return []

    if relevance_scores is None:
        relevance_scores = [0.0] * len(documents)

    combined: List[Tuple[Document, float]] = []
    for document, relevance in zip(documents, relevance_scores):
        try:
            relevance_value = float(relevance)
        except (TypeError, ValueError):
            relevance_value = 0.0
        score = relevance_weight * relevance_value + freshness_weight * freshness_score(document)
        combined.append((document, score))

    combined.sort(key=lambda item: item[1], reverse=True)
    ranked = [document for document, _ in combined]
    if top_k is not None:
        return ranked[:top_k]
    return ranked


def rerank_with_freshness(
    query: str,
    documents: List[Document],
    reranker: CrossEncoderReranker,
    relevance_weight: float = 0.7,
    freshness_weight: float = 0.3,
    top_k: Optional[int] = None,
) -> List[Document]:
    """Cross-encoder relevance first, then a separate freshness ranking stage."""
    if not documents:
        return []

    try:
        scored = reranker.rerank(query=query, documents=documents, top_k=len(documents))
    except Exception:
        scored = [(document, 0.0) for document in documents]

    ranked_docs = [document for document, _ in scored]
    scores = [float(score) for _, score in scored]
    return apply_freshness_ranking(
        documents=ranked_docs,
        relevance_scores=scores,
        relevance_weight=relevance_weight,
        freshness_weight=freshness_weight,
        top_k=top_k,
    )
