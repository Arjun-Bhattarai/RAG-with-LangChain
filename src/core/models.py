from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document

LOCAL = "LOCAL"
WEB = "WEB"
HYBRID = "HYBRID"
VALID_ROUTES = (LOCAL, WEB, HYBRID)


@dataclass
class EvidenceEvaluation:
    sufficient: bool
    requires_more_retrieval: bool
    relevant_documents: List[Document] = field(default_factory=list)
    reason: str = ""


@dataclass
class PipelineResult:
    query: str
    route: str
    queries: List[str]
    answer: str
    context: str
    retrieved_documents: List[Document] = field(default_factory=list)
    reranked_documents: List[Document] = field(default_factory=list)
    evidence: List[Document] = field(default_factory=list)
    evaluation: Optional[dict] = None
    evidence_evaluation: Optional[EvidenceEvaluation] = None
    long_context_used: bool = False
    raptor_used: bool = False
