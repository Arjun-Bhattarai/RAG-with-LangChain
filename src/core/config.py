import os
from dataclasses import dataclass


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RAGConfig:
    """Central configuration for the integrated RAG pipeline."""

    ollama_model: str = "llama3:latest"
    embedding_model: str = "nomic-embed-text:latest"
    ollama_base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.0

    top_k: int = 5
    retrieval_k: int = 5
    rerank_top_k: int = 10
    final_top_k: int = 5
    wikipedia_top_k: int = 5
    web_top_k: int = 5
    max_query_variations: int = 5

    freshness_weight: float = 0.3
    relevance_weight: float = 0.7

    raptor_clusters: int = 3
    raptor_k: int = 3
    index_batch_size: int = 32
    chunk_size: int = 256
    chunk_overlap: int = 50

    min_evidence_docs: int = 1
    long_context_min_chars: int = 4000
    max_retrieval_retries: int = 1

    enable_multi_query: bool = True
    enable_raptor: bool = True
    enable_crag: bool = True
    enable_self_rag: bool = True
    enable_long_context: bool = True
    enable_evaluation: bool = True

    @classmethod
    def from_env(cls) -> "RAGConfig":
        defaults = cls()
        return cls(
            ollama_model=_env_str("OLLAMA_MODEL", defaults.ollama_model),
            embedding_model=_env_str("EMBEDDING_MODEL", defaults.embedding_model),
            ollama_base_url=_env_str("OLLAMA_BASE_URL", defaults.ollama_base_url),
            temperature=_env_float("TEMPERATURE", defaults.temperature),
            top_k=_env_int("TOP_K", defaults.top_k),
            retrieval_k=_env_int("RETRIEVAL_K", defaults.retrieval_k),
            rerank_top_k=_env_int("RERANK_TOP_K", defaults.rerank_top_k),
            final_top_k=_env_int("FINAL_TOP_K", defaults.final_top_k),
            wikipedia_top_k=_env_int("WIKIPEDIA_TOP_K", defaults.wikipedia_top_k),
            web_top_k=_env_int("WEB_TOP_K", defaults.web_top_k),
            max_query_variations=_env_int(
                "MAX_QUERY_VARIATIONS", defaults.max_query_variations
            ),
            freshness_weight=_env_float("FRESHNESS_WEIGHT", defaults.freshness_weight),
            relevance_weight=_env_float("RELEVANCE_WEIGHT", defaults.relevance_weight),
            raptor_clusters=_env_int("RAPTOR_CLUSTERS", defaults.raptor_clusters),
            raptor_k=_env_int("RAPTOR_K", defaults.raptor_k),
            enable_multi_query=_env_bool("ENABLE_MULTI_QUERY", defaults.enable_multi_query),
            enable_raptor=_env_bool("ENABLE_RAPTOR", defaults.enable_raptor),
            enable_crag=_env_bool("ENABLE_CRAG", defaults.enable_crag),
            enable_self_rag=_env_bool("ENABLE_SELF_RAG", defaults.enable_self_rag),
            enable_long_context=_env_bool(
                "ENABLE_LONG_CONTEXT", defaults.enable_long_context
            ),
            enable_evaluation=_env_bool("ENABLE_EVALUATION", defaults.enable_evaluation),
        )
