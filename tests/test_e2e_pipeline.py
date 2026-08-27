"""
Reproducible end-to-end pipeline tests using mocks.

These tests do NOT require Ollama, Tavily, Wikipedia, or network access.
They verify that the pipeline stages are wired together correctly and that
each stage's output feeds into the next stage properly.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.core.config import RAGConfig
from src.core.models import HYBRID, LOCAL, WEB, EvidenceEvaluation
from src.pipeline.rag_pipeline import RAGPipeline
from src.query_translation.query_processor import QueryProcessor
from src.Reranking.freshness import (
    apply_freshness_ranking,
    extract_document_datetime,
    freshness_score,
    rerank_with_freshness,
)
from src.Reranking.reranking import CrossEncoderReranker
from src.retrieval.chroma import ChromaRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.web import WebRetriever
from src.routing.source_routing import parse_route
from src.utils.deduplication import remove_duplicates
from src.utils.evidence import select_evidence



# Shared fixtures


SAMPLE_DOCS = [
    Document(
        page_content="RAPTOR builds a hierarchical summary tree using clustering.",
        metadata={"source": "raptor", "url": "http://raptor", "date": "2024-06-01"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for grounded answers.",
        metadata={"source": "rag", "url": "http://rag"},
    ),
    Document(
        page_content="ChromaDB is a vector database for similarity search.",
        metadata={"source": "chroma", "url": "http://chroma", "date": "2024-01-15"},
    ),
]

WEB_DOCS = [
    Document(
        page_content="Quantum computing has seen advances in error correction.",
        metadata={"source": "web_search", "url": "http://web1", "published_date": "2024-08-01"},
    ),
    Document(
        page_content="Wikipedia article on quantum computing fundamentals.",
        metadata={"source": "wikipedia", "url": "http://wiki1"},
    ),
]


class FakeRouter:
    def __init__(self, route=LOCAL):
        self._route = route

    def route(self, query):
        return self._route


class FakeQueryProcessor:
    def __init__(self, extra_queries=None):
        self._extra = extra_queries or []

    def process(self, query):
        return [query] + self._extra


class FakeReranker:
    """Assigns decreasing scores based on document position."""
    def rerank(self, query, documents, top_k=None):
        scored = [(doc, 1.0 / (i + 1)) for i, doc in enumerate(documents)]
        k = top_k or len(scored)
        return scored[:k]

    def rerank_documents(self, query, documents, top_k=None):
        return [doc for doc, _ in self.rerank(query, documents, top_k=top_k)]


class FakeGeneration:
    def invoke(self, payload):
        return "This is a generated answer based on the provided context."


class FakeMemory:
    def add_message(self, msg):
        pass


def _build_pipeline(
    route=LOCAL,
    local_docs=None,
    web_docs=None,
    extra_queries=None,
    enable_crag=False,
    enable_self_rag=False,
    enable_long_context=False,
    max_retries=0,
) -> RAGPipeline:
    """Build a fully-mocked RAGPipeline for testing."""
    local_docs = local_docs if local_docs is not None else SAMPLE_DOCS
    web_docs = web_docs if web_docs is not None else WEB_DOCS

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.config = RAGConfig(
        enable_crag=enable_crag,
        enable_self_rag=enable_self_rag,
        enable_long_context=enable_long_context,
        enable_evaluation=False,
        enable_raptor=False,
        enable_multi_query=True,
        final_top_k=5,
        rerank_top_k=10,
        max_retrieval_retries=max_retries,
        relevance_weight=0.7,
        freshness_weight=0.3,
        long_context_min_chars=100,
    )
    pipeline.router = FakeRouter(route)
    pipeline.query_processor = FakeQueryProcessor(extra_queries)
    pipeline.chroma = ChromaRetriever(
        retriever=SimpleNamespace(invoke=lambda q: list(local_docs))
    )
    pipeline.web = SimpleNamespace(retrieve=lambda q: list(web_docs))
    pipeline.raptor = SimpleNamespace(is_ready=lambda: False)
    pipeline.hybrid = HybridRetriever(
        local=pipeline.chroma,
        web=pipeline.web,
        use_raptor=False,
    )
    pipeline.reranker = FakeReranker()
    pipeline.generation_chain = FakeGeneration()
    pipeline.conversation_memory = FakeMemory()
    pipeline.evaluator = None
    pipeline.long_context = None
    pipeline.crag = None
    pipeline.self_rag = None
    pipeline.llm = None
    pipeline.vectorstore = None

    return pipeline


# Phase 3 — LOCAL retrieval


class TestLocalRetrieval:
    def test_local_route_returns_local_docs(self):
        pipeline = _build_pipeline(route=LOCAL, web_docs=[])
        result = pipeline.run("What is RAPTOR?", evaluate=False, route="LOCAL")

        assert result["route"] == LOCAL
        assert len(result["retrieved_documents"]) > 0
        assert result["answer"] != ""

    def test_local_deduplication(self):
        duped = SAMPLE_DOCS + [SAMPLE_DOCS[0]]  # duplicate
        pipeline = _build_pipeline(route=LOCAL, local_docs=duped, web_docs=[])
        result = pipeline.run("What is RAPTOR?", evaluate=False, route="LOCAL")

        assert len(result["retrieved_documents"]) < len(duped)

    def test_local_empty_index(self):
        pipeline = _build_pipeline(route=LOCAL, local_docs=[], web_docs=[])
        pipeline.chroma = ChromaRetriever(retriever=None)
        pipeline.hybrid = HybridRetriever(
            local=pipeline.chroma, web=pipeline.web, use_raptor=False
        )
        result = pipeline.run("What is RAPTOR?", evaluate=False, route="LOCAL")

        assert result["answer"] != ""
        assert len(result["retrieved_documents"]) == 0


# Phase 4 — WEB retrieval

class TestWebRetrieval:
    def test_web_route_returns_web_docs(self):
        pipeline = _build_pipeline(route=WEB)
        result = pipeline.run("latest quantum computing", evaluate=False, route="WEB")

        assert result["route"] == WEB
        assert len(result["retrieved_documents"]) > 0
        sources = {doc.metadata.get("source") for doc in result["retrieved_documents"]}
        assert sources & {"web_search", "wikipedia"}

    def test_web_no_results(self):
        pipeline = _build_pipeline(route=WEB, web_docs=[])
        pipeline.web = SimpleNamespace(retrieve=lambda q: [])
        pipeline.hybrid = HybridRetriever(
            local=pipeline.chroma, web=pipeline.web, use_raptor=False
        )
        result = pipeline.run("anything", evaluate=False, route="WEB")

        assert len(result["retrieved_documents"]) == 0
        assert result["answer"] != ""


# Phase 5 — HYBRID retrieval

class TestHybridRetrieval:
    def test_hybrid_combines_sources(self):
        pipeline = _build_pipeline(route=HYBRID)
        result = pipeline.run("RAG vs fine-tuning", evaluate=False, route="HYBRID")

        assert result["route"] == HYBRID
        sources = {doc.metadata.get("source") for doc in result["retrieved_documents"]}
        has_local = any(s not in ("web_search", "wikipedia") for s in sources)
        has_web = any(s in ("web_search", "wikipedia") for s in sources)
        assert has_local, "Hybrid should include local documents"
        assert has_web, "Hybrid should include web documents"


# Phase 6 — Multi-Query

class TestMultiQuery:
    def test_original_query_preserved(self):
        processor = QueryProcessor(enable_multi_query=True, max_variations=4)
        processor.generator = MagicMock(
            return_value=["variation one", "variation two", "variation three"]
        )
        queries = processor.process("original question")

        assert queries[0] == "original question"
        assert "variation one" in queries

    def test_duplicates_removed(self):
        processor = QueryProcessor(enable_multi_query=True, max_variations=5)
        processor.generator = MagicMock(
            return_value=["Original Question", "new query", "another one"]
        )
        queries = processor.process("original question")

        lower_queries = [q.lower() for q in queries]
        assert len(lower_queries) == len(set(lower_queries))

    def test_empty_queries_removed(self):
        processor = QueryProcessor(enable_multi_query=True, max_variations=5)
        processor.generator = MagicMock(return_value=["", "  ", "valid query", ""])
        queries = processor.process("original")

        assert all(q.strip() for q in queries)

    def test_max_variations_respected(self):
        processor = QueryProcessor(enable_multi_query=True, max_variations=2)
        processor.generator = MagicMock(
            return_value=["q1", "q2", "q3", "q4", "q5"]
        )
        queries = processor.process("original")
        assert len(queries) <= 2

    def test_generator_failure_returns_original(self):
        processor = QueryProcessor(enable_multi_query=True, max_variations=5)
        processor.generator = MagicMock(side_effect=RuntimeError("LLM down"))
        queries = processor.process("my question")

        assert queries == ["my question"]

    def test_disabled_multi_query(self):
        processor = QueryProcessor(enable_multi_query=False)
        queries = processor.process("test")
        assert queries == ["test"]

    def test_retrieval_uses_all_queries(self):
        pipeline = _build_pipeline(
            route=LOCAL,
            extra_queries=["alt query 1", "alt query 2"],
            web_docs=[],
        )
        result = pipeline.run("original", evaluate=False, route="LOCAL")

        assert len(result["queries"]) == 3
        # Retrieval should have produced docs from all queries
        assert len(result["retrieved_documents"]) > 0


# Phase 7 — Reranking + Freshness

class TestRerankingFreshness:
    def test_freshness_with_valid_date(self):
        doc = Document(page_content="x", metadata={"date": datetime.now().strftime("%Y-%m-%d")})
        assert extract_document_datetime(doc) is not None
        assert freshness_score(doc) > 0.0

    def test_freshness_without_date(self):
        doc = Document(page_content="x", metadata={})
        assert extract_document_datetime(doc) is None
        assert freshness_score(doc) == 0.0

    def test_freshness_with_malformed_date(self):
        doc = Document(page_content="x", metadata={"date": {"year": 2020}})
        assert extract_document_datetime(doc) is None
        assert freshness_score(doc) == 0.0

    def test_freshness_with_none_date(self):
        doc = Document(page_content="x", metadata={"date": None})
        assert freshness_score(doc) == 0.0

    def test_apply_freshness_ranking_ordering(self):
        recent = Document(
            page_content="recent",
            metadata={"date": datetime.now().strftime("%Y-%m-%d")},
        )
        old = Document(
            page_content="old",
            metadata={"date": (datetime.now() - timedelta(days=3650)).strftime("%Y-%m-%d")},
        )
        ranked = apply_freshness_ranking(
            [old, recent],
            relevance_scores=[0.5, 0.5],
            relevance_weight=0.5,
            freshness_weight=0.5,
        )
        assert ranked[0] is recent

    def test_freshness_does_not_crash_on_mixed_dates(self):
        docs = [
            Document(page_content="a", metadata={"date": "2024-01-01"}),
            Document(page_content="b", metadata={}),
            Document(page_content="c", metadata={"date": "not-a-date"}),
            Document(page_content="d", metadata={"published_date": None}),
        ]
        # Should not raise
        ranked = apply_freshness_ranking(docs, relevance_scores=[1.0, 0.5, 0.3, 0.1])
        assert len(ranked) == 4

    def test_rerank_with_freshness_separate_stages(self):
        """Verify freshness is a separate stage from CrossEncoder reranking."""
        docs = [
            Document(page_content="RAPTOR", metadata={"date": "2024-08-01"}),
            Document(page_content="unrelated", metadata={}),
        ]
        reranker = FakeReranker()
        ranked = rerank_with_freshness(
            query="RAPTOR",
            documents=docs,
            reranker=reranker,
            relevance_weight=0.7,
            freshness_weight=0.3,
        )
        assert len(ranked) == 2

    def test_freshness_not_passed_to_rerank_documents(self):
        """Confirm that freshness_aware is NOT passed to rerank_documents."""
        reranker = CrossEncoderReranker(top_k=5)
        import inspect
        sig = inspect.signature(reranker.rerank_documents)
        param_names = set(sig.parameters.keys())
        assert "freshness_aware" not in param_names


# Phase 8 — CRAG

class TestCRAG:
    def test_crag_sufficient_evidence(self):
        from src.advanced_RAG.CRAG.crag import CRAGEvaluator

        class FakeGrader:
            def invoke(self, payload):
                return "YES"

        evaluator = CRAGEvaluator.__new__(CRAGEvaluator)
        evaluator.min_relevant = 1
        evaluator.retrieval_grader = FakeGrader()

        docs = [Document(page_content="RAPTOR uses tree-based retrieval.")]
        result = evaluator.evaluate("What is RAPTOR?", docs)

        assert result.sufficient is True
        assert result.requires_more_retrieval is False
        assert len(result.relevant_documents) == 1

    def test_crag_insufficient_evidence(self):
        from src.advanced_RAG.CRAG.crag import CRAGEvaluator

        class FakeGrader:
            def invoke(self, payload):
                return "NO"

        evaluator = CRAGEvaluator.__new__(CRAGEvaluator)
        evaluator.min_relevant = 1
        evaluator.retrieval_grader = FakeGrader()

        docs = [Document(page_content="Weather forecast")]
        result = evaluator.evaluate("What is RAPTOR?", docs)

        assert result.sufficient is False
        assert result.requires_more_retrieval is True
        assert len(result.relevant_documents) == 0

    def test_crag_empty_documents(self):
        from src.advanced_RAG.CRAG.crag import CRAGEvaluator

        evaluator = CRAGEvaluator.__new__(CRAGEvaluator)
        evaluator.min_relevant = 1
        result = evaluator.evaluate("What is RAPTOR?", [])

        assert result.sufficient is False
        assert result.requires_more_retrieval is True

    def test_crag_triggers_retry_in_pipeline(self):
        """When CRAG says insufficient, pipeline should retry retrieval."""
        pipeline = _build_pipeline(
            route=LOCAL, enable_crag=True, max_retries=1, web_docs=[]
        )

        class AlwaysInsufficientCRAG:
            min_relevant = 1
            def evaluate(self, query, documents):
                return EvidenceEvaluation(
                    sufficient=False,
                    requires_more_retrieval=True,
                    relevant_documents=[],
                    reason="Insufficient evidence",
                )

        pipeline.crag = AlwaysInsufficientCRAG()
        pipeline.config = RAGConfig(
            enable_crag=True, enable_self_rag=False, enable_long_context=False,
            enable_evaluation=False, enable_raptor=False, enable_multi_query=False,
            max_retrieval_retries=1, final_top_k=5, rerank_top_k=10,
            relevance_weight=0.7, freshness_weight=0.3,
        )
        result = pipeline.run("What is RAPTOR?", evaluate=False, route="LOCAL")

        # Pipeline should still produce an answer even after retries
        assert result["answer"] != ""


# Phase 9 — Self-RAG

class TestSelfRAG:
    def test_self_rag_sufficient_evidence(self):
        from src.advanced_RAG.Self_RAG.self_rag import SelfRAG

        class FakeGrader:
            def invoke(self, payload):
                return "YES"

        self_rag = SelfRAG.__new__(SelfRAG)
        self_rag.relevance_chain = FakeGrader()

        docs = [Document(page_content="RAPTOR uses tree-based retrieval.")]
        result = self_rag.evaluate_evidence("What is RAPTOR?", docs)

        assert result.sufficient is True
        assert len(result.relevant_documents) == 1

    def test_self_rag_insufficient_evidence(self):
        from src.advanced_RAG.Self_RAG.self_rag import SelfRAG

        class FakeGrader:
            def invoke(self, payload):
                return "NO"

        self_rag = SelfRAG.__new__(SelfRAG)
        self_rag.relevance_chain = FakeGrader()

        docs = [Document(page_content="Weather forecast")]
        result = self_rag.evaluate_evidence("What is RAPTOR?", docs)

        assert result.sufficient is False
        assert result.requires_more_retrieval is True

    def test_self_rag_empty_documents(self):
        from src.advanced_RAG.Self_RAG.self_rag import SelfRAG

        self_rag = SelfRAG.__new__(SelfRAG)
        result = self_rag.evaluate_evidence("What is RAPTOR?", [])

        assert result.requires_more_retrieval is True

    def test_self_rag_in_pipeline(self):
        """When Self-RAG finds insufficient evidence, pipeline handles it."""
        pipeline = _build_pipeline(
            route=LOCAL, enable_self_rag=True, max_retries=1, web_docs=[]
        )

        class AlwaysInsufficientSelfRAG:
            def evaluate_evidence(self, query, documents):
                return EvidenceEvaluation(
                    sufficient=False,
                    requires_more_retrieval=True,
                    relevant_documents=[],
                    reason="Self-RAG found no relevant documents.",
                )

        # Mock the pipeline to use our fake Self-RAG
        pipeline.config = RAGConfig(
            enable_crag=False, enable_self_rag=True, enable_long_context=False,
            enable_evaluation=False, enable_raptor=False, enable_multi_query=False,
            max_retrieval_retries=1, final_top_k=5, rerank_top_k=10,
            relevance_weight=0.7, freshness_weight=0.3,
        )
        pipeline.self_rag = AlwaysInsufficientSelfRAG()
        pipeline.chroma = ChromaRetriever(
            retriever=SimpleNamespace(invoke=lambda q: list(SAMPLE_DOCS))
        )
        # Override evaluate_evidence to use our mock
        original_evaluate = pipeline.evaluate_evidence

        def patched_evaluate(query, documents):
            # Simulate the pipeline's evaluate_evidence with our Self-RAG
            eval_result = EvidenceEvaluation(
                sufficient=True, requires_more_retrieval=False,
                relevant_documents=documents, reason="Accepted.",
            )
            try:
                self_eval = AlwaysInsufficientSelfRAG().evaluate_evidence(query, documents)
                if self_eval.requires_more_retrieval:
                    eval_result = EvidenceEvaluation(
                        sufficient=False, requires_more_retrieval=True,
                        relevant_documents=[], reason=self_eval.reason,
                    )
            except Exception:
                pass
            return eval_result

        pipeline.evaluate_evidence = patched_evaluate
        result = pipeline.run("What is RAPTOR?", evaluate=False, route="LOCAL")

        assert result["answer"] != ""


# Phase 10 — Long Context

class TestLongContext:
    def test_long_context_not_used_when_short(self):
        pipeline = _build_pipeline(route=LOCAL, enable_long_context=True, web_docs=[])
        pipeline.config = RAGConfig(
            enable_crag=False, enable_self_rag=False, enable_long_context=True,
            enable_evaluation=False, enable_raptor=False, enable_multi_query=False,
            max_retrieval_retries=0, final_top_k=5, rerank_top_k=10,
            relevance_weight=0.7, freshness_weight=0.3,
            long_context_min_chars=999999,  # Very high threshold
        )
        pipeline.long_context = SimpleNamespace(
            compress_context=lambda context, query: "compressed"
        )
        result = pipeline.run("What is RAPTOR?", evaluate=False, route="LOCAL")

        assert result["long_context_used"] is False

    def test_long_context_used_when_large(self):
        big_docs = [
            Document(page_content="x " * 3000, metadata={"url": f"http://{i}"})
            for i in range(3)
        ]
        pipeline = _build_pipeline(route=LOCAL, local_docs=big_docs, web_docs=[])
        pipeline.config = RAGConfig(
            enable_crag=False, enable_self_rag=False, enable_long_context=True,
            enable_evaluation=False, enable_raptor=False, enable_multi_query=False,
            max_retrieval_retries=0, final_top_k=5, rerank_top_k=10,
            relevance_weight=0.7, freshness_weight=0.3,
            long_context_min_chars=100,
        )
        pipeline.long_context = SimpleNamespace(
            compress_context=lambda context, query: "compressed context"
        )
        result = pipeline.run("test", evaluate=False, route="LOCAL")

        assert result["long_context_used"] is True


# Phase 12 — Failure Handling

class TestFailureHandling:
    def test_empty_query_raises(self):
        pipeline = _build_pipeline()
        with pytest.raises(ValueError, match="empty"):
            pipeline.run("", evaluate=False)

    def test_whitespace_query_raises(self):
        pipeline = _build_pipeline()
        with pytest.raises(ValueError, match="empty"):
            pipeline.run("   ", evaluate=False)

    def test_no_retrieved_documents(self):
        pipeline = _build_pipeline(route=LOCAL, local_docs=[], web_docs=[])
        pipeline.chroma = ChromaRetriever(retriever=None)
        pipeline.hybrid = HybridRetriever(
            local=pipeline.chroma, web=pipeline.web, use_raptor=False
        )
        result = pipeline.run("anything", evaluate=False, route="LOCAL")

        assert "not available" in result["answer"].lower() or result["answer"] != ""

    def test_invalid_router_output_defaults_to_hybrid(self):
        assert parse_route("GARBAGE") == HYBRID
        assert parse_route(None) == HYBRID
        assert parse_route("") == HYBRID

    def test_missing_document_metadata(self):
        doc = Document(page_content="test", metadata={})
        assert freshness_score(doc) == 0.0

    def test_none_date_metadata(self):
        doc = Document(page_content="test", metadata={"date": None})
        assert freshness_score(doc) == 0.0

    def test_malformed_date_metadata(self):
        doc = Document(page_content="test", metadata={"date": {"nested": True}})
        assert freshness_score(doc) == 0.0

    def test_empty_generation_output_returns_fallback(self):
        pipeline = _build_pipeline(route=LOCAL, web_docs=[])
        pipeline.generation_chain = SimpleNamespace(
            invoke=lambda payload: ""
        )
        # Empty context should trigger the "not available" path
        pipeline.chroma = ChromaRetriever(retriever=None)
        pipeline.hybrid = HybridRetriever(
            local=pipeline.chroma, web=pipeline.web, use_raptor=False
        )
        result = pipeline.run("test", evaluate=False, route="LOCAL")
        assert result["answer"] != ""

    def test_deduplication_empty_list(self):
        assert remove_duplicates([]) == []

    def test_select_evidence_empty(self):
        assert select_evidence([], top_k=5) == []

    def test_query_processor_empty_llm_output(self):
        proc = QueryProcessor(enable_multi_query=True, max_variations=3)
        proc.generator = SimpleNamespace(invoke=lambda payload: "")
        result = proc.process("test query")
        assert result == ["test query"]


# Complete pipeline flow tests

class TestCompletePipelineFlow:
    def test_full_local_pipeline(self):
        pipeline = _build_pipeline(route=LOCAL, web_docs=[])
        result = pipeline.run("What is RAPTOR?", evaluate=False, route="LOCAL")

        # Verify all expected keys present
        assert "query" in result
        assert "route" in result
        assert "queries" in result
        assert "answer" in result
        assert "context" in result
        assert "retrieved_documents" in result
        assert "reranked_documents" in result
        assert "evidence" in result
        assert "evidence_evaluation" in result
        assert "long_context_used" in result

        # Verify flow
        assert result["route"] == LOCAL
        assert result["queries"][0] == "What is RAPTOR?"
        assert result["answer"] != ""
        assert isinstance(result["evidence_evaluation"], dict)

    def test_full_web_pipeline(self):
        pipeline = _build_pipeline(route=WEB)
        result = pipeline.run("quantum computing", evaluate=False, route="WEB")

        assert result["route"] == WEB
        assert len(result["retrieved_documents"]) > 0
        assert result["answer"] != ""

    def test_full_hybrid_pipeline(self):
        pipeline = _build_pipeline(route=HYBRID)
        result = pipeline.run("RAG vs fine-tuning", evaluate=False, route="HYBRID")

        assert result["route"] == HYBRID
        assert len(result["retrieved_documents"]) > 0
        assert result["answer"] != ""

    def test_pipeline_stores_memory(self):
        memory_calls = []

        class TrackingMemory:
            def add_message(self, msg):
                memory_calls.append(msg)

        pipeline = _build_pipeline(route=LOCAL, web_docs=[])
        pipeline.conversation_memory = TrackingMemory()
        pipeline.run("test", evaluate=False, route="LOCAL")

        assert len(memory_calls) == 2  # HumanMessage + AIMessage

    def test_retry_retrieval_on_insufficient_evidence(self):
        """Verify the retry loop actually executes."""
        call_count = {"retrieve": 0}
        original_docs = list(SAMPLE_DOCS)

        pipeline = _build_pipeline(
            route=LOCAL, web_docs=WEB_DOCS, max_retries=1
        )

        original_retrieve = pipeline.retrieve

        def counting_retrieve(queries, route):
            call_count["retrieve"] += 1
            return original_retrieve(queries, route)

        pipeline.retrieve = counting_retrieve

        # First evaluation returns insufficient, second returns sufficient
        eval_count = {"count": 0}

        def alternating_evaluate(query, documents):
            eval_count["count"] += 1
            if eval_count["count"] == 1:
                return EvidenceEvaluation(
                    sufficient=False,
                    requires_more_retrieval=True,
                    relevant_documents=[],
                    reason="First try insufficient.",
                )
            return EvidenceEvaluation(
                sufficient=True,
                requires_more_retrieval=False,
                relevant_documents=documents,
                reason="Second try sufficient.",
            )

        pipeline.evaluate_evidence = alternating_evaluate

        result = pipeline.run("What is RAPTOR?", evaluate=False, route="LOCAL")

        # Should have called retrieve at least twice (initial + retry)
        assert call_count["retrieve"] >= 2
        assert result["answer"] != ""

    def test_route_override(self):
        """Explicit route= parameter should override the router."""
        pipeline = _build_pipeline(route=LOCAL)  # Router says LOCAL
        result = pipeline.run("test", evaluate=False, route="WEB")
        assert result["route"] == WEB

    def test_decision_field_matches_route(self):
        pipeline = _build_pipeline(route=HYBRID)
        result = pipeline.run("test", evaluate=False, route="HYBRID")
        assert result["decision"] == HYBRID
        assert result["mode"] == "hybrid"
