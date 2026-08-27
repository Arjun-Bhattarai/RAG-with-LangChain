"""
End-to-end diagnostic for the RAG pipeline.

Runs real queries through every pipeline stage and reports what actually
happened.  Stages that depend on an unavailable service are reported as
BLOCKED rather than silently skipped.

Usage:
    .venv\\Scripts\\python.exe tests\\e2e_diagnostic.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from langchain_core.documents import Document


# Helpers

@dataclass
class StageReport:
    name: str
    status: str = "PENDING"        # PASS / FAIL / BLOCKED / SKIPPED
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_ms: float = 0.0


def _run_stage(name: str, fn, *args, **kwargs) -> StageReport:
    """Run *fn* and capture its result or failure into a StageReport."""
    report = StageReport(name=name)
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        report.status = "PASS"
        if isinstance(result, dict):
            report.data = result
        elif isinstance(result, str):
            report.detail = result
    except Exception as exc:
        report.status = "FAIL"
        report.error = f"{type(exc).__name__}: {exc}"
        report.detail = traceback.format_exc()
    report.duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    return report


def _print_report(report: StageReport) -> None:
    icon = {"PASS": "✅", "FAIL": "❌", "BLOCKED": "🚫", "SKIPPED": "⏭️"}.get(
        report.status, "❓"
    )
    print(f"\n{'=' * 72}")
    print(f"{icon}  {report.name}  [{report.status}]  ({report.duration_ms:.0f} ms)")
    print("=" * 72)
    if report.data:
        for k, v in report.data.items():
            if isinstance(v, list) and len(v) > 3:
                print(f"  {k}: [{len(v)} items]")
            else:
                print(f"  {k}: {v}")
    if report.detail and report.status != "PASS":
        for line in report.detail.strip().splitlines()[-6:]:
            print(f"  {line}")
    if report.error:
        print(f"  ERROR: {report.error}")


# Service availability checks

def check_ollama() -> bool:
    """Return True if Ollama is reachable."""
    try:
        import requests
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        return r.ok
    except Exception:
        return False


def check_tavily() -> bool:
    """Return True if the Tavily key is configured and a trivial search works."""
    import os
    key = os.getenv("TAVILY_API_KEY", "")
    if not key:
        return False
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=key)
        result = client.search(query="test", max_results=1, search_depth="basic")
        return isinstance(result, dict) and "results" in result
    except Exception:
        return False


def check_wikipedia() -> bool:
    try:
        import requests
        r = requests.get(
            "https://en.wikipedia.org/w/rest.php/v1/search/page",
            params={"q": "test", "limit": 1},
            timeout=10,
        )
        return r.ok
    except Exception:
        return False


# Phase 3 — LOCAL retrieval

def test_local_retrieval(pipeline) -> Dict[str, Any]:
    """Build index from sample docs, then query LOCAL."""
    sample_docs = [
        Document(
            page_content=(
                "RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) "
                "builds a hierarchical tree of document summaries using clustering and "
                "recursive summarization. Leaf nodes contain chunk summaries, while "
                "higher-level nodes contain cluster summaries. At query time, RAPTOR "
                "retrieves from multiple levels of the tree to provide both detailed "
                "and high-level context."
            ),
            metadata={"source": "raptor_paper", "date": datetime.now().strftime("%Y-%m-%d")},
        ),
        Document(
            page_content=(
                "RAG (Retrieval-Augmented Generation) combines a retriever component "
                "with a language model generator. The retriever fetches relevant "
                "documents from a knowledge base, and the generator uses those "
                "documents as context to produce grounded answers."
            ),
            metadata={"source": "rag_overview", "date": "2024-01-15"},
        ),
        Document(
            page_content=(
                "Vector databases store high-dimensional embeddings and support "
                "efficient similarity search. ChromaDB is a popular open-source "
                "vector database that integrates well with LangChain."
            ),
            metadata={"source": "vectordb_guide"},
        ),
        Document(
            page_content=(
                "Self-RAG introduces a self-reflective mechanism where the language "
                "model evaluates whether retrieval is needed, grades the relevance "
                "of retrieved documents, and checks if its own answer is supported "
                "by the evidence."
            ),
            metadata={"source": "self_rag_paper", "date": "2024-03-01"},
        ),
        Document(
            page_content=(
                "CRAG (Corrective Retrieval Augmented Generation) adds an evaluation "
                "step after retrieval. If the retrieved documents are insufficient "
                "or irrelevant, the system performs corrective retrieval to find "
                "better evidence before generating the final answer."
            ),
            metadata={"source": "crag_paper", "published_date": "2024-02-10"},
        ),
    ]

    # Build the local index
    pipeline.build_index(sample_docs)

    # Run query with forced LOCAL route
    query = "What is RAPTOR?"
    result = pipeline.run(query, evaluate=False, route="LOCAL")

    return {
        "query": result["query"],
        "route": result["route"],
        "query_count": len(result["queries"]),
        "queries": result["queries"][:5],
        "retrieved_count": len(result["retrieved_documents"]),
        "reranked_count": len(result["reranked_documents"]),
        "evidence_count": len(result["evidence"]),
        "evidence_eval": result["evidence_evaluation"],
        "long_context_used": result["long_context_used"],
        "answer_preview": str(result["answer"])[:300],
        "answer_length": len(str(result["answer"])),
    }


# Phase 4 — WEB retrieval

def test_web_retrieval(pipeline) -> Dict[str, Any]:
    query = "What are the latest developments in quantum computing?"
    result = pipeline.run(query, evaluate=False, route="WEB")

    return {
        "query": result["query"],
        "route": result["route"],
        "query_count": len(result["queries"]),
        "retrieved_count": len(result["retrieved_documents"]),
        "retrieved_sources": list({
            doc.metadata.get("source", "unknown")
            for doc in result["retrieved_documents"]
        }),
        "reranked_count": len(result["reranked_documents"]),
        "evidence_count": len(result["evidence"]),
        "evidence_eval": result["evidence_evaluation"],
        "long_context_used": result["long_context_used"],
        "answer_preview": str(result["answer"])[:300],
    }


# Phase 5 — HYBRID retrieval

def test_hybrid_retrieval(pipeline) -> Dict[str, Any]:
    query = "How does RAG compare to fine-tuning for knowledge-intensive tasks?"
    result = pipeline.run(query, evaluate=False, route="HYBRID")

    sources = [doc.metadata.get("source", "unknown") for doc in result["retrieved_documents"]]
    has_local = any(s not in ("web_search", "wikipedia") for s in sources)
    has_web = any(s in ("web_search", "wikipedia") for s in sources)

    return {
        "query": result["query"],
        "route": result["route"],
        "query_count": len(result["queries"]),
        "retrieved_count": len(result["retrieved_documents"]),
        "has_local_docs": has_local,
        "has_web_docs": has_web,
        "source_types": list(set(sources)),
        "reranked_count": len(result["reranked_documents"]),
        "evidence_count": len(result["evidence"]),
        "evidence_eval": result["evidence_evaluation"],
        "answer_preview": str(result["answer"])[:300],
    }


# Phase 6 — Multi-Query

def test_multi_query(pipeline) -> Dict[str, Any]:
    query = "What is RAPTOR and how does it improve retrieval?"
    queries = pipeline.query_processor.process(query)

    return {
        "original_query": query,
        "generated_count": len(queries),
        "original_preserved": queries[0] == query if queries else False,
        "queries": queries[:6],
        "max_variations": pipeline.query_processor.max_variations,
        "respects_max": len(queries) <= pipeline.query_processor.max_variations,
        "no_empty_queries": all(q.strip() for q in queries),
        "no_duplicates": len(queries) == len(set(q.lower() for q in queries)),
    }


# Phase 7 — Reranking + Freshness
# 

def test_reranking_freshness() -> Dict[str, Any]:
    from src.Reranking.reranking import CrossEncoderReranker
    from src.Reranking.freshness import (
        rerank_with_freshness,
        freshness_score,
        extract_document_datetime,
    )

    docs = [
        Document(
            page_content="RAPTOR builds a hierarchical summary tree for retrieval.",
            metadata={"date": datetime.now().strftime("%Y-%m-%d")},
        ),
        Document(
            page_content="Unrelated sports news about football matches.",
            metadata={},
        ),
        Document(
            page_content="RAG systems combine retrieval with generation.",
            metadata={"published_date": "not-a-real-date"},
        ),
        Document(
            page_content="Vector databases enable efficient similarity search.",
            metadata={"date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")},
        ),
    ]

    # Check date extraction
    dates = [extract_document_datetime(d) for d in docs]
    freshness_scores = [freshness_score(d) for d in docs]

    # Check reranking
    reranker = CrossEncoderReranker(top_k=4)
    query = "What is RAPTOR?"

    ranked = rerank_with_freshness(
        query=query,
        documents=docs,
        reranker=reranker,
        relevance_weight=0.7,
        freshness_weight=0.3,
        top_k=4,
    )

    return {
        "input_count": len(docs),
        "date_extracted": [d is not None for d in dates],
        "freshness_scores": [round(s, 4) for s in freshness_scores],
        "no_crash_on_missing_date": True,
        "no_crash_on_malformed_date": True,
        "reranked_count": len(ranked),
        "reranked_order": [doc.page_content[:40] for doc in ranked],
        "freshness_is_separate_stage": True,
    }


# Phase 8 — CRAG

def test_crag(pipeline) -> Dict[str, Any]:
    good_docs = [
        Document(page_content="RAPTOR builds a hierarchical summary tree using clustering."),
        Document(page_content="RAPTOR uses recursive summarization for multi-level retrieval."),
    ]
    poor_docs = [
        Document(page_content="Today's weather forecast calls for sunny skies."),
        Document(page_content="The stock market closed higher on Tuesday."),
    ]

    eval_good = pipeline.crag.evaluate("What is RAPTOR?", good_docs)
    eval_poor = pipeline.crag.evaluate("What is RAPTOR?", poor_docs)
    eval_empty = pipeline.crag.evaluate("What is RAPTOR?", [])

    return {
        "good_evidence_sufficient": eval_good.sufficient,
        "good_evidence_relevant_count": len(eval_good.relevant_documents),
        "good_evidence_reason": eval_good.reason,
        "poor_evidence_sufficient": eval_poor.sufficient,
        "poor_evidence_requires_retry": eval_poor.requires_more_retrieval,
        "poor_evidence_reason": eval_poor.reason,
        "empty_evidence_requires_retry": eval_empty.requires_more_retrieval,
    }


# Phase 9 — Self-RAG

def test_self_rag(pipeline) -> Dict[str, Any]:
    from src.advanced_RAG.Self_RAG.self_rag import SelfRAG

    good_docs = [
        Document(page_content="RAPTOR builds a hierarchical summary tree using clustering."),
    ]
    poor_docs = [
        Document(page_content="Today's weather forecast calls for sunny skies."),
    ]

    # Build a SelfRAG with a static retriever
    class StaticRetriever:
        def invoke(self, query):
            return good_docs

    self_rag = SelfRAG(llm=pipeline.llm, retriever=StaticRetriever())
    eval_good = self_rag.evaluate_evidence("What is RAPTOR?", good_docs)
    eval_poor = self_rag.evaluate_evidence("What is RAPTOR?", poor_docs)
    eval_empty = self_rag.evaluate_evidence("What is RAPTOR?", [])

    return {
        "good_evidence_sufficient": eval_good.sufficient,
        "good_evidence_count": len(eval_good.relevant_documents),
        "good_reason": eval_good.reason,
        "poor_evidence_sufficient": eval_poor.sufficient,
        "poor_requires_retry": eval_poor.requires_more_retrieval,
        "poor_reason": eval_poor.reason,
        "empty_requires_retry": eval_empty.requires_more_retrieval,
    }


# Phase 10 — Long Context

def test_long_context(pipeline) -> Dict[str, Any]:
    from src.advanced_RAG.Long_Context.long_context import LongContext

    lc = LongContext(model=pipeline.config.ollama_model)

    # Build a large context (> long_context_min_chars)
    evidence = [
        Document(page_content=f"Document {i}: " + "x " * 300)
        for i in range(10)
    ]

    from src.pipeline.generation_rag import format_docs
    context = format_docs(evidence)
    context_size = len(context)

    compressed = lc.compress_context(context=context, query="What is RAPTOR?")
    compressed_size = len(compressed)

    return {
        "original_context_size": context_size,
        "compressed_context_size": compressed_size,
        "compression_ratio": round(compressed_size / max(context_size, 1), 3),
        "compression_executed": True,
        "threshold_chars": pipeline.config.long_context_min_chars,
        "context_exceeds_threshold": context_size >= pipeline.config.long_context_min_chars,
    }


# Phase 12 — Failure Handling

def test_failure_handling(pipeline) -> Dict[str, Any]:
    results = {}

    # 1. Empty query
    try:
        pipeline.run("", evaluate=False)
        results["empty_query"] = "FAIL — no error raised"
    except ValueError as e:
        results["empty_query"] = f"PASS — ValueError: {e}"
    except Exception as e:
        results["empty_query"] = f"FAIL — unexpected: {type(e).__name__}: {e}"

    # 2. Whitespace-only query
    try:
        pipeline.run("   ", evaluate=False)
        results["whitespace_query"] = "FAIL — no error raised"
    except ValueError:
        results["whitespace_query"] = "PASS"
    except Exception as e:
        results["whitespace_query"] = f"FAIL — {type(e).__name__}: {e}"

    # 3. No local documents (without building index, Chroma returns empty)
    from src.pipeline.rag_pipeline import RAGPipeline, RAGConfig
    empty_pipeline = RAGPipeline.__new__(RAGPipeline)
    empty_pipeline.config = RAGConfig(
        enable_crag=False, enable_self_rag=False, enable_long_context=False,
        enable_evaluation=False, enable_raptor=False, enable_multi_query=False,
        max_retrieval_retries=0,
    )
    from types import SimpleNamespace
    from src.retrieval.chroma import ChromaRetriever
    from src.retrieval.hybrid import HybridRetriever
    empty_pipeline.router = SimpleNamespace(route=lambda q: "LOCAL")
    empty_pipeline.query_processor = SimpleNamespace(process=lambda q: [q])
    empty_pipeline.chroma = ChromaRetriever(retriever=None)
    empty_pipeline.web = SimpleNamespace(retrieve=lambda q: [])
    empty_pipeline.raptor = SimpleNamespace(is_ready=lambda: False)
    empty_pipeline.hybrid = HybridRetriever(
        local=empty_pipeline.chroma, web=empty_pipeline.web, use_raptor=False
    )
    empty_pipeline.reranker = SimpleNamespace(
        rerank=lambda **kw: [], rerank_documents=lambda **kw: []
    )
    empty_pipeline.generation_chain = SimpleNamespace(
        invoke=lambda payload: "No information available."
    )
    empty_pipeline.conversation_memory = SimpleNamespace(add_message=lambda m: None)
    empty_pipeline.evaluator = None
    empty_pipeline.long_context = None
    empty_pipeline.crag = None
    empty_pipeline.self_rag = None
    try:
        r = empty_pipeline.run("What is RAPTOR?", evaluate=False)
        results["no_local_docs"] = f"PASS — answer: {str(r['answer'])[:80]}"
    except Exception as e:
        results["no_local_docs"] = f"FAIL — {type(e).__name__}: {e}"

    # 4. Invalid router output
    from src.routing.source_routing import parse_route
    results["invalid_route_parsed"] = parse_route("GARBAGE_ROUTE_XYZ")

    # 5. Missing document metadata
    from src.Reranking.freshness import freshness_score, extract_document_datetime
    doc_no_meta = Document(page_content="test", metadata={})
    doc_none_date = Document(page_content="test", metadata={"date": None})
    doc_bad_date = Document(page_content="test", metadata={"date": {"nested": True}})
    results["missing_metadata_freshness"] = freshness_score(doc_no_meta)
    results["none_date_freshness"] = freshness_score(doc_none_date)
    results["bad_date_freshness"] = freshness_score(doc_bad_date)
    results["no_crash_on_bad_metadata"] = "PASS"

    # 6. Empty LLM query-generation output
    from src.query_translation.query_processor import QueryProcessor
    proc = QueryProcessor(enable_multi_query=True, max_variations=3)
    proc.generator = SimpleNamespace(invoke=lambda payload: "")
    result = proc.process("test query")
    results["empty_llm_output"] = f"PASS — queries: {result}"

    # 7. Deduplication with empty list
    from src.utils.deduplication import remove_duplicates
    results["dedup_empty"] = f"PASS — {remove_duplicates([])}"

    return results


# Main diagnostic runner

def main():
    print("=" * 72)
    print("  RAG PIPELINE END-TO-END DIAGNOSTIC")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # ── Service availability ──────────────────────────────────────────────
    print("\n>>> Checking service availability...")
    ollama_ok = check_ollama()
    tavily_ok = check_tavily()
    wiki_ok = check_wikipedia()

    print(f"  Ollama:    {'✅ available' if ollama_ok else '❌ unavailable'}")
    print(f"  Tavily:    {'✅ available' if tavily_ok else '❌ unavailable'}")
    print(f"  Wikipedia: {'✅ available' if wiki_ok else '❌ unavailable'}")

    if not ollama_ok:
        print("\n❌ Ollama is required for pipeline operation. Aborting.")
        print("   Start Ollama and ensure llama3:latest + nomic-embed-text:latest are pulled.")
        sys.exit(1)

    all_reports: List[StageReport] = []

    # ── Create pipeline ───────────────────────────────────────────────────
    print("\n>>> Creating RAGPipeline...")
    from src.pipeline.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline(
        enable_raptor=False,         # Skip RAPTOR for speed
        enable_evaluation=False,     # Skip evaluation metrics
    )
    print("  Pipeline created.")

    # ── Phase 7: Reranking + Freshness (no Ollama dependency) ─────────
    report = _run_stage("Phase 7: Reranking + Freshness", test_reranking_freshness)
    _print_report(report)
    all_reports.append(report)

    # ── Phase 12: Failure Handling ────────────────────────────────────
    report = _run_stage("Phase 12: Failure Handling", test_failure_handling, pipeline)
    _print_report(report)
    all_reports.append(report)

    # ── Phase 6: Multi-Query ──────────────────────────────────────────
    report = _run_stage("Phase 6: Multi-Query", test_multi_query, pipeline)
    _print_report(report)
    all_reports.append(report)

    # ── Phase 3: LOCAL retrieval ──────────────────────────────────────
    report = _run_stage("Phase 3: LOCAL Retrieval", test_local_retrieval, pipeline)
    _print_report(report)
    all_reports.append(report)

    # ── Phase 4: WEB retrieval ────────────────────────────────────────
    if tavily_ok or wiki_ok:
        report = _run_stage("Phase 4: WEB Retrieval", test_web_retrieval, pipeline)
    else:
        report = StageReport(name="Phase 4: WEB Retrieval", status="BLOCKED",
                             error="Tavily and Wikipedia both unavailable")
    _print_report(report)
    all_reports.append(report)

    # ── Phase 5: HYBRID retrieval ─────────────────────────────────────
    if tavily_ok or wiki_ok:
        report = _run_stage("Phase 5: HYBRID Retrieval", test_hybrid_retrieval, pipeline)
    else:
        report = StageReport(name="Phase 5: HYBRID Retrieval", status="BLOCKED",
                             error="Web services unavailable for hybrid test")
    _print_report(report)
    all_reports.append(report)

    # ── Phase 8: CRAG ─────────────────────────────────────────────────
    report = _run_stage("Phase 8: CRAG Evaluation", test_crag, pipeline)
    _print_report(report)
    all_reports.append(report)

    # ── Phase 9: Self-RAG ─────────────────────────────────────────────
    report = _run_stage("Phase 9: Self-RAG Evaluation", test_self_rag, pipeline)
    _print_report(report)
    all_reports.append(report)

    # ── Phase 10: Long Context ────────────────────────────────────────
    report = _run_stage("Phase 10: Long Context", test_long_context, pipeline)
    _print_report(report)
    all_reports.append(report)

    # ── Phase 11: Complete pipeline queries ───────────────────────────
    def test_complete_pipeline():
        results = {}
        # LOCAL query
        r = pipeline.run("What is RAPTOR?", evaluate=False, route="LOCAL")
        results["local_query"] = r["query"]
        results["local_route"] = r["route"]
        results["local_answer_preview"] = str(r["answer"])[:200]
        results["local_retrieved"] = len(r["retrieved_documents"])
        results["local_reranked"] = len(r["reranked_documents"])

        # WEB query (if available)
        if tavily_ok or wiki_ok:
            r = pipeline.run("What is the current state of AI regulation?", evaluate=False, route="WEB")
            results["web_query"] = r["query"]
            results["web_route"] = r["route"]
            results["web_answer_preview"] = str(r["answer"])[:200]
            results["web_retrieved"] = len(r["retrieved_documents"])
        else:
            results["web_status"] = "BLOCKED"

        # HYBRID query (if available)
        if tavily_ok or wiki_ok:
            r = pipeline.run(
                "How does RAG compare to fine-tuning for knowledge-intensive tasks?",
                evaluate=False, route="HYBRID",
            )
            results["hybrid_query"] = r["query"]
            results["hybrid_route"] = r["route"]
            results["hybrid_answer_preview"] = str(r["answer"])[:200]
            results["hybrid_retrieved"] = len(r["retrieved_documents"])
        else:
            results["hybrid_status"] = "BLOCKED"

        return results

    report = _run_stage("Phase 11: Complete Pipeline", test_complete_pipeline)
    _print_report(report)
    all_reports.append(report)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for r in all_reports:
        icon = {"PASS": "✅", "FAIL": "❌", "BLOCKED": "🚫"}.get(r.status, "❓")
        print(f"  {icon}  {r.name}: {r.status} ({r.duration_ms:.0f} ms)")

    passed = sum(1 for r in all_reports if r.status == "PASS")
    failed = sum(1 for r in all_reports if r.status == "FAIL")
    blocked = sum(1 for r in all_reports if r.status == "BLOCKED")
    print(f"\n  Total: {len(all_reports)}  |  PASS: {passed}  |  FAIL: {failed}  |  BLOCKED: {blocked}")


if __name__ == "__main__":
    main()
