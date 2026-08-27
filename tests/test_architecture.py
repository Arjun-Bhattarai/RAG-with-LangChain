"""Offline tests for the modular RAG architecture."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.core.config import RAGConfig
from src.core.models import HYBRID, LOCAL, WEB
from src.query_translation.query_processor import QueryProcessor
from src.Reranking.freshness import extract_document_datetime, freshness_score, apply_freshness_ranking
from src.Reranking.reranking import CrossEncoderReranker
from src.retrieval.chroma import ChromaRetriever
from src.retrieval.hybrid import HybridRetriever
from src.routing.source_routing import parse_route
from src.utils.deduplication import remove_duplicates
from src.utils.evidence import select_evidence


def test_imports():
    import src.advanced_RAG.CRAG.crag as crag
    import src.advanced_RAG.Self_RAG.self_rag as self_rag
    import src.advanced_indexing.raptor as raptor
    import src.pipeline.rag_pipeline as rag_pipeline
    import src.query_translation.multi_query as multi_query
    import src.retrieval.web as web
    import src.routing.source_routing as source_routing

    assert hasattr(crag, "CRAGEvaluator")
    assert hasattr(self_rag.SelfRAG, "evaluate_evidence")
    assert hasattr(raptor, "RaptorIndexer")
    assert hasattr(rag_pipeline, "RAGPipeline")
    assert hasattr(multi_query, "create_multi_query_generator")
    assert hasattr(web, "WebRetriever")
    assert hasattr(source_routing, "SourceRouter")
    print("imports: ok")


def test_router_parsing():
    assert parse_route(SimpleNamespace(source="LOCAL")) == LOCAL
    assert parse_route("please use WEB search") == WEB
    assert parse_route("HYBRID please") == HYBRID
    assert parse_route("ANSWERABLE") == HYBRID
    assert parse_route("not a valid label") == HYBRID
    assert parse_route(None) == HYBRID
    print("router parsing: ok")


def test_query_processor_fallback():
    processor = QueryProcessor(enable_multi_query=False, max_variations=4)
    assert processor.process("What is RAPTOR?") == ["What is RAPTOR?"]

    processor.generator = MagicMock(side_effect=RuntimeError("llm down"))
    processor.enable_multi_query = True
    assert processor.process("hello") == ["hello"]

    processor.generator = MagicMock(return_value=["hello", "hi there", "greetings"])
    processed = processor.process("hello")
    assert processed[0] == "hello"
    assert "hi there" in processed
    print("query processor: ok")


def test_deduplication():
    docs = [
        Document(page_content="a", metadata={"url": "http://x"}),
        Document(page_content="a copy", metadata={"url": "http://x"}),
        Document(page_content="b", metadata={"title": "B"}),
        Document(page_content="b", metadata={"title": "B"}),
    ]
    unique = remove_duplicates(docs)
    assert len(unique) == 2
    print("deduplication: ok")


def test_freshness_with_and_without_dates():
    dated = Document(
        page_content="news",
        metadata={"published_date": datetime.now().strftime("%Y-%m-%d")},
    )
    undated = Document(page_content="old notes", metadata={})
    invalid = Document(page_content="bad", metadata={"published_date": {"year": 2020}})
    missing = Document(page_content="none", metadata={"date": None})

    assert extract_document_datetime(dated) is not None
    assert extract_document_datetime(undated) is None
    assert extract_document_datetime(invalid) is None
    assert extract_document_datetime(missing) is None
    assert freshness_score(undated) == 0.0
    assert freshness_score(dated) > 0.0

    ranked = apply_freshness_ranking(
        [undated, dated],
        relevance_scores=[0.5, 0.5],
        relevance_weight=0.7,
        freshness_weight=0.3,
    )
    assert ranked[0] is dated
    print("freshness: ok")


def test_cross_encoder_api():
    reranker = CrossEncoderReranker(top_k=2)
    assert callable(reranker.rerank_documents)
    empty = reranker.rerank_documents(query="q", documents=[])
    assert empty == []
    print("cross encoder api: ok")


def test_local_retrieval_without_index():
    retriever = ChromaRetriever(retriever=None)
    assert retriever.retrieve("anything") == []
    print("local retrieval empty index: ok")


def test_web_and_hybrid_retrieval():
    class FakeSource:
        def __init__(self, docs):
            self.docs = docs

        def retrieve(self, query):
            return self.docs

    web_docs = [Document(page_content="web", metadata={"source": "web_search", "url": "http://w"})]
    wiki_docs = [Document(page_content="wiki", metadata={"source": "wikipedia", "url": "http://wiki"})]
    local_docs = [Document(page_content="local", metadata={"source": "chroma"})]

    from src.retrieval.web import WebRetriever

    web = WebRetriever(wikipedia=FakeSource(wiki_docs), web_search=FakeSource(web_docs))
    web_results = web.retrieve("q")
    assert len(web_results) == 2

    local = ChromaRetriever(retriever=SimpleNamespace(invoke=lambda q: local_docs))
    hybrid = HybridRetriever(local=local, web=web, use_raptor=False)
    combined = hybrid.retrieve("q")
    assert len(combined) == 3
    print("web/hybrid retrieval: ok")


def test_evidence_selection():
    docs = [Document(page_content=str(i)) for i in range(8)]
    selected = select_evidence(docs, top_k=5)
    assert len(selected) == 5
    print("evidence selection: ok")


def test_crag_and_self_rag_evaluation():
    from src.advanced_RAG.CRAG.crag import CRAGEvaluator
    from src.advanced_RAG.Self_RAG.self_rag import SelfRAG

    class FakeLLM:
        def invoke(self, payload):
            text = str(payload)
            if "irrelevant" in text.lower():
                return SimpleNamespace(content="NO")
            return SimpleNamespace(content="YES")

        def __or__(self, other):
            return self

    class Grader:
        def invoke(self, payload):
            document = payload["document"]
            return "NO" if "irrelevant" in document.lower() else "YES"

    docs = [
        Document(page_content="Relevant facts about RAPTOR trees."),
        Document(page_content="irrelevant sports scores"),
    ]

    evaluator = CRAGEvaluator.__new__(CRAGEvaluator)
    evaluator.min_relevant = 1
    evaluator.retrieval_grader = Grader()
    result = evaluator.evaluate("What is RAPTOR?", docs)
    assert result.sufficient is True
    assert result.requires_more_retrieval is False
    assert len(result.relevant_documents) == 1

    empty = evaluator.evaluate("q", [])
    assert empty.requires_more_retrieval is True

    self_rag = SelfRAG.__new__(SelfRAG)
    self_rag.relevance_chain = Grader()
    self_eval = self_rag.evaluate_evidence("What is RAPTOR?", docs)
    assert self_eval.sufficient is True
    print("crag/self-rag evaluation: ok")


def test_pipeline_with_mocks():
    from src.pipeline.rag_pipeline import RAGPipeline

    class FakeRouter:
        def route(self, query):
            return LOCAL

    class FakeProcessor:
        def process(self, query):
            return [query, query + " alternative"]

    class FakeReranker:
        def rerank(self, query, documents, top_k=None):
            scored = [(doc, 1.0) for doc in documents]
            return scored[: top_k or len(scored)]

        def rerank_documents(self, query, documents, top_k=None):
            return [doc for doc, _ in self.rerank(query, documents, top_k=top_k)]

    class FakeLLM:
        def invoke(self, payload):
            return "grounded answer"

    local_docs = [
        Document(page_content="RAPTOR builds a summary tree.", metadata={"url": "http://a"}),
        Document(page_content="RAPTOR builds a summary tree.", metadata={"url": "http://a"}),
        Document(
            page_content="Cluster summary of RAPTOR.",
            metadata={"published_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")},
        ),
    ]

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.config = RAGConfig(
        enable_crag=False,
        enable_self_rag=False,
        enable_long_context=False,
        enable_evaluation=False,
        enable_raptor=False,
        enable_multi_query=True,
        final_top_k=5,
        rerank_top_k=5,
        max_retrieval_retries=0,
        relevance_weight=0.7,
        freshness_weight=0.3,
    )
    pipeline.router = FakeRouter()
    pipeline.query_processor = FakeProcessor()
    pipeline.hybrid = HybridRetriever(
        local=ChromaRetriever(retriever=SimpleNamespace(invoke=lambda q: local_docs)),
        web=SimpleNamespace(retrieve=lambda q: []),
        use_raptor=False,
    )
    pipeline.chroma = pipeline.hybrid.local
    pipeline.raptor = SimpleNamespace(is_ready=lambda: False)
    pipeline.reranker = FakeReranker()
    pipeline.generation_chain = FakeLLM()
    pipeline.conversation_memory = SimpleNamespace(add_message=lambda msg: None)
    pipeline.evaluator = None
    pipeline.long_context = None

    result = pipeline.run("What is RAPTOR?", evaluate=False, route="LOCAL")
    assert result["route"] == LOCAL
    assert result["answer"] == "grounded answer"
    assert result["queries"][0] == "What is RAPTOR?"
    assert len(result["retrieved_documents"]) == 2
    assert result["decision"] != "ANSWERABLE"
    print("complete pipeline: ok")


if __name__ == "__main__":
    test_imports()
    test_router_parsing()
    test_query_processor_fallback()
    test_deduplication()
    test_freshness_with_and_without_dates()
    test_cross_encoder_api()
    test_local_retrieval_without_index()
    test_web_and_hybrid_retrieval()
    test_evidence_selection()
    test_crag_and_self_rag_evaluation()
    test_pipeline_with_mocks()
    print("\nAll architecture tests passed.")
