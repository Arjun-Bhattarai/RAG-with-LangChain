"""
Modular RAG pipeline.

User Query
    -> Query Processing (rewrite / multi-query)
    -> Ollama Router (LOCAL / WEB / HYBRID)
    -> Retrieval (Chroma/RAPTOR, Tavily+Wikipedia, or both)
    -> Deduplicate
    -> Cross-Encoder rerank
    -> Freshness ranking
    -> Evidence selection
    -> CRAG / Self-RAG evaluation
    -> Optional extra retrieval
    -> Context builder / optional long-context compression
    -> Ollama generation
"""

from __future__ import annotations

import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.advanced_indexing.raptor import RaptorIndexer
from src.advanced_RAG.CRAG.crag import CRAGEvaluator
from src.advanced_RAG.Long_Context.long_context import LongContext
from src.advanced_RAG.Self_RAG.self_rag import SelfRAG
from src.core.config import RAGConfig
from src.core.models import (
    HYBRID,
    LOCAL,
    VALID_ROUTES,
    WEB,
    EvidenceEvaluation,
)
from src.evaluation.rag_evaluation import RAGEvaluator
from src.memory.memory import ConversationMemory
from src.pipeline.generation_rag import DEFAULT_GENERATION_PROMPT, create_answer_chain, format_docs
from src.pipeline.indexing_rag import build_vectorstore
from src.pipeline.retrieval_rag import create_retriever
from src.query_translation.query_processor import QueryProcessor
from src.Reranking.freshness import rerank_with_freshness
from src.Reranking.reranking import CrossEncoderReranker
from src.retrieval.chroma import ChromaRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.raptor import RaptorRetriever
from src.retrieval.web import WebRetriever
from src.routing.source_routing import SourceRouter
from src.utils.deduplication import remove_duplicates
from src.utils.evidence import select_evidence


class RAGPipeline:
    """Conditional RAG system: router selects retrieval, then evaluate and generate."""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        llm: Optional[Any] = None,
        embeddings: Optional[Any] = None,
        **overrides: Any,
    ):
        self.config = replace(config or RAGConfig.from_env(), **overrides)

        self.llm = llm or ChatOllama(
            model=self.config.ollama_model,
            temperature=self.config.temperature,
            base_url=self.config.ollama_base_url,
        )
        self.embeddings = embeddings or OllamaEmbeddings(
            model=self.config.embedding_model,
            base_url=self.config.ollama_base_url,
        )

        self.router = SourceRouter(llm=self.llm)
        self.query_processor = QueryProcessor(
            llm=self.llm,
            max_variations=self.config.max_query_variations,
            enable_multi_query=self.config.enable_multi_query,
        )

        self.vectorstore = None
        self.chroma = ChromaRetriever()
        self.web = WebRetriever(
            wikipedia_top_k=self.config.wikipedia_top_k,
            web_top_k=self.config.web_top_k,
        )
        self.raptor_indexer = None
        self.raptor = RaptorRetriever(k=self.config.raptor_k)
        if self.config.enable_raptor:
            self.raptor_indexer = RaptorIndexer(
                llm=self.llm,
                embeddings=self.embeddings,
                n_clusters=self.config.raptor_clusters,
            )
            self.raptor.indexer = self.raptor_indexer

        self.hybrid = HybridRetriever(
            local=self.chroma,
            web=self.web,
            raptor=self.raptor,
            use_raptor=self.config.enable_raptor,
        )

        self.reranker = CrossEncoderReranker(top_k=self.config.rerank_top_k)
        self.crag = CRAGEvaluator(llm=self.llm, min_relevant=self.config.min_evidence_docs)
        self.self_rag = None
        self.long_context = LongContext(model=self.config.ollama_model) if self.config.enable_long_context else None
        self.generation_chain = create_answer_chain(llm=self.llm, prompt=DEFAULT_GENERATION_PROMPT)
        self.conversation_memory = ConversationMemory()
        self.evaluator = None
        if self.config.enable_evaluation:
            self.evaluator = RAGEvaluator(
                model=self.config.ollama_model,
                temperature=self.config.temperature,
            )

    def build_index(self, documents: List[Document]):
        if not documents:
            raise ValueError("Cannot build index from empty documents.")

        self.vectorstore = build_vectorstore(
            documents=documents,
            embedding_model=self.embeddings,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            batch_size=self.config.index_batch_size,
        )
        retriever = create_retriever(vectorstore=self.vectorstore, k=self.config.retrieval_k)
        self.chroma.set_retriever(retriever)

        if self.config.enable_raptor:
            if self.raptor_indexer is None:
                self.raptor_indexer = RaptorIndexer(
                    llm=self.llm,
                    embeddings=self.embeddings,
                    n_clusters=self.config.raptor_clusters,
                )
                self.raptor.indexer = self.raptor_indexer
            try:
                self.raptor_indexer.build_tree(documents=documents)
            except Exception as exc:
                print(f"RAPTOR indexing failed: {exc}")

        return self.vectorstore

    def retrieve(self, queries: List[str], route: str) -> List[Document]:
        documents: List[Document] = []
        for query in queries:
            try:
                if route == LOCAL:
                    documents.extend(self.hybrid.retrieve_local(query))
                elif route == WEB:
                    documents.extend(self.hybrid.retrieve_web(query))
                else:
                    documents.extend(self.hybrid.retrieve(query))
            except Exception as exc:
                print(f"Retrieval failed for '{query}': {exc}")
        return documents

    def _retry_retrieval(self, query: str, queries: List[str], route: str) -> List[Document]:
        if route == LOCAL:
            return self.retrieve(queries or [query], WEB)
        if route == WEB:
            return self.retrieve(queries or [query], LOCAL)
        extra_queries = queries[1:] or [query]
        return self.retrieve(extra_queries, HYBRID)

    def evaluate_evidence(self, query: str, documents: List[Document]) -> EvidenceEvaluation:
        if not documents:
            return EvidenceEvaluation(
                sufficient=False,
                requires_more_retrieval=True,
                relevant_documents=[],
                reason="Empty retrieval results.",
            )

        evaluation = EvidenceEvaluation(
            sufficient=True,
            requires_more_retrieval=False,
            relevant_documents=documents,
            reason="Evidence accepted without LLM grading.",
        )

        if self.config.enable_crag:
            try:
                evaluation = self.crag.evaluate(query=query, documents=documents)
            except Exception as exc:
                print(f"CRAG evaluation failed: {exc}")

        if self.config.enable_self_rag:
            try:
                retriever = self.chroma.retriever
                if retriever is None:
                    class _StaticRetriever:
                        def invoke(self, _query):
                            return documents

                    retriever = _StaticRetriever()
                self.self_rag = SelfRAG(llm=self.llm, retriever=retriever)
                self_eval = self.self_rag.evaluate_evidence(query, documents)
                if self_eval.requires_more_retrieval:
                    evaluation = EvidenceEvaluation(
                        sufficient=False,
                        requires_more_retrieval=True,
                        relevant_documents=self_eval.relevant_documents or evaluation.relevant_documents,
                        reason=self_eval.reason,
                    )
                elif self_eval.relevant_documents:
                    evaluation.relevant_documents = self_eval.relevant_documents
                    evaluation.reason = f"{evaluation.reason} {self_eval.reason}".strip()
            except Exception as exc:
                print(f"Self-RAG evaluation failed: {exc}")

        return evaluation

    def build_context(self, query: str, evidence: List[Document]) -> tuple[str, bool]:
        context = format_docs(evidence)
        used_long_context = False
        if (
            self.config.enable_long_context
            and self.long_context is not None
            and context
            and len(context) >= self.config.long_context_min_chars
        ):
            try:
                context = self.long_context.compress_context(context=context, query=query)
                used_long_context = True
            except Exception as exc:
                print(f"Long-context compression failed: {exc}")
        return context, used_long_context

    def generate(self, query: str, context: str) -> str:
        if not context or not str(context).strip():
            return "The information is not available in the retrieved evidence."
        try:
            return self.generation_chain.invoke({"context": context, "question": query})
        except Exception as exc:
            print(f"Ollama generation failed: {exc}")
            return "The answer could not be generated because the language model failed."

    def store_memory(self, query: str, answer: str) -> None:
        self.conversation_memory.add_message(HumanMessage(content=query))
        self.conversation_memory.add_message(AIMessage(content=answer))

    def evaluate_response(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        if self.evaluator is None:
            return {}
        try:
            return {
                "context_relevance": self.evaluator.evaluate_context_relevance(
                    question=query, context=context
                ),
                "faithfulness": self.evaluator.evaluate_faithfulness(
                    context=context, answer=answer
                ),
                "answer_relevance": self.evaluator.evaluate_answer_relevance(
                    question=query, answer=answer
                ),
            }
        except Exception as exc:
            print(f"Response evaluation failed: {exc}")
            return {}

    def run(
        self,
        query: str,
        evaluate: Optional[bool] = None,
        route: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query = query.strip()
        queries = self.query_processor.process(query)

        selected_route = route.upper() if route else self.router.route(query)
        if selected_route not in VALID_ROUTES:
            selected_route = HYBRID

        documents = self.retrieve(queries, selected_route)
        documents = remove_duplicates(documents)

        try:
            ranked = rerank_with_freshness(
                query=query,
                documents=documents,
                reranker=self.reranker,
                relevance_weight=self.config.relevance_weight,
                freshness_weight=self.config.freshness_weight,
                top_k=self.config.rerank_top_k,
            )
        except Exception as exc:
            print(f"Reranker failed: {exc}")
            ranked = documents[: self.config.rerank_top_k]

        evidence = select_evidence(ranked, top_k=self.config.final_top_k)
        evaluation = self.evaluate_evidence(query, evidence)

        retries = 0
        while evaluation.requires_more_retrieval and retries < self.config.max_retrieval_retries:
            retries += 1
            extra = self._retry_retrieval(query, queries, selected_route)
            documents = remove_duplicates(documents + extra)
            try:
                ranked = rerank_with_freshness(
                    query=query,
                    documents=documents,
                    reranker=self.reranker,
                    relevance_weight=self.config.relevance_weight,
                    freshness_weight=self.config.freshness_weight,
                    top_k=self.config.rerank_top_k,
                )
            except Exception as exc:
                print(f"Reranker failed on retry: {exc}")
                ranked = documents[: self.config.rerank_top_k]
            evidence = select_evidence(ranked, top_k=self.config.final_top_k)
            evaluation = self.evaluate_evidence(query, evidence)

        if evaluation.relevant_documents:
            evidence = select_evidence(evaluation.relevant_documents, top_k=self.config.final_top_k)

        context, long_context_used = self.build_context(query, evidence)
        answer = self.generate(query, context)
        self.store_memory(query, answer)

        should_evaluate = self.config.enable_evaluation if evaluate is None else evaluate
        metrics = {}
        if should_evaluate:
            metrics = self.evaluate_response(query, answer, context)

        return {
            "query": query,
            "route": selected_route,
            "decision": selected_route,
            "mode": selected_route.lower(),
            "queries": queries,
            "answer": answer,
            "context": context,
            "retrieved_documents": documents,
            "reranked_documents": ranked,
            "evidence": evidence,
            "evidence_evaluation": asdict(evaluation),
            "long_context_used": long_context_used,
            "raptor_used": self.raptor.is_ready() and selected_route in (LOCAL, HYBRID),
            "evaluation": metrics,
        }


RAGIntegration = RAGPipeline
