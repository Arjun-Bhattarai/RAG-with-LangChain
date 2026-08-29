"""
Modular RAG pipeline.

Flow:

User Query
    ↓
Ollama Answerability Check
    ├── ANSWERABLE
    │       ↓
    │   Ollama Direct Answer
    │
    └── NOT_ANSWERABLE
            ↓
        Query Processing
            ↓
        LOCAL ChromaDB Retrieval
            ↓
        Reranking
            ↓
        CRAG / Self-RAG
            ↓
        Evidence sufficient?
            ├── YES → Generation
            │
            └── NO
                  ↓
              Web Retrieval
              ├── Tavily
              └── Wikipedia
                  ↓
              Reranking
                  ↓
              RAPTOR
                  ↓
              CRAG / Self-RAG
                  ↓
              Long Context
                  ↓
              Ollama Generation
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


# ---------------------------------------------------------
# LangChain
# ---------------------------------------------------------

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama, OllamaEmbeddings


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from src.advanced_indexing.raptor import RaptorIndexer

from src.advanced_RAG.CRAG.crag import CRAGEvaluator
from src.advanced_RAG.Long_Context.long_context import LongContext
from src.advanced_RAG.Self_RAG.self_rag import SelfRAG

from src.core.config import RAGConfig

from src.core.models import (
    HYBRID,
    LOCAL,
    WEB,
    VALID_ROUTES,
    EvidenceEvaluation,
)

from src.evaluation.rag_evaluation import RAGEvaluator

from src.memory.memory import ConversationMemory

from src.pipeline.generation_rag import (
    DEFAULT_GENERATION_PROMPT,
    create_answer_chain,
    format_docs,
)

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
    """
    Complete RAG pipeline.

    Priority:

    1. Ask Ollama if it already knows the answer.
    2. If not, search local ChromaDB.
    3. If local evidence is insufficient, use web retrieval.
    4. Apply reranking / RAPTOR / CRAG / Self-RAG / long context.
    5. Generate final answer with Ollama.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        llm: Optional[Any] = None,
        embeddings: Optional[Any] = None,
        **overrides: Any,
    ):

       
        # CONFIG
       

        self.config = replace(
            config or RAGConfig.from_env(),
            **overrides,
        )

       
        # OLLAMA
       

        self.llm = llm or ChatOllama(
            model=self.config.ollama_model,
            temperature=self.config.temperature,
            base_url=self.config.ollama_base_url,
        )

        self.embeddings = embeddings or OllamaEmbeddings(
            model=self.config.embedding_model,
            base_url=self.config.ollama_base_url,
        )

       
        # ANSWERABILITY CHECK
       

        self.answerability_prompt = ChatPromptTemplate.from_template(
            """
Determine whether you can answer the user's question reliably
using your existing knowledge.

Return ONLY one of these labels:

ANSWERABLE
NOT_ANSWERABLE

Rules:

- Return NOT_ANSWERABLE for current, live, latest, recent,
  today's, yesterday's, tomorrow's, or time-sensitive information.
- Return NOT_ANSWERABLE if you are uncertain.
- Return ANSWERABLE only when you are reasonably confident
  that your existing knowledge is sufficient.

Question:
{question}

Decision:
"""
        )

        self.answerability_chain = (
            self.answerability_prompt
            | self.llm
            | StrOutputParser()
        )

       
        # QUERY PROCESSING
       

        self.query_processor = QueryProcessor(
            llm=self.llm,
            max_variations=self.config.max_query_variations,
            enable_multi_query=self.config.enable_multi_query,
        )

       
        # ROUTER
       

        self.router = SourceRouter(
            llm=self.llm
        )

       
        # LOCAL RETRIEVAL
       

        self.vectorstore = None

        self.chroma = ChromaRetriever()

        # -----------------------------------------------------
        # IMPORTANT:
        # Load your existing ChromaDB here if your
        # ChromaRetriever supports persistent loading.
        #
        # Otherwise build_index() must be called before
        # querying.
        # -----------------------------------------------------

        try:
            self._load_existing_chroma()
        except Exception as exc:
            print(
                f"Warning: existing ChromaDB could not be loaded: {exc}"
            )

       
        # WEB RETRIEVAL
       

        self.web = WebRetriever(
            wikipedia_top_k=self.config.wikipedia_top_k,
            web_top_k=self.config.web_top_k,
        )

       
        # RAPTOR
       

        self.raptor_indexer = None

        self.raptor = RaptorRetriever(
            k=self.config.raptor_k
        )

        if self.config.enable_raptor:

            self.raptor_indexer = RaptorIndexer(
                llm=self.llm,
                embeddings=self.embeddings,
                n_clusters=self.config.raptor_clusters,
            )

            self.raptor.indexer = self.raptor_indexer

       
        # HYBRID RETRIEVER
       

        self.hybrid = HybridRetriever(
            local=self.chroma,
            web=self.web,
            raptor=self.raptor,
            use_raptor=self.config.enable_raptor,
        )

       
        # RERANKER
       

        self.reranker = CrossEncoderReranker(
            top_k=self.config.rerank_top_k
        )

       
        # CRAG
       

        self.crag = CRAGEvaluator(
            llm=self.llm,
            min_relevant=self.config.min_evidence_docs,
        )

       
        # SELF-RAG
       

        self.self_rag = None

       
        # LONG CONTEXT
       

        self.long_context = (
            LongContext(
                model=self.config.ollama_model
            )
            if self.config.enable_long_context
            else None
        )

       
        # GENERATION
       

        self.generation_chain = create_answer_chain(
            llm=self.llm,
            prompt=DEFAULT_GENERATION_PROMPT,
        )

       
        # MEMORY
       

        self.conversation_memory = ConversationMemory()

       
        # EVALUATION
       

        self.evaluator = None

        if self.config.enable_evaluation:

            self.evaluator = RAGEvaluator(
                model=self.config.ollama_model,
                temperature=self.config.temperature,
            )

    # CHROMA INITIALIZATION

    def _load_existing_chroma(self) -> None:
        """
        Load an existing persistent ChromaDB collection.

        IMPORTANT:
        The exact implementation depends on how your
        ChromaRetriever is written.

        If ChromaRetriever already loads the persistent
        collection in its constructor, nothing else is required.

        If it requires a retriever, this method should be
        connected to that implementation.
        """

        # If your ChromaRetriever already initializes itself,
        # this will simply use that retriever.

        if getattr(self.chroma, "retriever", None) is not None:

            print("Local ChromaDB retriever loaded.")

        else:

            print(
                "Local ChromaDB retriever is currently empty."
            )

    # BUILD INDEX

    def build_index(
        self,
        documents: List[Document],
    ):

        if not documents:
            raise ValueError(
                "Cannot build index from empty documents."
            )

        self.vectorstore = build_vectorstore(
            documents=documents,
            embedding_model=self.embeddings,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            batch_size=self.config.index_batch_size,
        )

        retriever = create_retriever(
            vectorstore=self.vectorstore,
            k=self.config.retrieval_k,
        )

        self.chroma.set_retriever(
            retriever
        )

        # -----------------------------------------------------
        # RAPTOR
        # -----------------------------------------------------

        if self.config.enable_raptor:

            if self.raptor_indexer is None:

                self.raptor_indexer = RaptorIndexer(
                    llm=self.llm,
                    embeddings=self.embeddings,
                    n_clusters=self.config.raptor_clusters,
                )

                self.raptor.indexer = self.raptor_indexer

            try:

                self.raptor_indexer.build_tree(
                    documents=documents
                )

            except Exception as exc:

                print(
                    f"RAPTOR indexing failed: {exc}"
                )

        return self.vectorstore

    # ANSWERABILITY

    def check_answerability(
        self,
        query: str,
    ) -> str:

        try:

            decision = (
                self.answerability_chain
                .invoke(
                    {
                        "question": query
                    }
                )
                .strip()
                .upper()
            )

        except Exception as exc:

            print(
                f"Ollama answerability check failed: {exc}"
            )

            return "NOT_ANSWERABLE"

        if "NOT_ANSWERABLE" in decision:

            return "NOT_ANSWERABLE"

        if "ANSWERABLE" in decision:

            return "ANSWERABLE"

        return "NOT_ANSWERABLE"

    # CURRENT QUERY DETECTION

    def is_current_query(
        self,
        query: str,
    ) -> bool:

        keywords = [
            "today",
            "current",
            "latest",
            "now",
            "live",
            "recent",
            "yesterday",
            "tomorrow",
        ]

        query_lower = query.lower()

        return any(
            keyword in query_lower
            for keyword in keywords
        )

    # RETRIEVAL

    def retrieve(
        self,
        queries: List[str],
        route: str,
    ) -> List[Document]:

        documents: List[Document] = []

        for query in queries:

            try:

                if route == LOCAL:

                    documents.extend(
                        self.hybrid.retrieve_local(
                            query
                        )
                    )

                elif route == WEB:

                    documents.extend(
                        self.hybrid.retrieve_web(
                            query
                        )
                    )

                else:

                    documents.extend(
                        self.hybrid.retrieve(
                            query
                        )
                    )

            except Exception as exc:

                print(
                    f"Retrieval failed for '{query}': {exc}"
                )

        return documents

    # WEB FALLBACK

    def retrieve_web_fallback(
        self,
        query: str,
        queries: List[str],
    ) -> List[Document]:

        print(
            "Local evidence insufficient."
        )

        print(
            "Starting web fallback..."
        )

        documents: List[Document] = []

        try:

            documents = self.retrieve(
                queries or [query],
                WEB,
            )

        except Exception as exc:

            print(
                f"Web fallback failed: {exc}"
            )

        return documents

    # RETRY

    def _retry_retrieval(
        self,
        query: str,
        queries: List[str],
        route: str,
    ) -> List[Document]:

        if route == LOCAL:

            return self.retrieve_web_fallback(
                query,
                queries,
            )

        if route == WEB:

            return self.retrieve(
                queries or [query],
                LOCAL,
            )

        extra_queries = (
            queries[1:]
            or [query]
        )

        return self.retrieve(
            extra_queries,
            HYBRID,
        )

    # EVIDENCE EVALUATION

    def evaluate_evidence(
        self,
        query: str,
        documents: List[Document],
    ) -> EvidenceEvaluation:

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

        # -----------------------------------------------------
        # CRAG
        # -----------------------------------------------------

        if self.config.enable_crag:

            try:

                evaluation = self.crag.evaluate(
                    query=query,
                    documents=documents,
                )

            except Exception as exc:

                print(
                    f"CRAG evaluation failed: {exc}"
                )

        # -----------------------------------------------------
        # SELF-RAG
        # -----------------------------------------------------

        if self.config.enable_self_rag:

            try:

                retriever = self.chroma.retriever

                if retriever is None:

                    class _StaticRetriever:

                        def invoke(
                            self,
                            _query,
                        ):
                            return documents

                    retriever = _StaticRetriever()

                self.self_rag = SelfRAG(
                    llm=self.llm,
                    retriever=retriever,
                )

                self_eval = (
                    self.self_rag.evaluate_evidence(
                        query,
                        documents,
                    )
                )

                if self_eval.requires_more_retrieval:

                    evaluation = EvidenceEvaluation(
                        sufficient=False,
                        requires_more_retrieval=True,
                        relevant_documents=(
                            self_eval.relevant_documents
                            or evaluation.relevant_documents
                        ),
                        reason=self_eval.reason,
                    )

                elif self_eval.relevant_documents:

                    evaluation.relevant_documents = (
                        self_eval.relevant_documents
                    )

                    evaluation.reason = (
                        f"{evaluation.reason} "
                        f"{self_eval.reason}"
                    ).strip()

            except Exception as exc:

                print(
                    f"Self-RAG evaluation failed: {exc}"
                )

        return evaluation

    # CONTEXT

    def build_context(
        self,
        query: str,
        evidence: List[Document],
    ) -> tuple[str, bool]:

        context = format_docs(
            evidence
        )

        used_long_context = False

        if (
            self.config.enable_long_context
            and self.long_context is not None
            and context
            and len(context)
            >= self.config.long_context_min_chars
        ):

            try:

                context = (
                    self.long_context.compress_context(
                        context=context,
                        query=query,
                    )
                )

                used_long_context = True

            except Exception as exc:

                print(
                    f"Long-context compression failed: {exc}"
                )

        return (
            context,
            used_long_context,
        )

    # GENERATION

    def generate(
        self,
        query: str,
        context: str,
    ) -> str:

        if not context or not str(context).strip():

            return (
                "The information is not available "
                "in the retrieved evidence."
            )

        try:

            return self.generation_chain.invoke(
                {
                    "context": context,
                    "question": query,
                }
            )

        except Exception as exc:

            print(
                f"Ollama generation failed: {exc}"
            )

            return (
                "The answer could not be generated "
                "because the language model failed."
            )

    # MEMORY

    def store_memory(
        self,
        query: str,
        answer: str,
    ) -> None:

        self.conversation_memory.add_message(
            HumanMessage(
                content=query
            )
        )

        self.conversation_memory.add_message(
            AIMessage(
                content=answer
            )
        )

    # RESPONSE EVALUATION

    def evaluate_response(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> Dict[str, Any]:

        if self.evaluator is None:

            return {}

        try:

            return {

                "context_relevance":
                    self.evaluator.evaluate_context_relevance(
                        question=query,
                        context=context,
                    ),

                "faithfulness":
                    self.evaluator.evaluate_faithfulness(
                        context=context,
                        answer=answer,
                    ),

                "answer_relevance":
                    self.evaluator.evaluate_answer_relevance(
                        question=query,
                        answer=answer,
                    ),
            }

        except Exception as exc:

            print(
                f"Response evaluation failed: {exc}"
            )

            return {}

    # MAIN PIPELINE

    def run(
        self,
        query: str,
        evaluate: Optional[bool] = None,
        route: Optional[str] = None,
    ) -> Dict[str, Any]:

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        query = query.strip()

       
        # STEP 1 — OLLAMA ANSWERABILITY
       

        print(
            "\nChecking Ollama answerability..."
        )

        answerability = (
            self.check_answerability(
                query
            )
        )

        print(
            f"Ollama decision: {answerability}"
        )

        # STEP 2 — DIRECT OLLAMA

        if answerability == "ANSWERABLE":

            print(
                "Answered directly by Ollama."
            )

            try:

                response = self.llm.invoke(
                    query
                )

                answer = response.content

            except Exception as exc:

                print(
                    f"Direct Ollama generation failed: {exc}"
                )

                answer = (
                    "The answer could not be generated."
                )

            self.store_memory(
                query,
                answer,
            )

            return {

                "query": query,

                "route": "OLLAMA",

                "decision": "ANSWERABLE",

                "mode": "direct_ollama",

                "queries": [query],

                "answer": answer,

                "context": "",

                "retrieved_documents": [],

                "reranked_documents": [],

                "evidence": [],

                "evidence_evaluation": {},

                "long_context_used": False,

                "raptor_used": False,

                "evaluation": {},
            }

        # STEP 3 — OLLAMA FAILED

        print(
            "Ollama cannot answer reliably."
        )

        print(
            "Starting local RAG..."
        )

        # STEP 4 — QUERY PROCESSING

        queries = (
            self.query_processor.process(
                query
            )
        )

        # STEP 5 — LOCAL FIRST

        selected_route = LOCAL

        if route:

            selected_route = route.upper()

        if selected_route not in VALID_ROUTES:

            selected_route = LOCAL

        # FORCE LOCAL FIRST.
        #
        # Even if the router says WEB, we don't go to the web
        # before checking local knowledge.
        # 

        local_documents = self.retrieve(
            queries,
            LOCAL,
        )

        local_documents = remove_duplicates(
            local_documents
        )

        print(
            f"Local documents retrieved: "
            f"{len(local_documents)}"
        )

        # STEP 6 — RERANK LOCAL DOCUMENTS

        documents = local_documents

        if documents:

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

                print(
                    f"Local reranker failed: {exc}"
                )

                ranked = documents[
                    :self.config.rerank_top_k
                ]

        else:

            ranked = []

        evidence = select_evidence(
            ranked,
            top_k=self.config.final_top_k,
        )

        # STEP 7 — CHECK LOCAL EVIDENCE

        evaluation = self.evaluate_evidence(
            query,
            evidence,
        )

        print(
            "Local evidence sufficient:",
            evaluation.sufficient,
        )

        print(
            "Requires more retrieval:",
            evaluation.requires_more_retrieval,
        )

        # STEP 8 — WEB FALLBACK

        if evaluation.requires_more_retrieval:

            print(
                "\nLocal RAG was insufficient."
            )

            print(
                "Falling back to external retrieval..."
            )

            web_documents = (
                self.retrieve_web_fallback(
                    query,
                    queries,
                )
            )

            web_documents = remove_duplicates(
                web_documents
            )

            print(
                f"Web documents retrieved: "
                f"{len(web_documents)}"
            )

             
            # Combine local + web evidence

            documents = remove_duplicates(
                local_documents
                + web_documents
            )

            # Rerank combined results

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

                print(
                    f"Combined reranker failed: {exc}"
                )

                ranked = documents[
                    :self.config.rerank_top_k
                ]

            evidence = select_evidence(
                ranked,
                top_k=self.config.final_top_k,
            )

            evaluation = self.evaluate_evidence(
                query,
                evidence,
            )

        # STEP 9 — FINAL EVIDENCE

        if evaluation.relevant_documents:

            evidence = select_evidence(
                evaluation.relevant_documents,
                top_k=self.config.final_top_k,
            )

        # STEP 10 — BUILD CONTEXT

        context, long_context_used = (
            self.build_context(
                query,
                evidence,
            )
        )

        # STEP 11 — GENERATE

        answer = self.generate(
            query,
            context,
        )

        # STEP 12 — MEMORY

        self.store_memory(
            query,
            answer,
        )

        # STEP 13 — EVALUATION

        should_evaluate = (
            self.config.enable_evaluation
            if evaluate is None
            else evaluate
        )

        metrics = {}

        if should_evaluate:

            metrics = self.evaluate_response(
                query,
                answer,
                context,
            )

        
        # RETURN
        

        return {

            "query": query,

            "route": "LOCAL"
            if not evaluation.requires_more_retrieval
            else "HYBRID",

            "decision": answerability,

            "mode": "local_rag"
            if not evaluation.requires_more_retrieval
            else "web_fallback_rag",

            "queries": queries,

            "answer": answer,

            "context": context,

            "retrieved_documents": documents,

            "reranked_documents": ranked,

            "evidence": evidence,

            "evidence_evaluation": asdict(
                evaluation
            ),

            "long_context_used":
                long_context_used,

            "raptor_used": (
                self.raptor.is_ready()
                and not evaluation.requires_more_retrieval
            ),

            "evaluation": metrics,
        }


RAGIntegration = RAGPipeline