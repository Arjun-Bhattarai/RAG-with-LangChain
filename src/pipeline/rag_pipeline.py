from __future__ import annotations

import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


# 
# PROJECT ROOT
# 

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


# 
# LANGCHAIN
# 

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama, OllamaEmbeddings


# 
# PROJECT IMPORTS
# 

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


# 
# PERSONAL QUERY GENERATION PROMPT
# (module-level so it isn't rebuilt on every call)
# 

PERSONAL_PROMPT = ChatPromptTemplate.from_template(
    """
You are Arjunx, the personal AI assistant for
Arjun Bhattarai.

Your task is to answer questions about Arjun Bhattarai
and Arjunx using the provided website evidence.

IMPORTANT RULES:

1. Use the website evidence as the primary source.
2. Do not invent information.
3. Do not guess.
4. Do not use unsupported facts about Arjun.
5. If the requested information is not available in the
   website evidence, clearly say so.
6. Answer naturally and concisely.
7. If the user asks who developed, created, built, or made
   Arjunx, identify Arjun Bhattarai as the developer ONLY
   when supported by the website evidence.
8. For Arjunx developer/creator questions, clearly state
   that Arjun Bhattarai is the developer/creator of Arjunx
   when supported by the evidence.
9. Always provide Arjun's personal website link at the
   end of the response.

Personal Website:
https://arjunbhattarai8.com.np

Website Evidence:
{context}

Question:
{question}

Answer:
"""
)


class RAGPipeline:

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        llm: Optional[Any] = None,
        embeddings: Optional[Any] = None,
        **overrides: Any,
    ):

        # CONFIGURATION

        self.config = replace(
            config or RAGConfig.from_env(),
            **overrides,
        )

        # ARJUNX PERSONAL WEBSITE

        self.personal_website_url = (
            "https://arjunbhattarai8.com.np"
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

    # 
    # CHROMA INITIALIZATION
    # 

    def _load_existing_chroma(self) -> None:

        if getattr(
            self.chroma,
            "retriever",
            None,
        ) is not None:

            print(
                "Local ChromaDB retriever loaded."
            )

        else:

            print(
                "Local ChromaDB retriever is currently empty."
            )

    # 
    # BUILD INDEX
    # 

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

        # RAPTOR INDEX

        if self.config.enable_raptor:

            if self.raptor_indexer is None:

                self.raptor_indexer = RaptorIndexer(
                    llm=self.llm,
                    embeddings=self.embeddings,
                    n_clusters=self.config.raptor_clusters,
                )

                self.raptor.indexer = (
                    self.raptor_indexer
                )

            try:

                self.raptor_indexer.build_tree(
                    documents=documents
                )

            except Exception as exc:

                print(
                    f"RAPTOR indexing failed: {exc}"
                )

        return self.vectorstore

    # 
    # PERSONAL QUERY DETECTION
    # 

    def is_personal_query(
        self,
        query: str,
    ) -> bool:
        """
        Detect questions about Arjun Bhattarai or Arjunx.

        Personal and Arjunx-related queries are always routed
        to Arjun Bhattarai's personal website.
        """

        query_lower = query.lower().strip()

        personal_keywords = [

            # ARJUN BHATTARAI — NAME / IDENTITY

            "arjun bhattarai",
            "who is arjun",
            "who's arjun",
            "who is arjun bhattarai",
            "about arjun",
            "tell me about arjun",
            "tell me about arjun bhattarai",

            # PROFILE

            "arjun profile",
            "arjun's profile",
            "arjun portfolio",
            "arjun's portfolio",
            "arjun's bio",
            "arjun bio",

            # SKILLS

            "arjun skills",
            "arjun's skills",
            "what skills does arjun have",
            "what are arjun's skills",
            "technologies arjun knows",
            "tech stack of arjun",

            # PROJECTS

            "arjun projects",
            "arjun's projects",
            "what projects has arjun built",
            "projects built by arjun",
            "projects created by arjun",
            "what has arjun built",

            # EDUCATION

            "arjun education",
            "arjun's education",
            "where did arjun study",
            "what did arjun study",

            # EXPERIENCE

            "arjun experience",
            "arjun's experience",
            "arjun work experience",
            "arjun's work experience",

            # WORK

            "what does arjun do",
            "what does arjun work on",
            "what is arjun working on",
            "what is arjun currently working on",

            # DEVELOPER / ENGINEER

            "arjun developer",
            "arjun programmer",
            "arjun engineer",
            "arjun software developer",
            "arjun full stack developer",

            # WEBSITE / PORTFOLIO

            "arjun website",
            "arjun's website",
            "arjun portfolio website",
            "arjun personal website",
            "arjun website link",
            "arjun portfolio link",

            # ARJUNX — DEVELOPER / CREATOR

            "who is the developer of arjunx",
            "who developed arjunx",
            "who is arjunx developer",
            "who developed arjunx ai",
            "who created arjunx",
            "who built arjunx",
            "who made arjunx",
            "who is the creator of arjunx",
            "who is the creator behind arjunx",
            "who is behind arjunx",
            "developer of arjunx",
            "creator of arjunx",
            "arjunx developer",
            "arjunx creator",
            "arjunx creator name",
            "arjunx built by",
            "arjunx created by",
            "arjunx developed by",
            "who made this arjunx",
            "who built this arjunx",
            "who created this arjunx",

        ]

        return any(
            keyword in query_lower
            for keyword in personal_keywords
        )

    # 
    # PERSONAL WEBSITE RETRIEVAL
    # 

    def retrieve_personal_website(
        self,
    ) -> List[Document]:
        """
        Retrieve readable content from Arjun's personal website.
        """

        print(
            "\nRetrieving Arjun Bhattarai's website..."
        )

        print(
            f"URL: {self.personal_website_url}"
        )

        try:

            response = requests.get(
                self.personal_website_url,
                timeout=10,
                headers={
                    "User-Agent": "Arjunx-RAG/1.0"
                },
            )

            response.raise_for_status()

            soup = BeautifulSoup(
                response.text,
                "html.parser",
            )

            # Remove non-content elements.

            for element in soup(
                [
                    "script",
                    "style",
                    "noscript",
                    "svg",
                    "nav",
                    "footer",
                ]
            ):

                element.decompose()

            text = soup.get_text(
                separator="\n",
                strip=True,
            )

            if not text:

                print(
                    "Personal website returned no readable content."
                )

                return []

            document = Document(
                page_content=text,
                metadata={
                    "source": "personal_website",
                    "title": (
                        "Arjun Bhattarai - Personal Website"
                    ),
                    "url": self.personal_website_url,
                },
            )

            print(
                "Personal website retrieved successfully."
            )

            print(
                f"Website content length: {len(text)} characters"
            )

            return [document]

        except Exception as exc:

            print(
                f"Personal website retrieval failed: {exc}"
            )

            return []

    # 
    # PERSONAL QUERY ANSWER
    # 

    def answer_personal_query(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        Answer questions about Arjun Bhattarai using
        information from his personal website.
        """

        documents = (
            self.retrieve_personal_website()
        )

        # WEBSITE FAILED

        if not documents:

            answer = (
                "I couldn't retrieve Arjun Bhattarai's "
                "personal website at the moment."
            )

            self.store_memory(
                query,
                answer,
            )

            return {

                "query": query,

                "route": "PERSONAL_WEBSITE",

                "decision": "PERSONAL",

                "mode": "personal_website",

                "queries": [query],

                "answer": answer,

                "website": self.personal_website_url,

                "website_url": self.personal_website_url,

                "context": "",

                "retrieved_documents": [],

                "reranked_documents": [],

                "evidence": [],

                "evidence_evaluation": {},

                "long_context_used": False,

                "raptor_used": False,

                "evaluation": {},
            }

        # RERANK WEBSITE CONTENT

        try:

            ranked = self.reranker.rerank_documents(
                query=query,
                documents=documents,
            )

        except Exception as exc:

            print(
                f"Personal website reranking failed: {exc}"
            )

            ranked = documents

        # SELECT EVIDENCE

        evidence = select_evidence(
            ranked,
            top_k=self.config.final_top_k,
        )

        # BUILD CONTEXT

        context, long_context_used = (
            self.build_context(
                query,
                evidence,
            )
        )

        # PERSONAL GENERATION

        personal_chain = (
            PERSONAL_PROMPT
            | self.llm
            | StrOutputParser()
        )

        # GENERATE

        try:

            answer = personal_chain.invoke(
                {
                    "context": context,
                    "question": query,
                }
            )

        except Exception as exc:

            print(
                f"Personal answer generation failed: {exc}"
            )

            answer = (
                "I couldn't generate an answer from "
                "Arjun's personal website."
            )

        # ADD WEBSITE LINK TO ANSWER

        answer = (
            f"{answer}\n\n"
            f"🌐 Arjun's Website: "
            f"{self.personal_website_url}"
        )

        # MEMORY

        self.store_memory(
            query,
            answer,
        )

        # RETURN

        return {

            "query": query,

            "route": "PERSONAL_WEBSITE",

            "decision": "PERSONAL",

            "mode": "personal_website",

            "queries": [query],

            "answer": answer,

            # Explicit website fields
            "website": self.personal_website_url,

            "website_url": self.personal_website_url,

            "context": context,

            "retrieved_documents": documents,

            "reranked_documents": ranked,

            "evidence": evidence,

            "evidence_evaluation": {

                "sufficient": bool(evidence),

                "requires_more_retrieval": (
                    not bool(evidence)
                ),

                "reason": (
                    "Answer generated from Arjun Bhattarai's "
                    "personal website."
                ),
            },

            "long_context_used": (
                long_context_used
            ),

            "raptor_used": False,

            "evaluation": {},
        }

    # 
    # ANSWERABILITY
    # 

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

    # 
    # CURRENT QUERY DETECTION
    # 

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

    # 
    # RETRIEVAL
    # 

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

    # 
    # WEB FALLBACK
    # 

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

    # 
    # RETRY RETRIEVAL
    # 

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

    # 
    # EVIDENCE EVALUATION
    # 

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

        # CRAG

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

        # SELF-RAG

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

    # 
    # CONTEXT
    # 

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

    # 
    # GENERATION
    # 

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

    # 
    # MEMORY
    # 

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

    # 
    # RESPONSE EVALUATION
    # 

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

    # 
    # MAIN PIPELINE
    # 

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

        # STEP 0 — PERSONAL QUERY

        print(
            "\nChecking whether this is a personal query..."
        )

        if self.is_personal_query(query):

            print(
                "Personal query detected."
            )

            print(
                "Using Arjun Bhattarai's personal website."
            )

            return self.answer_personal_query(
                query
            )

        print(
            "Not a personal query."
        )

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

                "website": self.personal_website_url,

                "website_url": self.personal_website_url,

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

        # Force local retrieval first.

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

        used_web = False

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

            if web_documents:

                used_web = True

            # COMBINE LOCAL + WEB

            documents = remove_duplicates(
                local_documents
                + web_documents
            )

            # RERANK COMBINED RESULTS

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

        # STEP 14 — RETURN RESULT

        return {

            "query": query,

            "route": (
                "HYBRID"
                if used_web
                else "LOCAL"
            ),

            "decision": answerability,

            "mode": (
                "web_fallback_rag"
                if used_web
                else "local_rag"
            ),

            "queries": queries,

            "answer": answer,

            # Always expose your website to the API/frontend.
            "website": self.personal_website_url,

            "website_url": self.personal_website_url,

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
                and not used_web
            ),

            "evaluation": metrics,
        }


# 
# COMPATIBILITY ALIAS
# 

RAGIntegration = RAGPipeline