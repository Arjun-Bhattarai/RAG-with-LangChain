"""
Reusable integrated RAG pipeline.

Flow
----
1. Ask local Ollama whether it can answer reliably.
2. If yes, answer directly with Ollama.
3. If no:
   - Retrieve from Wikipedia + Tavily.
   - Deduplicate external documents.
   - Build a Chroma vector store.
   - Perform Multi-Query retrieval.
   - Rerank retrieved documents.
   - Run RAPTOR for additional hierarchical evidence.
   - Run Self-RAG for additional verification/evidence.
   - Run Long Context processing.
   - Generate the final grounded answer with Ollama.
4. Store the conversation turn in memory.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama, OllamaEmbeddings


from data_source.wikipedia import WikipediaSource
from data_source.web_search import WebSearchSource


from src.pipeline.indexing_rag import build_vectorstore
from src.pipeline.retrieval_rag import (
    create_retriever,
)

from src.query_translation.multi_query import (
    create_multi_query_retrieval_chain,
)


from src.Reranking.reranking import CrossEncoderReranker
from src.advanced_indexing.raptor import RaptorIndexer


from src.advanced_RAG.Self_RAG.self_rag import SelfRAG
from src.advanced_RAG.Long_Context.long_context import LongContext


from src.memory.memory import ConversationMemory


from src.evaluation.rag_evaluation import RAGEvaluator


class RAGIntegration:
    """
    Reusable integrated RAG system.

    The system always asks the local Ollama model first.

    If Ollama can answer:
        Query -> Ollama -> Answer

    If Ollama cannot answer:
        Query
          -> Wikipedia + Tavily
          -> Deduplication
          -> Chroma
          -> Multi-Query
          -> Retrieval
          -> Reranking
          -> RAPTOR
          -> Self-RAG
          -> Long Context
          -> Ollama
          -> Answer
    """

    def __init__(
        self,
        llm_model: str = "llama3:latest",
        embedding_model: str = "nomic-embed-text:latest",
        ollama_base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.0,
        wikipedia_top_k: int = 5,
        web_top_k: int = 5,
        retrieval_k: int = 5,
        rerank_top_k: int = 5,
        raptor_clusters: int = 3,
        index_batch_size: int = 32,
        enable_raptor: bool = True,
        enable_self_rag: bool = True,
        enable_long_context: bool = True,
        enable_evaluation: bool = True,
    ):
        """
        Initialize the integrated RAG system.
        """

        # Configuration

        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature

        self.wikipedia_top_k = wikipedia_top_k
        self.web_top_k = web_top_k
        self.retrieval_k = retrieval_k
        self.rerank_top_k = rerank_top_k
        self.raptor_clusters = raptor_clusters
        self.index_batch_size = index_batch_size

        self.enable_raptor = enable_raptor
        self.enable_self_rag = enable_self_rag
        self.enable_long_context = enable_long_context
        self.enable_evaluation = enable_evaluation

        # Local Ollama

        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=self.temperature,
            base_url=self.ollama_base_url,
        )

        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.ollama_base_url,
        )

        # Data sources

        self.wikipedia = WikipediaSource(
            top_k=self.wikipedia_top_k,
        )

        self.web_search = WebSearchSource(
            top_k=self.web_top_k,
        )

        # Generation prompt

        self.generation_prompt = ChatPromptTemplate.from_template("""
You are a careful RAG assistant.

Answer the question using ONLY the provided evidence.

Evidence:
{context}

Question:
{question}

Rules:
- Do not invent facts.
- Do not use unsupported information.
- Prefer the most recent and directly relevant evidence.
- If the evidence is insufficient, say that the information
  is not available in the retrieved evidence.
- Answer clearly and concisely.

Answer:
""")

        self.generation_chain = self.generation_prompt | self.llm | StrOutputParser()

        # Ollama answerability check

        self.answerability_prompt = ChatPromptTemplate.from_template("""
Determine whether you can answer the user's question
reliably using your existing knowledge.

Important:
- If the question requires current, live, latest, today's,
  recent, or time-sensitive information, return NOT_ANSWERABLE.
- If you are uncertain, return NOT_ANSWERABLE.
- Return only one label.

ANSWERABLE
NOT_ANSWERABLE

Question:
{question}

Decision:
""")

        self.answerability_chain = (
            self.answerability_prompt | self.llm | StrOutputParser()
        )

        # Runtime components

        self.vectorstore = None
        self.retriever = None
        self.multi_query_retriever = None

        self.reranker = CrossEncoderReranker(
            top_k=self.rerank_top_k,
        )

        self.raptor = None
        self.self_rag = None
        self.long_context = None

        if self.enable_raptor:
            self.raptor = RaptorIndexer(
                llm=self.llm,
                embeddings=self.embeddings,
                n_clusters=self.raptor_clusters,
            )

        if self.enable_long_context:
            self.long_context = LongContext(
                model=self.llm_model,
            )

        # Memory

        self.conversation_memory = ConversationMemory()

        # Evaluation

        self.evaluator = None

        if self.enable_evaluation:
            self.evaluator = RAGEvaluator(
                model=self.llm_model,
                temperature=self.temperature,
            )

    # Answerability

    def check_answerability(
        self,
        query: str,
    ) -> str:
        """
        Ask local Ollama whether it can answer the query.
        """

        decision = self.answerability_chain.invoke({"question": query}).strip().upper()

        if "NOT_ANSWERABLE" in decision:
            return "NOT_ANSWERABLE"

        if "ANSWERABLE" in decision:
            return "ANSWERABLE"

        # Conservative fallback.
        return "NOT_ANSWERABLE"

    # External retrieval

    def retrieve_external_documents(
        self,
        query: str,
    ) -> List[Document]:
        """
        Retrieve documents from Wikipedia and Tavily.
        """

        wikipedia_documents = self.wikipedia.retrieve(query)

        web_documents = self.web_search.retrieve(query)

        documents = wikipedia_documents + web_documents

        return self.deduplicate_documents(documents)

    # Deduplication

    @staticmethod
    def deduplicate_documents(
        documents: List[Document],
    ) -> List[Document]:
        """
        Remove duplicate external search results.
        """

        unique_documents: Dict[str, Document] = {}

        for document in documents:

            url = document.metadata.get("url")

            if url:
                key = url
            else:
                key = (
                    document.metadata.get(
                        "title",
                        "",
                    )
                    + document.page_content[:200]
                )

            if key not in unique_documents:
                unique_documents[key] = document

        return list(unique_documents.values())

    # Vector store

    def build_index(
        self,
        documents: List[Document],
    ):
        """
        Build Chroma vector store and retriever.
        """

        if not documents:
            raise ValueError("Cannot build index from empty documents.")

        self.vectorstore = build_vectorstore(
            documents=documents,
            embedding_model=self.embeddings,
            batch_size=self.index_batch_size,
        )

        self.retriever = create_retriever(
            vectorstore=self.vectorstore,
            k=self.retrieval_k,
        )

        self.multi_query_retriever = create_multi_query_retrieval_chain(
            retriever=self.retriever,
            llm=self.llm,
        )

        return self.vectorstore

    # Standard + Multi-Query Retrieval

    def retrieve_documents(
        self,
        query: str,
    ) -> List[Document]:
        """
        Retrieve documents through the Multi-Query pipeline.
        """

        if self.multi_query_retriever is None:
            raise RuntimeError("Vector index has not been built.")

        documents = self.multi_query_retriever.invoke(query)

        return documents

    # Reranking

    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:
        """
        Rerank retrieved documents.
        """

        if not documents:
            return []

        return self.reranker.rerank_documents(
            query=query,
            documents=documents,
        )

    # RAPTOR

    def run_raptor(
        self,
        query: str,
        documents: List[Document],
    ) -> Dict[str, List[Any]]:
        """
        Build and query RAPTOR evidence.
        """

        if not self.enable_raptor:
            return {
                "leaf": [],
                "clusters": [],
            }

        if not documents:
            return {
                "leaf": [],
                "clusters": [],
            }

        if self.raptor is None:
            self.raptor = RaptorIndexer(
                llm=self.llm,
                embeddings=self.embeddings,
                n_clusters=self.raptor_clusters,
            )

        self.raptor.build_tree(documents=documents)

        return self.raptor.retrieve(
            query=query,
            k=3,
        )

    # Self-RAG

    def run_self_rag(
        self,
        query: str,
    ) -> str:
        """
        Run the existing Self-RAG implementation.
        """

        if not self.enable_self_rag:
            return ""

        if self.retriever is None:
            raise RuntimeError("Retriever must be created before Self-RAG.")

        self.self_rag = SelfRAG(
            llm=self.llm,
            retriever=self.retriever,
        )

        result = self.self_rag.invoke(
            query,
            max_retries=2,
        )

        return str(result)

    # Long Context

    def run_long_context(
        self,
        query: str,
        documents: List[Document],
    ) -> str:
        """
        Run the existing Long Context implementation.
        """

        if not self.enable_long_context:
            return ""

        if not documents:
            return ""

        if self.long_context is None:
            self.long_context = LongContext(
                model=self.llm_model,
            )

        result = self.long_context.run(
            documents=documents,
            query=query,
            compress=True,
        )

        if isinstance(result, dict):
            return str(result.get("answer", ""))

        return str(result)

    # Context construction

    @staticmethod
    def build_context(
        reranked_documents: List[Document],
        raptor_results: Optional[Dict[str, List[Any]]] = None,
        self_rag_answer: str = "",
        long_context_answer: str = "",
    ) -> str:
        """
        Build the final evidence context.
        """

        context_parts: List[str] = []

        # Reranked document evidence

        if reranked_documents:
            reranked_context = "\n\n".join(
                document.page_content for document in reranked_documents
            )

            if reranked_context:
                context_parts.append("Retrieved Evidence:\n" + reranked_context)

        # RAPTOR evidence

        if raptor_results:
            leaf_results = raptor_results.get(
                "leaf",
                [],
            )

            cluster_results = raptor_results.get(
                "clusters",
                [],
            )

            raptor_items = leaf_results + cluster_results

            if raptor_items:
                raptor_context = "\n\n".join(str(item) for item in raptor_items)

                context_parts.append("RAPTOR Evidence:\n" + raptor_context)

        # Self-RAG evidence

        if self_rag_answer:
            context_parts.append("Self-RAG Evidence:\n" + self_rag_answer)

        # Long-context evidence

        if long_context_answer:
            context_parts.append("Long-Context Evidence:\n" + long_context_answer)

        return "\n\n".join(context_parts)

    # Generation

    def generate_answer(
        self,
        query: str,
        context: str,
    ) -> str:
        """
        Generate a grounded answer from evidence.
        """

        if not context.strip():
            return "The information is not available " "in the retrieved evidence."

        return self.generation_chain.invoke(
            {
                "context": context,
                "question": query,
            }
        )

    # Direct Ollama

    def direct_answer(
        self,
        query: str,
    ) -> str:
        """
        Generate a direct answer using local Ollama.
        """

        response = self.llm.invoke(query)

        return response.content

    # Memory

    def store_memory(
        self,
        query: str,
        answer: str,
    ) -> None:
        """
        Store one conversation turn.
        """

        self.conversation_memory.add_message(HumanMessage(content=query))

        self.conversation_memory.add_message(AIMessage(content=answer))

    # Evaluation

    def evaluate_response(
        self,
        query: str,
        answer: str,
        context: str = "",
        mode: str = "direct",
    ) -> Dict[str, Any]:
        """
        Evaluate the generated answer.

        Direct mode:
            Answer relevance only.

        RAG mode:
            Context relevance
            Faithfulness
            Answer relevance
        """

        if self.evaluator is None:
            return {}

        results: Dict[str, Any] = {}

        if mode == "direct":

            results["answer_relevance"] = self.evaluator.evaluate_answer_relevance(
                question=query,
                answer=answer,
            )

            return results

        results["context_relevance"] = self.evaluator.evaluate_context_relevance(
            question=query,
            context=context,
        )

        results["faithfulness"] = self.evaluator.evaluate_faithfulness(
            context=context,
            answer=answer,
        )

        results["answer_relevance"] = self.evaluator.evaluate_answer_relevance(
            question=query,
            answer=answer,
        )

        return results

    # Main pipeline

    def run(
        self,
        query: str,
        evaluate: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete integrated RAG pipeline.

        Returns:
            {
                "query": ...,
                "decision": ...,
                "answer": ...,
                "mode": ...,
                "context": ...,
                "retrieved_documents": ...,
                "reranked_documents": ...,
                "raptor_results": ...,
                "self_rag_answer": ...,
                "long_context_answer": ...,
                "evaluation": ...,
            }
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query = query.strip()

        # Step 1: Ollama answerability

        decision = self.check_answerability(query)

        # Step 2: Direct local answer

        if decision == "ANSWERABLE":

            answer = self.direct_answer(query)

            self.store_memory(
                query=query,
                answer=answer,
            )

            should_evaluate = self.enable_evaluation if evaluate is None else evaluate

            evaluation = {}

            if should_evaluate:
                evaluation = self.evaluate_response(
                    query=query,
                    answer=answer,
                    mode="direct",
                )

            return {
                "query": query,
                "decision": decision,
                "mode": "direct",
                "answer": answer,
                "context": "",
                "retrieved_documents": [],
                "reranked_documents": [],
                "raptor_results": {},
                "self_rag_answer": "",
                "long_context_answer": "",
                "evaluation": evaluation,
            }

        # Step 3: External retrieval

        documents = self.retrieve_external_documents(query)

        if not documents:
            answer = (
                "I could not retrieve enough "
                "external information to answer "
                "the question."
            )

            self.store_memory(
                query=query,
                answer=answer,
            )

            return {
                "query": query,
                "decision": decision,
                "mode": "external",
                "answer": answer,
                "context": "",
                "retrieved_documents": [],
                "reranked_documents": [],
                "raptor_results": {},
                "self_rag_answer": "",
                "long_context_answer": "",
                "evaluation": {},
            }

        # Step 4: Build vector index

        self.build_index(documents)

        # Step 5: Multi-Query retrieval

        retrieved_documents = self.retrieve_documents(query)

        # Step 6: Reranking

        reranked_documents = self.rerank_documents(
            query=query,
            documents=retrieved_documents,
        )

        # Step 7: RAPTOR

        raptor_results = self.run_raptor(
            query=query,
            documents=documents,
        )

        # Step 8: Self-RAG

        self_rag_answer = ""

        if self.enable_self_rag:
            self_rag_answer = self.run_self_rag(query)

        # Step 9: Long context

        long_context_answer = ""

        if self.enable_long_context:
            long_context_answer = self.run_long_context(
                query=query,
                documents=reranked_documents,
            )

        # Step 10: Final context

        context = self.build_context(
            reranked_documents=reranked_documents,
            raptor_results=raptor_results,
            self_rag_answer=self_rag_answer,
            long_context_answer=long_context_answer,
        )

        # Step 11: Final Ollama generation

        answer = self.generate_answer(
            query=query,
            context=context,
        )

        # Step 12: Memory

        self.store_memory(
            query=query,
            answer=answer,
        )

        # Step 13: Evaluation

        should_evaluate = self.enable_evaluation if evaluate is None else evaluate

        evaluation = {}

        if should_evaluate:
            evaluation = self.evaluate_response(
                query=query,
                answer=answer,
                context=context,
                mode="rag",
            )

        return {
            "query": query,
            "decision": decision,
            "mode": "external_rag",
            "answer": answer,
            "context": context,
            "retrieved_documents": retrieved_documents,
            "reranked_documents": reranked_documents,
            "raptor_results": raptor_results,
            "self_rag_answer": self_rag_answer,
            "long_context_answer": long_context_answer,
            "evaluation": evaluation,
        }


# Optional command-line test


if __name__ == "__main__":

    rag = RAGIntegration()

    user_query = input("Ask a question: ")

    result = rag.run(
        user_query,
        evaluate=True,
    )

    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)
    print(result["decision"])

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result["answer"])

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    for metric, value in result["evaluation"].items():
        print(f"{metric}: {value}")
