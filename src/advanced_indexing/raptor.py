from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from sklearn.cluster import KMeans


# PROMPTS

CHUNK_SUMMARY_PROMPT_TEMPLATE = """
Summarize the following document chunk.

Keep the important information, facts, concepts, and relationships.
Do not add information that is not present in the document.

Document:
{doc}
"""

CHUNK_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    CHUNK_SUMMARY_PROMPT_TEMPLATE
)


CLUSTER_SUMMARY_PROMPT_TEMPLATE = """
Combine the following related summaries into one higher-level summary.

Preserve the important concepts, facts, and relationships.
Do not introduce information that is not present in the summaries.

Summaries:
{summaries}
"""

CLUSTER_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    CLUSTER_SUMMARY_PROMPT_TEMPLATE
)


RAPTOR_ANSWER_PROMPT_TEMPLATE = """
Answer the question using the following RAPTOR context.

Context:
{context}

Question:
{question}

Answer clearly and accurately.
"""

RAPTOR_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    RAPTOR_ANSWER_PROMPT_TEMPLATE
)


# RAPTOR INDEXER

class RaptorIndexer:

    def __init__(
        self,
        llm: Optional[Any] = None,
        embeddings: Optional[Any] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        n_clusters: int = 3,
        random_state: int = 42,
    ):
        # MODELS

        self.llm = llm or ChatOllama(
            model="llama3:latest",
            temperature=0,
        )

        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=embedding_model
        )

        self.n_clusters = n_clusters
        self.random_state = random_state

        # STORED RAPTOR DATA

        self.leaf_summaries: List[str] = []
        self.leaf_embeddings: List[List[float]] = []

        self.cluster_summaries: List[str] = []
        self.cluster_embeddings: List[List[float]] = []

        # INDEX STATUS

        self._ready = False

    # BUILD RAPTOR TREE

    def build_tree(
        self,
        documents: List[Document],
        max_concurrency: int = 4,
    ) -> None:
        """
        Build the RAPTOR tree.

        max_concurrency controls how many Ollama summarization
        requests can run concurrently.

        Recommended values for local Ollama:
            2 - 4

        Do not use extremely high concurrency because local
        Ollama models may become slower under heavy load.
        """

        if not documents:
            self._ready = False
            return

        # Reset previous index.

        self.leaf_summaries = []
        self.leaf_embeddings = []

        self.cluster_summaries = []
        self.cluster_embeddings = []

        self._ready = False

        print("\n========================================")
        print("Building RAPTOR index")
        print("========================================")

        print(
            f"Documents: {len(documents)}"
        )

        print(
            f"Max concurrency: {max_concurrency}"
        )

        # 1. LEAF CHUNK SUMMARIZATION

        print("\n[1/5] Generating leaf summaries...")

        summary_chain = (
            {
                "doc": lambda x: (
                    x.page_content
                    if hasattr(x, "page_content")
                    else str(x)
                )
            }
            | CHUNK_SUMMARY_PROMPT
            | self.llm
            | StrOutputParser()
        )

        try:
            self.leaf_summaries = summary_chain.batch(
                documents,
                config={
                    "max_concurrency": max_concurrency
                },
            )

        except Exception as exc:
            print(
                f"RAPTOR leaf summarization failed: {exc}"
            )
            self._ready = False
            return

        # Remove empty summaries.

        self.leaf_summaries = [
            summary.strip()
            for summary in self.leaf_summaries
            if summary and summary.strip()
        ]

        print(
            f"Leaf summaries created: "
            f"{len(self.leaf_summaries)}"
        )

        if not self.leaf_summaries:
            print("No leaf summaries generated.")
            return

        # 2. EMBED LEAF SUMMARIES

        print("\n[2/5] Embedding leaf summaries...")

        try:
            self.leaf_embeddings = (
                self.embeddings.embed_documents(
                    self.leaf_summaries
                )
            )

        except Exception as exc:
            print(
                f"RAPTOR leaf embedding failed: {exc}"
            )
            self._ready = False
            return

        print(
            f"Leaf embeddings created: "
            f"{len(self.leaf_embeddings)}"
        )

        # Convert to NumPy.

        embedding_matrix = np.asarray(
            self.leaf_embeddings,
            dtype=np.float32,
        )

        # Normalize embeddings.

        embedding_matrix = self._normalize_embeddings(
            embedding_matrix
        )

        # 3. KMEANS CLUSTERING

        print("\n[3/5] Clustering leaf summaries...")

        actual_clusters = min(
            self.n_clusters,
            len(self.leaf_summaries),
        )

        if actual_clusters <= 1:
            cluster_labels = np.zeros(
                len(self.leaf_summaries),
                dtype=int,
            )
        else:
            kmeans = KMeans(
                n_clusters=actual_clusters,
                random_state=self.random_state,
                n_init="auto",
            )

            cluster_labels = kmeans.fit_predict(
                embedding_matrix
            )

        clusters: Dict[int, List[str]] = {}

        for i, label in enumerate(cluster_labels):

            clusters.setdefault(
                int(label),
                [],
            ).append(
                self.leaf_summaries[i]
            )

        print(
            f"Clusters created: {len(clusters)}"
        )

        # 4. CLUSTER SUMMARIZATION

        print(
            "\n[4/5] Generating cluster summaries..."
        )

        cluster_summary_chain = (
            CLUSTER_SUMMARY_PROMPT
            | self.llm
            | StrOutputParser()
        )

        cluster_inputs = []

        for cluster_docs in clusters.values():

            combined_text = "\n\n".join(
                cluster_docs
            )

            cluster_inputs.append(
                {
                    "summaries": combined_text
                }
            )

        try:
            self.cluster_summaries = (
                cluster_summary_chain.batch(
                    cluster_inputs,
                    config={
                        "max_concurrency": max_concurrency
                    },
                )
            )

        except Exception as exc:
            print(
                f"RAPTOR cluster summarization failed: {exc}"
            )

            self.cluster_summaries = []

        self.cluster_summaries = [
            summary.strip()
            for summary in self.cluster_summaries
            if summary and summary.strip()
        ]

        print(
            f"Cluster summaries created: "
            f"{len(self.cluster_summaries)}"
        )

        # 5. EMBED CLUSTER SUMMARIES

        print(
            "\n[5/5] Embedding cluster summaries..."
        )

        if self.cluster_summaries:

            try:

                self.cluster_embeddings = (
                    self.embeddings.embed_documents(
                        self.cluster_summaries
                    )
                )

                print(
                    f"Cluster embeddings created: "
                    f"{len(self.cluster_embeddings)}"
                )

            except Exception as exc:

                print(
                    f"RAPTOR cluster embedding failed: {exc}"
                )

                self.cluster_embeddings = []

        # INDEX READY

        self._ready = bool(
            self.leaf_summaries
            and self.leaf_embeddings
        )

        print("\n========================================")

        if self._ready:
            print("RAPTOR index built successfully.")
        else:
            print("RAPTOR index build failed.")

        print("========================================")

    # NORMALIZE EMBEDDINGS

    @staticmethod
    def _normalize_embeddings(
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize embeddings for cosine similarity.
        """

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        norms = np.maximum(
            norms,
            1e-12,
        )

        return embeddings / norms

    # READY CHECK

    def is_ready(self) -> bool:
        """
        Return True when a usable RAPTOR index exists.
        """

        return self._ready

    # RETRIEVAL

    def retrieve(
        self,
        query: str,
        k: int = 3,
    ) -> Dict[str, List[str]]:
        """
        Retrieve top-k leaf and cluster summaries.
        """

        if not query or not query.strip():
            return {
                "leaf": [],
                "clusters": [],
            }

        query = query.strip()

        query_embedding = np.asarray(
            self.embeddings.embed_query(query),
            dtype=np.float32,
        )

        # Normalize query.

        query_norm = np.linalg.norm(
            query_embedding
        )

        if query_norm > 0:
            query_embedding = (
                query_embedding / query_norm
            )

        leaf_results: List[str] = []
        cluster_results: List[str] = []

        # LEAF RETRIEVAL

        if self.leaf_embeddings:

            leaf_matrix = np.asarray(
                self.leaf_embeddings,
                dtype=np.float32,
            )

            leaf_matrix = self._normalize_embeddings(
                leaf_matrix
            )

            leaf_scores = (
                leaf_matrix @ query_embedding
            )

            top_leaf_k = min(
                k,
                len(self.leaf_summaries),
            )

            leaf_indices = np.argsort(
                leaf_scores
            )[-top_leaf_k:][::-1]

            leaf_results = [
                self.leaf_summaries[i]
                for i in leaf_indices
            ]

        # CLUSTER RETRIEVAL

        if self.cluster_embeddings:

            cluster_matrix = np.asarray(
                self.cluster_embeddings,
                dtype=np.float32,
            )

            cluster_matrix = (
                self._normalize_embeddings(
                    cluster_matrix
                )
            )

            cluster_scores = (
                cluster_matrix @ query_embedding
            )

            top_cluster_k = min(
                k,
                len(self.cluster_summaries),
            )

            cluster_indices = np.argsort(
                cluster_scores
            )[-top_cluster_k:][::-1]

            cluster_results = [
                self.cluster_summaries[i]
                for i in cluster_indices
            ]

        return {
            "leaf": leaf_results,
            "clusters": cluster_results,
        }

    # ANSWER

    def answer(
        self,
        query: str,
        k: int = 3,
    ) -> str:
        """
        Retrieve RAPTOR context and generate an answer.
        """

        results = self.retrieve(
            query,
            k=k,
        )

        context = "\n\n".join(
            results["leaf"]
            + results["clusters"]
        )

        if not context.strip():

            return (
                "No relevant RAPTOR context "
                "was found."
            )

        answer_chain = (
            RAPTOR_ANSWER_PROMPT
            | self.llm
            | StrOutputParser()
        )

        return answer_chain.invoke(
            {
                "context": context,
                "question": query,
            }
        )