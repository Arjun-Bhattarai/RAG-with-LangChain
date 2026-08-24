from typing import Any, Dict, List, Optional
import numpy as np
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from sklearn.cluster import KMeans

CHUNK_SUMMARY_PROMPT_TEMPLATE = """Summarize the following document chunk.
Keep the important information and concepts.

Document:
{doc}
"""

CHUNK_SUMMARY_PROMPT = ChatPromptTemplate.from_template(CHUNK_SUMMARY_PROMPT_TEMPLATE)

CLUSTER_SUMMARY_PROMPT_TEMPLATE = """Combine the following related summaries into
one higher-level summary.

Preserve the important concepts and relationships.

Summaries:
{summaries}
"""

CLUSTER_SUMMARY_PROMPT = ChatPromptTemplate.from_template(CLUSTER_SUMMARY_PROMPT_TEMPLATE)

RAPTOR_ANSWER_PROMPT_TEMPLATE = """Answer the question using the following RAPTOR context.

Context:
{context}

Question:
{question}

Answer clearly and accurately.
"""

RAPTOR_ANSWER_PROMPT = ChatPromptTemplate.from_template(RAPTOR_ANSWER_PROMPT_TEMPLATE)


class RaptorIndexer:
    """Implements RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        embeddings: Optional[Any] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        n_clusters: int = 3,
        random_state: int = 42,
    ):
        self.llm = llm or ChatOllama(model="llama3:latest", temperature=0)
        self.embeddings = embeddings or HuggingFaceEmbeddings(model_name=embedding_model)
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.leaf_summaries: List[str] = []
        self.leaf_embeddings: List[List[float]] = []
        self.cluster_summaries: List[str] = []
        self.cluster_embeddings: List[List[float]] = []

    def build_tree(
        self,
        documents: List[Document],
        max_concurrency: int = 1,
    ):
        """Build the RAPTOR tree with leaf chunk summaries and higher-level cluster summaries."""
        # 1. Generate chunk summaries
        summary_chain = (
            {"doc": lambda x: x.page_content if hasattr(x, "page_content") else str(x)}
            | CHUNK_SUMMARY_PROMPT
            | self.llm
            | StrOutputParser()
        )
        self.leaf_summaries = summary_chain.batch(
            documents, config={"max_concurrency": max_concurrency}
        )

        if not self.leaf_summaries:
            return

        # 2. Embed leaf summaries
        self.leaf_embeddings = self.embeddings.embed_documents(self.leaf_summaries)
        embedding_matrix = np.array(self.leaf_embeddings)

        # 3. Cluster leaf summaries with KMeans
        actual_clusters = min(self.n_clusters, len(self.leaf_summaries))
        kmeans = KMeans(
            n_clusters=actual_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        cluster_labels = kmeans.fit_predict(embedding_matrix)

        clusters: Dict[int, List[str]] = {}
        for i, label in enumerate(cluster_labels):
            clusters.setdefault(int(label), []).append(self.leaf_summaries[i])

        # 4. Generate cluster higher-level summaries
        cluster_summary_chain = CLUSTER_SUMMARY_PROMPT | self.llm | StrOutputParser()
        self.cluster_summaries = []
        for _, cluster_docs in clusters.items():
            combined_text = "\n\n".join(cluster_docs)
            higher_summary = cluster_summary_chain.invoke({"summaries": combined_text})
            self.cluster_summaries.append(higher_summary)

        # 5. Embed higher-level cluster summaries
        self.cluster_embeddings = self.embeddings.embed_documents(self.cluster_summaries)

    def retrieve(self, query: str, k: int = 3) -> Dict[str, List[str]]:
        """Retrieve top-k leaf and higher-level cluster summaries for the query."""
        query_embedding = np.array(self.embeddings.embed_query(query))

        leaf_results: List[str] = []
        cluster_results: List[str] = []

        if self.leaf_embeddings:
            leaf_scores = np.dot(np.array(self.leaf_embeddings), query_embedding)
            top_leaf_k = min(k, len(self.leaf_summaries))
            leaf_indices = np.argsort(leaf_scores)[-top_leaf_k:][::-1]
            leaf_results = [self.leaf_summaries[i] for i in leaf_indices]

        if self.cluster_embeddings:
            cluster_scores = np.dot(np.array(self.cluster_embeddings), query_embedding)
            top_cluster_k = min(k, len(self.cluster_summaries))
            cluster_indices = np.argsort(cluster_scores)[-top_cluster_k:][::-1]
            cluster_results = [self.cluster_summaries[i] for i in cluster_indices]

        return {
            "leaf": leaf_results,
            "clusters": cluster_results,
        }

    def answer(self, query: str, k: int = 3) -> str:
        """Retrieve leaf and cluster contexts and synthesize final answer."""
        results = self.retrieve(query, k=k)
        context = "\n\n".join(results["leaf"] + results["clusters"])

        answer_chain = RAPTOR_ANSWER_PROMPT | self.llm | StrOutputParser()
        return answer_chain.invoke({"context": context, "question": query})
