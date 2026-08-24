# 🧠 RAG From Scratch

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-FC521F?style=flat-square)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Inference-Ollama_(Llama_3)-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.ai/)
[![ColBERT](https://img.shields.io/badge/Late_Interaction-ColBERT-red?style=flat-square)](https://github.com/stanford-futuredata/ColBERT)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

> A modular, hands-on implementation of **Retrieval-Augmented Generation (RAG)** systems built from the ground up. From basic vector search to agentic self-reflection, multi-turn memory, and quantitative evaluation.


> **Prerequisites & Foundation:** This project builds upon foundational language modeling principles covered in **[LLMs from Scratch](https://github.com/Arjun-Bhattarai/LLMs)** (covering Transformer internals, attention mechanisms, and GPT architectures).

---
## 📖 Step-by-Step Concepts Explained

---

### 1. 🚀 Core Baseline Pipeline

The foundation of every RAG system. It transforms raw text into indexed vectors and answers queries using grounded context.

* **Document Ingestion & Chunking (`01_basic_rag`, `02_indexing_rag`)**: Loads unstructured documents and splits them into smaller, token-aware passages so relevant information can fit cleanly inside LLM prompts.
* **Vector Indexing (`03_retrieval_rag`)**: Converts text chunks into dense numerical vectors (embeddings) stored in ChromaDB and retrieves top-$k$ nearest matches via Cosine Similarity or MMR (Maximal Marginal Relevance).
* **Grounded Generation (`04_generation_rag`)**: Passes the retrieved passages alongside the user query into a local LLM (`llama3`), enforcing strict prompt constraints to prevent hallucinations.

---

### 2. 🔄 Query Translation & Expansion

User questions are often vague or phrased differently than stored documents. Query translation optimizes the question *before* searching.

* **Multi-Query (`05_multi_query`)**: LLM generates 3–5 variations of the user prompt to search across multiple semantic angles at once.
* **RAG-Fusion (`06_Rag_fusion`)**: Runs multiple queries and merges all results using **Reciprocal Rank Fusion (RRF)** to elevate consistently high-ranking passages.
* **Query Decomposition (`07_decomposition`)**: Breaks complex, multi-hop questions into simpler sub-questions answered step-by-step.
* **Step-Back Prompting (`08_step_back`)**: Asks the LLM for high-level foundational concepts first, retrieving broad principles alongside specific facts.
* **HyDE (`09_HyDE`)**: Generates a hypothetical answer passage first, vectors it, and uses it to find real documents with similar semantic patterns.

---

### 3. 🔀 Query Routing

Not all queries should go to the same database or pipeline. Routing sends the query to the best destination.

* **Logical Routing (`10_logical_routing`)**: Uses LLM structured output / JSON schema to classify query intent and select target datasources (e.g., Python docs vs. SQL DB).
* **Semantic Routing (`11_semantic_routing`)**: Calculates cosine similarity against predefined topic embeddings for ultra-fast, zero-LLM classification.

---

### 4. 🏷️ Query Structuring & Metadata Filtering

* **Query Structuring (`12_query_structuring`)**: Separates natural text from filters (e.g., *"Show tutorials on Agents after June 2023"* becomes a keyword search for *"Agents"* with a metadata filter `{'year': {'$gte': 2023}}`).

---

### 5. 📚 Advanced Indexing Paradigms

Fixes the trade-off between small chunks (good for search) and large chunks (good for context).

* **Multi-Representational Indexing (`13_multi_representational_indexing`)**: Embeds concise summaries or small sub-chunks in Chroma, but returns the full parent document to the LLM.
* **RAPTOR (`14_Raptor`)**: Recursively clusters and summarizes document chunks into a tree hierarchy. Allows answering both high-level thematic questions and detailed queries.
* **ColBERT / Late Interaction (`15_ColBERT`)**: Compares queries and documents token-by-token (MaxSim operator) instead of compressing an entire document into a single vector.

---

### 6. 🎯 Reranking

* **Two-Stage Retrieval (`16_reranking`)**: First-stage retrieval pulls a broad pool (e.g., 20 candidates). A cross-encoder or RRF re-scores them to pick the top 3–5 highest-quality chunks, removing irrelevant noise before generation.

---

### 7. 🧠 Agentic & Corrective RAG

Replaces static, one-way pipelines with self-checking loops that inspect relevance and recover from bad retrievals.

* **Self-RAG (`Self_RAG`)**: Evaluates whether retrieval is needed, checks if retrieved docs are relevant, generates the answer, and verifies that the output is grounded (not hallucinated).
* **Corrective RAG (`CRAG`)**: Automatically triggers a fallback search query or broadens search depth if initial retrieved documents fail relevance checks.
* **Long-Context Compression (`Long_Context`)**: Extracts only query-relevant facts from large documents before sending them to the final model.

---

### 8. 💾 Conversational Memory

Maintains multi-turn context across conversations without overflowing the context window.

* **Conversation Buffer (`17_conversation_memory`)**: Stores the full sequential history of user and assistant turns.
* **Summarization Memory (`18_summarization_memory`)**: Keeps recent messages verbatim while maintaining a rolling summary of older turns.
* **Vector Memory (`19_vector_memory`)**: Stores past conversation turns in ChromaDB and semantically retrieves relevant past discussions when needed.

---

### 9. 📊 Evaluation & The RAG Triad

Quantifies system performance to measure retrieval accuracy and answer quality:

| Metric | Focus | What It Validates |
| :--- | :--- | :--- |
| **Context Relevance** | Retrieval | Is retrieved context concise and free of irrelevant noise? |
| **Faithfulness** | Generation | Are all claims in the answer supported by the retrieved context? |
| **Answer Relevance** | Generation | Does the answer directly and accurately address the user's question? |
| **Precision & Recall @ K** | Search | What fraction of retrieved documents are relevant, and were all relevant docs found? |

---

## 🛠️ Tech Stack

* **Orchestration**: LangChain Core / Community
* **Vector Database**: ChromaDB
* **LLMs & Embeddings**: Ollama (Llama 3, Nomic Embed Text), BGE Embeddings
* **Late Interaction Search**: ColBERT / RAGatouille
* **Data Parsing**: BeautifulSoup4, Tiktoken, Pydantic V2

---

## 🤝 Model Context Protocol (MCP)

This repository is ready for AI coding assistants via GitMCP:

```json
{
  "servers": {
    "RAG-from-scratch Docs": {
      "type": "sse",
      "url": "https://gitmcp.io/Arjun-Bhattarai/RAG-from-scratch"
    }
  }
}
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
