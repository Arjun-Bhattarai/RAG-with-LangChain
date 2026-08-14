> **Prerequisites:** This project builds on my **[LLMs](https://github.com/Arjun-Bhattarai/LLMs)** repository. Familiarity with Transformers, GPT models, and LLM fundamentals is recommended before diving in.

# RAG From Scratch

A hands-on, notebook-driven implementation of **Retrieval-Augmented Generation (RAG)** built from first principles using **LangChain**, **ChromaDB**, and **Ollama**. This repository systematically explores the full RAG landscape — from a bare-bones pipeline to cutting-edge retrieval strategies — through clear, well-documented Jupyter notebooks.

---

## Why This Repository?

Most RAG tutorials rely on high-level wrappers that hide what is actually happening. This repo is different — every concept is implemented step-by-step so you understand the *mechanics* behind each technique, not just how to call an API.

---

## Repository Structure

```
RAG-from-scratch/
│
├── pipeline/                   # Core RAG Pipeline (Notebooks 01–04)
│   ├── 01_basic_rag.ipynb
│   ├── 02_indexing_rag.ipynb
│   ├── 03_retrieval_rag.ipynb
│   └── 04_generation_rag.ipynb
│
├── query_translation/          # Advanced Query Techniques (Notebooks 05–09)
│   ├── 05_multi_query.ipynb
│   ├── 06_Rag_fusion.ipynb
│   ├── 07_decomposition.ipynb
│   ├── 08_step_back.ipynb
│   └── 09_HyDE.ipynb
│
├── routing/                    # Query Routing Strategies (Notebooks 10–11)
│   ├── 10_logical_routing.ipynb
│   └── 11_semantic_routing.ipynb
│
├── query_structuring/          # Structured Query Generation (Notebook 12)
│   └── 12_query_structuring.ipynb
│
├── advanced_indexing/          # Advanced Indexing Methods (Notebooks 13–15)
│   ├── 13_multi_representational_indexing.ipynb
│   ├── 14_Raptor.ipynb
│   └── 15_ColBERT.ipynb
│
├── requirements.txt
└── README.md
```

---

## What Is Implemented

### Core RAG Pipeline

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `basic_rag` | End-to-end RAG — load, chunk, embed, retrieve, generate |
| 02 | `indexing_rag` | Document chunking strategies and vector indexing with ChromaDB |
| 03 | `retrieval_rag` | Semantic retrieval, similarity search, and retriever configuration |
| 04 | `generation_rag` | Context-aware answer generation with Ollama (Llama 3) |

### Query Translation

Techniques to reformulate or expand user queries before retrieval, improving recall and relevance.

| # | Notebook | Description |
|---|----------|-------------|
| 05 | `multi_query` | Generate multiple query variants and merge results |
| 06 | `rag_fusion` | Reciprocal Rank Fusion over multiple retrieved result sets |
| 07 | `decomposition` | Break complex questions into sub-questions (sequential and parallel) |
| 08 | `step_back` | Abstract queries to a higher level before retrieval |
| 09 | `HyDE` | Hypothetical Document Embeddings — embed generated answers, not queries |

### Routing

Intelligently direct queries to the most relevant data source or retrieval strategy.

| # | Notebook | Description |
|---|----------|-------------|
| 10 | `logical_routing` | Rule-based routing using LLM-generated structured output |
| 11 | `semantic_routing` | Embedding-based routing to the best-matching prompt or datasource |

### Query Structuring

| # | Notebook | Description |
|---|----------|-------------|
| 12 | `query_structuring` | Convert natural language queries into structured filters for metadata-aware retrieval |

### Advanced Indexing

| # | Notebook | Description |
|---|----------|-------------|
| 13 | `multi_representational_indexing` | Index multiple document representations for richer retrieval |
| 14 | `RAPTOR` | Recursive abstractive processing — hierarchical summarization for long-context retrieval |
| 15 | `ColBERT` | Token-level late interaction retrieval for fine-grained semantic matching |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core language |
| **LangChain** | RAG orchestration and chains |
| **Ollama + Llama 3** | Local LLM inference |
| **ChromaDB** | Vector store for embeddings |
| **HuggingFace (BAAI/bge-small-en-v1.5)** | Open-source text embeddings |
| **ColBERT (RAGatouille)** | Late-interaction neural retrieval |
| **BeautifulSoup4** | Web document loading |
| **tiktoken** | Token counting and chunking |

'
> **Tip:** Each notebook is self-contained — you can run any section independently once you have completed the core pipeline (notebooks 01-04).

---

## Progress

| Module | Status |
|--------|--------|
| Basic RAG Pipeline | Complete |
| Indexing | Complete |
| Retrieval | Complete |
| Generation | Complete |
| Multi-Query Retrieval | Complete |
| RAG-Fusion | Complete |
| Query Decomposition | Complete |
| Step-Back Prompting | Complete |
| HyDE | Complete |
| Logical Routing | Complete |
| Semantic Routing | Complete |
| Query Structuring | Complete |
| Multi-Representational Indexing | Complete |
| RAPTOR | Complete |
| ColBERT | Complete |

More advanced RAG techniques (e.g., Corrective RAG, Self-RAG, Adaptive RAG) may be added as the project grows.

---

## MCP Integration

This repository is accessible to MCP-compatible AI assistants (Cursor, Claude Desktop, Windsurf, Cline, VS Code) via [GitMCP](https://gitmcp.io/Arjun-Bhattarai/RAG-from-scratch).

Add the following to your MCP client config:

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

## Goal

The goal of this repository is to learn, implement, and document modern Retrieval-Augmented Generation techniques from scratch — building a strong conceptual and practical foundation for production-ready RAG systems.

---

## Related Work

- [LLMs from Scratch](https://github.com/Arjun-Bhattarai/LLMs) — the prerequisite repository covering Transformer fundamentals, GPT training, and LLM internals.
