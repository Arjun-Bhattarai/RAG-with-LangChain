# 🧠 RAG From Scratch
## **Arjunx — Advanced Modular Retrieval-Augmented Generation System**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-FC521F?style=flat-square)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Inference-Ollama_(Llama_3)-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.ai/)
[![ColBERT](https://img.shields.io/badge/Late_Interaction-ColBERT-red?style=flat-square)](https://github.com/stanford-futuredata/ColBERT)

> **Prerequisites & Foundation:** This project builds upon foundational language modeling principles covered in **[LLMs from Scratch](https://github.com/Arjun-Bhattarai/LLMs)**, covering Transformer internals, attention mechanisms, and GPT architectures.

**Arjunx** is an advanced AI assistant built on top of a modular Retrieval-Augmented Generation (RAG) system developed from scratch.

The project explores, implements, evaluates, and integrates modern RAG techniques into a single adaptive pipeline.

Rather than relying on a simple retrieve → generate workflow, Arjunx combines query processing, intelligent source routing, advanced retrieval, deduplication, reranking, freshness scoring, CRAG, Self-RAG, long-context processing, failure handling, memory, evaluation, and local LLM inference through Ollama.

> **Goal:** Understand and engineer modern RAG systems from the ground up instead of treating RAG as a black box.

---

# ✨ Features

* 🔎 **Query Processing** — Multi-Query, RAG Fusion, Query Decomposition, Step-Back Prompting, HyDE
* 🧭 **Intelligent Routing** — Logical, Semantic, Local, Web, and Hybrid routing
* 📚 **Advanced Retrieval** — Chroma, RAPTOR, ColBERT, Hybrid Retrieval
* ♻️ **Deduplication** — Removes duplicate and overlapping evidence
* 🎯 **Reranking** — Cross-Encoder based relevance reranking
* 🕐 **Freshness Scoring** — Prioritizes recent information when required
* 🔄 **CRAG** — Corrective retrieval when evidence is insufficient
* 🧠 **Self-RAG** — Evaluates retrieved evidence and generated responses
* 📖 **Long Context** — Handles larger collections of relevant evidence
* 🛡️ **Failure Handling** — Supports `NOT_ANSWERABLE` when evidence is insufficient
* 🤖 **Local LLM** — Generation through Ollama
* 💾 **Conversation Memory** — Maintains conversational context
* 🧪 **Evaluation & Testing** — Architecture, end-to-end, and diagnostic validation
* 🌐 **Web Retrieval** — Tavily and Wikipedia integration
* ⚡ **FastAPI Backend** — Exposes Arjunx through a REST API
* 🖥️ **Web Interface** — Simple frontend for interacting with Arjunx

---

# 🏗️ Arjunx RAG Architecture

```text
                         USER QUERY
                              │
                              ▼
                    ┌──────────────────┐
                    │  QUERY PROCESSOR │
                    │    MULTI-QUERY   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   SOURCE ROUTER  │
                    │ LOCAL / WEB /    │
                    │ HYBRID           │
                    └────────┬─────────┘
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
             LOCAL          WEB        HYBRID
                │            │            │
             Chroma       Tavily       Chroma
             RAPTOR      Wikipedia      RAPTOR
             ColBERT                   Tavily
                                      Wikipedia
                └────────────┼────────────┘
                             ▼
                    RETRIEVED DOCUMENTS
                             │
                             ▼
                       DEDUPLICATION
                             │
                             ▼
                    CROSS-ENCODER
                       RERANKING
                             │
                             ▼
                      FRESHNESS SCORE
                             │
                             ▼
                       TOP EVIDENCE
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
                  CRAG              SELF-RAG
                    │                 │
                    └────────┬────────┘
                             │
                    EVIDENCE EVALUATION
                             │
                  ┌──────────┴──────────┐
                  │                     │
             SUFFICIENT           INSUFFICIENT
                  │                     │
                  │              ADDITIONAL RETRIEVAL
                  │                     │
                  │              RE-EVALUATION
                  │                     │
                  └──────────┬──────────┘
                             ▼
                     CONTEXT BUILDER
                             │
                             ▼
                       LONG CONTEXT
                       WHEN NEEDED
                             │
                             ▼
                    OLLAMA LOCAL LLM
                             │
                             ▼
                       FINAL ANSWER
                             │
                             ▼
                          ARJUNX
```

---

# 🔬 RAG Techniques

The project follows a progression from fundamental RAG to advanced retrieval and reasoning techniques.

| Category                | Techniques                                              |
| ----------------------- | ------------------------------------------------------- |
| **Fundamentals**        | Basic RAG, Indexing, Retrieval, Generation              |
| **Query Processing**    | Multi-Query, RAG Fusion, Decomposition, Step-Back, HyDE |
| **Query Understanding** | Query Structuring                                       |
| **Routing**             | Logical, Semantic, Source Routing                       |
| **Indexing**            | Multi-Representational Indexing, RAPTOR, ColBERT        |
| **Retrieval**           | Chroma, Web, Wikipedia, Hybrid Retrieval                |
| **Retrieval Quality**   | Deduplication, Cross-Encoder Reranking, Freshness       |
| **Advanced RAG**        | CRAG, Self-RAG, Long Context                            |
| **Memory**              | Conversation, Summarization, Vector Memory              |
| **Evaluation**          | Retrieval Evaluation, Pipeline Testing, Diagnostics     |
| **Application Layer**   | FastAPI API, Web Interface, Arjunx Assistant            |

---

# ⚙️ How It Works

Arjunx processes a user query through several stages.

### **1. Query Processing**

The original query is transformed into multiple query representations to improve retrieval coverage.

### **2. Source Routing**

The system determines whether the query should use **LOCAL**, **WEB**, or **HYBRID** retrieval.

### **3. Retrieval**

Relevant evidence is collected from local vector stores and/or external sources such as Tavily and Wikipedia.

### **4. Deduplication**

Duplicate and overlapping documents are removed.

### **5. Reranking**

A cross-encoder evaluates candidate documents and improves their relevance ordering.

### **6. Freshness**

Recent information can receive additional importance for time-sensitive queries.

### **7. Evidence Evaluation**

CRAG and Self-RAG evaluate whether the retrieved evidence is useful and sufficient.

### **8. Corrective Retrieval**

If the evidence is insufficient, additional retrieval can be triggered.

### **9. Context Construction**

The strongest evidence is assembled into the final context, using long-context processing when necessary.

### **10. Generation**

The final context is passed to a locally running LLM through Ollama.

### **11. Failure Handling**

If reliable evidence cannot be established, Arjunx can return `NOT_ANSWERABLE` instead of producing an unsupported answer.

### **12. Response**

The final grounded response is returned to the user through the Arjunx interface.

---

# 🧪 Validation

The integrated pipeline has been validated across its major components.

| Component            | Status |
| -------------------- | :----: |
| Multi-Query          |    ✅   |
| Local Retrieval      |    ✅   |
| Web Retrieval        |    ✅   |
| Hybrid Retrieval     |    ✅   |
| Reranking            |    ✅   |
| Freshness Scoring    |    ✅   |
| CRAG                 |    ✅   |
| Self-RAG             |    ✅   |
| Long Context         |    ✅   |
| Complete Pipeline    |    ✅   |
| Failure Handling     |    ✅   |
| Ollama Generation    |    ✅   |
| FastAPI Integration  |    ✅   |
| Arjunx Web Interface |    ✅   |

The repository includes architecture tests, end-to-end tests, and diagnostic validation.

---

# 🧰 Technology Stack

| Technology              | Purpose                         |
| ----------------------- | ------------------------------- |
| **Python**              | Core implementation             |
| **LangChain**           | RAG and LLM orchestration       |
| **ChromaDB**            | Vector storage and retrieval    |
| **Hugging Face**        | Embeddings and model components |
| **Ollama**              | Local LLM inference             |
| **Tavily**              | Web search                      |
| **Wikipedia**           | External knowledge retrieval    |
| **Cross-Encoder**       | Document reranking              |
| **RAPTOR**              | Hierarchical retrieval          |
| **ColBERT**             | Fine-grained retrieval          |
| **FastAPI**             | Backend API                     |
| **HTML/CSS/JavaScript** | Arjunx web interface            |

---

# 🌐 Arjunx Application

The RAG system is exposed through a FastAPI backend and a lightweight web interface.

```text
                    ┌─────────────────────┐
                    │      ARJUNX UI      │
                    │                     │
                    │  Ask a Question     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     FASTAPI API     │
                    │                     │
                    │     POST /chat/     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    RAG SERVICE      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   RAG PIPELINE      │
                    │                     │
                    │ Query → Retrieval   │
                    │ → Ranking → CRAG    │
                    │ → Context → Ollama  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     FINAL ANSWER    │
                    └─────────────────────┘
```

The application layer allows the integrated RAG system to be used as a real AI assistant rather than only through notebooks.

---

# 🤝 Model Context Protocol (MCP)

This repository is ready for AI coding assistants through **GitMCP**.

Add the following MCP server configuration:

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

This allows compatible AI coding assistants to access the repository as a contextual knowledge source.

---

# 🎯 Project Philosophy

This project is primarily a **learning and engineering project** focused on understanding modern RAG architectures.

The progression is:

**RAG Fundamentals → Query Processing → Routing → Advanced Indexing → Retrieval → Reranking → Evidence Evaluation → CRAG / Self-RAG → Long Context → Evaluation → Integration → Application**

The central idea is:

> **Better Query Understanding + Better Retrieval + Better Ranking + Better Evidence Evaluation + Better Context Management = More Reliable RAG**

Arjunx represents the application of these concepts into a single integrated AI assistant.

---

# 📌 Project Status

**Status: Core Integration & Application Completed ✅**

The major components of the integrated RAG pipeline have been implemented and validated, including:

**Query Processing · Routing · Local/Web/Hybrid Retrieval · Advanced Indexing · Deduplication · Reranking · Freshness · CRAG · Self-RAG · Long Context · Failure Handling · Ollama Generation · End-to-End Testing · FastAPI Integration · Arjunx Web Interface**

The project is now ready for further experimentation, optimization, deployment, and research.

---

# 👨‍💻 Author

**Arjun Bhattarai**

### Arjunx

> **An advanced, locally powered AI assistant built to explore and engineer modern Retrieval-Augmented Generation systems.**

---

# 🚀 Built to Understand RAG from the Ground Up

**From basic retrieval to an adaptive, evidence-aware, locally powered AI assistant.**

**Built by Arjun Bhattarai · Powered by RAG · Running locally with Ollama · Presented through Arjunx**
