> **Prerequisites:** This project builds on my **[LLMs](https://github.com/Arjun-Bhattarai/LLMs)** repository. Familiarity with Transformers, GPT models, and LLM fundamentals is recommended.

# RAG From Scratch

A hands-on implementation of **Retrieval-Augmented Generation (RAG)** from first principles using **LangChain**, **ChromaDB**, and **Ollama**. This repository focuses on understanding how modern RAG systems work through practical implementations rather than high-level abstractions.

---

## What I've Implemented

### Core RAG Pipeline
- Basic RAG setup
- Document loading and preprocessing
- Document chunking
- Embedding generation
- Vector indexing with ChromaDB
- Semantic retrieval
- Context-aware answer generation

### Advanced Retrieval Techniques
- Multi-Query Retrieval
- RAG-Fusion (Reciprocal Rank Fusion)
- Query Decomposition
  - Sequential Query Decomposition
  - Parallel Query Decomposition

### Additional Work
- Local inference using Ollama (Llama 3)
- HuggingFace Embeddings (`BAAI/bge-small-en-v1.5`)
- Modern LangChain implementation (without deprecated APIs)
- Detailed notebook explanations and experiments

---

## Tech Stack

- Python
- LangChain
- Ollama
- Llama 3
- ChromaDB
- HuggingFace Embeddings
- BeautifulSoup4
- tiktoken

---

## Current Status

- ✅ Basic RAG
- ✅ Indexing
- ✅ Retrieval
- ✅ Generation
- ✅ Multi-Query Retrieval
- ✅ RAG-Fusion
- ✅ Query Decomposition

More advanced RAG techniques will be added as the project progresses.

---

## Goal

The goal of this repository is to learn, implement, and document modern Retrieval-Augmented Generation techniques from scratch while building a strong foundation for production-ready RAG systems.