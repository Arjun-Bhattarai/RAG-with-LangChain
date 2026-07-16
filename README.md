> **Prerequisites:** This project builds on my **[LLMs](https://github.com/Arjun-Bhattarai/LLMs)** repository. Familiarity with Transformer architecture, GPT models, and the fundamentals of LLMs is recommended.

# RAG From Scratch: A Practical Learning Journey

his repository focuses on learning RAG from first principles through hands-on implementation, experiments, and detailed explanations rather than using high-level abstractions.

---

## Project Overview

This project implements a full **Retrieval-Augmented Generation (RAG)** pipeline from scratch using LangChain and a local Ollama model.  
The objective is to generate answers grounded in retrieved documents instead of relying only on model memory.

## What This Project Covers

- Core RAG concepts and practical implementation
- Document indexing for semantic search
- Retrieval of relevant chunks from indexed data
- Prompting with retrieved context
- Context-aware answer generation with an LLM
- Updated LangChain-compatible flow without deprecated Hub usage

## RAG Pipeline Diagram

### High-Level Flow

```text
User Question
    ↓
Retriever (semantic search on indexed chunks)
    ↓
Relevant Context Chunks
    ↓
Prompt Template (Context + Question)
    ↓
LLM (Ollama / Llama3)
    ↓
Final Grounded Answer
```

### Detailed Component Flow

```text
Knowledge Source Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Index
    ↓
Retriever
    ↓
(Top-K relevant chunks)

User Query ────────────────────────────────┐
                                            ├──> Prompt Construction
Retrieved Context ──────────────────────────┘
                    ↓
                LLM Inference
                    ↓
             Parsed Final Response
```

## Workflow Stages

1. **Basic Setup**  
   Establishes the foundational RAG flow and required integrations.

2. **Indexing**  
   Prepares and stores chunk embeddings for retrieval.

3. **Retrieval**  
   Selects relevant context chunks for an incoming query.

4. **Generation**  
   Combines question + context in a prompt and generates the final answer.

## Design Decisions

- **Grounded responses:** Answers are tied to retrieved context.
- **Deterministic behavior:** Low temperature for consistent outputs.
- **Local inference:** Uses Ollama-hosted model for local execution.
- **Modern compatibility:** Uses local `ChatPromptTemplate` approach.

## Expected Output

For each user query, the system:
- retrieves semantically relevant content,
- injects it into the prompt,
- and returns a concise, context-aware answer.

## Tech Stack

- Python
- LangChain
- Ollama
- Llama 3
- HuggingFace Embeddings (BAAI/bge-small-en-v1.5)
- ChromaDB
- tiktoken
- BeautifulSoup4

## Current Limitations

- Quality is dependent on retrieval quality and indexed document relevance.
- If context is missing, answers may be uncertain.
- Local model speed depends on hardware capability.

## Future Improvements

- Add reranking for better retrieval precision
- Add source citation in generated answers
- Add evaluation for retrieval and answer faithfulness
- Improve query handling for ambiguous or multi-hop questions

## Conclusion

This repository demonstrates a practical, modular, and modern RAG implementation path.  
It is intended as both a learning resource and a solid base for building production-ready retrieval-augmented AI applications.
