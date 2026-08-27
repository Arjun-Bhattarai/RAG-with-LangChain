# 🧠 RAG From Scratch

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-FC521F?style=flat-square)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Inference-Ollama_(Llama_3)-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.ai/)
[![ColBERT](https://img.shields.io/badge/Late_Interaction-ColBERT-red?style=flat-square)](https://github.com/stanford-futuredata/ColBERT)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

A modular, end-to-end Retrieval-Augmented Generation (RAG) system built from scratch, exploring and integrating modern RAG techniques into a single local pipeline.

Goes beyond a basic retrieve → generate loop by combining query transformation, intelligent source routing, advanced indexing, multi-source retrieval, deduplication, cross-encoder reranking, freshness scoring, CRAG, Self-RAG, long-context handling, failure handling, and local inference via Ollama.

> **Prerequisites & Foundation:** This project builds upon foundational language modeling principles covered in **[LLMs from Scratch](https://github.com/Arjun-Bhattarai/LLMs)** (covering Transformer internals, attention mechanisms, and GPT architectures).

Overview

A basic RAG pipeline (Query → Retrieve → Context → LLM → Answer) breaks down when queries are ambiguous, evidence is scattered or outdated, or the context grows too large. This project tackles those problems by implementing each technique as an independent module, then combining the relevant ones into a single integrated pipeline — with a focus on modularity, retrieval quality, evidence evaluation, and reliability.

Key Features
Query Processing — multi-query generation, decomposition, step-back prompting, HyDE, RAG Fusion, query structuring
Routing — logical, semantic, and source routing (LOCAL / WEB / HYBRID)
Retrieval — Chroma vector retrieval, Wikipedia, Tavily web search, hybrid retrieval
Advanced Indexing — multi-representational indexing, RAPTOR, ColBERT
Retrieval Quality — deduplication, cross-encoder reranking, freshness scoring
Advanced RAG — Corrective RAG (CRAG), Self-RAG, long-context processing, evidence evaluation
Generation — local LLM inference via Ollama
Testing — architecture tests, end-to-end pipeline tests, diagnostics
Architecture
User Query → Multi-Query Processor → Source Router (LOCAL / WEB / HYBRID)
    → Retrieval (Chroma/RAPTOR, Tavily/Wikipedia) → Deduplication
    → Cross-Encoder Reranking → Freshness Scoring → Top Evidence
    → CRAG + Self-RAG Evaluation → Sufficient? → [No: retrieve again]
    → Context Builder → Long Context (if needed) → Ollama → Final Answer

If evidence is deemed insufficient after evaluation, the system returns NOT_ANSWERABLE instead of overclaiming.

Project Structure
RAG-from-scratch/
├── data_source/          # Web search & Wikipedia sources
├── integration/          # Final integrated pipeline (script + notebook)
├── notebooks/            # Individual technique experiments
├── src/
│   ├── pipeline/          # Core RAG stages
│   ├── query_translation/ # Multi-query, RAG Fusion, decomposition, step-back, HyDE
│   ├── query_structuring/
│   ├── routing/           # Logical, semantic, source routing
│   ├── advanced_indexing/ # Multi-rep indexing, RAPTOR, ColBERT
│   ├── retrieval/         # Chroma, hybrid, web retrievers
│   ├── Reranking/          # Cross-encoder + freshness
│   ├── advanced_RAG/       # CRAG, Self-RAG, long context
│   ├── evaluation/
│   ├── memory/             # Conversation, summarization, vector memory
│   └── utils/               # Deduplication, evidence helpers
└── tests/                 # Architecture, e2e, diagnostic tests
Technologies
Technology	Purpose
Python	Core language
LangChain	RAG/LLM orchestration
ChromaDB	Vector storage & retrieval
Hugging Face	Embeddings
Ollama	Local LLM inference
Tavily / Wikipedia	Web retrieval
Cross-Encoder	Reranking
RAPTOR / ColBERT	Advanced indexing

FastAPI is planned for a future API layer and is not yet implemented.

Installation
bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RAG-from-scratch

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

Create a .env file:

env
TAVILY_API_KEY=your_tavily_api_key

Pull the required local model:

bash
ollama pull <MODEL_NAME>
Running
bash
# Integrated pipeline
python integration/integration.py

# Or explore the notebook
jupyter lab integration/integrated_rag.ipynb

Individual technique experiments live under notebooks/ (e.g. 05_multi_query.ipynb, 14_Raptor.ipynb, CRAG.ipynb).

Testing
bash
python tests/test_architecture.py   # architecture validation
python tests/test_e2e_pipeline.py   # end-to-end pipeline
python tests/e2e_diagnostic.py      # component diagnostics

Latest diagnostic run: all major phases (Multi-Query, LOCAL/WEB/HYBRID Retrieval, Reranking + Freshness, CRAG, Self-RAG, Long Context, Complete Pipeline, Failure Handling) passed.

Example

Query: "What is the latest situation of the floods in Nepal, including the death toll, missing people, worst-affected areas, and main causes?"

The system routed to WEB retrieval, evaluated the gathered evidence via CRAG and Self-RAG, and — because the evidence wasn't sufficient to confidently answer a fast-changing situation — returned NOT_ANSWERABLE rather than guessing. This is the failure-handling principle in action: insufficient evidence → don't overclaim.

Design Principles
Modularity — each technique is an independent, swappable component
Separation of concerns — query processing, routing, retrieval, ranking, evaluation, generation are cleanly split
Evidence-based generation — the LLM answers from retrieved evidence, not just internal knowledge
Reliability — CRAG/Self-RAG evaluate evidence before generation; the system can decline to answer
Local-first — Ollama removes dependence on a hosted LLM API
Future Improvements
FastAPI backend + frontend
Recall@K / Precision@K, MRR/NDCG evaluation
Latency benchmarking, source credibility scoring
Additional local embedding models
Production deployment & observability
Status

Core integrated pipeline is implemented and validated: multi-query, LOCAL/WEB/HYBRID retrieval, reranking + freshness, CRAG, Self-RAG, long context, failure handling, and the full end-to-end pipeline all pass diagnostics.

License

No license currently specified.

Author

Arjun Bhattarai — GitHub: Arjun-Bhattarai