"""
Integrated RAG entrypoint.

Re-exports the modular pipeline so notebooks and scripts can keep using
`from integration.integration import RAGIntegration`.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.rag_pipeline import RAGIntegration, RAGPipeline

__all__ = ["RAGIntegration", "RAGPipeline"]


if __name__ == "__main__":
    rag = RAGPipeline()

    user_query = input("Ask a question: ")

    result = rag.run(
        user_query,
        evaluate=False,
    )

    print("\n" + "=" * 70)
    print("ROUTE")
    print("=" * 70)
    print(result["route"])

    print("queries:", result["queries"])

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result["answer"])