from typing import Any, Dict, List, Optional, Union
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

PHYSICS_TEMPLATE = """You are a very smart physics professor.

Answer physics questions in a simple and accurate way.

Question:
{query}
"""

MATH_TEMPLATE = """You are a skilled mathematician.

Solve mathematical problems step by step.

Question:
{query}
"""

DEFAULT_PROMPT_TEMPLATES = [
    PHYSICS_TEMPLATE,
    MATH_TEMPLATE,
]


def cosine_similarity_matrix(a: Union[List, np.ndarray], b: Union[List, np.ndarray]) -> np.ndarray:
    """Calculate normalized cosine similarity matrix between two sets of vectors."""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)

    if a_arr.ndim == 1:
        a_arr = a_arr[np.newaxis, :]
    if b_arr.ndim == 1:
        b_arr = b_arr[np.newaxis, :]

    norm_a = np.linalg.norm(a_arr, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b_arr, axis=1, keepdims=True)

    norm_a[norm_a == 0] = 1e-10
    norm_b[norm_b == 0] = 1e-10

    a_norm = a_arr / norm_a
    b_norm = b_arr / norm_b

    return np.dot(a_norm, b_norm.T)


class SemanticRouter:
    """Routes queries to the most semantically relevant prompt template based on embedding similarity."""

    def __init__(
        self,
        prompt_templates: Optional[List[str]] = None,
        embeddings: Optional[Any] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        llm: Optional[Any] = None,
    ):
        self.templates = prompt_templates or DEFAULT_PROMPT_TEMPLATES
        self.embeddings = embeddings or HuggingFaceEmbeddings(model_name=embedding_model)
        self.llm = llm or ChatOllama(model="llama3:latest", temperature=0)
        self.prompt_embeddings = self.embeddings.embed_documents(self.templates)

    def route_prompt(self, input_data: Union[str, Dict[str, Any]]) -> PromptTemplate:
        """Find the prompt template with highest cosine similarity to input query."""
        if isinstance(input_data, dict):
            query = input_data.get("query", input_data.get("question", ""))
        else:
            query = str(input_data)

        query_embedding = self.embeddings.embed_query(query)
        similarity = cosine_similarity_matrix([query_embedding], self.prompt_embeddings)[0]
        most_similar_idx = int(np.argmax(similarity))
        selected_template = self.templates[most_similar_idx]

        return PromptTemplate.from_template(selected_template)

    def create_chain(self):
        """Create a runnable chain for semantic routing + LLM response."""
        return (
            {"query": RunnablePassthrough()}
            | RunnableLambda(self.route_prompt)
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, query: str) -> str:
        """Route query and generate response."""
        chain = self.create_chain()
        return chain.invoke(query)
