from typing import Any, List, Optional, Union
import numpy as np
import tiktoken
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import TokenTextSplitter


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Return the number of tokens in a text string using tiktoken."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def cosine_similarity(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
    """Calculate the cosine similarity between two 1D vectors."""
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))


def split_documents(
    documents: List[Document],
    chunk_size: int = 256,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Split documents into token-based chunks using TokenTextSplitter."""
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


def build_vectorstore(
    documents: List[Document],
    embedding_model: Union[str, Any] = "BAAI/bge-small-en-v1.5",
    chunk_size: int = 256,
    chunk_overlap: int = 50,
    collection_name: Optional[str] = None,
) -> Chroma:
    """Split documents and index them into a Chroma vectorstore."""
    splits = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if isinstance(embedding_model, str):
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    else:
        embeddings = embedding_model

    kwargs = {"documents": splits, "embedding": embeddings}
    if collection_name:
        kwargs["collection_name"] = collection_name

    vectorstore = Chroma.from_documents(**kwargs)
    return vectorstore
