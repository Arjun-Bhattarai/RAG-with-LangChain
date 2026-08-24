from typing import Any, List
from langchain_core.documents import Document


def create_retriever(
    vectorstore: Any,
    k: int = 1,
    **search_kwargs: Any,
):
    """Create a retriever from a vectorstore with specified top-k."""
    kwargs = {"k": k, **search_kwargs}
    return vectorstore.as_retriever(search_kwargs=kwargs)


def retrieve_documents(
    retriever: Any,
    query: str,
) -> List[Document]:
    """Retrieve relevant documents for a given query."""
    return retriever.invoke(query)
