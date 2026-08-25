from typing import Any, List, Optional
import requests


def get_wikipedia_page(title: str, user_agent: str = "RAG_ColBERT/0.0.1") -> str:
    """Fetch full text content of a Wikipedia page using the Wikipedia API."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }
    headers = {"User-Agent": user_agent}
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract", "")


def chunk_text_fixed_length(text: str, chunk_size: int = 180) -> List[str]:
    """Split text into fixed-character chunks matching ColBERT tutorial splitting."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


class ColBERTIndexer:
    """Wrapper around PyLate ColBERT model and PLAID index for neural token-level retrieval."""

    def __init__(
        self,
        model_name_or_path: str = "colbert-ir/colbertv2.0",
        index_folder: str = "colbert-index",
        index_name: str = "default_index",
        override: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.index_folder = index_folder
        self.index_name = index_name
        self.override = override
        self._model: Optional[Any] = None
        self._index: Optional[Any] = None
        self._retriever: Optional[Any] = None
        self.chunks: List[str] = []

    @property
    def model(self):
        """Lazy load the ColBERT model."""
        if self._model is None:
            from pylate import models
            self._model = models.ColBERT(model_name_or_path=self.model_name_or_path)
        return self._model

    def index_documents(
        self,
        chunks: List[str],
        document_ids: Optional[List[str]] = None,
        show_progress_bar: bool = False,
    ):
        """Encode chunks and add them into the PLAID index."""
        from pylate import indexes, retrieve

        self.chunks = chunks
        if document_ids is None:
            document_ids = [str(i) for i in range(len(chunks))]

        document_embeddings = self.model.encode(
            chunks,
            is_query=False,
            show_progress_bar=show_progress_bar,
        )

        self._index = indexes.PLAID(
            index_folder=self.index_folder,
            index_name=self.index_name,
            override=self.override,
        )

        self._index.add_documents(
            documents_ids=document_ids,
            documents_embeddings=document_embeddings,
        )
        self._retriever = retrieve.ColBERT(index=self._index)

    def retrieve(
        self,
        query: str,
        k: int = 3,
        show_progress_bar: bool = False,
    ) -> List[dict]:
        """Retrieve top-k relevant chunks with similarity scores using token-level ColBERT retrieval."""
        if self._retriever is None:
            raise ValueError("Index has not been built yet. Call index_documents first.")

        query_embedding = self.model.encode(
            [query],
            is_query=True,
            show_progress_bar=show_progress_bar,
        )

        results = self._retriever.retrieve(
            queries_embeddings=query_embedding,
            k=k,
        )
        return results[0] if results else []
