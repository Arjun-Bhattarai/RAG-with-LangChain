import uuid
from typing import Any, List, Optional
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.stores import InMemoryByteStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama

try:
    # pyrefly: ignore [missing-import]
    from langchain.retrievers import MultiVectorRetriever
except ImportError:
    try:
        from langchain_classic.retrievers import MultiVectorRetriever
    except ImportError:
        from langchain_community.retrievers import MultiVectorRetriever

SUMMARY_PROMPT_TEMPLATE = """Summarize the following document briefly:

{doc}
"""

DEFAULT_SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_PROMPT_TEMPLATE)


def create_summary_chain(
    llm: Optional[Any] = None,
    prompt: Optional[ChatPromptTemplate] = None,
):
    """Create a chain that generates concise summaries for document chunks."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if prompt is None:
        prompt = DEFAULT_SUMMARY_PROMPT

    return (
        {"doc": lambda x: x.page_content if hasattr(x, "page_content") else str(x)}
        | prompt
        | llm
        | StrOutputParser()
    )


class MultiRepresentationalIndexer:
    """Manages Multi-Representational Indexing where summaries are embedded into vectorstore while full documents are retained in a docstore."""

    def __init__(
        self,
        embeddings: Optional[Any] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        llm: Optional[Any] = None,
        collection_name: str = "summaries",
        id_key: str = "doc_id",
        byte_store: Optional[Any] = None,
    ):
        self.embeddings = embeddings or HuggingFaceEmbeddings(model_name=embedding_model)
        self.llm = llm or ChatOllama(model="llama3:latest", temperature=0)
        self.id_key = id_key
        self.collection_name = collection_name
        self.byte_store = byte_store or InMemoryByteStore()
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            byte_store=self.byte_store,
            id_key=self.id_key,
        )

    def generate_summaries(
        self,
        documents: List[Document],
        max_concurrency: int = 1,
    ) -> List[str]:
        """Generate summaries for all provided documents using LLM."""
        chain = create_summary_chain(llm=self.llm)
        return chain.batch(documents, config={"max_concurrency": max_concurrency})

    def index(
        self,
        documents: List[Document],
        summaries: Optional[List[str]] = None,
        max_concurrency: int = 1,
    ) -> MultiVectorRetriever:
        """Index documents by linking generated summaries in vectorstore to full documents in docstore."""
        if summaries is None:
            summaries = self.generate_summaries(documents, max_concurrency=max_concurrency)

        doc_ids = [str(uuid.uuid4()) for _ in documents]
        summary_docs = [
            Document(page_content=summary, metadata={self.id_key: doc_ids[i]})
            for i, summary in enumerate(summaries)
        ]

        self.retriever.vectorstore.add_documents(summary_docs)
        self.retriever.docstore.mset(list(zip(doc_ids, documents)))
        return self.retriever

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve original documents given a query searching against summaries."""
        return self.retriever.invoke(query)

    def as_retriever(self) -> MultiVectorRetriever:
        """Return the configured MultiVectorRetriever."""
        return self.retriever
