# Local Ollama LLM use garna
from langchain_ollama import ChatOllama


def create_context_chunks(text, chunk_size=500):
    """Split a long text into word-based chunks."""
    words = str(text).split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def select_relevant_context(chunks, keywords):
    """Keep chunks that contain any of the provided keywords."""
    selected_chunks = []
    for chunk in chunks:
        if any(keyword.lower() in chunk.lower() for keyword in keywords):
            selected_chunks.append(chunk)
    return selected_chunks


class LongContext:

    def __init__(
        self,
        model="llama3:latest",
        temperature=0,
    ):
        # Local Ollama LLM load garne
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
        )

    def build_context(self, documents):
        # Retrieved documents lai single long context ma combine garne
        context_parts = []

        for document in documents:

            # LangChain Document object bata content nikalne
            if hasattr(document, "page_content"):
                context_parts.append(
                    document.page_content
                )

            # Normal string bhaye directly use garne
            else:
                context_parts.append(
                    str(document)
                )

        return "\n\n".join(context_parts)

    def create_context_chunks(self, text, chunk_size=500):
        return create_context_chunks(text, chunk_size=chunk_size)

    def select_relevant_context(self, chunks, keywords):
        return select_relevant_context(chunks, keywords)

    def compress_context(self, context, query):
        # Large context bata query-relevant information extract garne
        prompt = f"""
You are a context compression component in a RAG system.

Extract only the information from the provided context that is relevant
to answering the user's question.

Rules:
- Use only information explicitly present in the context.
- Do not add outside knowledge.
- Do not answer the question.
- Preserve important details.
- Remove irrelevant information.
- Keep the compressed context factual and concise.

Context:
{context}

Question:
{query}

Relevant Context:
"""

        # LLM bata relevant context extract garne
        response = self.llm.invoke(prompt)

        return response.content

    def generate(self, context, query):
        # Long context use garera final answer generate garne
        prompt = f"""
Answer the user's question using only the provided context.

Rules:
- Use only the provided context.
- Do not invent information.
- If the answer is not present in the context, say that the
  information is not available in the provided context.
- Give a clear and concise answer.

Context:
{context}

Question:
{query}

Answer:
"""

        # Final answer generate garne
        response = self.llm.invoke(prompt)

        return response.content

    def run(
        self,
        documents,
        query,
        compress=True,
    ):
        # Retrieved documents lai long context ma combine garne
        context = self.build_context(
            documents
        )

        # Large context lai optionally compress garne
        if compress:
            context = self.compress_context(
                context=context,
                query=query,
            )

        # Context bata final answer generate garne
        answer = self.generate(
            context=context,
            query=query,
        )

        # Integration ko lagi structured result return garne
        return {
            "query": query,
            "context": context,
            "answer": answer,
        }