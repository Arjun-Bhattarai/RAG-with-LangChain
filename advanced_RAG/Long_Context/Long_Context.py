# Local Ollama LLM use garna
from langchain_ollama import ChatOllama


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