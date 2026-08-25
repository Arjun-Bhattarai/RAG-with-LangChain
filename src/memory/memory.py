from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma


class ConversationMemory:

    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def get_messages(self):
        return self.messages

    def clear(self):
        self.messages = []

    def chat(self, user_message, llm=None):
        if llm is None:
            llm = create_llm()

        self.add_message(HumanMessage(content=user_message))
        response = llm.invoke(self.get_messages())
        self.add_message(AIMessage(content=response.content))
        return response.content


class SummarizationMemory:

    def __init__(self, llm, max_recent_messages=4):
        self.llm = llm
        self.summary = ""
        self.recent_messages = []
        self.max_recent_messages = max_recent_messages

    def _format_messages(self, messages):

        formatted = []

        for message in messages:

            if isinstance(message, HumanMessage):
                role = "User"

            elif isinstance(message, AIMessage):
                role = "Assistant"

            else:
                role = "Message"

            formatted.append(
                f"{role}: {message.content}"
            )

        return "\n".join(formatted)

    def _summarize_messages(self, messages):

        conversation = self._format_messages(messages)

        prompt = f"""
You are a conversation memory summarizer.

Summarize ONLY the information explicitly stated in the conversation.

IMPORTANT RULES:
- Treat the conversation as the only source of truth.
- Do not use outside knowledge.
- Do not explain concepts mentioned by the user.
- Do not add definitions or facts that the user did not state.
- Do not infer additional information.
- Do not correct the user.
- Preserve the user's exact meaning.
- Keep important user facts, preferences, goals, requirements, and decisions.
- Remove small talk and repetitive information.

Conversation:
{conversation}

Write a short factual summary containing only information explicitly present
in the conversation.

Summary:
"""

        response = self.llm.invoke(prompt)

        return response.content

    def add_message(self, message):

        self.recent_messages.append(message)

        if len(self.recent_messages) > self.max_recent_messages:
            self._summarize_old_messages()

    def _summarize_old_messages(self):

        old_messages = self.recent_messages[
            :-self.max_recent_messages
        ]

        if self.summary:

            old_messages = [
                HumanMessage(
                    content=f"Existing summary:\n{self.summary}"
                )
            ] + old_messages

        self.summary = self._summarize_messages(
            old_messages
        )

        self.recent_messages = self.recent_messages[
            -self.max_recent_messages:
        ]

    def get_context(self):

        context = []

        if self.summary:

            context.append(
                HumanMessage(
                    content=f"Conversation summary:\n{self.summary}"
                )
            )

        context.extend(self.recent_messages)

        return context

    def clear(self):

        self.summary = ""
        self.recent_messages = []

    def chat(self, user_message):
        human_message = HumanMessage(content=user_message)
        self.add_message(human_message)
        context = self.get_context()
        response = self.llm.invoke(context)
        self.add_message(AIMessage(content=response.content))
        return response.content


class VectorMemory:

    def __init__(
        self,
        embedding_model="nomic-embed-text",
        collection_name="conversation_vector_memory",
    ):

        self.embeddings = OllamaEmbeddings(
            model=embedding_model
        )

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )

    def store_memory(self, message):

        self.vector_store.add_texts(
            texts=[message.content],
            metadatas=[
                {
                    "role": message.__class__.__name__
                }
            ],
        )

    def retrieve_memories(self, query, k=3):

        return self.vector_store.similarity_search(
            query,
            k=k,
        )

    def clear(self):

        self.vector_store.delete_collection()

    def chat(self, user_message, llm=None, k=3):
        if llm is None:
            llm = create_llm()

        memories = self.retrieve_memories(user_message, k=k)
        memory_context = "\n".join(memory.page_content for memory in memories)
        prompt = f"""
Use the provided memories to answer the user's question.

Memories:
{memory_context}

User question:
{user_message}

Answer:
"""
        response = llm.invoke(prompt)
        self.store_memory(HumanMessage(content=user_message))
        self.store_memory(AIMessage(content=response.content))
        return response.content


def create_llm(
    model="llama3:latest",
    temperature=0,
):

    return ChatOllama(
        model=model,
        temperature=temperature,
    )