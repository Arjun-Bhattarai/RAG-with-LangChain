from typing import Any, List, Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

DEFAULT_STEP_BACK_EXAMPLES = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "What can the members of The Police do?",
    },
    {
        "input": "Jan Sindel was born in what country?",
        "output": "What is Jan Sindel's personal history?",
    },
]

STEP_BACK_SYSTEM_PROMPT = """You are an expert in world knowledge.

Your task is to convert a specific user question into a broader, more general step-back question that is easier to answer.

Here are a few examples:
"""

STEP_BACK_RESPONSE_PROMPT_TEMPLATE = """You are an expert in world knowledge.

I am going to ask you a question.

Your response should be comprehensive and should use the provided contexts whenever they are relevant. If any context is not relevant, ignore it.

Normal Context:
{normal_context}

Step-Back Context:
{step_back_context}

Original Question:
{question}

Answer:
"""

STEP_BACK_RESPONSE_PROMPT = ChatPromptTemplate.from_template(STEP_BACK_RESPONSE_PROMPT_TEMPLATE)


def create_step_back_prompt(
    examples: Optional[List[dict]] = None,
    system_prompt: str = STEP_BACK_SYSTEM_PROMPT,
) -> ChatPromptTemplate:
    """Create the few-shot ChatPromptTemplate for step-back query generation."""
    if examples is None:
        examples = DEFAULT_STEP_BACK_EXAMPLES

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("user", "{question}"),
        ]
    )


def create_step_back_generator(
    llm: Optional[Any] = None,
    prompt: Optional[ChatPromptTemplate] = None,
):
    """Create a chain that generates a broader step-back question."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if prompt is None:
        prompt = create_step_back_prompt()

    return prompt | llm | StrOutputParser()


def create_step_back_rag_chain(
    retriever: Any,
    llm: Optional[Any] = None,
    step_back_prompt: Optional[ChatPromptTemplate] = None,
    response_prompt: Optional[ChatPromptTemplate] = None,
):
    """Create a Step-Back RAG chain retrieving context from both original and step-back queries."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if response_prompt is None:
        response_prompt = STEP_BACK_RESPONSE_PROMPT

    step_back_generator = create_step_back_generator(llm=llm, prompt=step_back_prompt)

    chain = (
        {
            "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
            "step_back_context": step_back_generator | retriever,
            "question": lambda x: x["question"],
        }
        | response_prompt
        | llm
        | StrOutputParser()
    )
    return chain
