# CRAG ko full pipeline implementation.
# Yo code ma document retrieval, grading,
# corrective retrieval ra final answer generation ko process implement gareko cha.

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# LOCAL LLM


# Local LLM ko lagi Ollama ko llama3 model use gareko
llm = ChatOllama(
    model="llama3:latest",
    temperature=0
)


# RETRIEVAL GRADER

# Retrieved document question sanga relevant cha ki chaina
# bhanera check garna prompt banayeko
grader_prompt = ChatPromptTemplate.from_template(
    """
You are a retrieval relevance grader.

Your task is to determine whether the following
document is relevant to the user's question.

Question:
{question}

Document:
{document}

If the document contains information that can help
answer the question, respond with:

YES

Otherwise respond with:

NO

Respond with only YES or NO.
"""
)

# Grader prompt lai LLM sanga connect gareko
retrieval_grader = (
    grader_prompt
    | llm
    | StrOutputParser()
)



# GENERATION CHAIN


# Relevant context ko basis ma final answer generate garna prompt banayeko
generation_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful RAG assistant.

Answer the question using only the provided context.

Context:
{context}

Question:
{question}

If the context does not contain enough information
to answer the question, say that you do not have
enough information.

Answer:
"""
)

# Generation prompt lai LLM sanga connect gareko
generation_chain = (
    generation_prompt
    | llm
    | StrOutputParser()
)


# CRAG PIPELINE


# Yaha CRAG ko full pipeline implementation ho.
# Yo code ma document retrieval, grading,
# corrective retrieval ra final answer generation implement gareko cha.

def crag(question, retriever, vectorstore):

    # STEP 1: Initial retrieval
    # Chroma vectorstore bata user ko question ko
    # basis ma documents retrieve gareko
    retrieved_docs = retriever.invoke(question)

    print(
        f"Initial retrieved documents: "
        f"{len(retrieved_docs)}"
    )


    # STEP 2: Retrieved documents ko grading
    # Relevant documents matra select gareko
    relevant_docs = []

    for i, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        # Document question sanga relevant cha ki chaina
        # bhanera LLM grader bata check gareko
        grade = retrieval_grader.invoke({
            "question": question,
            "document": doc.page_content
        })

        # LLM ko output clean gareko
        grade = grade.strip().upper()

        print(
            f"Document {i} grade: {grade}"
        )

        # Relevant document matra list ma rakheko
        if grade == "YES":
            relevant_docs.append(doc)


    # STEP 3: Corrective retrieval
    # Relevant document bhetiyena bhane corrective retrieval gareko
    if len(relevant_docs) == 0:

        print(
            "No relevant documents found."
        )

        print(
            "Applying corrective retrieval..."
        )

        # Dherai documents retrieve garna
        # corrective retriever banayeko
        corrective_retriever = (
            vectorstore.as_retriever(
                search_kwargs={
                    "k": 10
                }
            )
        )

        # Corrective retrieval gareko
        relevant_docs = (
            corrective_retriever.invoke(
                question
            )
        )

    else:

        print(
            f"Found {len(relevant_docs)} "
            f"relevant documents."
        )


    # STEP 4: Context construction
    # Relevant documents lai euta context ma combine gareko
    context = "\n\n".join(
        doc.page_content
        for doc in relevant_docs
    )


    # STEP 5: Final answer generation
    # Context ra question LLM lai diyera answer generate gareko
    answer = generation_chain.invoke({
        "context": context,
        "question": question
    })

    return answer