from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# SELF-RAG KO REUSABLE IMPLEMENTATION

def _normalize_yes_no(value):
    """Normalize LLM grader output to YES, NO, or UNKNOWN."""
    text = str(value).strip().upper()
    if text.startswith("YES"):
        return "YES"
    if text.startswith("NO"):
        return "NO"
    return "UNKNOWN"


# RETRIEVAL DECISION

# Question ko answer garna retrieval required cha ki chaina bhanera decide garne prompt

retrieval_decision_prompt = ChatPromptTemplate.from_template("""
You are a Self-RAG retrieval decision system.

Determine whether external documents are required
to answer the following question accurately.

Question:
{question}

Respond with only YES or NO.
""")


# DOCUMENT RELEVANCE GRADER

# Retrieved document question sanga relevant cha ki chaina check garne prompt

relevance_prompt = ChatPromptTemplate.from_template("""
You are a document relevance evaluator.

Question:
{question}

Document:
{document}

Does this document contain information useful for answering
the question?

Respond with only YES or NO.
""")


# 
# ANSWER GENERATION

# Retrieved context use garera answer generate garne prompt

generation_prompt = ChatPromptTemplate.from_template("""
You are a helpful RAG assistant.

Answer the question using the provided context.

If the context does not contain enough information,
do not invent information.

Context:
{context}

Question:
{question}

Give a concise and accurate answer.
""")


# ANSWER SUPPORT GRADER

# Generated answer retrieved context bata supported cha ki chaina check garne prompt

support_prompt = ChatPromptTemplate.from_template("""
You are a Self-RAG critic.

Determine whether the answer is fully supported by
the provided context.

Context:
{context}

Answer:
{answer}

Respond with only YES or NO.
""")


# ANSWER USEFULNESS GRADER

# Answer le user ko question properly answer gareko cha ki chaina check garne prompt

usefulness_prompt = ChatPromptTemplate.from_template("""
You are a Self-RAG answer evaluator.

Determine whether the answer directly and adequately
answers the user's question.

Question:
{question}

Answer:
{answer}

Respond with only YES or NO.
""")


# 
# QUERY REFINEMENT

# Answer reliable chaina bhane better retrieval ko lagi query improve garne prompt

rewrite_prompt = ChatPromptTemplate.from_template("""
You are a query refinement system.

Rewrite the question so that it is more specific
and easier for a retrieval system to find relevant information.

Original question:
{question}

Return only the improved search query.
""")


# 
# SELF-RAG CLASS

class SelfRAG:

    def __init__(self, llm, retriever):
        """
        Self-RAG system initialize garne.

        llm:
            Local Ollama LLM

        retriever:
            Existing RAG retriever
        """

        self.llm = llm
        self.retriever = retriever

        # Sabai prompts lai existing LLM sanga connect gareko

        self.retrieval_decision_chain = (
            retrieval_decision_prompt
            | self.llm
            | StrOutputParser()
        )

        self.relevance_chain = (
            relevance_prompt
            | self.llm
            | StrOutputParser()
        )

        self.generation_chain = (
            generation_prompt
            | self.llm
            | StrOutputParser()
        )

        self.support_chain = (
            support_prompt
            | self.llm
            | StrOutputParser()
        )

        self.usefulness_chain = (
            usefulness_prompt
            | self.llm
            | StrOutputParser()
        )

        self.rewrite_chain = (
            rewrite_prompt
            | self.llm
            | StrOutputParser()
        )


    # RETRIEVAL DECISION

    def should_retrieve(self, question):

        # Question ko lagi retrieval required cha ki chaina decide gareko

        decision = self.retrieval_decision_chain.invoke({
            "question": question
        })

        return _normalize_yes_no(decision)


    # DOCUMENT RETRIEVAL

    def retrieve(self, question):

        # Existing retriever bata documents retrieve gareko

        documents = self.retriever.invoke(question)

        return documents


    # DOCUMENT GRADING

    def grade_documents(self, question, documents):

        # Relevant documents matra select garna list banayeko

        relevant_documents = []

        # Each document lai individually evaluate gareko

        for i, document in enumerate(documents, start=1):

            grade = self.relevance_chain.invoke({
                "question": question,
                "document": document.page_content
            })

            grade = _normalize_yes_no(grade)

            print(f"Document {i} relevance: {grade}")

            # Relevant document matra rakheko

            if grade == "YES":
                relevant_documents.append(document)

        return relevant_documents


    # CONTEXT BUILDING
  

    def build_context(self, documents):

        # Relevant documents ko content combine garera context banayeko

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        return context



    # ANSWER GENERATION
  

    def generate_answer(self, question, documents):

        # Documents bata context banayeko

        context = self.build_context(documents)

        # Context ra question LLM lai diyera answer generate gareko

        answer = self.generation_chain.invoke({
            "context": context,
            "question": question
        })

        return answer



    # ANSWER SUPPORT CHECK


    def check_support(self, answer, documents):

        # Documents bata context banayeko

        context = self.build_context(documents)

        # Answer context bata supported cha ki chaina check gareko

        result = self.support_chain.invoke({
            "context": context,
            "answer": answer
        })

        return _normalize_yes_no(result)



    # ANSWER USEFULNESS CHECK


    def check_usefulness(self, question, answer):

        # Answer le question properly answer gareko cha ki chaina check gareko

        result = self.usefulness_chain.invoke({
            "question": question,
            "answer": answer
        })

        return _normalize_yes_no(result)


  
    # QUERY REFINEMENT

    def refine_query(self, question):

        # Better retrieval ko lagi question rewrite gareko

        improved_question = self.rewrite_chain.invoke({
            "question": question
        })

        return improved_question.strip()



    # COMPLETE SELF-RAG PIPELINE


    def invoke(self, question, max_retries=2):

        print("\n========== SELF-RAG ==========")

        # STEP 1:
        # Retrieval required cha ki chaina decide gareko

        decision = self.should_retrieve(question)

        print(f"Retrieval required: {decision}")


        # STEP 2:
        # Retrieval required chaina bhane direct answer generate garne

        if decision == "NO":

            answer = self.generation_chain.invoke({
                "context": "",
                "question": question
            })

            return answer


        current_question = question


        # STEP 3:
        # Self-RAG le maximum retry samma correction garna sakcha

        for attempt in range(max_retries + 1):

            print(
                f"\n========== Attempt {attempt + 1} =========="
            )

            # Documents retrieve gareko

            documents = self.retrieve(current_question)

            print(
                f"Retrieved documents: {len(documents)}"
            )


            # STEP 4:
            # Retrieved documents ko relevance check gareko

            relevant_documents = self.grade_documents(
                current_question,
                documents
            )

            print(
                f"Relevant documents: "
                f"{len(relevant_documents)}"
            )


            # Relevant documents chaina bhane query refine garne

            if not relevant_documents:

                if attempt < max_retries:

                    print(
                        "No relevant documents found."
                    )

                    current_question = self.refine_query(
                        current_question
                    )

                    print(
                        f"Refined query: "
                        f"{current_question}"
                    )

                    continue

                return (
                    "I could not find enough "
                    "relevant information."
                )


            # STEP 5:
            # Relevant documents bata answer generate gareko

            answer = self.generate_answer(
                current_question,
                relevant_documents
            )

            print("\nGenerated answer:")
            print(answer)


            # STEP 6:
            # Answer context bata supported cha ki chaina check gareko

            support = self.check_support(
                answer,
                relevant_documents
            )

            print(
                f"\nAnswer supported: {support}"
            )


            # STEP 7:
            # Answer question lai properly answer garcha ki gardaina check gareko

            usefulness = self.check_usefulness(
                current_question,
                answer
            )

            print(
                f"Answer useful: {usefulness}"
            )


            # STEP 8:
            # Answer reliable cha bhane final answer return gareko

            if support == "YES" and usefulness == "YES":

                print(
                    "\nSelf-RAG accepted the answer."
                )

                return answer


            # STEP 9:
            # Answer reliable chaina bhane query refine garne

            if attempt < max_retries:

                print(
                    "\nSelf-RAG rejected the answer."
                )

                current_question = self.refine_query(
                    current_question
                )

                print(
                    f"Refined query: "
                    f"{current_question}"
                )


        # Maximum retries pachi pani reliable answer napaye fallback

        return (
            "The answer could not be verified "
            "against the retrieved context."
        )
