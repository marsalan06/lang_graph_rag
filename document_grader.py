import logging
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from typing_extensions import TypedDict
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the state dictionary
class GraphState(TypedDict):
    query: str
    retrieved_docs: list
    relevant_docs: list

# Define structured output model
class DocumentGrader(BaseModel):
    """
    LLM response model for grading document relevance.
    """
    grade: Literal["relevant", "irrelevant"] = Field(
        ..., description="Score as 'relevant' if the document matches the query; otherwise, 'irrelevant'."
    )

class DocumentGradingPipeline:
    """
    Implements a LangGraph pipeline for filtering relevant documents using an LLM-based grader.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125", 
            temperature=0, 
            openai_api_key=Config.OPENAI_API_KEY
        )

        self.grader_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant for a tutoring platform. Your task is to decide whether a given document is **relevant** to a user's question.

                Return **only** a JSON object like:
                {{ "grade": "relevant" }} or {{ "grade": "irrelevant" }}

                ### Mark a document as "relevant" if:
                - It explains the **same concept**, principle, rule, or formula — even with different numbers or phrasing.
                - It gives a **similar solved example**, math problem, or code pattern that can help solve or understand the query.
                - It offers **step-by-step reasoning**, explanations, or analogies that are **methodologically useful**.
                - It includes **code**, equations, or logic that applies to the problem — even if not identical.

                ### Mark as "irrelevant" if:
                - It discusses **completely different topics**, unrelated examples, or unrelated theory.
                - It includes **trivia, historical facts, or definitions** not useful for answering or solving the query.
                - It is **conceptually unrelated** or misleading.

                DO NOT explain. ONLY return JSON like: {{ "grade": "relevant" }} or {{ "grade": "irrelevant" }}"""),
                            ("human", "Query: {query}\n\nDocument: {document}")
        ])

        self.grader_chain = self.grader_prompt | self.llm | JsonOutputParser(pydantic_object=DocumentGrader)

    def grade_documents(self, state: GraphState) -> GraphState:
        """
        Grades all retrieved documents and filters out irrelevant ones using the LLM.
        """
        query = state["query"]
        retrieved_docs = state["retrieved_docs"]
        relevant_docs = []

        if not retrieved_docs:
            return {**state, "relevant_docs": []}  # Return empty list if no documents found

        for doc in retrieved_docs:
            try:
                response = self.grader_chain.invoke({
                    "query": query,
                    "document": doc.page_content
                })

                # Handle both dict and Pydantic return types
                grade_value = response.get("grade") if isinstance(response, dict) else response.grade
                if grade_value == "relevant":
                    relevant_docs.append(doc)

            except Exception as e:
                print(f"❌ Error grading document: {e}")

        return {**state, "relevant_docs": relevant_docs}

    def run(self, query, retrieved_docs):
        """
        Runs the document grading pipeline.
        """
        inputs = {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "relevant_docs": []
        }

        for output in self.app.stream(inputs):
            pass  # Process execution flow

        return output
