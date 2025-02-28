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

        # ✅ Fix: Corrected prompt to prevent "missing grade" error
        self.grader_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a grader assessing document relevance. If the document's sementic sense is relevant to the query, "
                           "grade it as 'relevant'. Otherwise, mark it as 'irrelevant'. Respond in JSON format with key grade and its value either relevant or irrelevant,"
                            "Be relaxed in marking it interms of semantic sense"),
                ("human", "Query: {query}\n\nDocument: {document}")
            ]
        )

        # ✅ Fix: Use JsonOutputParser to ensure JSON response format
        self.grader_chain = self.grader_prompt | self.llm | JsonOutputParser(pydantic_object=DocumentGrader)

        # # Define LangGraph pipeline
        # self.workflow = StateGraph(GraphState)
        # self.workflow.add_node("grade_documents", self.grade_documents)
        # self.workflow.set_entry_point("grade_documents")
        # self.app = self.workflow.compile()

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
                # ✅ Fix: Ensuring that only "query" and "document" are sent to the LLM
                response = self.grader_chain.invoke({
                    "query": query,
                    "document": doc.page_content
                })
                
                # ✅ Fix: Structured parsing ensures a valid response
                if response.grade == "relevant":
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
