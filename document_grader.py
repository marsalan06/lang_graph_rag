import logging
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
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

# Define a structured response model for LLM output
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
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=Config.OPENAI_API_KEY)

        # Define LLM prompt
        self.grader_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a grader assessing document relevance. If a document is relevant to the query, "
                           "grade it as 'relevant'. Otherwise, mark it as 'irrelevant'. Respond in JSON format."),
                ("human", "Query: {query}\n\nDocument: {document}")
            ]
        )

        # Define LLM-powered grader chain
        self.grader_chain = self.grader_prompt | self.llm.with_structured_output(DocumentGrader, method="json_mode")

        # Define LangGraph pipeline
        self.workflow = StateGraph(GraphState)
        self.workflow.add_node("grade_documents", self.grade_documents)
        self.workflow.set_entry_point("grade_documents")
        self.app = self.workflow.compile()

    def grade_documents(self, state: GraphState) -> GraphState:
        """
        Grades all retrieved documents and filters out irrelevant ones using the LLM.
        """
        logging.info("🎯 Grading retrieved documents for relevance.")
        query = state["query"]
        retrieved_docs = state["retrieved_docs"]
        relevant_docs = []

        for doc in retrieved_docs:
            try:
                response = self.grader_chain.invoke({"query": query, "document": doc.page_content})
                if response.grade == "relevant":
                    logging.info(f"✅ Document marked as relevant: {doc.page_content[:100]}...")
                    relevant_docs.append(doc)
                else:
                    logging.warning(f"⚠️ Document marked as irrelevant: {doc.page_content[:100]}...")
            except Exception as e:
                logging.error(f"❌ Error grading document: {e}")

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
