import logging
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from query_rewriter import QueryRewritePipeline
from retriever import Retriever
from document_grader import DocumentGradingPipeline
from response_generator import ResponseGenerationPipeline
from input_analyzer import InputAnalyzer
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from pydantic import BaseModel, Field  # ✅ Fix for Pydantic v2 warning

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class GraphState(TypedDict):
    query: str
    rewritten_queries: list
    retrieved_docs: list
    relevant_docs: list
    attempt_count: int
    response: str
    input_type: str

class CRAGPipeline:
    """
    Implements a correct CRAG pipeline using LangGraph.
    """

    def __init__(self):
        self.input_analyzer = InputAnalyzer()
        self.query_rewriter = QueryRewritePipeline()
        self.retriever = Retriever()
        self.document_grader = DocumentGradingPipeline()
        self.response_generator = ResponseGenerationPipeline()
        self.workflow = StateGraph(GraphState)

        # Define processing steps
        self.workflow.add_node("analyze_input", self.analyze_input)
        self.workflow.add_node("retrieve_documents", self.retrieve_documents)
        self.workflow.add_node("grade_documents", self.grade_documents)
        self.workflow.add_node("rewrite_query", self.rewrite_query)
        self.workflow.add_node("generate_response", self.generate_response)

        # ✅ Corrected Execution Flow
        # self.workflow.add_edge("analyze_input", "retrieve_documents")  # Default: proceed as a question
        self.workflow.add_conditional_edges(
            "analyze_input",
            self.decide_analysis_result,
            {
                "retrieve_documents": "retrieve_documents",  # If input is a question
                "generate_response": "generate_response"  # If input is a pleasantry
            }
        )
        self.workflow.add_edge("retrieve_documents", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.decide_next_step,
            {
                "generate_response": "generate_response",  # ✅ If relevant, generate response
                "rewrite_query": "rewrite_query",  # ❌ If not relevant, rewrite and retry
                "apology": "generate_response"  # ❌ If max retries, return apology
            }
        )

        self.workflow.add_edge("rewrite_query", "retrieve_documents")  # ✅ Retry after rewriting
        self.workflow.add_edge("generate_response", END)

        # ✅ Set correct entry point (starts with retrieval)
        self.workflow.set_entry_point("analyze_input")
        self.app = self.workflow.compile()

    def analyze_input(self, state: GraphState) -> GraphState:
        """
        Determines if the input is a question or a pleasantry.
        """
        logging.info(f"🔍 Analyzing input: {state['query']}")
        input_type = self.input_analyzer.analyze_input(state["query"])

        logging.info(f"📌 Input classified as: {input_type}")
        return {**state, "input_type": input_type}
    
    def retrieve_documents(self, state: GraphState) -> GraphState:
        """
        Retrieves documents from Pinecone.
        """
        logging.info(f"🔍 Retrieving documents for query: {state['query']}")
        metadata_filter = {"source": {"$eq": "software_design_dev"}}
        retrieved_docs = self.retriever.retrieve_relevant_docs(state["query"], namespace="SE_Software_Engineering", metadata_filter=metadata_filter, k=3)

        logging.info(f"📄 Retrieved {len(retrieved_docs)} documents.")
        return {**state, "retrieved_docs": retrieved_docs, "relevant_docs": []}

    def grade_documents(self, state: GraphState) -> GraphState:
        """
        Filters out irrelevant documents before response generation.
        """
        logging.info("🎯 Grading retrieved documents for relevance.")
        query = state["query"]
        retrieved_docs = state["retrieved_docs"]
        relevant_docs = []

        if not retrieved_docs:
            logging.warning("⚠️ No documents retrieved. Skipping grading step.")
            return {**state, "relevant_docs": []}

        for doc in retrieved_docs:
            try:
                response = self.document_grader.grader_chain.invoke(
                    {"query": query, "document": doc.page_content}
                )

                logging.info(f"📌 LLM Response: {response}")
                print("-------response.grade-------", type(response))

                if response["grade"] == "relevant":
                    logging.info(f"✅ Document marked as relevant: {doc.page_content[:200]}...")
                    relevant_docs.append(doc)
                else:
                    logging.warning(f"⚠️ Document marked as irrelevant: {doc.page_content[:200]}...")

            except Exception as e:
                logging.error(f"❌ Error grading document: {e}", exc_info=True)

        logging.info(f"✅ {len(relevant_docs)} relevant documents found out of {len(retrieved_docs)} retrieved.")
        return {**state, "relevant_docs": relevant_docs}

    def rewrite_query(self, state: GraphState) -> GraphState:
        attempt = state["attempt_count"]
        if attempt >= 2:
            logging.warning("⚠️ Max rewrite attempts reached. Returning apology message.")
            return {**state, "attempt_count": attempt}

        logging.info(f"🔄 Rewriting query attempt {attempt + 1}")
        rewritten_query = self.query_rewriter.run(state["query"])
        if rewritten_query != state["query"]:
            state["rewritten_queries"].append(rewritten_query)
            logging.info(f"✅ Rewritten Query: {rewritten_query}")
            return {**state, "query": rewritten_query, "attempt_count": attempt + 1}
        return {**state, "attempt_count": attempt + 1}
    
    def decide_analysis_result(self, state: GraphState) -> str:
        if state["input_type"] == "pleasantry":
            logging.info("💬 Detected pleasantry. Proceeding to generate response.")
            return "generate_response"
        logging.info("❓ Detected question. Proceeding to retrieve documents.")
        return "retrieve_documents"
    
    
    def decide_next_step(self, state: GraphState) -> str:
        """
        Decides whether to generate a response, retry retrieval, or return an apology.
        """
        if state["relevant_docs"]:
            logging.info("✅ Relevant documents found. Proceeding to response generation.")
            return "generate_response"

        if state["attempt_count"] < 2:
            logging.warning("⚠️ No relevant documents found. Retrying with a rewritten query.")
            return "rewrite_query"

        logging.warning("⚠️ Max retrieval attempts reached. Returning apology message.")
        return "apology"

    def generate_response(self, state: GraphState) -> GraphState:
        logging.info("📝 Generating response...")
        # Let the response generator handle both questions and pleasantries
        response = self.response_generator.run(state["query"], state["relevant_docs"])
        logging.info(f"✅ Generated response: {response[:100]}...")
        return {**state, "response": response}
    

    def run(self, query: str) -> str:
        """
        Runs the complete CRAG pipeline.
        """
        inputs = {
            "query": query,
            "rewritten_queries": [],
            "retrieved_docs": [],
            "relevant_docs": [],
            "attempt_count": 0,
            "response": "",
            "input_type": ""
        }

        final_response = "⚠️ No valid response found."

        for output in self.app.stream(inputs):
            logging.info(f"📥 Pipeline Output: {output}")
            # Check if the response is nested under 'generate_response'
            if "generate_response" in output and "response" in output["generate_response"]:
                final_response = output["generate_response"]["response"]

        return final_response

def display_graph(pipeline, save_path="crag_graph.png"):
    """
    Generates and saves the LangGraph execution graph.
    """
    logging.info("📊 Generating CRAG pipeline graph...")
    
    plot = pipeline.app.get_graph().draw_mermaid_png()
    img = Image.open(BytesIO(plot))

    img.save(save_path)
    logging.info(f"✅ Graph saved to {save_path}")

    # plt.figure(figsize=(12, 8))
    # plt.imshow(img)
    # plt.axis("off")
    # plt.show(block=False)
