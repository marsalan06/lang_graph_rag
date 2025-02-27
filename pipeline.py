import logging
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from query_rewriter import QueryRewritePipeline
from retriever import Retriever
from document_grader import DocumentGradingPipeline
from response_generator import ResponseGenerationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class GraphState(TypedDict):
    query: str
    rewritten_queries: list
    retrieved_docs: list
    relevant_docs: list
    attempt_count: int
    response: str

class CRAGPipeline:
    """
    Implements a complete CRAG pipeline using LangGraph.
    """

    def __init__(self):
        self.query_rewriter = QueryRewritePipeline()
        self.retriever = Retriever()
        self.document_grader = DocumentGradingPipeline()
        self.response_generator = ResponseGenerationPipeline()
        self.workflow = StateGraph(GraphState)

        # Define processing steps
        self.workflow.add_node("rewrite_query", self.rewrite_query)
        self.workflow.add_node("retrieve_documents", self.retrieve_documents)
        self.workflow.add_node("grade_documents", self.grade_documents)
        self.workflow.add_node("generate_response", self.generate_response)

        # Define execution flow
        self.workflow.add_edge("rewrite_query", "retrieve_documents")
        self.workflow.add_edge("retrieve_documents", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.decide_next_step,
            {
                "retry": "rewrite_query",  # Now correctly looping back if attempt count < 2
                "generate_response": "generate_response",
                "apology": "generate_response"
            }
        )
        self.workflow.add_edge("generate_response", END)

        # Set entry point
        self.workflow.set_entry_point("rewrite_query")
        self.app = self.workflow.compile()

    def rewrite_query(self, state: GraphState) -> GraphState:
        """
        Rewrites the user query up to 2 times if retrieval fails.
        """
        logging.info(f"🔄 Rewriting query attempt {state['attempt_count'] + 1}")

        rewritten_query = self.query_rewriter.run(state["query"])
        if rewritten_query != state["query"]:  # Ensure a new query is generated
            state["rewritten_queries"].append(rewritten_query)
            state["query"] = rewritten_query

        logging.info(f"✅ Rewritten Query: {rewritten_query}")
        return {**state, "attempt_count": state["attempt_count"] + 1}

    def retrieve_documents(self, state: GraphState) -> GraphState:
        """
        Retrieves documents from Pinecone.
        """
        logging.info(f"🔍 Retrieving documents for query: {state['query']}")
        metadata_filter = {"source": {"$eq": "software_design_dev"}}
        retrieved_docs = self.retriever.retrieve_relevant_docs(state["query"],namespace="SE_Software_Engineering", metadata_filter=metadata_filter, k=3)

        logging.info(f"📄 Retrieved {len(retrieved_docs)} documents.")
        return {**state, "retrieved_docs": retrieved_docs, "relevant_docs": []}

    def grade_documents(self, state: GraphState) -> GraphState:
        """
        Filters out irrelevant documents before response generation.
        """
        logging.info("🎯 Grading retrieved documents for relevance.")
        graded_output = self.document_grader.run(state["query"], state["retrieved_docs"])

        logging.info(f"✅ {len(graded_output['relevant_docs'])} relevant documents found.")
        return {**state, "relevant_docs": graded_output["relevant_docs"]}

    def decide_next_step(self, state: GraphState) -> str:
        """
        Decides whether to generate a response, retry query rewriting, or return an apology.
        """
        if state["relevant_docs"]:
            logging.info("✅ Relevant documents found. Proceeding to response generation.")
            return "generate_response"

        if state["attempt_count"] < 2:
            logging.warning("⚠️ No relevant documents found. Retrying with a rewritten query.")
            return "retry"

        logging.warning("⚠️ Max rewrite attempts reached. Returning apology message.")
        return "apology"

    def generate_response(self, state: GraphState) -> GraphState:
        """
        Generates the final response.
        """
        logging.info("📝 Generating response...")

        if not state["relevant_docs"]:
            response = "I’m sorry, I don’t have enough information on this topic."
            logging.warning("⚠️ No relevant documents found. Returning apology message.")
        else:
            response = self.response_generator.run(state["query"], state["relevant_docs"])

        logging.info(f"✅ Generated response: {response[:100]}...")  # Log first 100 chars
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
            "response": ""
        }

        final_output = None  # Ensure final response is stored

        for output in self.app.stream(inputs):
            print("------output----", output, flush=True)
            final_output = output  # Capture the latest state
            print("----final output----", final_output, flush=True)

        print("----final output type----", type(final_output), flush=True)

        # Ensure the final response is extracted correctly
        if final_output and isinstance(final_output, dict):
            for key, value in final_output.items():
                if isinstance(value, dict) and "response" in value and value["response"]:
                    print("-----Extracted response-----", flush=True)
                    logging.info(f"🔹 Final Output: {value['response'][:100]}...")  # Log output
                    return value["response"]

        logging.warning("⚠️ No valid response found. Returning default apology message.")
        return "⚠️ I’m sorry, I don’t have enough information on this topic."

