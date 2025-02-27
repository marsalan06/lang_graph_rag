import logging
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from config import Config

class QueryRewriter:
    """
    A query rewriter that optimizes user queries for better retrieval.
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=Config.OPENAI_API_KEY)

        self.rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a query rewriter that improves retrieval.
                If a query is already optimized, modify it slightly to increase retrieval success."""),
                ("human", "Original query: {query} \n\n Improve this query for better retrieval."),
            ]
        )

        self.rewriter_chain = self.rewrite_prompt | self.llm | StrOutputParser()

    def rewrite_query(self, query: str) -> str:
        """
        Optimizes a query using LLM. Ensures that a rewritten query is actually different.
        """
        try:
            improved_query = self.rewriter_chain.invoke({"query": query}).strip()

            if improved_query.lower() == query.lower():
                logging.warning("⚠️ Query rewriter returned an identical query. Modifying manually.")
                improved_query = query + " in simple terms"

            logging.info(f"✅ Rewritten Query: {improved_query}")
            return improved_query

        except Exception as e:
            logging.error(f"Query rewriting failed: {e}")
            return query

class GraphState(TypedDict):
    query: str
    rewritten_queries: list
    attempt_count: int

class QueryRewritePipeline:
    """
    Implements a LangGraph pipeline for query rewriting with a max of 2 attempts.
    """

    def __init__(self):
        self.rewriter = QueryRewriter()
        self.workflow = StateGraph(GraphState)

        self.workflow.add_node("rewrite", self.rewrite)
        self.workflow.add_conditional_edges(
            "rewrite", self.should_rewrite, {"retry": "rewrite", "end": END}
        )

        self.workflow.set_entry_point("rewrite")
        self.app = self.workflow.compile()

    def rewrite(self, state: GraphState) -> GraphState:
        new_query = self.rewriter.rewrite_query(state["query"])
        state["rewritten_queries"].append(new_query)
        return {**state, "query": new_query, "attempt_count": state["attempt_count"] + 1}

    def should_rewrite(self, state: GraphState) -> str:
        if state["attempt_count"] >= 2:
            return "end"
        return "retry" if state["rewritten_queries"][-1] != state["query"] else "end"

    def run(self, query: str) -> str:
        inputs = {"query": query, "rewritten_queries": [], "attempt_count": 0}
        for output in self.app.stream(inputs):
            pass  
        return output["query"]
