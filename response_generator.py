import logging
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config

class GraphState(TypedDict):
    """
    Defines the state dictionary for LangGraph execution.
    """
    query: str
    relevant_docs: list
    response: str
    messages: list  # Added for consistency with pipeline

class ResponseGenerator:
    """
    Generates responses based on relevant retrieved documents.
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=Config.OPENAI_API_KEY)

        # Define response generation prompt
        # Updated prompt with graceful fallback instruction
        self.prompt = ChatPromptTemplate.from_messages([
                ("system", """
            You are a helpful and intelligent assistant. Use only the provided `context` and `conversation history` to respond to the user's query.

            Guidelines:
            - If the input is a greeting (e.g., "hi", "hello") or a pleasantry (e.g., "thank you"), respond naturally and politely. No context is required.
            - Otherwise, use logical reasoning, examples, or step-by-step breakdowns from the `context` to answer the query.
            - If the query involves a problem or a task (e.g., calculation, analysis, or code), look for related patterns or methods in the context and apply them accordingly.
            - Do **not** guess or use external knowledge not present in the provided information.
            - If no relevant information is available, respond:
            "I don't have enough information in the provided context to answer that. Could you clarify or provide more details?"

            Format output clearly and professionally. Use structured steps or bullet points if it helps improve clarity.
            """),
                ("human", "Conversation History:\n{history}\n\nQuery: {query}\n\nContext:\n{context}\n\nAnswer:")
        ])


        self.response_chain = self.prompt | self.llm | StrOutputParser()

    def generate_response(self, query, relevant_docs, messages):
        context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else "No relevant data."
        # Format the conversation history as a string
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]) if messages else "No prior conversation."
        print("----history----")
        print(history)
        return self.response_chain.invoke({"query": query, "context": context, "history": history})

class ResponseGenerationPipeline:
    """
    Implements a LangGraph pipeline for generating responses.
    """

    def __init__(self):
        self.generator = ResponseGenerator()
        self.workflow = StateGraph(GraphState)

        self.workflow.add_node("generate_response", self.generate_response)

        # Entry point
        self.workflow.set_entry_point("generate_response")

        # Compile LangGraph pipeline
        self.app = self.workflow.compile()

    def generate_response(self, state: GraphState) -> GraphState:
        # Generate response based on relevant documents and conversation history
        response = self.generator.generate_response(state["query"], state["relevant_docs"], state["messages"])
        return {**state, "response": response}

    def run(self, query, relevant_docs, messages=None):
        # Initialize state and execute pipeline
        inputs = {
            "query": query,
            "relevant_docs": relevant_docs,
            "response": "",
            "messages": messages or []
        }
        for output in self.app.stream(inputs):
            pass  # Process execution flow
        return output["response"]
