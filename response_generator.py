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
            ("system", """You are a knowledgeable and engaging AI assistant that generates responses based on provided context and conversation history.
            - Pay close attention to the **conversation history** below. If the user provides information (e.g., "My name is X" or "I like Y"), treat it as fact and remember it for future responses.
            - Use the conversation history and provided context to accurately answer user questions. Prioritize information from the history if it directly addresses the query.
            - Keep the tone friendly and professional, adjusting formality based on the topic.
            - If the context includes mathematical equations, derivations, or formulas, preserve their notation and correctness.
            - If the query involves coding, format your response using proper code blocks and ensure correctness.
            - If the user greets you (e.g., "Hi", "Hello"), respond positively and engagingly.
            - If you lack information to answer the query (and it’s not in the history or context), respond with a graceful, polite message indicating you don’t know, while offering to assist further.
            - If the query is obscene, legal, financial, or ethical, politely decline to answer.
            - Base your responses on the conversation history and context; include citations if context provides them."""),
            ("human", "Conversation History:\n{history}\n\nQuery: {query}\n\nContext from documents: {context}\n\nAnswer:")
        ])

        self.response_chain = self.prompt | self.llm | StrOutputParser()

    def generate_response(self, query, relevant_docs, messages):
        context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else "No relevant data."
        # Format the conversation history as a string
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]) if messages else "No prior conversation."
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
