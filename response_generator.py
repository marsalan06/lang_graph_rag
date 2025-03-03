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

class ResponseGenerator:
    """
    Generates responses based on relevant retrieved documents.
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=Config.OPENAI_API_KEY)

        # Define response generation prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a **knowledgeable and engaging AI assistant** that generates responses based on provided context.
            - Use the provided context to **accurately answer** user questions.
            - Keep the tone **friendly and professional**, adjusting formality based on the topic.
            - If the context includes **mathematical equations**, **derivations**, or **formulas**, preserve their **notation and correctness**.
            - If the query involves **coding or programming**, format your response using **proper code blocks** and ensure correctness.
            - If the user greets you (e.g., "Hi", "Hello"), respond positively and engagingly.
            - If you **don’t know** the answer, reply with: "Hmm, I’m not sure about that, but I’d love to help with something else!".
            - If the query is **obscene, legal, financial, or ethical in nature**, politely decline to answer.
            - Always base your responses strictly on the provided **context** and include citations when necessary."""),
            ("human", "Query: {query}\n\nContext: {context}\n\nAnswer:")
        ])

        self.response_chain = self.prompt | self.llm | StrOutputParser()

    def generate_response(self, query, relevant_docs):
        """
        Generates the final response.
        """
        context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else "No relevant data."
        return self.response_chain.invoke({"query": query, "context": context})

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
        """
        Generates the final response based on relevant documents.
        """
        response = self.generator.generate_response(state["query"], state["relevant_docs"])
        return {**state, "response": response}

    def run(self, query, relevant_docs):
        """
        Runs the response generation pipeline.
        """
        inputs = {"query": query, "relevant_docs": relevant_docs, "response": ""}
        for output in self.app.stream(inputs):
            pass  # Process execution flow
        return output["response"]
