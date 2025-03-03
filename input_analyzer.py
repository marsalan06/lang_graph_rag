import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from config import Config

class InputType(BaseModel):
    """
    LLM classification model for identifying user input type.
    """
    type: Literal["question", "pleasantry"] = Field(
        ..., description="Classify as 'question' if the input requires a factual or informational answer, otherwise 'pleasantry'."
    )

class InputAnalyzer:
    """
    Analyzes the user input to determine whether it's a query or a pleasantry.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0,
            openai_api_key=Config.OPENAI_API_KEY
        )

        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI that classifies user inputs as either:
            - 'question' if it seeks information, clarification, or problem-solving.
            - 'pleasantry' if it is a greeting, small talk, or social nicety (e.g., 'Hello', 'Hi').
            Respond strictly in JSON format with the key 'type'."""),
            ("human", "User input: {user_input}")
        ])

        self.analysis_chain = self.analysis_prompt | self.llm | JsonOutputParser()

    def analyze_input(self, user_input: str) -> str:
        """
        Determines whether input is a question or a pleasantry.
        """
        try:
            raw_response = self.analysis_chain.invoke({"user_input": user_input})

            # ✅ Explicitly parse into Pydantic model to avoid dict attribute error
            parsed_response = InputType(**raw_response)

            return parsed_response.type  # Returns 'question' or 'pleasantry'
        except Exception as e:
            logging.error(f"❌ Error analyzing input: {e}")
            return "question"  # Default to question if error occurs
