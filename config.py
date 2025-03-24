import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
    FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")
    LMS_AUTH_URL = os.getenv("LMS_AUTH_URL")
    FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY")
    FIREBASE_AUTH_DOMAIN = os.getenv("FIREBASE_AUTH_DOMAIN")
    FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")



    @staticmethod
    def validate():
        if not Config.PINECONE_API_KEY or not Config.OPENAI_API_KEY:
            raise ValueError("Missing API Keys! Please check your .env file.")

# Validate API keys on import
Config.validate()
